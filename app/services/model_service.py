from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
import json
import logging
import re
from typing import Literal

from fastapi import HTTPException
from langsmith import trace, tracing_context

from app.agent.state import AgentState, PlannedAction, UserIntent
from app.core.config import Settings
from app.core.langsmith import is_langsmith_enabled
from app.prompts.tutor_prompts import (
    build_action_instruction,
    build_action_user_prompt,
    build_routing_instruction,
    build_routing_user_prompt,
)
from app.schemas.chat import ChatRequest

from openai import OpenAI

logger = logging.getLogger(__name__)

ConfidenceLevel = Literal["high", "medium", "low"]
SourceMode = Literal["openai"]


@dataclass(frozen=True)
class RouteDecision:
    user_intent: UserIntent
    planned_action: PlannedAction
    route_reason: str
    topic_hint: str | None
    confidence: ConfidenceLevel
    source_mode: SourceMode


@dataclass(frozen=True)
class ActionResponse:
    topic: str | None
    message: str
    follow_up_question: str | None
    confidence: ConfidenceLevel
    safety_notes: str
    source_mode: SourceMode


class ModelService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._openai_call_semaphore = asyncio.Semaphore(max(1, settings.openai_max_concurrency))
        self._text_client: OpenAI | None = None
        self._vision_client: OpenAI | None = None

    async def classify_next_action(
        self,
        chat_request: ChatRequest,
        state: AgentState,
    ) -> RouteDecision:
        with tracing_context(
            enabled=is_langsmith_enabled(self.settings),
            project_name=self.settings.langsmith_project,
        ):
            with trace(
                "classify_next_action",
                run_type="chain",
                inputs={
                    "text": chat_request.text or "",
                    "has_image": bool(chat_request.image_base64 or chat_request.image_url),
                    "current_topic": state.get("current_topic"),
                    "last_agent_question": state.get("last_agent_question"),
                },
                metadata={"component": "model_service"},
            ) as run:
                async with self._openai_call_semaphore:
                    result = self._openai_route_decision(chat_request, state)
                run.end(
                    outputs={
                        "planned_action": result.planned_action,
                        "user_intent": result.user_intent,
                        "route_reason": result.route_reason,
                        "source_mode": result.source_mode,
                    }
                )
                return result

    async def generate_action_response(
        self,
        chat_request: ChatRequest,
        state: AgentState,
        action: PlannedAction,
    ) -> ActionResponse:
        with tracing_context(
            enabled=is_langsmith_enabled(self.settings),
            project_name=self.settings.langsmith_project,
        ):
            with trace(
                "generate_action_response",
                run_type="chain",
                inputs={
                    "action": action,
                    "text": chat_request.text or "",
                    "topic": state.get("current_topic") or state.get("pending_topic"),
                    "has_image": bool(chat_request.image_base64 or chat_request.image_url),
                },
                metadata={"component": "model_service"},
            ) as run:
                async with self._openai_call_semaphore:
                    result = self._openai_action_response(chat_request, state, action)
                run.end(
                    outputs={
                        "topic": result.topic,
                        "source_mode": result.source_mode,
                        "has_follow_up_question": bool(result.follow_up_question),
                    }
                )
                return result

    def _openai_route_decision(
        self,
        chat_request: ChatRequest,
        state: AgentState,
    ) -> RouteDecision:
        parsed = self._call_openai_json(
            model=self.settings.openai_planning_model,
            schema_name="xiaozhi_route_decision",
            schema={
                "type": "object",
                "properties": {
                    "user_intent": {
                        "type": "string",
                        "enum": [
                            "greeting",
                            "object_learning",
                            "direct_question",
                            "answer_attempt",
                            "unclear",
                            "fallback",
                        ],
                    },
                    "planned_action": {
                        "type": "string",
                        "enum": [
                            "greet",
                            "explain_and_ask",
                            "answer_question",
                            "evaluate_answer",
                            "clarify",
                            "fallback",
                        ],
                    },
                    "route_reason": {"type": "string"},
                    "topic_hint": {"type": "string"},
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                    },
                },
                "required": [
                    "user_intent",
                    "planned_action",
                    "route_reason",
                    "topic_hint",
                    "confidence",
                ],
                "additionalProperties": False,
            },
            instruction=build_routing_instruction(),
            prompt=build_routing_user_prompt(chat_request, state),
            chat_request=chat_request,
        )
        return RouteDecision(
            user_intent=self._normalize_user_intent(parsed.get("user_intent")),
            planned_action=self._normalize_action(parsed.get("planned_action")),
            route_reason=self._clean_text(parsed.get("route_reason"), "Route selected."),
            topic_hint=self._optional_text(parsed.get("topic_hint")),
            confidence=self._normalize_confidence(parsed.get("confidence")),
            source_mode="openai",
        )

    def _openai_action_response(
        self,
        chat_request: ChatRequest,
        state: AgentState,
        action: PlannedAction,
    ) -> ActionResponse:
        parsed = self._call_openai_json(
            model=self.settings.openai_model,
            schema_name=f"xiaozhi_action_{action}",
            schema={
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "message": {"type": "string"},
                    "follow_up_question": {"type": "string"},
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                    },
                    "safety_notes": {"type": "string"},
                },
                "required": [
                    "topic",
                    "message",
                    "follow_up_question",
                    "confidence",
                    "safety_notes",
                ],
                "additionalProperties": False,
            },
            instruction=build_action_instruction(action),
            prompt=build_action_user_prompt(chat_request, state, action),
            chat_request=chat_request,
        )
        # print(f'prompt:{build_action_user_prompt(chat_request, state, action)}')
        return ActionResponse(
            topic=self._optional_text(parsed.get("topic")) or state.get("current_topic"),
            message=self._clean_text(parsed.get("message"), "Let us learn one small thing together."),
            follow_up_question=self._optional_text(parsed.get("follow_up_question")),
            confidence=self._normalize_confidence(parsed.get("confidence")),
            safety_notes=self._clean_text(parsed.get("safety_notes"), ""),
            source_mode="openai",
        )

    def _call_openai_json(
        self,
        model: str,
        schema_name: str,
        schema: dict,
        instruction: str,
        prompt: str,
        chat_request: ChatRequest,
    ) -> dict:
        with tracing_context(
            enabled=is_langsmith_enabled(self.settings),
            project_name=self.settings.langsmith_project,
        ):
            with trace(
                "openai_responses_call",
                run_type="llm",
                inputs={
                    "schema_name": schema_name,
                    "model": model,
                    "text": chat_request.text or "",
                    "has_image": bool(chat_request.image_base64 or chat_request.image_url),
                },
                metadata={"provider": "llm.chat_completions"},
            ) as run:
                payload = self._build_openai_payload(
                    model,
                    schema_name,
                    schema,
                    instruction,
                    prompt,
                    chat_request,
                )
                try:
                    is_vision = bool(chat_request.image_base64 or chat_request.image_url)
                    client = self._get_vision_client() if is_vision else self._get_text_client()
                    model_name = "hunyuan-t1-vision-20250916" if is_vision else model
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=payload["messages"],
                        stream=False,
                        timeout=self._resolve_request_timeout(chat_request),
                    )
                    raw_body = self._extract_completion_content(
                        response.choices[0].message.content if response.choices else None
                    )
                except Exception as exc:
                    raise HTTPException(
                        status_code=502,
                        detail=f"LLM API request failed: {exc}",
                    ) from exc

                text_output = raw_body
                if not text_output:
                    raise HTTPException(status_code=502, detail="LLM API returned empty content.")
                parsed_output = self._parse_json_from_text(text_output)
                run.end(outputs=parsed_output)
                return parsed_output

    def _get_text_client(self) -> OpenAI:
        if self._text_client is None:
            self._text_client = OpenAI(
                api_key=self.settings.llm_api_key,
                base_url=self.settings.llm_base_url,
            )
        return self._text_client

    def _get_vision_client(self) -> OpenAI:
        if self._vision_client is None:
            self._vision_client = OpenAI(
                api_key=self.settings.vllm_api_key,
                base_url="https://api.hunyuan.cloud.tencent.com/v1",
            )
        return self._vision_client

    def _build_openai_payload(
        self,
        model: str,
        schema_name: str,
        schema: dict,
        instruction: str,
        prompt: str,
        chat_request: ChatRequest,
    ) -> dict:
        content: list[dict[str, object]] = [{"type": "text", "text": prompt}]
        if chat_request.image_base64:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{chat_request.image_mime_type};base64,{chat_request.image_base64}",
                        "detail": "auto",
                    },
                }
            )
        elif chat_request.image_url:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": str(chat_request.image_url), "detail": "auto"},
                }
            )
        return {
            "model": model,
            "messages": [
                {"role": "system", "content": self._build_schema_instruction(instruction, schema)},
                {"role": "user", "content": content},
            ],
        }

    def _extract_completion_content(self, content: object) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            return "\n".join(parts).strip()
        return ""

    def _build_schema_instruction(self, instruction: str, schema: dict) -> str:
        schema_json = json.dumps(schema, ensure_ascii=False)
        return (
            f"{instruction}\n\n"
            "Output requirements:\n"
            "1. Reply with a single JSON object only.\n"
            "2. Do not include markdown, code fences, or extra commentary.\n"
            f"3. Follow this JSON schema exactly: {schema_json}"
        )

    def _parse_json_from_text(self, text: str) -> dict:
        raw = text.strip()
        if not raw:
            raise HTTPException(status_code=502, detail="LLM API returned empty text content.")

        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        fence_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", raw, flags=re.IGNORECASE)
        if fence_match:
            candidate = fence_match.group(1).strip()
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = raw[start : end + 1]
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

        raise HTTPException(
            status_code=502,
            detail=f"LLM API returned non-JSON content: {raw[:300]}",
        )

    def _resolve_request_timeout(self, chat_request: ChatRequest) -> int:
        if chat_request.image_base64:
            return self.settings.openai_image_request_timeout_seconds
        return self.settings.request_timeout_seconds

    def _clean_text(self, value: object, fallback: str) -> str:
        if not isinstance(value, str):
            return fallback
        cleaned = " ".join(value.strip().split())
        return cleaned or fallback

    def _optional_text(self, value: object) -> str | None:
        if not isinstance(value, str):
            return None
        cleaned = " ".join(value.strip().split())
        return cleaned or None

    def _normalize_confidence(self, value: object) -> ConfidenceLevel:
        return value if isinstance(value, str) and value in {"high", "medium", "low"} else "medium"

    def _normalize_user_intent(self, value: object) -> UserIntent:
        return value if isinstance(value, str) and value in {"greeting", "object_learning", "direct_question", "answer_attempt", "unclear", "fallback"} else "fallback"

    def _normalize_action(self, value: object) -> PlannedAction:
        return value if isinstance(value, str) and value in {"greet", "explain_and_ask", "answer_question", "evaluate_answer", "clarify", "fallback"} else "fallback"

    def decode_image_for_debug(self, image_base64: str) -> bytes:
        return base64.b64decode(image_base64, validate=True)
