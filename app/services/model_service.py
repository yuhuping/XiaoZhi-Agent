from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
from datetime import datetime
import inspect
import json
import logging
from pathlib import Path
import tempfile
from typing import Any, Awaitable, Callable, Literal, TypeVar
from uuid import uuid4

from fastapi import HTTPException
from langsmith import trace, tracing_context
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from app.agent.state import AgentState
from app.core.config import Settings
from app.core.langsmith import is_langsmith_enabled
from app.prompts.plan_prompts import (
    build_execute_instruction,
    build_execute_user_prompt,
    build_plan_instruction,
    build_plan_user_prompt,
)
from app.prompts.tutor_prompts import (
    build_reason_instruction,
    build_reason_user_prompt,
    build_response_instruction,
    build_response_user_prompt,
)
from app.schemas.chat import ActType, ChatRequest, ConfidenceLevel
from app.schemas.llm_outputs import EpisodicSummary, PlanResult, ReactResponse, TopicSummary

_T = TypeVar("_T", bound=BaseModel)

logger = logging.getLogger(__name__)

DeltaCallback = Callable[[str], Awaitable[None] | None]


@dataclass(frozen=True)
class ReasonDecision:
    decision: str
    selected_act: ActType
    tool_name: str | None
    tool_input: dict[str, Any]
    route_reason: str
    topic_hint: str | None
    confidence: ConfidenceLevel


@dataclass(frozen=True)
class ResponseDraft:
    topic: str | None
    message: str
    follow_up_question: str | None
    confidence: ConfidenceLevel
    safety_notes: str


class ModelService:
    def __init__(self, settings: Settings, skill_registry=None) -> None:
        self.settings = settings
        self.skill_registry = skill_registry
        self._openai_call_semaphore = asyncio.Semaphore(max(1, settings.openai_max_concurrency))
        self._planning_text_llm: ChatOpenAI | None = None
        self._planning_vision_llm: ChatOpenAI | None = None

    async def reason_next_action(
        self,
        chat_request: ChatRequest,
        state: AgentState,
        tools: list[StructuredTool] | None = None,
    ) -> ReasonDecision:
        with tracing_context(
            enabled=is_langsmith_enabled(self.settings),
            project_name=self.settings.langsmith_project,
        ):
            with trace(
                "reason_next_action",
                run_type="chain",
                inputs={
                    "text": chat_request.text or "",
                    "mode": chat_request.mode,
                    "has_image": bool(chat_request.image_base64 or chat_request.image_url),
                    "current_topic": state.get("current_topic"),
                },
                metadata={"component": "model_service"},
            ) as run:
                reason_instruction = build_reason_instruction(chat_request.mode)
                reason_prompt = build_reason_user_prompt(chat_request, state)
                tool_names = [tool.name for tool in tools or []]
                logger.info(
                    "[react_reason_prompt] mode=%s tool_names=%s instruction_chars=%s prompt_chars=%s",
                    chat_request.mode,
                    tool_names,
                    len(reason_instruction),
                    len(reason_prompt),
                )
                self._dump_debug_json(
                    "react_reason_prompt",
                    {
                        "mode": chat_request.mode,
                        "instruction": reason_instruction,
                        "prompt": reason_prompt,
                        "tool_names": tool_names,
                    },
                )
                async with self._openai_call_semaphore:
                    ai_message = await self._call_llm_with_tools(
                        model=self.settings.llm_model,
                        instruction=reason_instruction,
                        prompt=reason_prompt,
                        chat_request=chat_request,
                        tools=tools or [],
                    )
                tool_name, tool_input = self._extract_first_tool_call(ai_message)
                selected_act = self._select_act_from_tool(tool_name)
                decision_text = self._extract_ai_text(ai_message)
                route_reason = self._build_route_reason(selected_act, tool_name, decision_text)
                logger.info(
                    "[react_reason_result] selected_act=%s tool_name=%r tool_input=%s decision=%r route_reason=%r",
                    selected_act,
                    tool_name,
                    self._preview_text(json.dumps(tool_input, ensure_ascii=False, default=str), limit=220),
                    self._preview_text(decision_text, limit=180),
                    route_reason,
                )
                decision = ReasonDecision(
                    decision=decision_text,
                    selected_act=selected_act,
                    tool_name=tool_name,
                    tool_input=tool_input,
                    route_reason=route_reason,
                    topic_hint=self._infer_topic_hint(chat_request, state),
                    confidence="medium",
                )
                run.end(
                    outputs={
                        "selected_act": decision.selected_act,
                        "tool_name": decision.tool_name,
                        "route_reason": decision.route_reason,
                    }
                )
                return decision

    async def generate_final_response(
        self,
        chat_request: ChatRequest,
        state: AgentState,
        on_delta: DeltaCallback | None = None,
    ) -> ResponseDraft:
        with tracing_context(
            enabled=is_langsmith_enabled(self.settings),
            project_name=self.settings.langsmith_project,
        ):
            with trace(
                "generate_final_response",
                run_type="chain",
                inputs={
                    "mode": chat_request.mode,
                    "selected_act": state.get("selected_act"),
                    "topic": state.get("current_topic"),
                    "has_rag": bool(state.get("retrieved_chunks")),
                },
                metadata={"component": "model_service"},
            ) as run:
                async with self._openai_call_semaphore:
                    if on_delta is None:
                        result = self._call_llm_json(
                            model_class=ReactResponse,
                            instruction=build_response_instruction(chat_request.mode),
                            prompt=build_response_user_prompt(chat_request, state),
                            chat_request=chat_request,
                        )
                        response = ResponseDraft(
                            topic=self._optional_text(result.topic) or state.get("current_topic"),
                            message=self._clean_text(result.message, "Let us learn one small thing together."),
                            follow_up_question=self._optional_text(result.follow_up_question),
                            confidence=self._normalize_confidence(result.confidence),
                            safety_notes=self._clean_text(result.safety_notes, ""),
                        )
                    else:
                        streamed_message = await self._stream_final_response_text(
                            chat_request=chat_request,
                            state=state,
                            on_delta=on_delta,
                        )
                        response = ResponseDraft(
                            topic=state.get("current_topic"),
                            message=self._clean_text(streamed_message, "Let us learn one small thing together."),
                            follow_up_question=None,
                            confidence="medium",
                            safety_notes="",
                        )
                run.end(
                    outputs={
                        "topic": response.topic,
                        "confidence": response.confidence,
                        "has_follow_up_question": bool(response.follow_up_question),
                    }
                )
                return response

    async def generate_plan(self, chat_request: ChatRequest, state: AgentState) -> dict[str, Any]:
        """Generate a plan for education mode. Returns {"steps": [...], "needs_retrieval": bool, "retrieval_query": str}."""
        with tracing_context(
            enabled=is_langsmith_enabled(self.settings),
            project_name=self.settings.langsmith_project,
        ):
            with trace(
                "generate_plan",
                run_type="chain",
                inputs={"text": chat_request.text or "", "mode": chat_request.mode},
                metadata={"component": "model_service"},
            ) as run:
                async with self._openai_call_semaphore:
                    try:
                        plan = self._call_llm_json(
                            model_class=PlanResult,
                            instruction=build_plan_instruction(),
                            prompt=build_plan_user_prompt(chat_request, state),
                            chat_request=chat_request,
                        )
                    except Exception as exc:
                        logger.warning("generate_plan: structured output failed, falling back to single step: %s", exc)
                        plan = PlanResult()
                    steps = plan.steps[:5] or ["直接回答用户问题"]
                    result = {
                        "steps": steps,
                        "needs_retrieval": plan.needs_retrieval,
                        "retrieval_query": plan.retrieval_query,
                    }
                    run.end(outputs={"step_count": len(result["steps"]), "needs_retrieval": result["needs_retrieval"]})
                    return result

    async def execute_plan(
        self,
        chat_request: ChatRequest,
        state: AgentState,
        on_delta: DeltaCallback | None = None,
    ) -> str:
        """Execute a plan by streaming the solution. Returns the full text."""
        instruction = build_execute_instruction()
        prompt = build_execute_user_prompt(chat_request, state)
        human_content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        image_content = self._build_image_content_from_request_or_history(chat_request, state)
        if image_content is not None:
            human_content.append(image_content)
            llm = self._get_planning_vision_llm()
        else:
            llm = self._get_planning_text_llm()

        fragments: list[str] = []
        async for chunk in llm.astream(
            [
                SystemMessage(content=instruction),
                HumanMessage(content=human_content),
            ]
        ):
            delta = self._extract_stream_chunk_text(chunk.content)
            if not delta:
                continue
            fragments.append(delta)
            if on_delta:
                emitted = on_delta(delta)
                if inspect.isawaitable(emitted):
                    await emitted
        return "".join(fragments).strip()

    async def _stream_final_response_text(
        self,
        chat_request: ChatRequest,
        state: AgentState,
        on_delta: DeltaCallback,
    ) -> str:
        # Skill 专属 prompt 路径：skill 自己提供 instruction 和 user prompt
        if state.get("selected_act") == "skill" and self.skill_registry:
            skill = self.skill_registry.find_skill_by_tool_name(state.get("selected_tool"))
            if skill:
                skill_instruction = skill.get_response_instruction()
                skill_prompt = skill.get_response_user_prompt(state)
                instruction = skill_instruction or (
                    f"{build_response_instruction(chat_request.mode)}\n"
                    "Output plain text only. Do not output JSON, markdown, or code fences."
                )
                prompt = skill_prompt or build_response_user_prompt(
                    chat_request=chat_request,
                    state=state,
                )
                # logger.info(
                #     "[skill_respond] tool=%r instruction=%r prompt_preview=%r",
                #     state.get("selected_tool"),
                #     instruction[:120],
                #     prompt[:300],
                # )
            else:
                instruction = (
                    f"{build_response_instruction(chat_request.mode)}\n"
                    "Output plain text only. Do not output JSON, markdown, or code fences."
                )
                prompt = build_response_user_prompt(
                    chat_request=chat_request,
                    state=state,
                )
        else:
            instruction = (
                f"{build_response_instruction(chat_request.mode)}\n"
                "Output plain text only. Do not output JSON, markdown, or code fences."
            )
            prompt = build_response_user_prompt(
                chat_request=chat_request,
                state=state,
            )
        logger.info(
            "[react_response_prompt] selected_act=%s selected_tool=%r observation=%r",
            state.get("selected_act"),
            state.get("selected_tool"),
            self._preview_text(str(state.get("observation_summary") or ""), limit=120),
            # self._preview_text(instruction, limit=180),
            # self._preview_text(prompt, limit=320),
        )
        self._dump_debug_json(
            "react_response_prompt",
            {
                "mode": chat_request.mode,
                "selected_act": state.get("selected_act"),
                "selected_tool": state.get("selected_tool"),
                "instruction": instruction,
                "prompt": prompt,
                "observation_summary": state.get("observation_summary"),
            },
        )
        human_content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        image_content = self._build_image_content_from_request_or_history(chat_request, state)
        if image_content is not None:
            human_content.append(image_content)
            llm = self._get_planning_vision_llm()
        else:
            llm = self._get_planning_text_llm()

        fragments: list[str] = []
        try:
            async for chunk in llm.astream(
                [
                    SystemMessage(content=instruction),
                    HumanMessage(content=human_content),
                ]
            ):
                delta = self._extract_stream_chunk_text(chunk.content)
                if not delta:
                    continue
                fragments.append(delta)
                emitted = on_delta(delta)
                if inspect.isawaitable(emitted):
                    await emitted
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"LLM stream request failed: {exc}",
            ) from exc
        return "".join(fragments).strip()

    def _build_image_content_from_request_or_history(
        self,
        chat_request: ChatRequest,
        state: AgentState,
    ) -> dict[str, Any] | None:
        if chat_request.image_base64:
            return {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{chat_request.image_mime_type};base64,{chat_request.image_base64}",
                    "detail": "auto",
                },
            }
        if chat_request.image_url:
            return {
                "type": "image_url",
                "image_url": {"url": str(chat_request.image_url), "detail": "auto"},
            }

        history = state.get("history", [])
        if not isinstance(history, list):
            return None
        for turn in reversed(history):
            if not isinstance(turn, dict):
                continue
            image_base64 = turn.get("image_base64")
            image_url = turn.get("image_url")
            image_mime_type = turn.get("image_mime_type")
            if isinstance(image_base64, str) and image_base64.strip():
                if not isinstance(image_mime_type, str) or not image_mime_type.strip():
                    continue
                return {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{image_mime_type};base64,{image_base64}",
                        "detail": "auto",
                    },
                }
            if isinstance(image_url, str) and image_url.strip():
                return {
                    "type": "image_url",
                    "image_url": {"url": image_url, "detail": "auto"},
                }
        return None

    async def _call_llm_with_tools(
        self,
        model: str,
        instruction: str,
        prompt: str,
        chat_request: ChatRequest,
        tools: list[StructuredTool],
    ) -> AIMessage:
        with tracing_context(
            enabled=is_langsmith_enabled(self.settings),
            project_name=self.settings.langsmith_project,
        ):
            with trace(
                "llm_planning_bind_tools",
                run_type="llm",
                inputs={
                    "model": model,
                    "text": chat_request.text or "",
                    "has_image": bool(chat_request.image_base64 or chat_request.image_url),
                    "tool_count": len(tools),
                },
                metadata={"provider": "llm.bind_tools"},
            ) as run:
                human_content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
                if chat_request.image_base64:
                    human_content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": (
                                    f"data:{chat_request.image_mime_type};"
                                    f"base64,{chat_request.image_base64}"
                                ),
                                "detail": "auto",
                            },
                        }
                    )
                elif chat_request.image_url:
                    human_content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": str(chat_request.image_url), "detail": "auto"},
                        }
                    )
                if chat_request.image_url or chat_request.image_base64:
                    base_model = self._get_planning_vision_llm()
                else:
                    base_model = self._get_planning_text_llm()
                runnable = base_model.bind_tools(tools) if tools else base_model
                try:
                    # 中文注释: 将 bind_tools 输入写入临时 JSON，避免终端长文本难读。  
                    self._dump_debug_json(
                        "llm_bind_tools_input",
                        {
                            "model": model,
                            "instruction": instruction,
                            "human_content": human_content,
                            "tool_names": [tool.name for tool in tools],
                        },
                    )

                    response = await runnable.ainvoke(
                        [
                            SystemMessage(content=instruction),
                            HumanMessage(content=human_content),
                        ]
                    )
                    
                except Exception as exc:
                    raise HTTPException(
                        status_code=502,
                        detail=f"LLM bind_tools request failed: {exc}",
                    ) from exc

                if not isinstance(response, AIMessage):
                    raise HTTPException(
                        status_code=502,
                        detail="LLM bind_tools returned non-AIMessage response.",
                    )
                run.end(
                    outputs={
                        "tool_calls": response.tool_calls,
                        "content": self._extract_ai_text(response),
                    }
                )
                return response

    def _call_llm_json(
        self,
        model_class: type[_T],
        instruction: str,
        prompt: str,
        chat_request: ChatRequest,
    ) -> _T:
        is_vision = bool(chat_request.image_base64 or chat_request.image_url)
        llm = self._get_planning_vision_llm() if is_vision else self._get_planning_text_llm()
        # json_mode: Dashscope 要求 messages 里含 "json" 一词；在 SystemMessage 末尾附加提示即可满足。
        structured_llm = llm.with_structured_output(model_class, method="json_mode")

        user_content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if chat_request.image_base64:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{chat_request.image_mime_type};base64,{chat_request.image_base64}",
                        "detail": "auto",
                    },
                }
            )
        elif chat_request.image_url:
            user_content.append(
                {"type": "image_url", "image_url": {"url": str(chat_request.image_url), "detail": "auto"}}
            )

        messages = [SystemMessage(content=f"{instruction}\nRespond with JSON."), HumanMessage(content=user_content)]
        # 中文注释: 将 structured_output 调用输入写入临时 JSON，终端只看路径。
        self._dump_debug_json(
            "llm_chat_completions_payload",
            {
                "model_class": model_class.__name__,
                "instruction": instruction,
                "prompt": prompt,
                "has_image": is_vision,
            },
        )
        return structured_llm.invoke(messages)

    def _get_planning_text_llm(self) -> ChatOpenAI:
        if self._planning_text_llm is None:
            self._planning_text_llm = ChatOpenAI(
                api_key=self.settings.llm_api_key,
                base_url=self.settings.llm_base_url,
                model=self.settings.llm_model,
                timeout=self.settings.request_timeout_seconds,
            )
        return self._planning_text_llm
    
    def _get_planning_vision_llm(self) -> ChatOpenAI:
        if self._planning_vision_llm is None:
            self._planning_vision_llm = ChatOpenAI(
                api_key=self.settings.vllm_api_key,
                base_url=self.settings.vllm_base_url,
                model=self.settings.vllm_model,
                timeout=self.settings.request_timeout_seconds,
            )
        return self._planning_vision_llm

    def _extract_stream_chunk_text(self, content: object) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item, str):
                    parts.append(item)
            return "".join(parts)
        return ""

    def _resolve_request_timeout(self, chat_request: ChatRequest) -> int:
        if chat_request.image_base64:
            return self.settings.openai_image_request_timeout_seconds
        return self.settings.request_timeout_seconds

    def _extract_ai_text(self, message: AIMessage) -> str:
        content = message.content
        if isinstance(content, str):
            return self._clean_text(content, "respond_directly")
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            if parts:
                return self._clean_text(" ".join(parts), "respond_directly")
        if message.tool_calls:
            return "use_tool"
        return "respond_directly"

    def _preview_text(self, value: object, limit: int = 160) -> str:
        if not isinstance(value, str):
            return "<non_text>"
        compact = " ".join(value.strip().split())
        if len(compact) <= limit:
            return compact or "<empty>"
        return f"{compact[:limit]}..."

    def _extract_first_tool_call(self, message: AIMessage) -> tuple[str | None, dict[str, Any]]:
        tool_calls = message.tool_calls or []
        if not tool_calls:
            return None, {}
        first_call = tool_calls[0]
        raw_name = first_call.get("name")
        tool_name = raw_name if isinstance(raw_name, str) and raw_name.strip() else None
        raw_args = first_call.get("args")
        tool_args = raw_args if isinstance(raw_args, dict) else {}
        return tool_name, tool_args

    def _select_act_from_tool(self, tool_name: str | None) -> ActType:
        if tool_name == "retrieve_knowledge":
            logger.info("[reason_route] tool=%r -> act=retrieve_knowledge (local_rag)", tool_name)
            return "retrieve_knowledge"
        if tool_name == "tavily_search":
            logger.info("[reason_route] tool=%r -> act=tavily_search (web_search)", tool_name)
            return "tavily_search"
        if tool_name == "read_memory_bundle":
            logger.info("[reason_route] tool=%r -> act=read_memory", tool_name)
            return "read_memory"
        if tool_name is not None:
            logger.info("[reason_route] tool=%r -> act=skill", tool_name)
            return "skill"
        logger.info("[reason_route] tool=None -> act=direct")
        return "direct"

    def _build_route_reason(
        self,
        selected_act: ActType,
        tool_name: str | None,
        decision_text: str,
    ) -> str:
        if selected_act == "direct":
            return "model selected direct response without tool call"
        if selected_act == "read_memory":
            return "model selected memory read before final response"
        if tool_name:
            return f"model selected tool call via bind_tools ({tool_name})"
        return self._clean_text(decision_text, "model selected tool call")

    def _infer_topic_hint(self, chat_request: ChatRequest, state: AgentState) -> str | None:
        for candidate in (
            state.get("detected_object"),
            state.get("topic_hint"),
            state.get("current_topic"),
        ):
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        text = (chat_request.text or "").strip()
        if not text:
            return None
        return text.split()[0][:32]

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
        if isinstance(value, str) and value in {"high", "medium", "low"}:
            return value
        return "medium"

    async def summarize_episodic_batch(
        self,
        events: list[dict[str, Any]],
        mode: str,
    ) -> dict[str, Any] | None:
        """压缩情景记忆：使用LLM将10条事件总结为1条。"""
        normalized: list[dict[str, str]] = []
        for item in events:
            if not isinstance(item, dict):
                continue
            text = self._clean_text(item.get("text"), "")
            if not text:
                continue
            normalized.append(
                {
                    "role": self._clean_text(item.get("role"), "unknown"),
                    "topic": self._clean_text(item.get("topic"), ""),
                    "mode": self._clean_text(item.get("mode"), mode),
                    "ts": self._clean_text(item.get("ts"), ""),
                    "text": text[:260],
                }
            )
        if len(normalized) < 2:
            return None

        mode_value: Literal["education", "companion", "parent"]
        if mode in {"education", "companion", "parent"}:
            mode_value = mode
        else:
            mode_value = "education"

        prompt_lines = [
            (
                f"- ts={item['ts']} role={item['role']} topic={item['topic']} "
                f"mode={item['mode']} text={item['text']}"
            )
            for item in normalized
        ]
        prompt = (
            "Compress the episodic memory batch into one concise memory item for a child-learning agent.\n"
            "Keep core facts, progress clues, and Use Chinese.\n"
            + "\n".join(prompt_lines)
        )

        try:
            with tracing_context(
                enabled=is_langsmith_enabled(self.settings),
                project_name=self.settings.langsmith_project,
            ):
                with trace(
                    "summarize_episodic_batch",
                    run_type="chain",
                    inputs={"event_count": len(normalized), "mode": mode_value},
                    metadata={"component": "model_service"},
                ) as run:
                    async with self._openai_call_semaphore:
                        ep = self._call_llm_json(
                            model_class=EpisodicSummary,
                            instruction=(
                                "You are a memory compressor. "
                                "Summarize a batch of tutoring events into one episodic memory."
                            ),
                            prompt=prompt,
                            chat_request=ChatRequest(text="compress episodic memory batch", mode=mode_value),
                        )
                    summary = self._clean_text(ep.summary, "")
                    if not summary:
                        return None
                    key_points: list[str] = []
                    for point in ep.key_points:
                        text = self._clean_text(point, "")
                        if text and text not in key_points:
                            key_points.append(text)
                    result = {
                        "summary": summary[:520],
                        "topic_hint": self._optional_text(ep.topic_hint),
                        "key_points": key_points[:8],
                    }
                    run.end(outputs=result)
                    return result
        except Exception as exc:
            logger.warning("episodic llm compact failed: %s", exc)
            return None

    async def summarize_topic_history(self, topic_events: list[dict[str, Any]], mode: str) -> str:
        normalized = self._normalize_topic_events(topic_events)
        if not normalized:
            return ""

        mode_value: Literal["education", "companion", "parent"]
        if mode in {"education", "companion", "parent"}:
            mode_value = mode
        else:
            mode_value = "education"

        prompt_lines = [f"- topic={item['topic']} mode={item['mode']} ts={item['ts']}" for item in normalized]
        prompt = (
            "You are a Personal Information Organizer, specialized in accurately storing facts, user memories, and preferences. Your primary role is to extract relevant pieces of information from conversations and organize them into distinct, manageable facts. This allows for easy retrieval and personalization in future interactions.\n"
            + "\n".join(prompt_lines)
        )
        try:
            ts = self._call_llm_json(
                model_class=TopicSummary,
                instruction="You are a memory compressor for child tutoring sessions.",
                prompt=prompt,
                chat_request=ChatRequest(text="compress memory history", mode=mode_value),
            )
            summary = self._clean_text(ts.summary, "")
            if summary:
                return summary[:220]
        except Exception as exc:
            logger.warning("memory summary via llm failed: %s", exc)
        return self._fallback_topic_summary(normalized)

    def _normalize_topic_events(self, topic_events: list[dict[str, Any]]) -> list[dict[str, str]]:
        normalized: list[dict[str, str]] = []
        for item in topic_events:
            if not isinstance(item, dict):
                continue
            topic = item.get("topic")
            if not isinstance(topic, str) or not topic.strip():
                continue
            normalized.append(
                {
                    "topic": topic.strip(),
                    "mode": str(item.get("mode") or "").strip(),
                    "ts": str(item.get("ts") or "").strip(),
                }
            )
        return normalized

    def _fallback_topic_summary(self, topic_events: list[dict[str, str]]) -> str:
        if not topic_events:
            return ""
        ordered_topics: list[str] = []
        for item in topic_events:
            topic = item["topic"]
            if topic not in ordered_topics:
                ordered_topics.append(topic)
        top_topics = ", ".join(ordered_topics[:4])
        return f"Recent long-term interests include: {top_topics}."

    def _dump_debug_json(self, tag: str, data: dict[str, Any]) -> None:
        # 中文注释: 调试输出目录，使用系统临时目录避免污染项目文件。
        debug_dir = Path(tempfile.gettempdir()) / "xiaozhi_llm_debug"
        # 中文注释: 文件名用时间戳 + 随机后缀，避免并发覆盖。
        file_name = f"{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{uuid4().hex[:8]}.json"
        file_path = debug_dir / file_name
        try:
            debug_dir.mkdir(parents=True, exist_ok=True)
            file_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            logger.info("LLM debug input saved: %s", file_path)
        except Exception as exc:
            logger.warning("write llm debug json failed: %s", exc)

    def decode_image_for_debug(self, image_base64: str) -> bytes:
        return base64.b64decode(image_base64, validate=True)
