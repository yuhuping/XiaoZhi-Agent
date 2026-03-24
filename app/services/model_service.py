from __future__ import annotations

from dataclasses import dataclass
import base64
import json
import logging
from typing import Literal
from urllib import error, request

from fastapi import HTTPException

from app.core.config import Settings
from app.prompts.tutor_prompts import (
    build_child_tutor_instruction,
    build_user_prompt,
)
from app.schemas.chat import ChatRequest

logger = logging.getLogger(__name__)

ConfidenceLevel = Literal["high", "medium", "low"]
SourceMode = Literal["mock", "openai"]


@dataclass(frozen=True)
class ModelOutput:
    topic: str
    explanation: str
    follow_up_question: str
    confidence: ConfidenceLevel
    safety_notes: str
    source_mode: SourceMode


class ModelService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def generate_child_response(self, chat_request: ChatRequest) -> ModelOutput:
        if self._should_use_mock():
            return self._mock_response(chat_request)

        try:
            return self._openai_response(chat_request)
        except Exception as exc:
            logger.exception("openai call failed, falling back to mock mode: %s", exc)
            return self._mock_response(chat_request)

    def _should_use_mock(self) -> bool:
        return self.settings.mock_mode or not self.settings.openai_api_key

    def _openai_response(self, chat_request: ChatRequest) -> ModelOutput:
        payload = self._build_openai_payload(chat_request)
        payload_bytes = json.dumps(payload).encode("utf-8")
        endpoint = f"{self.settings.openai_api_base}/responses"
        http_request = request.Request(
            url=endpoint,
            data=payload_bytes,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.settings.openai_api_key or ''}",
            },
            method="POST",
        )

        try:
            with request.urlopen(
                http_request,
                timeout=self.settings.request_timeout_seconds,
            ) as response:
                raw_body = response.read().decode("utf-8")
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="ignore")
            raise HTTPException(
                status_code=502,
                detail=f"OpenAI API request failed: {details or exc.reason}",
            ) from exc
        except error.URLError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"OpenAI API connection failed: {exc.reason}",
            ) from exc

        return self._parse_openai_response(raw_body)

    def _build_openai_payload(self, chat_request: ChatRequest) -> dict:
        content: list[dict[str, object]] = [
            {
                "type": "input_text",
                "text": build_user_prompt(
                    chat_request.text,
                    bool(chat_request.image_base64),
                    chat_request.age_hint,
                ),
            }
        ]
        if chat_request.image_base64:
            content.append(
                {
                    "type": "input_image",
                    "image_url": (
                        f"data:{chat_request.image_mime_type};base64,"
                        f"{chat_request.image_base64}"
                    ),
                    "detail": "auto",
                }
            )
        elif chat_request.image_url:
            content.append(
                {
                    "type": "input_image",
                    "image_url": str(chat_request.image_url),
                    "detail": "auto",
                }
            )

        return {
            "model": self.settings.openai_model,
            "input": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": build_child_tutor_instruction(),
                        }
                    ],
                },
                {"role": "user", "content": content},
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "child_explain_and_ask",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "topic": {"type": "string"},
                            "explanation": {"type": "string"},
                            "follow_up_question": {"type": "string"},
                            "confidence": {
                                "type": "string",
                                "enum": ["high", "medium", "low"],
                            },
                            "safety_notes": {"type": "string"},
                        },
                        "required": [
                            "topic",
                            "explanation",
                            "follow_up_question",
                            "confidence",
                            "safety_notes",
                        ],
                        "additionalProperties": False,
                    },
                }
            },
        }

    def _parse_openai_response(self, raw_body: str) -> ModelOutput:
        parsed = json.loads(raw_body)
        text_output = self._extract_openai_text(parsed)
        if not text_output:
            raise HTTPException(status_code=502, detail="OpenAI API returned empty content.")

        structured = json.loads(text_output)
        return ModelOutput(
            topic=self._clean_text(structured.get("topic"), fallback="topic"),
            explanation=self._clean_text(
                structured.get("explanation"),
                fallback="This is something interesting to learn about.",
            ),
            follow_up_question=self._ensure_question(
                self._clean_text(
                    structured.get("follow_up_question"),
                    fallback="What do you notice about it?",
                )
            ),
            confidence=self._normalize_confidence(structured.get("confidence")),
            safety_notes=self._clean_text(structured.get("safety_notes"), fallback=""),
            source_mode="openai",
        )

    def _extract_openai_text(self, parsed: dict) -> str:
        output_text = parsed.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        output_items = parsed.get("output")
        if not isinstance(output_items, list):
            return ""

        text_chunks: list[str] = []
        for item in output_items:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "message":
                continue
            content_items = item.get("content")
            if not isinstance(content_items, list):
                continue
            for content in content_items:
                if not isinstance(content, dict):
                    continue
                content_type = content.get("type")
                if content_type == "output_text":
                    text_value = content.get("text")
                    if isinstance(text_value, str) and text_value.strip():
                        text_chunks.append(text_value.strip())
                elif content_type == "refusal":
                    refusal = content.get("refusal")
                    if isinstance(refusal, str) and refusal.strip():
                        raise HTTPException(
                            status_code=502,
                            detail=f"OpenAI API refusal: {refusal.strip()}",
                        )
        return "\n".join(text_chunks).strip()

    def _mock_response(self, chat_request: ChatRequest) -> ModelOutput:
        if self._looks_unsafe(chat_request.text):
            return ModelOutput(
                topic="safe learning topic",
                explanation=(
                    "We should talk about safe and simple learning ideas. "
                    "We can learn about animals, colors, or fruit instead."
                ),
                follow_up_question="Would you like to talk about a cat or an apple?",
                confidence="high",
                safety_notes="Redirected away from unsafe or adult content.",
                source_mode="mock",
            )

        topic = self._detect_topic(chat_request)
        explanation = self._generate_explanation(topic, used_image=bool(chat_request.image_base64))
        follow_up = self._generate_follow_up_question(topic)
        confidence: ConfidenceLevel = "medium" if chat_request.image_base64 else "high"
        return ModelOutput(
            topic=topic,
            explanation=explanation,
            follow_up_question=follow_up,
            confidence=confidence,
            safety_notes="",
            source_mode="mock",
        )

    def _detect_topic(self, chat_request: ChatRequest) -> str:
        text = (chat_request.text or "").lower()
        keyword_map = {
            "apple": "apple",
            "cat": "cat",
            "dog": "dog",
            "car": "car",
            "bus": "bus",
            "banana": "banana",
            "moon": "moon",
            "star": "star",
            "flower": "flower",
            "tree": "tree",
        }
        for keyword, topic in keyword_map.items():
            if keyword in text:
                return topic
        if chat_request.image_base64 or chat_request.image_url:
            return "picture"
        return "topic"

    def _generate_explanation(self, topic: str, used_image: bool) -> str:
        explanations = {
            "apple": "An apple is a fruit. It is often round, crunchy, and sweet.",
            "cat": "A cat is a small animal with soft fur. Many cats like to jump and explore.",
            "dog": "A dog is an animal that can run, play, and learn from people.",
            "car": "A car is a vehicle that helps people travel from one place to another.",
            "bus": "A bus is a big vehicle that can carry many people together.",
            "banana": "A banana is a fruit with a soft inside and a yellow peel.",
            "moon": "The moon is the bright object we often see in the night sky.",
            "star": "A star is a shining light in the sky that is very far away.",
            "flower": "A flower is part of a plant and can have many bright colors.",
            "tree": "A tree is a tall plant with a trunk, branches, and leaves.",
            "picture": "This looks like something we can learn from together. We can talk about what stands out in the image.",
            "topic": "This is something fun to learn about. We can describe it in a simple way.",
        }
        explanation = explanations.get(topic, explanations["topic"])
        if used_image and topic != "picture":
            return f"I think this image shows a {topic}. {explanation}"
        return explanation

    def _generate_follow_up_question(self, topic: str) -> str:
        questions = {
            "apple": "What color do you think an apple can be?",
            "cat": "What sound does a cat make?",
            "dog": "Can you name something a dog likes to do?",
            "car": "What sound do you think a car makes?",
            "bus": "Have you seen a bus on the road before?",
            "banana": "What color is a banana when it is ripe?",
            "moon": "Do you see the moon in the day or at night most often?",
            "star": "Can you point to the sky where stars appear at night?",
            "flower": "What flower color do you like most?",
            "tree": "What grows on some trees?",
            "picture": "What is the first thing you notice in the picture?",
            "topic": "What do you notice about it?",
        }
        return questions.get(topic, questions["topic"])

    def _looks_unsafe(self, text: str | None) -> bool:
        if not text:
            return False
        lowered = text.lower()
        unsafe_keywords = {
            "kill",
            "weapon",
            "sex",
            "nude",
            "suicide",
            "blood",
            "hurt someone",
        }
        return any(keyword in lowered for keyword in unsafe_keywords)

    def _clean_text(self, value: object, fallback: str) -> str:
        if not isinstance(value, str):
            return fallback
        cleaned = " ".join(value.strip().split())
        return cleaned or fallback

    def _normalize_confidence(self, value: object) -> ConfidenceLevel:
        if isinstance(value, str) and value in {"high", "medium", "low"}:
            return value
        return "medium"

    def _ensure_question(self, value: str) -> str:
        return value if value.endswith("?") else f"{value}?"

    def decode_image_for_debug(self, image_base64: str) -> bytes:
        return base64.b64decode(image_base64, validate=True)
