from __future__ import annotations

import re

from app.agent.state import AgentState, PlannedAction, state_to_request
from app.services.model_service import ActionResponse, ModelService, RouteDecision


class BasicTools:
    def __init__(self, model_service: ModelService) -> None:
        self.model_service = model_service

    def detect_object(self, state: AgentState) -> str | None:
        text = (state.get("text_input") or "").lower()
        keyword_map = {
            "apple": "apple",
            "苹果": "apple",
            "cat": "cat",
            "猫": "cat",
            "dog": "dog",
            "狗": "dog",
            "car": "car",
            "汽车": "car",
            "bus": "bus",
            "公交": "bus",
            "banana": "banana",
            "香蕉": "banana",
            "moon": "moon",
            "月亮": "moon",
            "star": "star",
            "星星": "star",
            "flower": "flower",
            "花": "flower",
            "tree": "tree",
            "树": "tree",
        }
        for keyword, topic in keyword_map.items():
            if keyword in text:
                return topic
        if state.get("image_base64") or state.get("image_url"):
            return "picture"
        return None

    def perceive_signals(self, state: AgentState) -> list[str]:
        text = (state.get("text_input") or "").strip()
        lowered = text.lower()
        signals: list[str] = []
        if state.get("image_base64") or state.get("image_url"):
            signals.append("has_image")
        if not text:
            signals.append("empty_text")
            return signals
        if self._is_pure_greeting(lowered):
            signals.append("greeting_candidate")
        if "?" in text or any(token in text for token in ("什么", "为什么", "怎么", "哪里", "谁")):
            signals.append("question_candidate")
        if self._looks_like_answer(text):
            signals.append("answer_candidate")
        if lowered in {"this one", "what about this", "and then", "this?", "that one"} or text in {
            "这个呢",
            "然后呢",
            "啥意思",
            "什么意思",
        }:
            signals.append("clarify_candidate")
        if self.detect_object(state):
            signals.append("topic_candidate")
        return signals

    def load_age_policy(self, age_band: str) -> dict[str, str]:
        return {
            "age_band": age_band,
            "tone": "short, warm, supportive, educational",
            "question_style": "ask one easy follow-up question when helpful",
        }

    async def classify_next_action(self, state: AgentState) -> RouteDecision:
        request = state_to_request(state)
        return await self.model_service.classify_next_action(request, state)

    async def generate_action_response(
        self,
        state: AgentState,
        action: PlannedAction,
    ) -> ActionResponse:
        request = state_to_request(state)
        return await self.model_service.generate_action_response(request, state, action)

    def _is_pure_greeting(self, lowered: str) -> bool:
        normalized = re.sub(r"[!,.?~，。！？\s]+", "", lowered)
        return normalized in {
            "hi",
            "hello",
            "hey",
            "你好",
            "嗨",
            "哈喽",
            "早上好",
            "晚上好",
        }

    def _looks_like_answer(self, text: str) -> bool:
        cleaned = text.strip()
        if not cleaned or "?" in cleaned:
            return False
        word_count = len(cleaned.split())
        if any(token in cleaned for token in ("because", "因为", "我觉得", "it is", "它是")):
            return True
        return word_count <= 5 or len(cleaned) <= 10
