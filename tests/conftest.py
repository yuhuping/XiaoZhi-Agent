import os

import pytest

from app.services.model_service import ModelService


def _fake_call_openai_json(
    self: ModelService,
    model: str,
    schema_name: str,
    schema: dict,
    instruction: str,
    prompt: str,
    chat_request,
) -> dict:
    text = (chat_request.text or "").strip().lower()
    has_image = bool(chat_request.image_base64 or chat_request.image_url)

    if schema_name == "xiaozhi_route_decision":
        if text in {"hi", "hello", "hey", "你好", "您好"}:
            return {
                "user_intent": "greeting",
                "planned_action": "greet",
                "route_reason": "simple greeting",
                "topic_hint": "",
                "confidence": "high",
            }
        if text == "red":
            return {
                "user_intent": "answer_attempt",
                "planned_action": "evaluate_answer",
                "route_reason": "short answer to prior question",
                "topic_hint": "apple",
                "confidence": "medium",
            }
        if has_image:
            return {
                "user_intent": "object_learning",
                "planned_action": "explain_and_ask",
                "route_reason": "image input detected",
                "topic_hint": "picture",
                "confidence": "medium",
            }
        if "apple" in text:
            return {
                "user_intent": "object_learning",
                "planned_action": "explain_and_ask",
                "route_reason": "apple topic requested",
                "topic_hint": "apple",
                "confidence": "high",
            }
        return {
            "user_intent": "direct_question",
            "planned_action": "answer_question",
            "route_reason": "default direct question",
            "topic_hint": "",
            "confidence": "medium",
        }

    if schema_name.startswith("xiaozhi_action_"):
        action = schema_name.replace("xiaozhi_action_", "", 1)
        topic = "apple" if "apple" in text else ("picture" if has_image else "topic")
        if action == "greet":
            return {
                "topic": "",
                "message": "Hello! Nice to learn with you.",
                "follow_up_question": "What do you want to learn today?",
                "confidence": "high",
                "safety_notes": "",
            }
        if action == "evaluate_answer":
            return {
                "topic": topic,
                "message": "Good try. Let us think one more step.",
                "follow_up_question": "Can you explain a little more?",
                "confidence": "medium",
                "safety_notes": "",
            }
        if action == "explain_and_ask":
            return {
                "topic": topic,
                "message": "An apple is a fruit. It can be red or green.",
                "follow_up_question": "What color apple do you like?",
                "confidence": "high",
                "safety_notes": "",
            }
        if action == "answer_question":
            return {
                "topic": topic,
                "message": "That is a good question. Here is a simple answer.",
                "follow_up_question": "Do you want one more example?",
                "confidence": "medium",
                "safety_notes": "",
            }
        if action == "clarify":
            return {
                "topic": topic,
                "message": "I can help. Please tell me a bit more.",
                "follow_up_question": "Can you share one more detail?",
                "confidence": "medium",
                "safety_notes": "",
            }
        return {
            "topic": topic,
            "message": "Let us keep learning together.",
            "follow_up_question": "What should we learn next?",
            "confidence": "medium",
            "safety_notes": "",
        }

    raise AssertionError(f"Unexpected schema_name: {schema_name}")


@pytest.fixture(autouse=True)
def patch_model_service(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY") or "test-key")
    monkeypatch.setattr(ModelService, "_call_openai_json", _fake_call_openai_json)
