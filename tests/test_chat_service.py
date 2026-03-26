import asyncio
import pytest

from app.core.config import Settings
from app.schemas.chat import ChatRequest
from app.services.model_service import ModelService


def test_classify_next_action_routes_greeting() -> None:
    service = ModelService(Settings(openai_api_key="test-key"))
    request = ChatRequest(text="hello")

    result = asyncio.run(service.classify_next_action(request, {"session_id": "route-1"}))

    assert result.planned_action == "greet"
    assert result.user_intent == "greeting"
    assert result.source_mode == "openai"


def test_generate_action_response_returns_openai_mode() -> None:
    service = ModelService(Settings(openai_api_key="test-key"))
    request = ChatRequest(text="Tell me about an apple")

    result = asyncio.run(
        service.generate_action_response(
            request,
            {"session_id": "gen-1", "current_topic": "apple"},
            "explain_and_ask",
        )
    )

    assert result.source_mode == "openai"
    assert result.topic in {"apple", "picture", "topic"}
    assert result.follow_up_question


def test_parse_json_from_fenced_text() -> None:
    service = ModelService(Settings(openai_api_key="test-key"))
    raw_body = """```json
{"planned_action":"greet","user_intent":"greeting","route_reason":"simple greeting","topic_hint":"","confidence":"high"}
```"""

    result = service._parse_json_from_text(raw_body)

    assert result["planned_action"] == "greet"


def test_openai_payload_uses_planning_model_override() -> None:
    settings = Settings(
        openai_api_key="test-key",
        openai_model="gpt-5.4",
        openai_planning_model="gpt-4.1-mini",
    )
    service = ModelService(settings)
    payload = service._build_openai_payload(
        model=settings.openai_planning_model,
        schema_name="xiaozhi_route_decision",
        schema={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        instruction="Route the turn.",
        prompt="Current user text: hello",
        chat_request=ChatRequest(text="hello"),
    )

    assert payload["model"] == "gpt-4.1-mini"


def test_image_request_timeout_uses_image_timeout_setting() -> None:
    settings = Settings(
        openai_api_key="test-key",
        request_timeout_seconds=30,
        openai_image_request_timeout_seconds=90,
    )
    service = ModelService(settings)

    text_request = ChatRequest(text="hello")
    image_request = ChatRequest(
        text="what is this",
        image_base64="aGVsbG8=",
        image_mime_type="image/png",
    )

    assert service._resolve_request_timeout(text_request) == 30
    assert service._resolve_request_timeout(image_request) == 90


def test_chat_request_rejects_oversized_base64_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_UPLOAD_IMAGE_BYTES", "4")

    with pytest.raises(ValueError, match="too large"):
        ChatRequest(
            text="hello",
            image_base64="aGVsbG8=",
            image_mime_type="image/png",
        )
