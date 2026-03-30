import asyncio
import pytest

from app.core.config import Settings
from app.schemas.chat import ChatRequest
from app.services.model_service import ModelService


def test_reason_next_action_routes_to_retrieval_for_apple_topic() -> None:
    service = ModelService(Settings(openai_api_key="test-key"))
    request = ChatRequest(text="Tell me about an apple", mode="education")

    result = asyncio.run(service.reason_next_action(request, {"session_id": "route-1"}))

    assert result.selected_act == "retrieve_knowledge"
    assert result.tool_name == "retrieve_knowledge"
    assert result.source_mode == "llm"


def test_reason_next_action_can_select_tavily_tool() -> None:
    service = ModelService(Settings(openai_api_key="test-key"))
    request = ChatRequest(text="latest science news for kids", mode="education")

    result = asyncio.run(service.reason_next_action(request, {"session_id": "route-tavily"}))

    assert result.selected_act == "retrieve_knowledge"
    assert result.tool_name == "tavily_search"


def test_generate_final_response_returns_llm_mode() -> None:
    service = ModelService(Settings(openai_api_key="test-key"))
    request = ChatRequest(text="Tell me about an apple", mode="education")

    result = asyncio.run(
        service.generate_final_response(
            request,
            {
                "session_id": "gen-1",
                "selected_act": "retrieve_knowledge",
                "retrieved_chunks": [{"source": "bootstrap.txt", "score": 1.0, "snippet": "apple"}],
            },
        )
    )

    assert result.source_mode == "llm"
    assert result.message
    assert result.follow_up_question


def test_parse_json_from_fenced_text() -> None:
    service = ModelService(Settings(openai_api_key="test-key"))
    raw_body = """```json
{"decision":"respond_directly","selected_act":"direct","tool_name":"","tool_input":{},"route_reason":"simple","topic_hint":"","confidence":"high"}
```"""

    result = service._parse_json_from_text(raw_body)

    assert result["selected_act"] == "direct"


def test_parse_json_from_text_accepts_trailing_extra_brace() -> None:
    service = ModelService(Settings(openai_api_key="test-key"))
    raw_body = (
        '{"topic":"微博热搜","message":"实时热搜每分钟更新。","follow_up_question":"您想了解哪个话题？",'
        '"confidence":"medium","safety_notes":"热搜内容可能包含成人话题，建议家长陪同查看。"}}'
    )

    result = service._parse_json_from_text(raw_body)

    assert result["topic"] == "微博热搜"
    assert result["confidence"] == "medium"


def test_llm_payload_uses_planning_model_override() -> None:
    settings = Settings(
        openai_api_key="test-key",
        openai_model="Qwen3-235B-A22B",
        openai_planning_model="Qwen3-30B-A3B",
    )
    service = ModelService(settings)
    payload = service._build_payload(
        model=settings.openai_planning_model,
        schema={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        instruction="Route the turn.",
        prompt="Current user text: hello",
        chat_request=ChatRequest(text="hello"),
    )

    assert payload["model"] == "Qwen3-30B-A3B"


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
