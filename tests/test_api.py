from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_check() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_direct_flow_returns_v13_response_shape() -> None:
    response = client.post(
        "/api/v1/chat/explain-and-ask",
        json={"text": "hello", "age_hint": "4-6", "mode": "companion"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "companion"
    assert payload["message"]
    assert payload["react"]["selected_act"] == "direct"
    assert payload["metadata"]["source_mode"] == "llm"
    assert payload["metadata"]["workflow_trace"] == [
        "understand",
        "state_update",
        "chatbot",
        "observe",
        "respond",
        "memory_update",
        "response",
    ]
    assert payload["metadata"]["dialogue_stage"] == "responded"


def test_text_learning_flow_routes_to_retrieval() -> None:
    response = client.post(
        "/api/v1/chat/explain-and-ask",
        json={"text": "This looks like an apple.", "age_hint": "4-6", "mode": "education"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["react"]["selected_act"] == "retrieve_knowledge"
    assert payload["grounding"]["used_rag"] in {True, False}
    assert payload["metadata"]["rag_enabled"] is True


def test_invalid_input_requires_text_or_image() -> None:
    response = client.post("/api/v1/chat/explain-and-ask", json={})
    assert response.status_code == 422


def test_invalid_placeholder_image_is_rejected() -> None:
    response = client.post(
        "/api/v1/chat/explain-and-ask",
        json={
            "text": "What is this apple?",
            "image_base64": "string",
            "image_mime_type": "string",
        },
    )
    assert response.status_code == 422


def test_api_rejects_oversized_base64_payload(monkeypatch) -> None:
    monkeypatch.setenv("MAX_UPLOAD_IMAGE_BYTES", "4")
    response = client.post(
        "/api/v1/chat/explain-and-ask",
        json={
            "text": "hello",
            "image_base64": "aGVsbG8=",
            "image_mime_type": "image/png",
        },
    )
    assert response.status_code == 422
    assert "too large" in response.text


def test_image_url_input_is_accepted() -> None:
    response = client.post(
        "/api/v1/chat/explain-and-ask",
        json={
            "text": "What is in this picture?",
            "image_url": "https://example.com/cat.png",
            "mode": "education",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["metadata"]["used_image"] is True
    assert payload["react"]["selected_act"] == "retrieve_knowledge"


def test_session_memory_can_route_to_read_memory() -> None:
    first = client.post(
        "/api/v1/chat/explain-and-ask",
        json={"text": "Tell me about an apple.", "session_id": "api-memory-1", "mode": "education"},
    )
    assert first.status_code == 200

    second = client.post(
        "/api/v1/chat/explain-and-ask",
        json={"text": "Red", "session_id": "api-memory-1", "mode": "education"},
    )
    assert second.status_code == 200
    payload = second.json()
    assert payload["react"]["selected_act"] == "read_memory"
    assert payload["memory"]["session_updated"] is True


def test_time_sensitive_query_in_education_mode_can_route_to_tavily() -> None:
    response = client.post(
        "/api/v1/chat/explain-and-ask",
        json={"text": "What is the latest space news?", "mode": "education"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["react"]["selected_act"] == "retrieve_knowledge"
    assert payload["react"]["tool_name"] == "tavily_search"
    assert payload["react"]["tool_success"] in {True, False}
    assert payload["message"]


def test_parent_mode_returns_parent_facing_response() -> None:
    response = client.post(
        "/api/v1/chat/explain-and-ask",
        json={
            "text": "我是家长，孩子最近不爱阅读怎么办？",
            "mode": "parent",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "parent"
    assert payload["message"]
    assert payload["react"]["selected_act"] == "direct"


def test_root_route_returns_workflow_ready_status() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "XiaoZhi Playground" in response.text


def test_app_initializes_chat_service_on_startup() -> None:
    with TestClient(app) as startup_client:
        assert hasattr(startup_client.app.state, "chat_service")
