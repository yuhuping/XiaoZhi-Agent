from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_check() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_greet_flow_returns_chat_action() -> None:
    response = client.post(
        "/api/v1/chat/explain-and-ask",
        json={"text": "hello", "age_hint": "4-6"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["action"] == "greet"
    assert payload["message"]
    assert payload["metadata"]["source_mode"] == "openai"
    assert payload["metadata"]["workflow_trace"] == [
        "perception",
        "state_update",
        "planning",
        "action_router",
        "greet",
        "response",
    ]
    assert payload["metadata"]["planned_action"] == "greet"
    assert payload["metadata"]["dialogue_stage"] == "responded"


def test_text_learning_flow_routes_to_explain_and_ask() -> None:
    response = client.post(
        "/api/v1/chat/explain-and-ask",
        json={"text": "This looks like an apple.", "age_hint": "4-6"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["action"] == "explain_and_ask"
    assert payload["topic"] == "apple"
    assert payload["follow_up_question"].endswith("?")
    assert payload["metadata"]["planned_action"] == "explain_and_ask"


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
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["metadata"]["used_image"] is True
    assert payload["action"] == "explain_and_ask"
    assert payload["follow_up_question"].endswith("?")


def test_session_memory_can_route_to_evaluate_answer() -> None:
    first = client.post(
        "/api/v1/chat/explain-and-ask",
        json={"text": "Tell me about an apple.", "session_id": "api-memory-1"},
    )
    assert first.status_code == 200

    second = client.post(
        "/api/v1/chat/explain-and-ask",
        json={"text": "Red", "session_id": "api-memory-1"},
    )
    assert second.status_code == 200
    payload = second.json()
    assert payload["action"] == "evaluate_answer"
    assert payload["metadata"]["planned_action"] == "evaluate_answer"


def test_root_route_returns_workflow_ready_status() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "XiaoZhi Playground" in response.text


def test_app_initializes_chat_service_on_startup() -> None:
    with TestClient(app) as startup_client:
        assert hasattr(startup_client.app.state, "chat_service")
