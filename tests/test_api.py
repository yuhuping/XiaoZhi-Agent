from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_check() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_explain_and_ask_text_flow() -> None:
    response = client.post(
        "/api/v1/chat/explain-and-ask",
        json={"text": "This looks like an apple.", "age_hint": "4-6"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["topic"] == "apple"
    assert payload["explanation"]
    assert payload["follow_up_question"].endswith("?")
    assert payload["metadata"]["source_mode"] == "mock"
    assert payload["metadata"]["used_image"] is False


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
    assert payload["follow_up_question"].endswith("?")
