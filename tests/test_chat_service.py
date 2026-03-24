from app.core.config import Settings
from app.schemas.chat import ChatRequest
from app.services.model_service import ModelService


def test_mock_service_with_image_sets_used_topic() -> None:
    service = ModelService(Settings(mock_mode=True))
    request = ChatRequest(
        image_base64="aGVsbG8=",
        image_mime_type="image/png",
    )

    result = service._mock_response(request)

    assert result.topic == "picture"
    assert result.source_mode == "mock"
    assert result.follow_up_question.endswith("?")


def test_mock_service_with_image_url_sets_used_topic() -> None:
    service = ModelService(Settings(mock_mode=True))
    request = ChatRequest(
        image_url="https://example.com/flower.jpg",
    )

    result = service._mock_response(request)

    assert result.topic == "picture"
    assert result.source_mode == "mock"
    assert result.follow_up_question.endswith("?")


def test_mock_service_redirects_unsafe_content() -> None:
    service = ModelService(Settings(mock_mode=True))
    request = ChatRequest(text="Tell me about a weapon.")

    result = service._mock_response(request)

    assert result.topic == "safe learning topic"
    assert "safe" in result.safety_notes.lower()


def test_openai_response_parser_reads_output_array() -> None:
    service = ModelService(Settings(mock_mode=False, openai_api_key="test-key"))
    raw_body = """
    {
      "output": [
        {
          "type": "message",
          "content": [
            {
              "type": "output_text",
              "text": "{\\"topic\\":\\"apple\\",\\"explanation\\":\\"An apple is a fruit.\\",\\"follow_up_question\\":\\"What color can an apple be?\\",\\"confidence\\":\\"high\\",\\"safety_notes\\":\\"\\"}"
            }
          ]
        }
      ]
    }
    """

    result = service._parse_openai_response(raw_body)

    assert result.source_mode == "openai"
    assert result.topic == "apple"
    assert result.follow_up_question.endswith("?")
