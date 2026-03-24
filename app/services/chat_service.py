import logging
from functools import lru_cache

from app.core.config import get_settings
from app.schemas.chat import ChatMetadata, ChatRequest, ChatResponse
from app.services.model_service import ModelOutput, ModelService

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self, model_service: ModelService) -> None:
        self.model_service = model_service

    async def explain_and_ask(self, request: ChatRequest) -> ChatResponse:
        logger.info("model call started")
        result = await self.model_service.generate_child_response(request)
        logger.info("model call completed")
        return self._to_response(result, request)

    def _to_response(self, result: ModelOutput, request: ChatRequest) -> ChatResponse:
        return ChatResponse(
            topic=result.topic,
            explanation=result.explanation,
            follow_up_question=result.follow_up_question,
            metadata=ChatMetadata(
                source_mode=result.source_mode,
                confidence=result.confidence,
                safety_notes=result.safety_notes,
                used_image=bool(request.image_base64 or request.image_url),
            ),
        )


@lru_cache(maxsize=1)
def get_chat_service() -> ChatService:
    settings = get_settings()
    return ChatService(model_service=ModelService(settings))
