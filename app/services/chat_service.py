import logging

from fastapi import Request
from app.agent.graph import AgentGraph
from app.agent.state import build_initial_state
from app.core.config import get_settings
from app.core.langsmith import is_langsmith_enabled
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.model_service import ModelService
from app.services.session_store import SessionStore
from langsmith import trace, tracing_context

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self, graph: AgentGraph, model_service: ModelService) -> None:
        self.graph = graph
        self.model_service = model_service

    async def explain_and_ask(self, request: ChatRequest) -> ChatResponse:
        settings = self.model_service.settings
        with tracing_context(
            enabled=is_langsmith_enabled(settings),
            project_name=settings.langsmith_project,
            metadata={"component": "chat_service"},
        ):
            with trace(
                "chat_request",
                run_type="chain",
                inputs={
                    "text": request.text or "",
                    "has_image": bool(request.image_base64 or request.image_url),
                    "age_hint": request.age_hint,
                    "session_id": request.session_id,
                },
                metadata={"endpoint": "/api/v1/chat/explain-and-ask"},
            ) as run:
                state = build_initial_state(request)
                final_state = await self.graph.run(state=state)
                response = ChatResponse.model_validate(final_state["final_response"])
                run.end(outputs=response.model_dump())
                return response


def create_chat_service() -> ChatService:
    logger.info("initializing chat service")
    settings = get_settings()
    model_service = ModelService(settings)
    session_store = SessionStore()
    service = ChatService(
        graph=AgentGraph(model_service=model_service, session_store=session_store),
        model_service=model_service,
    )
    logger.info("chat service initialized")
    return service


def get_chat_service(request: Request) -> ChatService:
    service = getattr(request.app.state, "chat_service", None)
    if service is None:
        service = create_chat_service()
        request.app.state.chat_service = service
    return service
