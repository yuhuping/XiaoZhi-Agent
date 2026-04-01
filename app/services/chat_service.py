import logging
from collections.abc import Awaitable, Callable
from pathlib import Path

from fastapi import Request
from langsmith import trace, tracing_context

from app.agent.graph import AgentGraph
from app.agent.state import build_initial_state
from app.core.config import get_settings
from app.core.langsmith import is_langsmith_enabled
from app.memory import MemoryConfig, MemoryManager, MemoryTool
from app.rag.retriever import LocalKnowledgeRetriever
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.model_service import ModelService
from app.tools.basic_tools import BasicTools

logger = logging.getLogger(__name__)


class ChatService:
    """聊天服务：封装工作流调用。"""

    def __init__(self, graph: AgentGraph, model_service: ModelService) -> None:
        self.graph = graph
        self.model_service = model_service

    async def explain_and_ask(self, request: ChatRequest) -> ChatResponse:
        """执行一次对话请求。"""
        return await self.explain_and_ask_stream(request=request, on_delta=None)

    async def explain_and_ask_stream(
        self,
        request: ChatRequest,
        on_delta: Callable[[str], Awaitable[None] | None] | None,
    ) -> ChatResponse:
        """执行一次对话请求，可选逐段回调。"""
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
                state["stream_delta_writer"] = on_delta
                state["rag_enabled"] = settings.rag_enabled
                final_state = await self.graph.run(state=state)
                response = ChatResponse.model_validate(final_state["final_response"])
                run.end(outputs=response.model_dump())
                return response


def create_chat_service() -> ChatService:
    """创建聊天服务及其依赖。"""
    logger.info("initializing chat service")
    settings = get_settings()
    model_service = ModelService(settings)

    memory_db = Path(settings.memory_db_path)
    if settings.memory_reset_on_start and memory_db.exists():
        # 重要变量：按配置重建memory库。
        memory_db.unlink(missing_ok=True)

    memory_config = MemoryConfig(
        db_path=str(memory_db),
        index_dir=settings.memory_index_dir,
        working_memory_capacity=settings.memory_working_capacity,
        working_memory_ttl_minutes=settings.memory_working_ttl_minutes,
        consolidate_working_threshold=settings.memory_consolidate_working_threshold,
        consolidate_episodic_threshold=settings.memory_consolidate_episodic_threshold,
        forget_default_max_age_days=settings.memory_forget_max_age_days,
    )
    memory_manager = MemoryManager(config=memory_config)
    memory_tool = MemoryTool(manager=memory_manager)

    retriever = LocalKnowledgeRetriever.from_kg_dir(
        kg_dir=settings.kg_dir,
        auto_bootstrap=settings.kg_auto_bootstrap,
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
        auto_refresh_interval_seconds=settings.rag_refresh_interval_seconds,
    )
    tools = BasicTools(
        model_service=model_service,
        memory_tool=memory_tool,
        retriever=retriever,
    )
    service = ChatService(
        graph=AgentGraph(model_service=model_service, tools=tools),
        model_service=model_service,
    )
    logger.info("chat service initialized")
    return service


def get_chat_service(request: Request) -> ChatService:
    """获取或懒创建服务实例。"""
    service = getattr(request.app.state, "chat_service", None)
    if service is None:
        service = create_chat_service()
        request.app.state.chat_service = service
    return service
