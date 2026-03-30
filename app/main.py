from contextlib import asynccontextmanager

from fastapi import FastAPI
from langsmith.middleware import TracingMiddleware

from app.api.chat import router as chat_router
from app.core.config import get_settings
from app.core.langsmith import configure_langsmith, is_langsmith_enabled
from app.api.health import router as health_router
from app.core.logging import configure_logging
from app.services.chat_service import create_chat_service


configure_logging()
settings = get_settings()
configure_langsmith(settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.chat_service = create_chat_service()
    yield


app = FastAPI(
    title="XiaoZhi API",
    version="1.3.0",
    description="A controlled ReAct child-learning agent prototype with local RAG and lightweight memory.",
    lifespan=lifespan,
)

if is_langsmith_enabled(settings):
    app.add_middleware(TracingMiddleware)

app.include_router(health_router)
app.include_router(chat_router)
