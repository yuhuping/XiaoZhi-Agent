from fastapi import FastAPI

from app.api.chat import router as chat_router
from app.api.health import router as health_router
from app.core.logging import configure_logging


configure_logging()

app = FastAPI(
    title="XiaoZhi API",
    version="1.1.0",
    description="A child-friendly multimodal learning companion prototype.",
)

app.include_router(health_router)
app.include_router(chat_router)
