from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/")
async def root() -> dict[str, object]:
    return {
        "name": "XiaoZhi API",
        "version": "1.1.0",
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
        "main_endpoint": "/api/v1/chat/explain-and-ask",
    }


@router.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}
