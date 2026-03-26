from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse, HTMLResponse

router = APIRouter(tags=["health"])
FRONTEND_PATH = Path(__file__).resolve().parents[1] / "frontend" / "index.html"
LOGO_PATH = Path(__file__).resolve().parents[1] / "logo.png"


@router.get("/")
async def root() -> HTMLResponse:
    return HTMLResponse(FRONTEND_PATH.read_text(encoding="utf-8"))


@router.get("/logo.png")
async def logo() -> FileResponse:
    return FileResponse(LOGO_PATH, media_type="image/png")


@router.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}
