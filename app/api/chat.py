from __future__ import annotations

import asyncio
import json
import logging
from contextlib import suppress
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.schemas.chat import ChatRequest
from app.services.chat_service import ChatService, get_chat_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])
_STREAM_END = object()


def _format_sse_event(payload: dict[str, object]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _stream_chat_response(
    request: ChatRequest,
    service: ChatService,
) -> AsyncIterator[str]:
    queue: asyncio.Queue[dict[str, object] | object] = asyncio.Queue()

    async def on_delta(delta: str) -> None:
        if delta:
            await queue.put({"delta": delta})

    async def produce() -> None:
        try:
            await service.explain_and_ask_stream(request=request, on_delta=on_delta)
        except Exception:
            logger.exception("chat streaming failed")
        finally:
            await queue.put({"done": True})
            await queue.put(_STREAM_END)

    task = asyncio.create_task(produce())
    try:
        while True:
            item = await queue.get()
            if item is _STREAM_END:
                break
            yield _format_sse_event(item)
    finally:
        if not task.done():
            task.cancel()
        with suppress(asyncio.CancelledError):
            await task


@router.post("/explain-and-ask")
async def explain_and_ask(
    request: ChatRequest,
    service: ChatService = Depends(get_chat_service),
) -> StreamingResponse:
    logger.info("request received for explain-and-ask")
    return StreamingResponse(
        _stream_chat_response(request=request, service=service),
        media_type="text/event-stream",
    )
