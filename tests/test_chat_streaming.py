from __future__ import annotations

import asyncio
import json
from typing import Any

from app.api.chat import _stream_chat_response
from app.schemas.chat import ChatRequest


class FakeStreamingService:
    async def explain_and_ask_stream(self, request: ChatRequest, on_delta, on_event) -> None:
        await on_event(
            {
                "type": "workflow",
                "event": "planning_started",
                "phase": "planning",
                "status": "active",
                "title": "正在规划学习路线",
            }
        )
        await on_delta("蜗牛的壳")


class FailingStreamingService:
    async def explain_and_ask_stream(self, request: ChatRequest, on_delta, on_event) -> None:
        await on_event(
            {
                "type": "workflow",
                "event": "planning_started",
                "phase": "planning",
                "status": "active",
                "title": "正在规划学习路线",
            }
        )
        raise RuntimeError("provider unavailable")


def _decode_sse(raw_event: str) -> dict[str, Any]:
    return json.loads(raw_event.removeprefix("data: ").strip())


def test_stream_preserves_workflow_delta_and_done_order() -> None:
    request = ChatRequest(text="为什么蜗牛背着壳？", mode="education")

    async def collect() -> list[dict[str, Any]]:
        events = []
        async for raw_event in _stream_chat_response(request, FakeStreamingService()):  # type: ignore[arg-type]
            events.append(_decode_sse(raw_event))
        return events

    events = asyncio.run(collect())

    assert [event["type"] for event in events] == ["workflow", "delta", "done"]
    assert events[0]["phase"] == "planning"
    assert events[1]["delta"] == "蜗牛的壳"
    assert events[2]["done"] is True
    assert events[2]["success"] is True


def test_stream_failure_emits_error_without_false_done() -> None:
    request = ChatRequest(text="为什么蜗牛背着壳？", mode="education")

    async def collect() -> list[dict[str, Any]]:
        events = []
        async for raw_event in _stream_chat_response(request, FailingStreamingService()):  # type: ignore[arg-type]
            events.append(_decode_sse(raw_event))
        return events

    events = asyncio.run(collect())

    assert [event["type"] for event in events] == ["workflow", "error"]
    assert events[1] == {
        "type": "error",
        "error": "请求处理失败，请稍后重试。",
        "success": False,
    }


def test_playground_contains_plan_execute_learning_route() -> None:
    html = ("app/frontend/index.html")
    content = open(html, encoding="utf-8-sig").read()

    assert "Plan & Execute" in content
    assert "learning-route" in content
    assert "updateWorkflowUI" in content
