from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

from app.agent.nodes.execute import ExecuteNode
from app.agent.nodes.plan import PlanNode


def _make_state(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "session_id": "test-session",
        "profile_id": "default_child",
        "user_input": "什么是勾股定理",
        "latest_user_text": "什么是勾股定理",
        "text_input": "什么是勾股定理",
        "interaction_mode": "education",
        "current_topic": None,
        "Memory": {},
        "workflow_trace": [],
        "messages": [],
        "plan_steps": [],
        "plan_raw": "",
        "retrieved_chunks": [],
        "child_age_band": "3-8",
        "image_base64": None,
        "image_url": None,
        "image_mime_type": None,
        "selected_act": "direct",
        "selected_tool": None,
        "tool_input": {},
    }
    base.update(overrides)
    return base


class TestPlanNode:
    def test_plan_generates_steps_and_retrieval(self) -> None:
        """Plan with needs_retrieval=True should set tool_calls and selected_act."""
        fake_plan = {
            "steps": ["理解勾股定理的定义", "用简单例子说明"],
            "needs_retrieval": True,
            "retrieval_query": "勾股定理",
        }
        tools = AsyncMock()
        tools.generate_plan = AsyncMock(return_value=fake_plan)
        node = PlanNode(tools)
        state = _make_state()

        result = asyncio.run(node(state))

        assert result["plan_steps"] == ["理解勾股定理的定义", "用简单例子说明"]
        assert result["selected_act"] == "retrieve_knowledge"
        assert result["selected_tool"] == "retrieve_knowledge"
        assert result["tool_input"]["query"] == "勾股定理"
        assert len(result["messages"]) == 1
        assert result["messages"][0].tool_calls  # has tool_calls

    def test_plan_no_retrieval_goes_direct(self) -> None:
        """Plan with needs_retrieval=False should set selected_act=direct, no tool_calls."""
        fake_plan = {
            "steps": ["计算3+5"],
            "needs_retrieval": False,
            "retrieval_query": "",
        }
        tools = AsyncMock()
        tools.generate_plan = AsyncMock(return_value=fake_plan)
        node = PlanNode(tools)
        state = _make_state(user_input="3+5等于几", latest_user_text="3+5等于几", text_input="3+5等于几")

        result = asyncio.run(node(state))

        assert result["plan_steps"] == ["计算3+5"]
        assert result["selected_act"] == "direct"
        assert result["selected_tool"] is None
        assert result["messages"][0].tool_calls == []


class TestExecuteNode:
    def test_execute_streams_and_writes_result(self) -> None:
        """ExecuteNode should call execute_plan with stream_delta_writer and write execution_result."""
        tools = AsyncMock()
        tools.execute_plan = AsyncMock(return_value="第一步：3+5=8。答案是8。")
        node = ExecuteNode(tools)
        delta_writer = AsyncMock()
        state = _make_state(
            plan_steps=["计算3+5"],
            stream_delta_writer=delta_writer,
        )

        result = asyncio.run(node(state))

        tools.execute_plan.assert_called_once()
        assert result["execution_result"] == "第一步：3+5=8。答案是8。"
        assert result["message_draft"] == "第一步：3+5=8。答案是8。"
        assert result["dialogue_stage"] == "responded"

    def test_execute_without_stream_writer(self) -> None:
        """ExecuteNode works even when stream_delta_writer is None."""
        tools = AsyncMock()
        tools.execute_plan = AsyncMock(return_value="答案是42。")
        node = ExecuteNode(tools)
        state = _make_state(
            plan_steps=["回答问题"],
            stream_delta_writer=None,
        )

        result = asyncio.run(node(state))

        assert result["execution_result"] == "答案是42。"
        assert result["message_draft"] == "答案是42。"


class TestPlanNodeFallback:
    def test_plan_fallback_on_empty_steps(self) -> None:
        """If generate_plan returns empty steps, PlanNode should fallback."""
        fake_plan = {"steps": [], "needs_retrieval": False, "retrieval_query": ""}
        tools = AsyncMock()
        tools.generate_plan = AsyncMock(return_value=fake_plan)
        node = PlanNode(tools)
        state = _make_state()

        result = asyncio.run(node(state))

        assert result["selected_act"] == "direct"
        assert len(result["plan_steps"]) >= 1  # at least fallback step
