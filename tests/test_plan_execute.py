from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

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
