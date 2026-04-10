from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock

from app.agent.graph import should_continue_react
from app.agent.nodes.observe import ObserveNode
from app.agent.nodes.reason import ReasonNode


@dataclass(frozen=True)
class FakeReasonDecision:
    decision: str = "need to search"
    selected_act: str = "tavily_search"
    tool_name: str | None = "tavily_search"
    tool_input: dict[str, Any] | None = None
    route_reason: str = "test"
    topic_hint: str | None = None
    confidence: str = "medium"

    def __post_init__(self):
        if self.tool_input is None:
            object.__setattr__(self, "tool_input", {"query": "test query"})


def _make_state(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "session_id": "test-session",
        "profile_id": "default_child",
        "user_input": "北京天气",
        "latest_user_text": "北京天气",
        "text_input": "北京天气",
        "interaction_mode": "parent",
        "current_topic": None,
        "Memory": {},
        "workflow_trace": [],
        "messages": [],
        "react_iteration": 0,
        "react_max_iterations": 3,
        "react_history": [],
        "selected_act": "direct",
        "selected_tool": None,
        "tool_input": {},
        "child_age_band": "3-8",
        "image_base64": None,
        "image_url": None,
        "image_mime_type": None,
        "topic_hint": None,
    }
    base.update(overrides)
    return base


class TestReasonNodeIteration:
    def test_reason_increments_iteration_and_appends_history(self) -> None:
        tools = AsyncMock()
        tools.reason_next_action = AsyncMock(return_value=FakeReasonDecision())
        node = ReasonNode(tools)
        state = _make_state(react_iteration=0, react_history=[])

        result = asyncio.run(node(state))

        assert result["react_iteration"] == 1
        assert len(result["react_history"]) == 1
        entry = result["react_history"][0]
        assert entry["iteration"] == 1
        assert entry["thought"] == "need to search"
        assert entry["action"] == "tavily_search"

    def test_reason_second_iteration(self) -> None:
        tools = AsyncMock()
        tools.reason_next_action = AsyncMock(return_value=FakeReasonDecision(
            decision="search again",
            tool_input={"query": "more details"},
        ))
        node = ReasonNode(tools)
        existing_history = [{"iteration": 1, "thought": "first", "action": "tavily_search", "action_input": {"query": "q1"}, "observation": "got 2 results"}]
        state = _make_state(react_iteration=1, react_history=existing_history)

        result = asyncio.run(node(state))

        assert result["react_iteration"] == 2
        assert len(result["react_history"]) == 2
        assert result["react_history"][0] == existing_history[0]
        assert result["react_history"][1]["iteration"] == 2


class TestObserveNodeHistory:
    def test_observe_appends_observation_to_history(self) -> None:
        node = ObserveNode()
        history = [{"iteration": 1, "thought": "search", "action": "tavily_search", "action_input": {}}]
        state = _make_state(
            selected_act="tavily_search",
            selected_tool="tavily_search",
            react_history=history,
            messages=[],
        )

        result = asyncio.run(node(state))

        assert result["react_history"][0]["observation"] == result["observation_summary"]


class TestShouldContinueReact:
    def test_continues_when_under_max(self) -> None:
        state = _make_state(react_iteration=1, react_max_iterations=3, selected_act="tavily_search")
        assert should_continue_react(state) == "reason"

    def test_stops_at_max_iterations(self) -> None:
        state = _make_state(react_iteration=3, react_max_iterations=3, selected_act="tavily_search")
        assert should_continue_react(state) == "respond"

    def test_stops_on_direct_act(self) -> None:
        state = _make_state(react_iteration=1, react_max_iterations=3, selected_act="direct")
        assert should_continue_react(state) == "respond"
