from __future__ import annotations

import logging

from app.agent.state import AgentState, append_trace
from app.tools.basic_tools import BasicTools, ToolCall

logger = logging.getLogger(__name__)


class ActRetrieveNode:
    def __init__(self, tools: BasicTools) -> None:
        self.tools = tools

    async def __call__(self, state: AgentState) -> AgentState:
        logger.info("entering node=act_retrieve")
        tool_input = dict(state.get("tool_input") or {})
        if "query" not in tool_input:
            tool_input["query"] = state.get("latest_user_text") or state.get("user_input") or ""
        tool_name = state.get("selected_tool") or "retrieve_knowledge"
        if tool_name not in {"retrieve_knowledge", "tavily_search"}:
            tool_name = "retrieve_knowledge"
        result = self.tools.run_tool(ToolCall(name=tool_name, args=tool_input), state)
        return {
            "selected_tool": tool_name,
            "tool_input": tool_input,
            "tool_result": result.data,
            "tool_success": result.success,
            "retrieved_chunks": result.data.get("results", []),
            "dialogue_stage": "acted",
            "workflow_trace": append_trace(state, "act_retrieve"),
        }
