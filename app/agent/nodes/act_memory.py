from __future__ import annotations

import logging

from app.agent.state import AgentState, append_trace
from app.tools.basic_tools import BasicTools, ToolCall

logger = logging.getLogger(__name__)


class ActMemoryNode:
    def __init__(self, tools: BasicTools) -> None:
        self.tools = tools

    async def __call__(self, state: AgentState) -> AgentState:
        logger.info("entering node=act_memory")
        bundle_result = self.tools.run_tool(
            ToolCall(
                name="read_memory_bundle",
                args={
                    "session_id": state["session_id"],
                    "profile_id": state["profile_id"],
                },
            ),
            state,
        )
        bundle = bundle_result.data if bundle_result.success else {}
        session_data = bundle.get("session", {}) if isinstance(bundle, dict) else {}
        profile_data = bundle.get("profile", {}) if isinstance(bundle, dict) else {}
        return {
            "selected_tool": "read_memory",
            "tool_input": state.get("tool_input", {}),
            "tool_result": {"session": session_data, "profile": profile_data},
            "tool_success": bundle_result.success,
            "short_Memory": session_data if isinstance(session_data, dict) else {},
            "Memory": profile_data if isinstance(profile_data, dict) else {},
            "dialogue_stage": "acted",
            "workflow_trace": append_trace(state, "act_memory"),
        }
