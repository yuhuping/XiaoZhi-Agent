from __future__ import annotations

import logging

from app.agent.state import AgentState, append_trace

logger = logging.getLogger(__name__)


class ActDirectNode:
    async def __call__(self, state: AgentState) -> AgentState:
        logger.info("entering node=act_direct")
        return {
            "selected_tool": None,
            "tool_result": {},
            "tool_success": True,
            "dialogue_stage": "acted",
            "workflow_trace": append_trace(state, "act_direct"),
        }

