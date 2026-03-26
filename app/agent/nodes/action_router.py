from __future__ import annotations

import logging

from app.agent.state import AgentState, append_trace

logger = logging.getLogger(__name__)


class ActionRouterNode:
    async def __call__(self, state: AgentState) -> AgentState:
        logger.info("entering node=action_router")
        return {
            "dialogue_stage": "routed",
            "workflow_trace": append_trace(state, "action_router"),
        }
