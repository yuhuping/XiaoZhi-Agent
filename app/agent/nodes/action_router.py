from __future__ import annotations

import logging

from app.agent.state import AgentState, append_trace

logger = logging.getLogger(__name__)


class ActionRouterNode:
    async def __call__(self, state: AgentState) -> AgentState:
        logger.info("entering node=act_router")
        return {
            "dialogue_stage": "acted",
            "workflow_trace": append_trace(state, "act_router"),
        }
