from __future__ import annotations

import logging

from app.agent.state import AgentState, append_trace
from app.tools.basic_tools import BasicTools

logger = logging.getLogger(__name__)


class PlanningNode:
    def __init__(self, tools: BasicTools) -> None:
        self.tools = tools

    async def __call__(self, state: AgentState) -> AgentState:
        logger.info("entering node=planning")
        decision = await self.tools.classify_next_action(state)
        return {
            "user_intent": decision.user_intent,
            "planned_action": decision.planned_action,
            "route_reason": decision.route_reason,
            "topic_hint": decision.topic_hint or state.get("topic_hint"),
            "current_topic": state.get("current_topic") or decision.topic_hint,
            "confidence": decision.confidence,
            "source_mode": decision.source_mode,
            "dialogue_stage": "planned",
            "workflow_trace": append_trace(state, "planning"),
        }
