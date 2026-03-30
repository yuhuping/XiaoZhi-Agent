from __future__ import annotations

import logging

from app.agent.state import AgentState, append_trace
from app.tools.basic_tools import BasicTools

logger = logging.getLogger(__name__)


class UnderstandNode:
    def __init__(self, tools: BasicTools) -> None:
        self.tools = tools

    async def __call__(self, state: AgentState) -> AgentState:
        logger.info("entering node=understand")
        topic_hint = self.tools.detect_object(state)
        signals = self.tools.perceive_signals(state)
        return {
            "current_topic": topic_hint,
            "detected_object": topic_hint,
            "topic_hint": topic_hint,
            "perception_signals": signals,
            "dialogue_stage": "understood",
            "workflow_trace": append_trace(state, "understand"),
        }

