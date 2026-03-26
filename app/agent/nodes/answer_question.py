from __future__ import annotations

import logging

from app.agent.state import AgentState, append_trace
from app.tools.basic_tools import BasicTools

logger = logging.getLogger(__name__)


class AnswerQuestionNode:
    def __init__(self, tools: BasicTools) -> None:
        self.tools = tools

    async def __call__(self, state: AgentState) -> AgentState:
        logger.info("entering node=answer_question")
        result = await self.tools.generate_action_response(state, action="answer_question")
        return {
            "current_topic": result.topic or state.get("current_topic"),
            "source_mode": result.source_mode,
            "confidence": result.confidence,
            "safety_notes": result.safety_notes,
            "message_draft": result.message,
            "follow_up_question": result.follow_up_question,
            "workflow_trace": append_trace(state, "answer_question"),
        }
