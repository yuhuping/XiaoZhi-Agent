from __future__ import annotations

import logging

from app.agent.state import AgentState, append_trace, emit_stream_event
from app.tools.basic_tools import BasicTools

logger = logging.getLogger(__name__)


class ExecuteNode:
    def __init__(self, tools: BasicTools) -> None:
        self.tools = tools

    async def __call__(self, state: AgentState) -> AgentState:
        logger.info("entering node=execute")
        on_delta = state.get("stream_delta_writer")
        result_text = await self.tools.execute_plan(state, on_delta=on_delta)
        await emit_stream_event(
            state,
            {
                "type": "workflow",
                "event": "workflow_completed",
                "phase": "complete",
                "status": "completed",
                "title": "学习路线已完成",
            },
        )
        return {
            "execution_result": result_text,
            "message_draft": result_text,
            "confidence": "medium",
            "safety_notes": "",
            "dialogue_stage": "responded",
            "workflow_trace": append_trace(state, "execute"),
        }
