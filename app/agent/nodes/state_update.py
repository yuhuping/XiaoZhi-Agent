from __future__ import annotations

import logging

from app.agent.state import AgentState, append_trace
from app.tools.basic_tools import BasicTools, ToolCall

logger = logging.getLogger(__name__)


class StateUpdateNode:
    """状态更新：读取记忆包并补齐对话上下文。"""

    def __init__(self, tools: BasicTools) -> None:
        self.tools = tools

    async def __call__(self, state: AgentState) -> AgentState:
        """执行状态更新。"""
        logger.info("entering node=state_update")
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
        session = bundle.get("session", {}) if isinstance(bundle, dict) else {}
        profile = bundle.get("profile", {}) if isinstance(bundle, dict) else {}

        history = session.get("recent_turns", []) if isinstance(session, dict) else []
        if not isinstance(history, list):
            history = []

        last_agent_question = None
        pending_topic = state.get("current_topic")
        for turn in reversed(history):
            if not isinstance(turn, dict):
                continue
            if turn.get("role") != "assistant":
                continue
            if turn.get("asked_question"):
                last_agent_question = str(turn.get("asked_question"))
            pending_topic = turn.get("topic") or pending_topic
            break

        current_topic = state.get("current_topic") or pending_topic
        return {
            "history": history,
            "turn_index": (len(history) // 2) + 1,
            "last_agent_question": last_agent_question,
            "pending_topic": pending_topic,
            "current_topic": current_topic,
            "short_Memory": session if isinstance(session, dict) else {},
            "Memory": profile if isinstance(profile, dict) else {},
            "dialogue_stage": "state_ready",
            "workflow_trace": append_trace(state, "state_update"),
        }
