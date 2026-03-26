from __future__ import annotations

import logging

from app.agent.state import AgentState, PlannedAction, append_trace
from app.services.session_store import SessionStore

logger = logging.getLogger(__name__)


class StateUpdateNode:
    def __init__(self, session_store: SessionStore) -> None:
        self.session_store = session_store

    async def __call__(self, state: AgentState) -> AgentState:
        logger.info("entering node=state_update")
        session_id = state["session_id"]
        history = self.session_store.get_history(session_id)

        last_agent_question = None
        last_agent_action: PlannedAction | None = None
        pending_topic = state.get("current_topic")
        for turn in reversed(history):
            if turn.get("role") != "assistant":
                continue
            last_agent_question = turn.get("asked_question")
            action = turn.get("action")
            if action in {
                "greet",
                "explain_and_ask",
                "answer_question",
                "evaluate_answer",
                "clarify",
                "fallback",
            }:
                last_agent_action = action
            pending_topic = turn.get("topic") or pending_topic
            break

        current_topic = state.get("current_topic") or state.get("detected_object") or pending_topic
        return {
            "history": history,
            "turn_index": (len(history) // 2) + 1,
            "last_agent_question": last_agent_question,
            "last_agent_action": last_agent_action,
            "pending_topic": pending_topic,
            "current_topic": current_topic,
            "dialogue_stage": "state_ready",
            "workflow_trace": append_trace(state, "state_update"),
        }
