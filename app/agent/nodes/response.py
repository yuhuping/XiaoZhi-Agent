from __future__ import annotations

import logging

from app.agent.state import AgentState, append_trace, utc_now_iso
from app.schemas.chat import ChatMetadata, ChatResponse
from app.services.session_store import SessionStore

logger = logging.getLogger(__name__)


class ResponseNode:
    def __init__(self, session_store: SessionStore) -> None:
        self.session_store = session_store

    async def __call__(self, state: AgentState) -> AgentState:
        logger.info("entering node=response")
        workflow_trace = append_trace(state, "response")
        planned_action = state.get("planned_action") or "fallback"
        response = ChatResponse(
            session_id=state["session_id"],
            action=planned_action,
            message=state.get("message_draft") or "Let us learn one small thing together.",
            follow_up_question=state.get("follow_up_question"),
            topic=state.get("current_topic") or state.get("pending_topic"),
            metadata=ChatMetadata(
                source_mode=state.get("source_mode") or "openai",
                confidence=state.get("confidence") or "medium",
                safety_notes=state.get("safety_notes", ""),
                used_image=bool(state.get("image_base64") or state.get("image_url")),
                dialogue_stage="responded",
                planned_action=planned_action,
                workflow_trace=workflow_trace,
                input_modality=state.get("input_modality", "text"),
                route_reason=state.get("route_reason", ""),
            ),
        )
        assistant_question = response.follow_up_question
        self.session_store.append_turns(
            state["session_id"],
            [
                {
                    "role": "user",
                    "text": state.get("latest_user_text") or "[image input]",
                    "action": None,
                    "topic": state.get("current_topic"),
                    "asked_question": None,
                    "timestamp": utc_now_iso(),
                },
                {
                    "role": "assistant",
                    "text": response.message,
                    "action": planned_action,
                    "topic": response.topic,
                    "asked_question": assistant_question,
                    "timestamp": utc_now_iso(),
                },
            ],
        )
        return {
            "final_response": response.model_dump(),
            "dialogue_stage": "responded",
            "workflow_trace": workflow_trace,
        }
