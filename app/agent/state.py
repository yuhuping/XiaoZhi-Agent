from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, TypedDict
from uuid import uuid4

from app.schemas.chat import ChatRequest

InputModality = Literal["text", "image", "multimodal"]
DialogueStage = Literal[
    "received",
    "perceived",
    "state_ready",
    "planned",
    "routed",
    "responded",
]
UserIntent = Literal[
    "greeting",
    "object_learning",
    "direct_question",
    "answer_attempt",
    "unclear",
    "fallback",
]
PlannedAction = Literal[
    "greet",
    "explain_and_ask",
    "answer_question",
    "evaluate_answer",
    "clarify",
    "fallback",
]
ConversationRole = Literal["user", "assistant"]


class ConversationTurn(TypedDict, total=False):
    role: ConversationRole
    text: str
    action: PlannedAction | None
    topic: str | None
    asked_question: str | None
    timestamp: str


class AgentState(TypedDict, total=False):
    session_id: str
    turn_index: int
    user_input: str
    latest_user_text: str | None
    text_input: str | None
    normalized_text: str
    image_base64: str | None
    image_url: str | None
    image_mime_type: str | None
    input_modality: InputModality
    child_age_band: str
    current_topic: str | None
    detected_object: str | None
    topic_hint: str | None
    dialogue_stage: DialogueStage
    user_intent: UserIntent | None
    planned_action: PlannedAction | None
    route_reason: str
    perception_signals: list[str]
    history: list[ConversationTurn]
    last_agent_question: str | None
    last_agent_action: PlannedAction | None
    pending_topic: str | None
    message_draft: str | None
    follow_up_question: str | None
    final_response: dict[str, Any]
    source_mode: str | None
    confidence: str | None
    safety_notes: str
    workflow_trace: list[str]


def build_initial_state(request: ChatRequest, session_id: str | None = None) -> AgentState:
    has_text = bool(request.text and request.text.strip())
    has_image = bool(request.image_base64 or request.image_url)
    if has_text and has_image:
        modality: InputModality = "multimodal"
    elif has_image:
        modality = "image"
    else:
        modality = "text"

    normalized_text = (request.text or "").strip()
    return {
        "session_id": session_id or request.session_id or str(uuid4()),
        "turn_index": 0,
        "user_input": normalized_text,
        "latest_user_text": normalized_text or None,
        "text_input": request.text,
        "normalized_text": normalized_text.lower(),
        "image_base64": request.image_base64,
        "image_url": str(request.image_url) if request.image_url else None,
        "image_mime_type": request.image_mime_type,
        "input_modality": modality,
        "child_age_band": request.age_hint or "3-8",
        "dialogue_stage": "received",
        "route_reason": "",
        "perception_signals": [],
        "history": [],
        "message_draft": None,
        "follow_up_question": None,
        "final_response": {},
        "safety_notes": "",
        "workflow_trace": [],
    }


def append_trace(state: AgentState, node_name: str) -> list[str]:
    return [*state.get("workflow_trace", []), node_name]


def state_to_request(state: AgentState) -> ChatRequest:
    return ChatRequest(
        text=state.get("text_input"),
        image_base64=state.get("image_base64"),
        image_url=state.get("image_url"),
        image_mime_type=state.get("image_mime_type"),
        age_hint=state.get("child_age_band"),
        session_id=state.get("session_id"),
    )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
