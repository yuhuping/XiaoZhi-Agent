from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Literal, TypedDict
from uuid import uuid4

from app.schemas.chat import ActType, ChatRequest, InputModality, InteractionMode

DialogueStage = Literal[
    "received",
    "understood",
    "state_ready",
    "reasoned",
    "acted",
    "observed",
    "responded",
]
ConversationRole = Literal["user", "assistant"]
SourceMode = Literal["llm"]
ConfidenceLevel = Literal["high", "medium", "low"]
StreamDeltaWriter = Callable[[str], Awaitable[None] | None]
DEFAULT_CHILD_PROFILE_ID = "default_child"
DEFAULT_PARENT_PROFILE_ID = "default_parent"
# 按模式映射默认画像ID，避免家长模式写入儿童画像。
DEFAULT_PROFILE_ID_BY_MODE: dict[InteractionMode, str] = {
    "education": DEFAULT_CHILD_PROFILE_ID,
    "companion": DEFAULT_CHILD_PROFILE_ID,
    "parent": DEFAULT_PARENT_PROFILE_ID,
}


class ConversationTurn(TypedDict, total=False):
    role: ConversationRole
    text: str
    image_base64: str | None
    image_url: str | None
    image_mime_type: str | None
    topic: str | None
    asked_question: str | None
    mode: InteractionMode
    timestamp: str


class AgentState(TypedDict, total=False):
    session_id: str
    profile_id: str
    turn_index: int
    user_input: str
    latest_user_text: str | None
    text_input: str | None
    normalized_text: str
    image_base64: str | None
    image_url: str | None
    image_mime_type: str | None
    input_modality: InputModality
    interaction_mode: InteractionMode
    child_age_band: str
    current_topic: str | None
    detected_object: str | None
    topic_hint: str | None
    dialogue_stage: DialogueStage
    perception_signals: list[str]
    history: list[ConversationTurn]
    last_agent_question: str | None
    pending_topic: str | None
    route_reason: str
    react_decision: str
    selected_act: ActType
    selected_tool: str | None
    tool_input: dict[str, Any]
    tool_result: dict[str, Any]
    tool_success: bool
    retrieved_chunks: list[dict[str, Any]]
    observation_summary: str
    short_Memory: dict[str, Any]
    Memory: dict[str, Any]
    rag_enabled: bool
    message_draft: str | None
    follow_up_question: str | None
    confidence: ConfidenceLevel
    safety_notes: str
    source_mode: SourceMode
    memory_session_updated: bool
    memory_profile_updated: bool
    memory_written_types: list[str]
    memory_consolidated_count: int
    memory_forgotten_count: int
    stream_delta_writer: StreamDeltaWriter | None
    final_response: dict[str, Any]
    workflow_trace: list[str]
    messages: list[Any]


def resolve_default_profile_id(mode: InteractionMode) -> str:
    # 根据交互模式返回默认画像ID。
    return DEFAULT_PROFILE_ID_BY_MODE.get(mode, DEFAULT_CHILD_PROFILE_ID)


def resolve_profile_id(profile_id: str | None, mode: InteractionMode) -> str:
    # 若前端未传画像ID，则按模式兜底。
    raw_profile_id = (profile_id or "").strip()
    if raw_profile_id:
        return raw_profile_id
    return resolve_default_profile_id(mode)


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
    resolved_session_id = session_id or request.session_id or str(uuid4())
    resolved_profile_id = resolve_profile_id(profile_id=request.profile_id, mode=request.mode)
    return {
        "session_id": resolved_session_id,
        "profile_id": resolved_profile_id,
        "turn_index": 0,
        "user_input": normalized_text,
        "latest_user_text": normalized_text or None,
        "text_input": request.text,
        "normalized_text": normalized_text.lower(),
        "image_base64": request.image_base64,
        "image_url": str(request.image_url) if request.image_url else None,
        "image_mime_type": request.image_mime_type,
        "input_modality": modality,
        "interaction_mode": request.mode,
        "child_age_band": request.age_hint or "3-8",
        "dialogue_stage": "received",
        "route_reason": "",
        "perception_signals": [],
        "history": [],
        "message_draft": None,
        "follow_up_question": None,
        "confidence": "medium",
        "safety_notes": "",
        "source_mode": "llm",
        "react_decision": "respond_directly",
        "selected_act": "direct",
        "selected_tool": None,
        "tool_input": {},
        "tool_result": {},
        "tool_success": False,
        "retrieved_chunks": [],
        "observation_summary": "",
        "short_Memory": {},
        "Memory": {},
        "rag_enabled": True,
        "memory_session_updated": False,
        "memory_profile_updated": False,
        "memory_written_types": [],
        "memory_consolidated_count": 0,
        "memory_forgotten_count": 0,
        "stream_delta_writer": None,
        "final_response": {},
        "workflow_trace": [],
        "messages": [],
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
        profile_id=state.get("profile_id"),
        mode=state.get("interaction_mode", "education"),
    )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
