from __future__ import annotations

import textwrap

from app.agent.state import AgentState
from app.schemas.chat import ChatRequest, InteractionMode


def build_reason_instruction(mode: InteractionMode) -> str:
    if mode == "education":
        mode_rule = "Education mode: guide thinking first, then answer with one simple follow-up question."
    elif mode == "companion":
        mode_rule = "Companion mode: keep it warm and natural, answer directly when appropriate."
    else:
        mode_rule = (
            "Parent mode: Do your best to address any questions raised by the parents!"
            "not just a child-education helper. Use a professional, friendly, and efficient tone. "
            "Be maximally helpful!"
        )
    return textwrap.dedent(
        f"""
        You are XiaoZhi, an assistant designed for children ages 3 to 8 or their parents depend on follow mode.
        {mode_rule}
        Decide whether to answer directly or call one bound tool when grounding or memory lookup is needed.
        Retrieval policy:
        - Prefer local knowledge retrieval for stable educational facts.
        - Prefer web search only for recent, time-sensitive, or current-event questions.
        Memory policy:
        - Use memory lookup only when short child input depends on prior context.
        Keep planning robust and concise.
        """
    ).strip()


def build_reason_user_prompt(chat_request: ChatRequest, state: AgentState) -> str:
    age_band = _resolve_age_band(chat_request)
    return textwrap.dedent(
        f"""
        Mode: {chat_request.mode}
        Age band: {age_band}
        User text: {(chat_request.text or "").strip() or "No text provided."}
        Has image: {"yes" if chat_request.image_base64 or chat_request.image_url else "no"}
        Current topic: {state.get("current_topic") or state.get("topic_hint") or "unknown"}
        Last assistant question: {state.get("last_agent_question") or "none"}
        Perception signals: {", ".join(state.get("perception_signals", [])) or "none"}
        Memory:
        {state.get("Memory") or {}}
        Recent history:
        {_format_history(state)}
        """
    ).strip()


def build_response_instruction(mode: InteractionMode) -> str:
    if mode == "education":
        mode_rule = "Education mode: explain briefly, guide with one easy follow-up question."
    elif mode == "companion":
        mode_rule = "Companion mode: respond naturally and warmly, follow-up question is optional."
    else:
        mode_rule = (
            "Parent mode: act as a capable personal assistant for the parent or caregiver, "
            "not just a child-education helper. Use a professional, friendly, and efficient tone. "
            "Prioritize direct answers, practical suggestions, clear reasoning, and actionable next steps. "
            "Be maximally helpful within safety limits, and avoid child-directed wording unless explicitly needed."
        )
    return textwrap.dedent(
        f"""
        You are XiaoZhi, an assistant designed for children ages 3 to 8 or their parents depend on follow mode.
        {mode_rule}
        Keep response short, safe, warm, and clear.
        Avoid adult, unsafe, manipulative, or scary content.
        In parent mode, prioritize clarity, key facts, and actionable suggestions.
        """
    ).strip()


def build_response_user_prompt(
    chat_request: ChatRequest,
    state: AgentState,
    include_json_contract: bool = True,
) -> str:
    age_band = _resolve_age_band(chat_request)
    base_prompt = textwrap.dedent(
        f"""
        Mode: {chat_request.mode}
        Age band: {age_band}
        User text: {(chat_request.text or "").strip() or "No text provided."}
        Current topic: {state.get("current_topic") or "unknown"}
        ReAct decision: {state.get("react_decision") or "none"}
        Selected act: {state.get("selected_act") or "direct"}
        Observation summary: {state.get("observation_summary") or "none"}
        Retrieved context:
        {_format_retrieved_chunks(state)}
        Memory:
        {state.get("Memory") or {}}
        Recent history:
        {_format_history(state)}
        """
    ).strip()
    if not include_json_contract:
        return base_prompt
    return (
        f"{base_prompt}\n\n"
        "Return JSON:\n"
        "- topic: short noun phrase or empty string\n"
        "- message: final child-friendly reply\n"
        "- follow_up_question: one short question or empty string\n"
        "- confidence: high | medium | low\n"
        "- safety_notes: short string, empty if no issue"
    )


def _format_history(state: AgentState) -> str:
    history = state.get("history", [])
    # print(f'debug: history={history}')
    if not history:
        return "No prior turns."
    lines = []
    for turn in history[-6:]:
        role = turn.get("role", "unknown")
        text = turn.get("text", "")
        lines.append(f"- {role}: {text}")
    return "\n".join(lines)


def _resolve_age_band(chat_request: ChatRequest) -> str:
    if chat_request.mode == "parent":
        return "成年人"
    return chat_request.age_hint or "3-8"


def _format_retrieved_chunks(state: AgentState) -> str:
    chunks = state.get("retrieved_chunks", [])
    if not chunks:
        return "No retrieval."
    lines = []
    for chunk in chunks[:3]:
        lines.append(
            f"- source={chunk.get('source')} score={chunk.get('score')} snippet={chunk.get('snippet')}"
        )
    return "\n".join(lines)
