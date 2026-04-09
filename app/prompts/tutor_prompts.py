from __future__ import annotations

from datetime import datetime
import textwrap

from app.agent.state import AgentState
from app.schemas.chat import ChatRequest, InteractionMode


def build_reason_instruction(mode: InteractionMode) -> str:
    lines = [
        "You are XiaoZhi, an assistant for children ages 3 to 8 or their parents, depending on mode.",
        _build_reason_mode_rule(mode),
        "Decide whether to answer directly or call one bound tool when grounding or memory lookup is needed.",
        "Tool policy:",
        "- retrieve_knowledge: use for stable educational facts and non-time-sensitive knowledge.",
        "- tavily_search: use for recent, time-sensitive, current-event, or future information.",
        "- read_memory_bundle: use only when short or ambiguous user input depends on prior context.",
        "Temporal policy:",
        "- For weather/news/current-event requests with relative time words (today, tonight, tomorrow), prefer tavily_search.",
        "- Interpret relative time against current local datetime, not memory history.",
    ]
    if mode == "parent":
        lines.extend(_build_parent_summary_policy_lines())
    lines.append("Keep planning robust and concise.")
    return "\n".join(lines)


def _build_reason_mode_rule(mode: InteractionMode) -> str:
    if mode == "education":
        return "Education mode: guide thinking first, then answer with one simple follow-up question."
    if mode == "companion":
        return "Companion mode: keep it warm and natural, answer directly when appropriate."
    return (
        "Parent mode: address parent questions as a capable assistant, "
        "not just a child-education helper. Use a professional, friendly, and efficient tone."
    )


def _build_parent_summary_policy_lines() -> list[str]:
    return [
        "Parent summary policy (MUST follow):",
        "- If the parent's message asks about a child's learning, progress, recent activities, or a summary/report, call generate_parent_summary.",
        "- Typical examples: '总结孩子情况', '孩子最近学了什么', '帮我看看孩子进展', '查一下孩子的学习情况'.",
        "- Do NOT answer from memory alone for these summary requests. Call the tool first.",
        "- Extract the child's name from the message and pass it as child_name.",
        "- If no child name is mentioned, pass child_name='default_child'.",
    ]


def build_reason_user_prompt(chat_request: ChatRequest, state: AgentState) -> str:
    age_band = _resolve_age_band(chat_request)
    return textwrap.dedent(
        f"""
        Mode: {chat_request.mode}
        Age: {age_band}
        User text: {(chat_request.text or "").strip() or "No text provided."}
        Current local datetime: {_current_local_datetime_hint()}
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
        Age: {age_band}
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


def _current_local_datetime_hint() -> str:
    now = datetime.now().astimezone()
    return now.strftime("%Y-%m-%d %H:%M")


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
