from __future__ import annotations

import textwrap

from app.agent.state import AgentState, PlannedAction
from app.schemas.chat import ChatRequest


def build_routing_instruction() -> str:
    return textwrap.dedent(
        """
        You are the routing brain for XiaoZhi, a child-learning assistant for children aged 3 to 8.
        Choose the next action for the current turn.
        Priorities:
        1. Pure greetings like "hello" or "你好" should route to greet.
        2. Image recognition or simple "what is this" learning should route to explain_and_ask.
        3. Direct knowledge questions should route to answer_question.
        4. If the child is answering the assistant's previous question, route to evaluate_answer.
        5. If the input depends on missing context, route to clarify.
        6. Use fallback only when no better action fits or safety requires redirection.
        Do not treat a greeting as a vocabulary word to explain.
        """
    ).strip()


def build_routing_user_prompt(chat_request: ChatRequest, state: AgentState) -> str:
    history_text = _format_history(state)
    return textwrap.dedent(
        f"""
        Child age range: {chat_request.age_hint or "3-8"}
        Current user text: {(chat_request.text or "").strip() or "No text provided."}
        Has image input: {"yes" if chat_request.image_base64 or chat_request.image_url else "no"}
        Topic hint: {state.get("current_topic") or state.get("topic_hint") or "unknown"}
        Last assistant action: {state.get("last_agent_action") or "none"}
        Last assistant question: {state.get("last_agent_question") or "none"}
        Perception signals: {", ".join(state.get("perception_signals", [])) or "none"}
        Recent history:
        {history_text}

        Return JSON with:
        - user_intent: greeting | object_learning | direct_question | answer_attempt | unclear | fallback
        - planned_action: greet | explain_and_ask | answer_question | evaluate_answer | clarify | fallback
        - route_reason: a short sentence
        - topic_hint: short noun phrase or empty string
        - confidence: high | medium | low
        """
    ).strip()


def build_action_instruction(action: PlannedAction) -> str:
    base = """
        You are XiaoZhi, a child-learning assistant for children aged 3 to 8.
        Keep the answer warm, short, educational, and easy to understand.
        Do not act like a parent, therapist, or best friend.
        Do not include unsafe, adult, scary, or manipulative content.
        Keep the reply natural and conversational.
    """
    action_rules = {
        "greet": """
            The child is greeting you.
            Reply with a short welcome message and optionally one gentle learning invitation.
            Do not explain the literal meaning of the greeting.
        """,
        "explain_and_ask": """
            Explain the object or topic simply in one or two short sentences.
            Ask exactly one easy follow-up question.
        """,
        "answer_question": """
            Answer the child's question directly in simple language.
            You may add one easy follow-up question if it helps continue learning.
        """,
        "evaluate_answer": """
            The child is answering your previous question.
            Start with encouragement, then give a light correction or extension if needed.
            You may ask one easy next question.
        """,
        "clarify": """
            Politely ask a short clarifying question because there is not enough context.
            Do not invent a full explanation.
        """,
        "fallback": """
            Redirect gently to a safe and simple learning exchange.
            Keep it short and supportive.
        """,
    }
    return textwrap.dedent(base + action_rules[action]).strip()


def build_action_user_prompt(
    chat_request: ChatRequest,
    state: AgentState,
    action: PlannedAction,
) -> str:
    history_text = _format_history(state)   # _format_history会返回近6轮的对话历史，格式化成字符串 | 有待优化
    return textwrap.dedent(
        f"""
        Child age range: {chat_request.age_hint or "3-8"}
        Current action: {action}
        Current user text: {(chat_request.text or "").strip() or "No text provided."}
        Has image input: {"yes" if chat_request.image_base64 or chat_request.image_url else "no"}
        Current topic: {state.get("current_topic") or state.get("pending_topic") or "unknown"}
        Last assistant question: {state.get("last_agent_question") or "none"}
        Recent history:
        {history_text}

        Return JSON with:
        - topic: short noun phrase, or empty string if not needed
        - message: the main assistant reply
        - follow_up_question: one short question or empty string
        - confidence: high | medium | low
        - safety_notes: short string, empty if no issue
        """
    ).strip()


def _format_history(state: AgentState) -> str:
    history = state.get("history", [])
    if not history:
        return "No prior turns."
    lines = []
    for turn in history[-6:]:
        role = turn.get("role", "unknown")
        text = turn.get("text", "")
        action = turn.get("action") or "none"
        lines.append(f"- {role} [{action}]: {text}")
    return "\n".join(lines)
