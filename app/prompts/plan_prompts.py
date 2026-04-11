from __future__ import annotations

import textwrap

from app.agent.state import AgentState
from app.schemas.chat import ChatRequest


def build_plan_instruction() -> str:
    # 节点：plan（PlanExecute 子图）
    # 每次 education 模式请求调用一次，生成解题步骤并决定是否需要 RAG 检索。
    return textwrap.dedent("""\
        You are XiaoZhi, an educational assistant for children ages 3-8.
        Analyze the user's question and create an internal solving plan.

        Rules:
        - Maximum 5 steps. Simple questions need only 1-2 steps.
        - Each step is a concise instruction for how to explain/solve this part.
        - Retrieval policy — set needs_retrieval=true for ANY of these topics:
          * Animals or plants: body structure, behavior, habitat, classification, life cycle, diet
          * Nature science: why/how questions about natural phenomena, ecosystems, biology
          * The knowledge base contains detailed animal and plant content (十万个为什么).
            Always prefer retrieval for animal/plant questions to give accurate, grounded answers.
          * Other factual lookups: history facts, geography, human body, astronomy
        - Set needs_retrieval=false ONLY for: pure math, logic puzzles, casual chat, or simple greetings.
        - When in doubt about animal or plant topics, always set needs_retrieval=true.
        - retrieval_query should be a short, focused noun phrase in the same language as the question.
        - Do NOT include a "give final answer" step — that is handled separately.
    """).strip()


def build_plan_user_prompt(chat_request: ChatRequest, state: AgentState) -> str:
    # 节点：plan（PlanExecute 子图）
    return textwrap.dedent(f"""\
        User question: {(chat_request.text or "").strip() or "No text provided."}
        Age: {chat_request.age_hint or "3-8"}
        Current topic: {state.get("current_topic") or "unknown"}
        Memory context: {state.get("Memory") or {}}
    """).strip()


def build_execute_instruction() -> str:
    # 节点：execute（PlanExecute 子图）
    # 每次 education 模式请求调用一次，流式生成面向儿童的解答内容。
    return textwrap.dedent("""\
        You are XiaoZhi, a warm and encouraging educational assistant for children ages 3-8.
        You have been given a step-by-step plan to explain/solve a question.
        Follow the plan and produce a complete, child-friendly explanation.

        Rules:
        - Use clear step markers (第一步、第二步... or Step 1、Step 2...) matching the user's language.
        - Explain each step in simple, encouraging language suitable for young children.
        - If reference material is provided, incorporate it naturally.
        - End with a brief summary of the answer.
        - Output plain text only. No JSON, no markdown fences.
    """).strip()


def build_execute_user_prompt(chat_request: ChatRequest, state: AgentState) -> str:
    # 节点：execute（PlanExecute 子图）
    steps_text = "\n".join(
        f"  {i+1}. {step}" for i, step in enumerate(state.get("plan_steps", []))
    )
    chunks = state.get("retrieved_chunks", [])
    if chunks:
        refs = "\n".join(
            f"  - {c.get('snippet', '')}" for c in chunks[:3]
        )
        reference_section = f"Reference material:\n{refs}"
    else:
        reference_section = "No reference material."

    return textwrap.dedent(f"""\
        User question: {(chat_request.text or "").strip() or "No text provided."}
        Age: {chat_request.age_hint or "3-8"}

        Plan steps:
        {steps_text}

        {reference_section}

        Memory context: {state.get("Memory") or {}}
        Recent history:
        {_format_history(state)}
    """).strip()


def _format_history(state: AgentState) -> str:
    history = state.get("history", [])
    if not history:
        return "No prior turns."
    lines = []
    for turn in history[-6:]:
        role = turn.get("role", "unknown")
        text = turn.get("text", "")
        lines.append(f"- {role}: {text}")
    return "\n".join(lines)
