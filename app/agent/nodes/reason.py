from __future__ import annotations

import logging

from langchain_core.messages import AIMessage

from app.agent.state import AgentState, append_trace
from app.tools.basic_tools import BasicTools

logger = logging.getLogger(__name__)


class ReasonNode:
    def __init__(self, tools: BasicTools) -> None:
        self.tools = tools

    async def __call__(self, state: AgentState) -> AgentState:
        logger.info("entering node=chatbot")
        decision = await self.tools.reason_next_action(state)
        decision_act = decision.selected_act
        decision_reason = decision.route_reason
        selected_tool = decision.tool_name
        tool_input = dict(decision.tool_input or {})

        # Prefer explicit tool selection over act label to avoid dropping valid tool calls.
        if selected_tool in {"retrieve_knowledge", "tavily_search"}:
            decision_act = "retrieve_knowledge"
        elif selected_tool == "read_memory_bundle":
            decision_act = "read_memory"

        if decision_act == "retrieve_knowledge":
            selected_tool = selected_tool or "retrieve_knowledge"
            if "query" not in tool_input or not str(tool_input.get("query") or "").strip():
                tool_input["query"] = state.get("latest_user_text") or state.get("user_input") or ""
        elif decision_act == "read_memory":
            selected_tool = selected_tool or "read_memory_bundle"
            tool_input.setdefault("session_id", state["session_id"])
            tool_input.setdefault("profile_id", state["profile_id"])
        else:
            selected_tool = None
            tool_input = {}

        if selected_tool:
            call_suffix = "retrieve" if decision_act == "retrieve_knowledge" else "memory"
            tool_calls = [
                {
                    "id": f"tool_call_{call_suffix}_1",
                    "name": selected_tool,
                    "args": tool_input,
                    "type": "tool_call",
                }
            ]
        else:
            tool_calls = []
        # print(f"ReasonNode decision: act={decision_act}, tool_calls={tool_calls}, reason={decision_reason}, input={tool_input}")
        ai_message = AIMessage(content=decision.decision, tool_calls=tool_calls)
        return {
            "react_decision": decision.decision,
            "selected_act": decision_act,
            "selected_tool": selected_tool,
            "tool_input": tool_input,
            "route_reason": decision_reason,
            "topic_hint": decision.topic_hint or state.get("topic_hint"),
            "current_topic": state.get("current_topic") or decision.topic_hint,
            "confidence": decision.confidence,
            "source_mode": decision.source_mode,
            "dialogue_stage": "reasoned",
            "messages": [ai_message],
            "workflow_trace": append_trace(state, "chatbot"),
        }
