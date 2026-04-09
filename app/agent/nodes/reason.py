from __future__ import annotations

import logging
import re
from datetime import date, datetime, timedelta
from typing import Any

from langchain_core.messages import AIMessage

from app.agent.state import AgentState, append_trace
from app.schemas.chat import ActType
from app.tools.basic_tools import BasicTools

logger = logging.getLogger(__name__)


class ReasonNode:
    RETRIEVE_TOOL = "retrieve_knowledge"
    WEB_SEARCH_TOOL = "tavily_search"
    MEMORY_TOOL = "read_memory_bundle"
    ABSOLUTE_DATE_PATTERNS = (
        re.compile(r"\b\d{4}[-/]\d{1,2}([-/]\d{1,2})?\b"),
        re.compile(r"\d{4}年\d{1,2}月(\d{1,2}日?)?"),
        re.compile(r"\d{1,2}月\d{1,2}日"),
    )

    def __init__(self, tools: BasicTools) -> None:
        self.tools = tools

    async def __call__(self, state: AgentState) -> AgentState:
        logger.info("entering node=chatbot")
        decision = await self.tools.reason_next_action(state)
        decision_act = self._override_act_from_selected_tool(
            decision.selected_act,
            decision.tool_name,
        )
        decision_reason = decision.route_reason
        selected_tool, tool_input = self._normalize_tool_call(
            state=state,
            decision_act=decision_act,
            selected_tool=decision.tool_name,
            tool_input=decision.tool_input,
        )
        if decision_act == "skill" and not selected_tool:
            # Guardrail: skill route without tool name cannot run in ToolNode.
            decision_act = "direct"

        tool_calls = self._build_tool_calls(
            decision_act=decision_act,
            selected_tool=selected_tool,
            tool_input=tool_input,
        )
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

    def _override_act_from_selected_tool(self, decision_act: ActType, selected_tool: str | None) -> ActType:
        """Prefer explicit tool selection over LLM act label."""
        if selected_tool == self.RETRIEVE_TOOL:
            return "retrieve_knowledge"
        if selected_tool == self.WEB_SEARCH_TOOL:
            return "tavily_search"
        if selected_tool == self.MEMORY_TOOL:
            return "read_memory"
        return decision_act

    def _normalize_tool_call(
        self,
        state: AgentState,
        decision_act: ActType,
        selected_tool: str | None,
        tool_input: dict[str, Any] | None,
    ) -> tuple[str | None, dict[str, Any]]:
        """Build final tool name + args used by ToolNode."""
        prepared_input = dict(tool_input or {})

        if decision_act == "retrieve_knowledge":
            tool_name = selected_tool or self.RETRIEVE_TOOL
            return tool_name, self._ensure_retrieval_query(prepared_input, state)

        if decision_act == "tavily_search":
            normalized_input: dict[str, Any] = {}
            if "top_k" in prepared_input:
                normalized_input["top_k"] = prepared_input.get("top_k")
            normalized_input["query"] = self._build_tavily_query(state, prepared_input)
            return self.WEB_SEARCH_TOOL, normalized_input

        if decision_act == "read_memory":
            return self.MEMORY_TOOL, self._build_memory_tool_input(state)

        if decision_act == "skill":
            # Skill act must have an explicit tool name from the planner.
            if selected_tool:
                return selected_tool, prepared_input
            return None, {}

        return None, {}

    def _ensure_retrieval_query(self, tool_input: dict[str, Any], state: AgentState) -> dict[str, Any]:
        """Fill query fallback for retrieval/search tools."""
        query = str(tool_input.get("query") or "").strip()
        if query:
            return tool_input
        tool_input["query"] = state.get("latest_user_text") or state.get("user_input") or ""
        return tool_input

    def _build_tavily_query(self, state: AgentState, tool_input: dict[str, Any]) -> str:
        """Force web-search query from user text, then normalize relative dates."""
        llm_query = str(tool_input.get("query") or "").strip()
        original_user_query = str(state.get("latest_user_text") or state.get("user_input") or "").strip()
        anchor_local = datetime.now().astimezone()
        normalized_query = self._normalize_relative_date_query(
            query=original_user_query,
            anchor_date=anchor_local.date(),
        )
        final_query = normalized_query or original_user_query or llm_query
        logger.info(
            "[reason_route] tavily query override: "
            "original_user_query=%r normalized_query=%r llm_query=%r "
            "anchor_datetime_local=%s timezone_source=server_local",
            original_user_query,
            final_query,
            llm_query,
            anchor_local.isoformat(timespec="seconds"),
        )
        return final_query

    def _normalize_relative_date_query(self, query: str, anchor_date: date) -> str:
        normalized = (query or "").strip()
        if not normalized:
            return ""
        if self._contains_absolute_date(normalized):
            return normalized

        replacements = [
            (("后天",), 2),
            (("明天", "明晚"), 1),
            (("今天", "今晚", "今夜"), 0),
            (("昨天", "昨晚"), -1),
        ]
        for tokens, delta_days in replacements:
            absolute_date = self._format_absolute_date(anchor_date + timedelta(days=delta_days))
            for token in tokens:
                normalized = normalized.replace(token, absolute_date)
        return normalized.strip()

    def _contains_absolute_date(self, query: str) -> bool:
        return any(pattern.search(query) for pattern in self.ABSOLUTE_DATE_PATTERNS)

    def _format_absolute_date(self, day: date) -> str:
        return day.strftime("%Y-%m-%d")

    def _build_tool_calls(
        self,
        decision_act: ActType,
        selected_tool: str | None,
        tool_input: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if not selected_tool:
            return []
        call_suffix = self._call_suffix_for_act(decision_act)
        return [
            {
                "id": f"tool_call_{call_suffix}_1",
                "name": selected_tool,
                "args": tool_input,
                "type": "tool_call",
            }
        ]

    def _call_suffix_for_act(self, decision_act: ActType) -> str:
        if decision_act == "retrieve_knowledge":
            return "retrieve"
        if decision_act == "read_memory":
            return "memory"
        if decision_act == "tavily_search":
            return "web"
        if decision_act == "skill":
            return "skill"
        return "tool"

    def _build_memory_tool_input(self, state: AgentState) -> dict[str, str]:
        # Memory tool IDs are system context, so we never trust model-generated overrides here.
        return {
            "session_id": state["session_id"],
            "profile_id": state["profile_id"],
        }
