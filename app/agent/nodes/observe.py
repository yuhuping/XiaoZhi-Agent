from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import ToolMessage

from app.agent.state import AgentState

logger = logging.getLogger(__name__)


class ObserveNode:
    async def __call__(self, state: AgentState) -> AgentState:
        logger.info("entering node=observe")
        selected_act = state.get("selected_act", "direct")
        messages = state.get("messages", [])
        last_tool_payload = self._extract_last_tool_payload(messages)

        updates: dict[str, Any] = {}
        if selected_act == "retrieve_knowledge":
            chunks = last_tool_payload.get("results", [])
            summary = f"retrieved {len(chunks)} chunks via {state.get('selected_tool') or 'tool'}"
            updates["retrieved_chunks"] = chunks if isinstance(chunks, list) else []
            updates["tool_result"] = last_tool_payload
            updates["tool_success"] = bool(last_tool_payload.get("tool_success", False))
        elif selected_act == "read_memory":
            session_snapshot = last_tool_payload.get("session", {})
            profile_snapshot = last_tool_payload.get("profile", {})
            summary = (
                "loaded memory "
                f"(session_turns={len(session_snapshot.get('recent_turns', []))}, "
                f"profile_fields={len(profile_snapshot.keys())})"
            )
            updates["short_Memory"] = (
                session_snapshot if isinstance(session_snapshot, dict) else {}
            )
            updates["Memory"] = (
                profile_snapshot if isinstance(profile_snapshot, dict) else {}
            )
            updates["tool_result"] = last_tool_payload
            updates["tool_success"] = bool(last_tool_payload.get("tool_success", False))
        else:
            summary = "no external tool used"
            updates["tool_result"] = {}
            updates["tool_success"] = True

        workflow_trace = state.get("workflow_trace", [])
        if selected_act != "direct" and "tools" not in workflow_trace:
            workflow_trace = [*workflow_trace, "tools"]
        updates.update(
            {
                "observation_summary": summary,
                "dialogue_stage": "observed",
                "workflow_trace": [*workflow_trace, "observe"],
            }
        )
        return updates

    def _extract_last_tool_payload(self, messages: list[Any]) -> dict[str, Any]:
        if not messages:
            return {}
        for message in reversed(messages):
            if not isinstance(message, ToolMessage):
                continue
            content = message.content
            text = content if isinstance(content, str) else ""
            if not text and isinstance(content, list):
                text_parts = [part for part in content if isinstance(part, str)]
                text = "\n".join(text_parts)
            if not text:
                return {}
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                return {"raw": text}
            return parsed if isinstance(parsed, dict) else {"raw": text}
        return {}
