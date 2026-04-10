from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage

from app.agent.state import AgentState, append_trace
from app.tools.basic_tools import BasicTools

logger = logging.getLogger(__name__)

RETRIEVE_TOOL = "retrieve_knowledge"


class PlanNode:
    def __init__(self, tools: BasicTools) -> None:
        self.tools = tools

    async def __call__(self, state: AgentState) -> AgentState:
        logger.info("entering node=plan")
        plan = await self.tools.generate_plan(state)
        steps = plan.get("steps", ["直接回答用户问题"])
        if not steps:
            steps = ["直接回答用户问题"]
        needs_retrieval = plan.get("needs_retrieval", False)
        retrieval_query = plan.get("retrieval_query", "")

        tool_calls: list[dict[str, Any]] = []
        if needs_retrieval and retrieval_query:
            selected_act = "retrieve_knowledge"
            selected_tool = RETRIEVE_TOOL
            tool_input: dict[str, Any] = {"query": retrieval_query}
            tool_calls = [
                {
                    "id": "tool_call_plan_retrieve_1",
                    "name": RETRIEVE_TOOL,
                    "args": tool_input,
                    "type": "tool_call",
                }
            ]
        else:
            selected_act = "direct"
            selected_tool = None
            tool_input = {}

        ai_message = AIMessage(
            content=f"Plan: {len(steps)} steps, retrieval={needs_retrieval}",
            tool_calls=tool_calls,
        )

        return {
            "plan_steps": steps,
            "plan_raw": str(plan),
            "selected_act": selected_act,
            "selected_tool": selected_tool,
            "tool_input": tool_input,
            "dialogue_stage": "reasoned",
            "messages": [ai_message],
            "workflow_trace": append_trace(state, "plan"),
        }
