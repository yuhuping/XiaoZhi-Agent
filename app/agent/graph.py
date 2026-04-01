from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langsmith import trace, tracing_context

from app.agent.nodes.memory_update import MemoryUpdateNode
from app.agent.nodes.memory_compact import MemoryCompactNode
from app.agent.nodes.observe import ObserveNode
from app.agent.nodes.reason import ReasonNode
from app.agent.nodes.respond import RespondNode
from app.agent.nodes.response import ResponseNode
from app.agent.nodes.state_update import StateUpdateNode
from app.agent.nodes.understand import UnderstandNode
from app.agent.state import AgentState
from app.core.langsmith import is_langsmith_enabled
from app.services.model_service import ModelService
from app.tools.basic_tools import BasicTools

logger = logging.getLogger(__name__)


class AgentGraph:
    def __init__(
        self,
        model_service: ModelService,
        tools: BasicTools,
    ) -> None:
        self.settings = model_service.settings
        builder = StateGraph(AgentState)

        builder.add_node("understand", UnderstandNode(tools))
        builder.add_node("state_update", StateUpdateNode(tools))
        builder.add_node("chatbot", ReasonNode(tools))
        builder.add_node("tools", ToolNode(tools=tools.as_langgraph_tools(), messages_key="messages"))
        builder.add_node("observe", ObserveNode())
        builder.add_node("respond", RespondNode(tools))
        builder.add_node("memory_update", MemoryUpdateNode(tools))
        builder.add_node("response", ResponseNode())
        builder.add_node("memory_compact", MemoryCompactNode(tools))

        builder.add_edge(START, "understand")
        builder.add_edge("understand", "state_update")
        builder.add_edge("state_update", "chatbot")
        builder.add_conditional_edges(
            "chatbot",
            tools_condition,
            {
                "tools": "tools",
                "__end__": "observe",
            },
        )
        builder.add_edge("tools", "observe")
        builder.add_edge("observe", "respond")
        builder.add_edge("respond", "memory_update")
        builder.add_edge("memory_update", "response")
        builder.add_edge("response", "memory_compact")
        builder.add_edge("memory_compact", END)

        self.graph = builder.compile()
        logger.info("agent graph compiled")


    async def run(self, state: AgentState) -> AgentState:
        logger.info("workflow started")
        with tracing_context(
            enabled=is_langsmith_enabled(self.settings),
            project_name=self.settings.langsmith_project,
        ):
            with trace(
                "agent_graph",
                run_type="chain",
                inputs={
                    "session_id": state.get("session_id"),
                    "user_input": state.get("user_input"),
                    "input_modality": state.get("input_modality"),
                    "mode": state.get("interaction_mode"),
                },
                metadata={"component": "langgraph"},
            ) as run:
                result = await self.graph.ainvoke(state)
                run.end(
                    outputs={
                        "selected_act": result.get("final_response", {})
                        .get("react", {})
                        .get("selected_act"),
                        "workflow_trace": result.get("final_response", {})
                        .get("metadata", {})
                        .get("workflow_trace", []),
                    }
                )
        logger.info("workflow completed")
        return result
