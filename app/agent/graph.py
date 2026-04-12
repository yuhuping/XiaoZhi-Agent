from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langsmith import trace, tracing_context

from app.agent.nodes.execute import ExecuteNode
from app.agent.nodes.memory_compact import MemoryCompactNode
from app.agent.nodes.memory_update import MemoryUpdateNode
from app.agent.nodes.observe import ObserveNode
from app.agent.nodes.plan import PlanNode
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


def route_by_mode(state: AgentState) -> str:
    if state.get("interaction_mode") == "education":
        return "plan_execute"
    return "react"


def plan_route(state: AgentState) -> str:
    if state.get("selected_act") == "retrieve_knowledge":
        return "tools"
    return "execute"


def should_continue_react(state: AgentState) -> str:
    iteration = state.get("react_iteration", 1)
    max_iter = state.get("react_max_iterations", 3)
    if iteration >= max_iter:
        return "respond"
    if state.get("selected_act") == "direct":
        return "respond"
    # skill 工具执行一次即可，成功后直接生成回复
    if state.get("selected_act") == "skill" and state.get("tool_success"):
        return "respond"
    return "reason"


class AgentGraph:
    def __init__(
        self,
        model_service: ModelService,
        tools: BasicTools,
        skill_registry=None,
    ) -> None:
        self.settings = model_service.settings
        tool_list = tools.as_all_langgraph_tools()

        # --- PlanExecute subgraph ---
        pe = StateGraph(AgentState)
        pe.add_node("plan", PlanNode(tools))
        pe.add_node("tools", ToolNode(tools=tool_list, messages_key="messages"))
        pe.add_node("observe", ObserveNode(skill_registry=skill_registry))
        pe.add_node("execute", ExecuteNode(tools))

        pe.set_entry_point("plan")
        pe.add_conditional_edges("plan", plan_route, {"tools": "tools", "execute": "execute"})
        pe.add_edge("tools", "observe")
        pe.add_edge("observe", "execute")
        pe.add_edge("execute", END)

        plan_execute_graph = pe.compile()

        # --- ReAct subgraph ---
        ra = StateGraph(AgentState)
        ra.add_node("reason", ReasonNode(tools))
        ra.add_node("tools", ToolNode(tools=tool_list, messages_key="messages"))
        ra.add_node("observe", ObserveNode(skill_registry=skill_registry))
        ra.add_node("respond", RespondNode(tools))

        ra.set_entry_point("reason")
        ra.add_conditional_edges("reason", tools_condition, {"tools": "tools", "__end__": "respond"})
        ra.add_edge("tools", "observe")
        ra.add_conditional_edges("observe", should_continue_react, {"reason": "reason", "respond": "respond"})
        ra.add_edge("respond", END)

        react_graph = ra.compile()

        # --- Top-level Router ---
        router = StateGraph(AgentState)
        router.add_node("understand", UnderstandNode(tools))
        router.add_node("state_update", StateUpdateNode(tools))
        router.add_node("plan_execute", plan_execute_graph)
        router.add_node("react", react_graph)
        router.add_node("memory_update", MemoryUpdateNode(tools))
        router.add_node("response", ResponseNode())
        router.add_node("memory_compact", MemoryCompactNode(tools))

        router.add_edge(START, "understand")
        router.add_edge("understand", "state_update")
        router.add_conditional_edges("state_update", route_by_mode, {
            "plan_execute": "plan_execute",
            "react": "react",
        })
        router.add_edge("plan_execute", "memory_update")
        router.add_edge("react", "memory_update")
        router.add_edge("memory_update", "response")
        router.add_edge("response", "memory_compact")
        router.add_edge("memory_compact", END)

        self.graph = router.compile()
        logger.info("agent graph compiled (router + plan_execute + react subgraphs)")

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
