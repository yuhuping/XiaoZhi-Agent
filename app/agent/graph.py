from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph
from langsmith import trace, tracing_context

from app.agent.nodes.action_router import ActionRouterNode
from app.agent.nodes.answer_question import AnswerQuestionNode
from app.agent.nodes.clarify import ClarifyNode
from app.agent.nodes.evaluate_answer import EvaluateAnswerNode
from app.agent.nodes.explain_and_ask import ExplainAndAskNode
from app.agent.nodes.fallback import FallbackNode
from app.agent.nodes.greet import GreetNode
from app.agent.nodes.perception import PerceptionNode
from app.agent.nodes.planning import PlanningNode
from app.agent.nodes.response import ResponseNode
from app.agent.nodes.state_update import StateUpdateNode
from app.agent.router import resolve_action_route
from app.agent.state import AgentState
from app.core.langsmith import is_langsmith_enabled
from app.services.model_service import ModelService
from app.services.session_store import SessionStore
from app.tools.basic_tools import BasicTools

logger = logging.getLogger(__name__)


class AgentGraph:
    def __init__(self, model_service: ModelService, session_store: SessionStore) -> None:
        self.settings = model_service.settings
        tools = BasicTools(model_service=model_service)
        builder = StateGraph(AgentState)

        builder.add_node("perception", PerceptionNode(tools))
        builder.add_node("state_update", StateUpdateNode(session_store))
        builder.add_node("planning", PlanningNode(tools))
        builder.add_node("action_router", ActionRouterNode())
        builder.add_node("greet", GreetNode(tools))
        builder.add_node("explain_and_ask", ExplainAndAskNode(tools))
        builder.add_node("answer_question", AnswerQuestionNode(tools))
        builder.add_node("evaluate_answer", EvaluateAnswerNode(tools))
        builder.add_node("clarify", ClarifyNode(tools))
        builder.add_node("fallback", FallbackNode(tools))
        builder.add_node("response", ResponseNode(session_store))

        builder.add_edge(START, "perception")
        builder.add_edge("perception", "state_update")
        builder.add_edge("state_update", "planning")
        builder.add_edge("planning", "action_router")
        builder.add_conditional_edges(
            "action_router",
            resolve_action_route,
            {
                "greet": "greet",
                "explain_and_ask": "explain_and_ask",
                "answer_question": "answer_question",
                "evaluate_answer": "evaluate_answer",
                "clarify": "clarify",
                "fallback": "fallback",
            },
        )
        builder.add_edge("greet", "response")
        builder.add_edge("explain_and_ask", "response")
        builder.add_edge("answer_question", "response")
        builder.add_edge("evaluate_answer", "response")
        builder.add_edge("clarify", "response")
        builder.add_edge("fallback", "response")
        builder.add_edge("response", END)

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
                },
                metadata={"component": "langgraph"},
            ) as run:
                result = await self.graph.ainvoke(state)
                run.end(
                    outputs={
                        "planned_action": result.get("final_response", {}).get("action"),
                        "workflow_trace": result.get("final_response", {})
                        .get("metadata", {})
                        .get("workflow_trace", []),
                    }
                )
        logger.info("workflow completed")
        return result
