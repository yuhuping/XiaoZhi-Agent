from __future__ import annotations

from app.agent.state import AgentState, PlannedAction

WORKFLOW_ROUTE = (
    "perception",
    "state_update",
    "planning",
    "action_router",
    "response",
)

ACTION_NODES: tuple[PlannedAction, ...] = (
    "greet",
    "explain_and_ask",
    "answer_question",
    "evaluate_answer",
    "clarify",
    "fallback",
)


def get_workflow_route() -> list[str]:
    return list(WORKFLOW_ROUTE)


def resolve_action_route(state: AgentState) -> PlannedAction:
    planned_action = state.get("planned_action")
    if planned_action in ACTION_NODES:
        return planned_action
    return "fallback"
