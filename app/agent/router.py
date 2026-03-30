from __future__ import annotations

from app.agent.state import AgentState

WORKFLOW_ROUTE = (
    "understand",
    "state_update",
    "chatbot",
    "tools",
    "observe",
    "respond",
    "memory_update",
    "response",
    "memory_compact",
)

ACT_ROUTE_MAP = {
    "direct": "act_direct",
    "retrieve_knowledge": "act_retrieve",
    "read_memory": "act_memory",
}


def get_workflow_route() -> list[str]:
    return list(WORKFLOW_ROUTE)


def resolve_act_route(state: AgentState) -> str:
    return ACT_ROUTE_MAP.get(state.get("selected_act", "direct"), "act_direct")
