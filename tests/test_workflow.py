import asyncio

from app.agent.graph import AgentGraph
from app.agent.state import build_initial_state
from app.core.config import Settings
from app.schemas.chat import ChatRequest
from app.services.model_service import ModelService
from app.services.session_store import SessionStore


def test_agent_state_initialization_for_multimodal_input() -> None:
    request = ChatRequest(
        text="Tell me about this apple.",
        image_url="https://example.com/apple.png",
        age_hint="4-6",
        session_id="session-1",
    )

    state = build_initial_state(request)

    assert state["input_modality"] == "multimodal"
    assert state["child_age_band"] == "4-6"
    assert state["session_id"] == "session-1"
    assert state["dialogue_stage"] == "received"


async def _run_greet_flow() -> None:
    graph = AgentGraph(
        ModelService(Settings(openai_api_key="test-key")),
        SessionStore(),
    )
    state = build_initial_state(ChatRequest(text="hello", session_id="wf-greet"))

    final_state = await graph.run(state=state)
    response = final_state["final_response"]

    assert response["action"] == "greet"
    assert response["metadata"]["planned_action"] == "greet"
    assert response["metadata"]["workflow_trace"] == [
        "perception",
        "state_update",
        "planning",
        "action_router",
        "greet",
        "response",
    ]
    assert response["message"]


async def _run_evaluate_answer_flow() -> None:
    session_store = SessionStore()
    graph = AgentGraph(
        ModelService(Settings(openai_api_key="test-key")),
        session_store,
    )
    first_state = build_initial_state(
        ChatRequest(text="Tell me about an apple.", session_id="wf-memory")
    )
    await graph.run(state=first_state)

    second_state = build_initial_state(ChatRequest(text="Red", session_id="wf-memory"))
    final_state = await graph.run(state=second_state)
    response = final_state["final_response"]

    assert response["action"] == "evaluate_answer"
    assert response["metadata"]["workflow_trace"] == [
        "perception",
        "state_update",
        "planning",
        "action_router",
        "evaluate_answer",
        "response",
    ]


def test_workflow_graph_greet_route() -> None:
    asyncio.run(_run_greet_flow())


def test_workflow_graph_can_use_session_memory() -> None:
    asyncio.run(_run_evaluate_answer_flow())
