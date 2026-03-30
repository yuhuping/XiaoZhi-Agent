import asyncio
import json
from pathlib import Path
import sqlite3
import tempfile

import pytest

from app.agent.graph import AgentGraph
from app.agent.state import build_initial_state
from app.core.config import Settings
from app.memory import MemoryConfig, MemoryManager, MemoryTool
from app.rag.retriever import LocalKnowledgeRetriever
from app.schemas.chat import ChatRequest
from app.services.model_service import ModelService
from app.tools.basic_tools import BasicTools


def _build_graph(tmp_dir: Path) -> AgentGraph:
    settings = Settings(
        openai_api_key="test-key",
        kg_dir=str(tmp_dir / "KG"),
        memory_db_path=str(tmp_dir / "memory.sqlite3"),
        memory_index_dir=str(tmp_dir / "memory_index"),
        memory_reset_on_start=False,
    )
    model_service = ModelService(settings)
    memory_config = MemoryConfig(
        db_path=settings.memory_db_path,
        index_dir=settings.memory_index_dir,
        working_memory_capacity=settings.memory_working_capacity,
        working_memory_ttl_minutes=settings.memory_working_ttl_minutes,
    )
    memory_manager = MemoryManager(memory_config)
    memory_tool = MemoryTool(memory_manager)
    retriever = LocalKnowledgeRetriever.from_kg_dir(settings.kg_dir, auto_bootstrap=True)
    tools = BasicTools(
        model_service=model_service,
        memory_tool=memory_tool,
        retriever=retriever,
    )
    return AgentGraph(model_service=model_service, tools=tools)


def _read_long_term_counts(db_path: Path, user_id: str) -> dict[str, int]:
    """读取长期记忆计数：用于断言每轮只落1条长期记录。"""
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT memory_type, COUNT(*)
            FROM memory_items
            WHERE archived=0 AND user_id=?
            GROUP BY memory_type
            """,
            (user_id,),
        ).fetchall()
    finally:
        conn.close()

    counts = {"episodic": 0, "semantic": 0, "perceptual": 0}
    for memory_type, raw_count in rows:
        if isinstance(memory_type, str) and memory_type in counts:
            counts[memory_type] = int(raw_count)
    return counts


def test_agent_state_initialization_for_multimodal_input() -> None:
    request = ChatRequest(
        text="Tell me about this apple.",
        image_url="https://example.com/apple.png",
        age_hint="4-6",
        session_id="session-1",
        mode="education",
        profile_id="kid-1",
    )

    state = build_initial_state(request)

    assert state["input_modality"] == "multimodal"
    assert state["child_age_band"] == "4-6"
    assert state["session_id"] == "session-1"
    assert state["interaction_mode"] == "education"
    assert state["profile_id"] == "kid-1"
    assert state["dialogue_stage"] == "received"


@pytest.mark.parametrize(
    ("mode", "expected_profile_id"),
    [
        ("education", "default_child"),
        ("companion", "default_child"),
        ("parent", "default_parent"),
    ],
)
def test_agent_state_uses_mode_aware_default_profile_id_when_missing(
    mode: str, expected_profile_id: str
) -> None:
    # 未传profile_id时，默认值应跟随mode。
    request = ChatRequest(
        text="hello",
        session_id="session-default-profile",
        mode=mode,
    )
    state = build_initial_state(request)
    assert state["profile_id"] == expected_profile_id


async def _run_direct_flow(tmp_dir: Path) -> None:
    graph = _build_graph(tmp_dir)
    state = build_initial_state(ChatRequest(text="hello", session_id="wf-direct", mode="companion"))

    final_state = await graph.run(state=state)
    response = final_state["final_response"]

    assert response["mode"] == "companion"
    assert response["react"]["selected_act"] == "direct"
    assert response["memory"]["profile_updated"] is True
    assert response["memory"]["written_types"] == ["working", "episodic"]
    assert response["metadata"]["workflow_trace"] == [
        "understand",
        "state_update",
        "chatbot",
        "observe",
        "respond",
        "memory_update",
        "response",
    ]
    assert response["message"]
    counts = _read_long_term_counts(tmp_dir / "memory.sqlite3", "default_child")
    assert counts["episodic"] == 1
    assert counts["semantic"] == 0
    assert counts["perceptual"] == 0


async def _run_react_memory_flow(tmp_dir: Path) -> None:
    graph = _build_graph(tmp_dir)
    first_state = build_initial_state(
        ChatRequest(text="Tell me about an apple.", session_id="wf-memory", mode="education")
    )
    await graph.run(state=first_state)

    second_state = build_initial_state(ChatRequest(text="Red", session_id="wf-memory", mode="education"))
    final_state = await graph.run(state=second_state)
    response = final_state["final_response"]

    assert response["react"]["selected_act"] == "read_memory"
    assert response["memory"]["session_updated"] is True
    assert response["memory"]["profile_updated"] is True
    assert response["memory"]["written_types"] == ["working", "episodic"]
    assert response["metadata"]["workflow_trace"] == [
        "understand",
        "state_update",
        "chatbot",
        "tools",
        "observe",
        "respond",
        "memory_update",
        "response",
    ]
    counts = _read_long_term_counts(tmp_dir / "memory.sqlite3", "default_child")
    assert counts["episodic"] == 2
    assert counts["semantic"] == 0
    assert counts["perceptual"] == 0


async def _run_tavily_route_flow(tmp_dir: Path) -> None:
    graph = _build_graph(tmp_dir)
    state = build_initial_state(
        ChatRequest(text="latest science news for kids", session_id="wf-tavily", mode="education")
    )

    final_state = await graph.run(state=state)
    response = final_state["final_response"]

    assert response["react"]["selected_act"] == "retrieve_knowledge"
    assert response["react"]["tool_name"] == "tavily_search"
    assert response["metadata"]["workflow_trace"] == [
        "understand",
        "state_update",
        "chatbot",
        "tools",
        "observe",
        "respond",
        "memory_update",
        "response",
    ]


async def _run_memory_compact_flow(tmp_dir: Path) -> None:
    graph = _build_graph(tmp_dir)
    for idx in range(12):
        state = build_initial_state(
            ChatRequest(
                text=f"apple learning turn {idx}",
                session_id="wf-compact",
                mode="education",
                profile_id="kid-compact",
            )
        )
        await graph.run(state=state)

    db_path = tmp_dir / "memory.sqlite3"
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT memory_id, metadata
            FROM memory_items
            WHERE archived=0 AND user_id=? AND memory_type='episodic'
            """,
            ("kid-compact",),
        ).fetchall()
    finally:
        conn.close()

    alive_ids = {str(row[0]) for row in rows}
    compacted_meta: list[dict[str, object]] = []
    for _, metadata_raw in rows:
        if not isinstance(metadata_raw, str):
            continue
        try:
            metadata = json.loads(metadata_raw)
        except Exception:
            continue
        if isinstance(metadata, dict) and metadata.get("compacted") is True:
            compacted_meta.append(metadata)

    assert compacted_meta, "memory_compact should create compacted episodic summaries"

    has_episodic_source = False
    for metadata in compacted_meta:
        source_types = metadata.get("source_memory_types")
        if isinstance(source_types, list):
            normalized_types = [str(item) for item in source_types]
            if "episodic" in normalized_types:
                has_episodic_source = True
            assert "semantic" not in normalized_types
        source_ids = metadata.get("source_memory_ids")
        if not isinstance(source_ids, list):
            continue
        normalized_ids = [str(item) for item in source_ids]
        assert normalized_ids, "compacted summary should keep source ids"
        assert all(item_id not in alive_ids for item_id in normalized_ids)

    assert has_episodic_source, "memory_compact should include episodic records in source batch"


async def _run_image_flow_single_long_term_record(tmp_dir: Path) -> None:
    graph = _build_graph(tmp_dir)
    state = build_initial_state(
        ChatRequest(
            text="What is in this picture?",
            image_url="https://example.com/cat.png",
            session_id="wf-image",
            mode="education",
            profile_id="kid-image",
        )
    )

    final_state = await graph.run(state=state)
    response = final_state["final_response"]

    assert response["memory"]["written_types"] == ["working", "episodic"]
    counts = _read_long_term_counts(tmp_dir / "memory.sqlite3", "kid-image")
    assert counts["episodic"] == 1
    assert counts["semantic"] == 0
    assert counts["perceptual"] == 0


def test_workflow_graph_direct_route() -> None:
    with tempfile.TemporaryDirectory() as raw_tmp:
        asyncio.run(_run_direct_flow(Path(raw_tmp)))


def test_workflow_graph_can_use_memory_route() -> None:
    with tempfile.TemporaryDirectory() as raw_tmp:
        asyncio.run(_run_react_memory_flow(Path(raw_tmp)))


def test_workflow_graph_can_use_tavily_route() -> None:
    with tempfile.TemporaryDirectory() as raw_tmp:
        asyncio.run(_run_tavily_route_flow(Path(raw_tmp)))


def test_workflow_graph_runs_llm_memory_compact() -> None:
    with tempfile.TemporaryDirectory() as raw_tmp:
        asyncio.run(_run_memory_compact_flow(Path(raw_tmp)))


def test_workflow_image_turn_still_writes_single_long_term_record() -> None:
    with tempfile.TemporaryDirectory() as raw_tmp:
        asyncio.run(_run_image_flow_single_long_term_record(Path(raw_tmp)))
