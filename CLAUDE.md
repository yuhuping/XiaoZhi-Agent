# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

# Run all tests
pytest

# Run a single test file
pytest tests/test_memory_update_node.py -v
```

The project venv is at `./venv` (not `.venv`).

## Architecture

小智 is a ReAct-based educational agent with three interaction modes: `education` (guided learning), `companion` (casual chat), and `parent` (practical advice + web search). 这个项目本质是为了找实习的，所以代码简洁稳定易懂符合现有技术逻辑要大于代码工程鲁棒性。

### Request Flow

`POST /api/v1/chat/explain-and-ask` → `ChatService` → `AgentGraph.run()` → SSE streaming response

The LangGraph pipeline is a DAG:

```
understand → state_update → chatbot ─┬→ tools → observe → respond → memory_update → response → memory_compact → END
                                      └→ observe (no tool call)
```

Each node is a class in `app/agent/nodes/`. The `chatbot` node (`ReasonNode`) decides whether to invoke a tool (ReAct reasoning). The `tools_condition` edge routes to `ToolNode` or directly to `observe`.

**Three act routes** resolved in `router.py`:
- `direct` → `act_direct` (LLM generates response directly)
- `retrieve_knowledge` → `act_retrieve` (RAG lookup)
- `read_memory` → `act_memory` (memory lookup)

### State (`app/agent/state.py`)

`AgentState` is a single `TypedDict` that flows through every node. Key fields:
- `interaction_mode` / `profile_id` — determines persona and which memory profile to write
- `selected_act` — set by `ReasonNode`, consumed by `respond` node
- `stream_delta_writer` — async callback injected by `ChatService` for SSE streaming
- `messages` — LangChain message list passed to `ToolNode`

Profile IDs are mode-scoped: `education`/`companion` → `default_child`, `parent` → `default_parent`.

### Memory (`app/memory/`)

Four-tier memory managed by `MemoryManager`:
- **Working** — in-memory, session-scoped, TTL-based (default 60 min, capacity 50)
- **Episodic** — SQLite + FAISS, long-term conversation history
- **Semantic** — SQLite + FAISS, fact/preference store
- **Perceptual** — SQLite + FAISS, image/object observations

Storage: SQLite at `data/memory.sqlite3`, FAISS index files at `data/memory_index/`.

`memory_compact` node (end of each turn) compresses batches of ≥10 old episodic/semantic entries into a single summary item.

### RAG (`app/rag/`)

FAISS-backed retriever over PDF/text files in `KG/`. Auto-bootstraps on startup (`KG_AUTO_BOOTSTRAP=true`). Controlled via `rag_enabled` flag in `AgentState` and `Settings`.

### Configuration (`app/core/config.py`)

`Settings` is a frozen dataclass loaded from `.env` via `get_settings()` (cached). All env vars are read at import time — restart the server after changing `.env`.

Key `.env` variables:
```env
LLM_API_KEY, LLM_BASE_URL, LLM_MODEL       # Main LLM
vllm_api_key, vllm_base_url, vllm_model    # Vision model
TAVILY_API_KEY                              # Web search (parent mode)
```

### Notes

- Commented-out `print()` calls in nodes are intentional debug hooks — do not remove them.
- LangSmith tracing is opt-in via `LANGSMITH_TRACING=true` + `LANGSMITH_API_KEY`.
- Architecture detail docs: `docs/architecture/graph.md`, `rag.md`, `memory.md`.
