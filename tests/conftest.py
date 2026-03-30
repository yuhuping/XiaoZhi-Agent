import os
import re
import hashlib

import numpy as np
import pytest
from langchain_core.messages import AIMessage

import app.rag.retriever as retriever_module
from app.rag.retriever import LocalKnowledgeRetriever
from app.services.model_service import ModelService


def _extract_selected_act(prompt: str) -> str:
    match = re.search(r"Selected act:\s*([a-z_]+)", prompt)
    if match:
        return match.group(1)
    return "direct"


def _fake_call_llm_json(
    self: ModelService,
    model: str,
    schema_name: str,
    schema: dict,
    instruction: str,
    prompt: str,
    chat_request,
) -> dict:
    text = (chat_request.text or "").strip().lower()
    has_image = bool(chat_request.image_base64 or chat_request.image_url)

    if schema_name == "xiaozhi_reason":
        if text in {"hi", "hello", "hey", "你好", "您好"}:
            return {
                "decision": "greeting_direct_reply",
                "selected_act": "direct",
                "tool_name": "",
                "tool_input": {},
                "route_reason": "simple greeting",
                "topic_hint": "",
                "confidence": "high",
            }
        if text == "red":
            return {
                "decision": "use_memory_to_continue",
                "selected_act": "read_memory",
                "tool_name": "read_session_memory",
                "tool_input": {},
                "route_reason": "short answer likely depends on context",
                "topic_hint": "apple",
                "confidence": "medium",
            }
        if has_image or "apple" in text or "moon" in text:
            return {
                "decision": "retrieve_for_grounding",
                "selected_act": "retrieve_knowledge",
                "tool_name": "retrieve_knowledge",
                "tool_input": {"query": chat_request.text or "image content"},
                "route_reason": "topic explanation benefits from grounding",
                "topic_hint": "apple" if "apple" in text else ("moon" if "moon" in text else "picture"),
                "confidence": "high",
            }
        if "latest" in text or "news" in text:
            return {
                "decision": "need_fresh_web_info",
                "selected_act": "retrieve_knowledge",
                "tool_name": "tavily_search",
                "tool_input": {"query": chat_request.text or "latest child news"},
                "route_reason": "question asks for recent information",
                "topic_hint": "news",
                "confidence": "medium",
            }
        return {
            "decision": "respond_directly",
            "selected_act": "direct",
            "tool_name": "",
            "tool_input": {},
            "route_reason": "direct short response is enough",
            "topic_hint": "",
            "confidence": "medium",
        }

    if schema_name == "xiaozhi_response":
        selected_act = _extract_selected_act(prompt)
        topic = "apple" if "apple" in text else ("moon" if "moon" in text else "topic")
        if selected_act == "read_memory":
            return {
                "topic": topic,
                "message": "你回答得很好，我们继续想一想。",
                "follow_up_question": "你还记得它还有什么特点吗？",
                "confidence": "medium",
                "safety_notes": "",
            }
        if chat_request.mode == "companion":
            return {
                "topic": topic,
                "message": "好问题呀，我来陪你一起聊聊。",
                "follow_up_question": "",
                "confidence": "medium",
                "safety_notes": "",
            }
        if chat_request.mode == "parent":
            return {
                "topic": topic,
                "message": "给您一个简要建议：先确认最新官方政策，再结合孩子当前基础做分步规划。",
                "follow_up_question": "您更希望先看政策查询路径，还是先看学习提升计划？",
                "confidence": "medium",
                "safety_notes": "",
            }
        return {
            "topic": topic,
            "message": "苹果是一种水果，常见颜色有红色和绿色。",
            "follow_up_question": "你最喜欢什么颜色的苹果？",
            "confidence": "high",
            "safety_notes": "",
        }

    if schema_name == "memory_summary":
        return {"summary": "recent topics were discussed in tutoring turns"}

    if schema_name == "memory_compact_summary":
        topic = "apple" if "apple" in prompt.lower() else "topic"
        return {
            "summary": f"compacted episodic memory about {topic}",
            "topic_hint": topic,
            "key_points": [
                "child asked short follow-up questions",
                "assistant provided guided hints",
            ],
        }

    raise AssertionError(f"Unexpected schema_name: {schema_name}")


async def _fake_call_llm_with_tools(
    self: ModelService,
    model: str,
    instruction: str,
    prompt: str,
    chat_request,
    tools,
) -> AIMessage:
    text = (chat_request.text or "").strip().lower()
    has_image = bool(chat_request.image_base64 or chat_request.image_url)

    if text in {"hi", "hello", "hey", "浣犲ソ", "鎮ㄥソ"}:
        return AIMessage(content="respond_directly", tool_calls=[])
    if text == "red":
        return AIMessage(
            content="read memory before follow-up",
            tool_calls=[
                {
                    "id": "tool_call_memory_1",
                    "name": "read_memory_bundle",
                    "args": {"session_id": chat_request.session_id or "", "profile_id": chat_request.profile_id or ""},
                    "type": "tool_call",
                }
            ],
        )
    if "latest" in text or "news" in text:
        return AIMessage(
            content="need fresh results",
            tool_calls=[
                {
                    "id": "tool_call_tavily_1",
                    "name": "tavily_search",
                    "args": {"query": chat_request.text or "latest child news"},
                    "type": "tool_call",
                }
            ],
        )
    if has_image or "apple" in text or "moon" in text:
        return AIMessage(
            content="retrieve for grounding",
            tool_calls=[
                {
                    "id": "tool_call_retrieve_1",
                    "name": "retrieve_knowledge",
                    "args": {"query": chat_request.text or "image content"},
                    "type": "tool_call",
                }
            ],
        )
    return AIMessage(content="respond_directly", tool_calls=[])


@pytest.fixture(autouse=True)
def patch_model_service(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY") or "test-key")
    monkeypatch.setenv("TAVILY_API_KEY", "")
    monkeypatch.setattr(ModelService, "_call_llm_json", _fake_call_llm_json)
    monkeypatch.setattr(ModelService, "_call_llm_with_tools", _fake_call_llm_with_tools)


@pytest.fixture(autouse=True)
def patch_retriever_embedding(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory) -> None:
    if os.getenv("RETRIEVER_TEST_USE_REAL_OPENAI", "0").strip() == "1":
        # 手工联调模式：不注入 fake embedding，直接走真实 retriever 配置
        return

    monkeypatch.setattr(retriever_module, "OPENAI_EMBEDDING_MODEL", "test-embedding-model")
    monkeypatch.setattr(retriever_module, "OPENAI_EMBEDDING_BASE_URL", "https://example.com/v1")
    monkeypatch.setattr(retriever_module, "OPENAI_EMBEDDING_API_KEY", "test-key")
    monkeypatch.setattr(
        retriever_module,
        "DEFAULT_INDEX_DIR",
        tmp_path_factory.mktemp("rag_index_global"),
    )

    class _DummyOpenAI:
        def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            self.embeddings = object()

    monkeypatch.setattr(retriever_module, "OpenAI", _DummyOpenAI)

    def fake_embed_texts(self: LocalKnowledgeRetriever, texts: list[str]) -> np.ndarray:
        dim = 64
        matrix = np.zeros((len(texts), dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in LocalKnowledgeRetriever._tokenize(text):
                digest = hashlib.md5(token.encode("utf-8")).digest()
                bucket = int.from_bytes(digest[:4], byteorder="little", signed=False) % dim
                matrix[row, bucket] += 1.0
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.maximum(norms, 1e-12)

    monkeypatch.setattr(LocalKnowledgeRetriever, "_embed_texts", fake_embed_texts)
