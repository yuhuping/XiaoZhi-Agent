from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.agent.state import AgentState, state_to_request
from app.memory.tool import MemoryTool
from app.rag.retriever import LocalKnowledgeRetriever
from app.services.model_service import ModelService, ReasonDecision, ResponseDraft
from app.tools.tavily_search import TavilySearchTool

DeltaCallback = Callable[[str], Awaitable[None] | None]


class RetrieveKnowledgeInput(BaseModel):
    """知识检索输入。"""

    query: str = Field(description="User question to search in local KG.")
    top_k: int = Field(default=3, ge=1, le=8)
    min_score: float | None = Field(default=None, ge=0)


class TavilySearchInput(BaseModel):
    """联网检索输入。"""

    query: str = Field(description="Search query for web search.")
    top_k: int = Field(default=3, ge=1, le=8)


class ReadMemoryInput(BaseModel):
    """记忆读取输入。"""

    session_id: str
    profile_id: str


@dataclass(frozen=True)
class ToolCall:
    """工具调用参数。"""

    name: str
    args: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    """工具调用结果。"""

    tool_name: str
    success: bool
    data: dict[str, Any]
    error: str | None = None


class BasicTools:
    """基础工具层：RAG、联网检索、Memory统一入口。"""

    def __init__(
        self,
        model_service: ModelService,
        memory_tool: MemoryTool,
        retriever: LocalKnowledgeRetriever,
    ) -> None:
        self.model_service = model_service
        self.memory_tool = memory_tool
        self.retriever = retriever
        self.tavily = TavilySearchTool(
            api_key=model_service.settings.tavily_api_key,
            base_url=model_service.settings.tavily_base_url,
            timeout_seconds=model_service.settings.tavily_timeout_seconds,
            max_results=model_service.settings.tavily_max_results,
            search_depth=model_service.settings.tavily_search_depth,
        )
        self._langgraph_tools = self._build_langgraph_tools()

    def as_langgraph_tools(self) -> list[StructuredTool]:
        """导出给LangGraph使用的工具列表。"""
        return self._langgraph_tools

    def _build_langgraph_tools(self) -> list[StructuredTool]:
        """构建LangGraph工具定义。"""
        return [
            StructuredTool.from_function(
                func=self._langgraph_retrieve_knowledge,
                name="retrieve_knowledge",
                description="Retrieve children's encyclopedic facts from local knowledge base.",
                args_schema=RetrieveKnowledgeInput,
            ),
            StructuredTool.from_function(
                func=self._langgraph_tavily_search,
                name="tavily_search",
                description="Search web results for recent or future information.",
                args_schema=TavilySearchInput,
            ),
            StructuredTool.from_function(
                func=self._langgraph_read_memory_bundle,
                name="read_memory_bundle",
                description="Read both session memory and profile memory for the current child.",
                args_schema=ReadMemoryInput,
            ),
        ]

    def _langgraph_retrieve_knowledge(
        self,
        query: str,
        top_k: int = 3,
        min_score: float | None = None,
    ) -> str:
        """LangGraph知识检索包装。"""
        result = self._retrieve_knowledge(query=query, top_k=top_k, min_score=min_score)
        return json.dumps(result, ensure_ascii=False)

    def _langgraph_tavily_search(self, query: str, top_k: int = 3) -> str:
        """LangGraph联网检索包装。"""
        result = self._tavily_search(query=query, top_k=top_k)
        return json.dumps(result, ensure_ascii=False)

    def _langgraph_read_memory_bundle(self, session_id: str, profile_id: str) -> str:
        """LangGraph记忆读取包装。"""
        result = self._read_memory_bundle(session_id=session_id, profile_id=profile_id)
        return json.dumps(result, ensure_ascii=False)

    def detect_object(self, state: AgentState) -> str | None:
        """对象检测：基于关键词做轻量主题猜测。"""
        text = (state.get("text_input") or "").lower()
        keyword_map = {
            "apple": "apple",
            "苹果": "apple",
            "cat": "cat",
            "猫": "cat",
            "dog": "dog",
            "狗": "dog",
            "car": "car",
            "汽车": "car",
            "bus": "bus",
            "公交": "bus",
            "banana": "banana",
            "香蕉": "banana",
            "moon": "moon",
            "月亮": "moon",
            "star": "star",
            "星星": "star",
            "flower": "flower",
            "花": "flower",
            "tree": "tree",
            "树": "tree",
        }
        for keyword, topic in keyword_map.items():
            if keyword in text:
                return topic
        if state.get("image_base64") or state.get("image_url"):
            return "picture"
        return None

    def perceive_signals(self, state: AgentState) -> list[str]:
        """感知信号：用于后续reason路由提示。"""
        text = (state.get("text_input") or "").strip()
        lowered = text.lower()
        signals: list[str] = []
        if state.get("image_base64") or state.get("image_url"):
            signals.append("has_image")
        if not text:
            signals.append("empty_text")
            return signals
        if self._is_pure_greeting(lowered):
            signals.append("greeting_candidate")
        if "?" in text or any(token in text for token in ("什么", "为什么", "怎么", "哪里", "谁")):
            signals.append("question_candidate")
        if self._looks_like_answer(text):
            signals.append("answer_candidate")
        if self.detect_object(state):
            signals.append("topic_candidate")
        return signals

    async def reason_next_action(self, state: AgentState) -> ReasonDecision:
        """调用规划模型选择下一步动作。"""
        request = state_to_request(state)
        return await self.model_service.reason_next_action(request, state, tools=self._langgraph_tools)

    async def generate_final_response(
        self,
        state: AgentState,
        on_delta: DeltaCallback | None = None,
    ) -> ResponseDraft:
        """调用回答模型生成最终回复。"""
        request = state_to_request(state)
        return await self.model_service.generate_final_response(request, state, on_delta=on_delta)

    def run_tool(self, call: ToolCall, state: AgentState) -> ToolResult:
        """统一工具调用入口。"""
        if call.name == "retrieve_knowledge":
            query = str(call.args.get("query") or state.get("latest_user_text") or "").strip()
            result = self._retrieve_knowledge(
                query=query,
                top_k=int(call.args.get("top_k") or self.model_service.settings.rag_top_k),
                min_score=(
                    float(call.args["min_score"])
                    if "min_score" in call.args and call.args["min_score"] is not None
                    else None
                ),
            )
            return ToolResult(tool_name="retrieve_knowledge", success=True, data=result)

        if call.name == "tavily_search":
            query = str(call.args.get("query") or state.get("latest_user_text") or "").strip()
            result = self._tavily_search(
                query=query,
                top_k=int(call.args.get("top_k") or self.model_service.settings.tavily_max_results),
            )
            return ToolResult(
                tool_name="tavily_search",
                success=bool(result.get("tool_success", False)),
                data=result,
                error=result.get("error"),
            )

        if call.name in {"read_memory_bundle", "read_session_memory", "read_profile_memory", "memory_execute"}:
            result = self._run_memory_tool(call=call, state=state)
            return ToolResult(
                tool_name=call.name,
                success=bool(result.get("success", False)),
                data=result.get("data", {}),
                error=result.get("error"),
            )

        return ToolResult(tool_name=call.name, success=False, data={}, error="Unknown tool name.")

    def _run_memory_tool(self, call: ToolCall, state: AgentState) -> dict[str, Any]:
        """运行memory工具：兼容旧调用名。"""
        user_id = str(state.get("profile_id") or "default_child")
        session_id = str(state.get("session_id") or "default_session")

        if call.name == "read_memory_bundle":
            return self.memory_tool.execute(
                "read_bundle",
                user_id=call.args.get("profile_id") or user_id,
                session_id=call.args.get("session_id") or session_id,
            )
        if call.name == "read_session_memory":
            return self.memory_tool.execute("read_session", user_id=user_id, session_id=session_id)
        if call.name == "read_profile_memory":
            return self.memory_tool.execute("read_profile", user_id=user_id)
        if call.name == "memory_execute":
            action = str(call.args.get("action") or "").strip()
            kwargs = dict(call.args.get("kwargs") or {})
            kwargs.setdefault("user_id", user_id)
            kwargs.setdefault("session_id", session_id)
            return self.memory_tool.execute(action, **kwargs)
        return {"success": False, "data": {}, "error": f"unsupported memory call: {call.name}"}

    def _retrieve_knowledge(self, query: str, top_k: int, min_score: float | None = None) -> dict[str, Any]:
        """本地RAG检索。"""
        normalized_query = (query or "").strip()
        if not self.model_service.settings.rag_enabled:
            return {
                "query": normalized_query,
                "results": [],
                "used_rag": False,
                "tool_success": False,
                "error": "RAG is disabled by RAG_ENABLED=false",
                "index_status": self.retriever.get_index_status(),
            }
        if not normalized_query:
            return {
                "query": "",
                "results": [],
                "used_rag": False,
                "tool_success": False,
                "error": "query is required",
                "index_status": self.retriever.get_index_status(),
            }

        score_threshold = min_score if min_score is not None else self.model_service.settings.rag_min_score
        results = self.retriever.retrieve(
            query=normalized_query,
            top_k=max(1, top_k),
            min_score=score_threshold,
        )
        return {
            "query": normalized_query,
            "results": results,
            "used_rag": bool(results),
            "tool_success": True,
            "index_status": self.retriever.get_index_status(),
        }

    def _tavily_search(self, query: str, top_k: int) -> dict[str, Any]:
        """联网检索。"""
        if not query:
            return {"query": "", "results": [], "used_rag": False, "tool_success": False, "error": "query is required"}
        if not self.tavily.enabled:
            return {
                "query": query,
                "results": [],
                "used_rag": False,
                "tool_success": False,
                "error": "TAVILY_API_KEY is not configured",
            }

        try:
            payload = self.tavily.search(query=query, max_results=top_k)
        except Exception as exc:
            return {"query": query, "results": [], "used_rag": False, "tool_success": False, "error": str(exc)}

        raw_results = payload.get("results")
        normalized: list[dict[str, object]] = []
        if isinstance(raw_results, list):
            for index, item in enumerate(raw_results):
                if not isinstance(item, dict):
                    continue
                url = str(item.get("url") or "")
                title = str(item.get("title") or "")
                content = str(item.get("content") or "")
                score_raw = item.get("score")
                score = float(score_raw) if isinstance(score_raw, (int, float)) else 0.0
                snippet = content[:277].rstrip() + "..." if len(content) > 280 else content
                normalized.append(
                    {
                        "chunk_id": f"tavily:{index}",
                        "source": url or title or "tavily",
                        "score": round(score, 4),
                        "snippet": snippet,
                    }
                )
        return {
            "query": query,
            "results": normalized,
            "used_rag": bool(normalized),
            "tool_success": True,
            "answer": payload.get("answer"),
        }

    def _read_memory_bundle(self, session_id: str, profile_id: str) -> dict[str, Any]:
        """读取会话+画像记忆。"""
        result = self.memory_tool.execute(
            "read_bundle",
            user_id=profile_id,
            session_id=session_id,
        )
        return result.get("data", {}) if result.get("success") else {"session": {}, "profile": {}, "tool_success": False}

    def _is_pure_greeting(self, lowered: str) -> bool:
        """判断是否纯问候语。"""
        normalized = re.sub(r"[!,.?~，。！？\s]+", "", lowered)
        return normalized in {"hi", "hello", "hey", "你好", "您好", "哈喽", "早上好", "晚上好"}

    def _looks_like_answer(self, text: str) -> bool:
        """判断是否简短回答句。"""
        cleaned = text.strip()
        if not cleaned or "?" in cleaned:
            return False
        word_count = len(cleaned.split())
        if any(token in cleaned for token in ("because", "因为", "我觉得", "it is", "它是")):
            return True
        return word_count <= 5 or len(cleaned) <= 10
