from __future__ import annotations

import json
import logging
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

# 延迟导入避免循环依赖
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from app.skills.registry import SkillRegistry

DeltaCallback = Callable[[str], Awaitable[None] | None]

logger = logging.getLogger(__name__)


class RetrieveKnowledgeInput(BaseModel):
    """知识检索输入。"""

    query: str = Field(
        description="Search query for local knowledge base. "
        "Provide 2-3 semantically equivalent phrasings separated by ||| "
        "(e.g. 'why is panda a treasure|||panda national treasure reason|||why protect pandas').",
    )
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


class DirectAnswerInput(BaseModel):
    """直接回答，无需调用任何外部工具。"""

    reason: str = Field(default="", description="Brief reason why direct answer is sufficient.")


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
        skill_registry: "SkillRegistry | None" = None,
    ) -> None:
        self.model_service = model_service
        self.memory_tool = memory_tool
        self.retriever = retriever
        self.skill_registry = skill_registry
        self.tavily = TavilySearchTool(
            api_key=model_service.settings.tavily_api_key,
            base_url=model_service.settings.tavily_base_url,
            timeout_seconds=model_service.settings.tavily_timeout_seconds,
            max_results=model_service.settings.tavily_max_results,
            search_depth=model_service.settings.tavily_search_depth,
        )
        self._langgraph_tools = self._build_langgraph_tools()

    def as_langgraph_tools(self, mode: str | None = None) -> list[StructuredTool]:
        """导出给LangGraph使用的工具列表，按 mode 过滤 skill tools。"""
        base = list(self._langgraph_tools)
        if self.skill_registry:
            base.extend(self.skill_registry.get_tools(mode=mode))
        return base

    def as_all_langgraph_tools(self) -> list[StructuredTool]:
        """导出全量工具列表（不按 mode 过滤），供 ToolNode 注册。"""
        base = list(self._langgraph_tools)
        if self.skill_registry:
            base.extend(self.skill_registry.get_all_tools())
        return base

    def _build_langgraph_tools(self) -> list[StructuredTool]:
        """构建LangGraph工具定义。"""
        return [
            StructuredTool.from_function(
                func=self._langgraph_direct_answer,
                name="direct_answer",
                description=(
                    "Use when you can answer directly from your own knowledge or vision "
                    "without needing external search or memory lookup. "
                    "Typical cases: image questions ('这是什么', 'what is this'), "
                    "simple factual questions you already know, or follow-up questions "
                    "where prior observation already provides enough context."
                ),
                args_schema=DirectAnswerInput,
            ),
            StructuredTool.from_function(
                func=self._langgraph_retrieve_knowledge,
                name="retrieve_knowledge",
                description="Retrieve children's encyclopedic facts from local knowledge base. "
                "Provide 2-3 semantically equivalent phrasings separated by |||.",
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

    def _langgraph_direct_answer(self, reason: str = "") -> str:
        """direct_answer 工具的 no-op 实现，ReasonNode 会将其拦截为 direct act，不会真正执行。"""
        return ""

    def _langgraph_retrieve_knowledge(
        self,
        query: str,
        top_k: int = 3,
        min_score: float | None = None,
    ) -> str:
        """LangGraph知识检索包装：按 ||| 分隔多查询，合并去重结果。"""
        raw_queries = [q.strip() for q in query.split("|||") if q.strip()]
        if not raw_queries:
            raw_queries = [query]
        # debug用
        # print(f'原始查询分解为 {len(raw_queries)} 个子查询: {raw_queries}')
        if self.model_service.settings.rag_multi_query_enabled:
            sub_queries = raw_queries
        else:
            sub_queries = raw_queries[:1]
        seen_ids: set[str] = set()
        merged: list[dict] = []
        base_result: dict = {}
        for q in sub_queries:
            r = self._retrieve_knowledge(query=q, top_k=top_k, min_score=min_score)
            if not base_result:
                base_result = r
            for chunk in r.get("results", []):
                cid = str(chunk.get("chunk_id") or "")
                if cid not in seen_ids:
                    seen_ids.add(cid)
                    merged.append(chunk)
        merged.sort(key=lambda x: float(x.get("score") or 0), reverse=True)
        result = {
            **base_result,
            "query": query,
            "expanded_queries": sub_queries,
            "results": merged[:top_k],
            "used_rag": bool(merged),
        }
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
        mode = state.get("interaction_mode", "education")
        tools = self.as_langgraph_tools(mode=mode)
        return await self.model_service.reason_next_action(request, state, tools=tools)

    async def generate_final_response(
        self,
        state: AgentState,
        on_delta: DeltaCallback | None = None,
    ) -> ResponseDraft:
        """调用回答模型生成最终回复。"""
        request = state_to_request(state)
        return await self.model_service.generate_final_response(request, state, on_delta=on_delta)

    async def generate_plan(self, state: AgentState) -> dict[str, Any]:
        from app.agent.state import state_to_request
        chat_request = state_to_request(state)
        return await self.model_service.generate_plan(chat_request, state)

    async def execute_plan(self, state: AgentState, on_delta=None) -> str:
        from app.agent.state import state_to_request
        chat_request = state_to_request(state)
        return await self.model_service.execute_plan(chat_request, state, on_delta=on_delta)

    def run_tool(self, call: ToolCall, state: AgentState) -> ToolResult:
        """统一工具调用入口。"""
        should_log = self._should_log_react_tool(call=call, state=state)
        if should_log:
            logger.info(
                "[react_tool_start] tool=%s args=%s",
                call.name,
                self._summarize_value(call.args),
            )
        if call.name == "retrieve_knowledge":
            query = str(call.args.get("query") or state.get("latest_user_text") or "").strip()
            top_k = int(call.args.get("top_k") or self.model_service.settings.rag_top_k)
            min_score = (
                float(call.args["min_score"])
                if "min_score" in call.args and call.args["min_score"] is not None
                else None
            )
            raw = self._langgraph_retrieve_knowledge(query=query, top_k=top_k, min_score=min_score)
            result = json.loads(raw)
            tool_result = ToolResult(tool_name="retrieve_knowledge", success=True, data=result)
            self._log_tool_result(tool_result, enabled=should_log)
            return tool_result

        if call.name == "tavily_search":
            query = str(call.args.get("query") or state.get("latest_user_text") or "").strip()
            result = self._tavily_search(
                query=query,
                top_k=int(call.args.get("top_k") or self.model_service.settings.tavily_max_results),
            )
            tool_result = ToolResult(
                tool_name="tavily_search",
                success=bool(result.get("tool_success", False)),
                data=result,
                error=result.get("error"),
            )
            self._log_tool_result(tool_result, enabled=should_log)
            return tool_result

        if call.name in {"read_memory_bundle", "read_session_memory", "read_profile_memory", "memory_execute"}:
            result = self._run_memory_tool(call=call, state=state)
            tool_result = ToolResult(
                tool_name=call.name,
                success=bool(result.get("success", False)),
                data=result.get("data", {}),
                error=result.get("error"),
            )
            self._log_tool_result(tool_result, enabled=should_log)
            return tool_result

        tool_result = ToolResult(tool_name=call.name, success=False, data={}, error="Unknown tool name.")
        self._log_tool_result(tool_result, enabled=should_log)
        return tool_result

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
            if action in {"add", "search", "summary", "remove"}:
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

    def _log_tool_result(self, result: ToolResult, enabled: bool) -> None:
        if not enabled:
            return
        logger.info(
            "[react_tool_end] tool=%s success=%s error=%r data=%s",
            result.tool_name,
            result.success,
            result.error,
            self._summarize_value(result.data),
        )

    def _should_log_react_tool(self, call: ToolCall, state: AgentState) -> bool:
        if call.name in {"retrieve_knowledge", "tavily_search"}:
            return True
        if call.name in {"read_memory_bundle", "read_session_memory", "read_profile_memory"}:
            workflow_trace = state.get("workflow_trace", [])
            return isinstance(workflow_trace, list) and "chatbot" in workflow_trace
        return False

    def _summarize_value(self, value: Any, limit: int = 320) -> str:
        try:
            rendered = json.dumps(value, ensure_ascii=False, default=str)
        except TypeError:
            rendered = repr(value)
        compact = " ".join(rendered.split())
        if len(compact) <= limit:
            return compact
        return f"{compact[:limit]}..."
