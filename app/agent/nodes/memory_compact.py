from __future__ import annotations

import logging
from typing import Any

from app.agent.state import AgentState
from app.memory.base import MemoryItem
from app.tools.basic_tools import BasicTools

logger = logging.getLogger(__name__)


class MemoryCompactNode:
    """记忆压缩节点：使用LLM将episodic+semantic记忆按10条压缩为1条episodic。"""

    def __init__(self, tools: BasicTools) -> None:
        self.tools = tools

    async def __call__(self, state: AgentState) -> AgentState:
        """执行压缩：LLM成功后删除原始10条并写回摘要。"""
        logger.info("entering node=memory_compact")
        user_id = str(state.get("profile_id") or "").strip()
        if not user_id:
            return {}

        mode = str(state.get("interaction_mode") or "education")
        manager = self.tools.memory_tool.manager
        compacted_batches = 0

        while True:
            batch = manager.get_episodic_compact_candidates(user_id=user_id, batch_size=10)
            if len(batch) < 10:
                break

            events = self._to_events(batch=batch, fallback_mode=mode)
            llm_result = await self.tools.model_service.summarize_episodic_batch(
                events=events,
                mode=mode,
            )
            if not llm_result:
                logger.warning("skip memory compact because llm summary is empty")
                break

            compact_result = manager.compact_episodic_batch(
                user_id=user_id,
                source_items=batch,
                summary_text=str(llm_result.get("summary") or ""),
                mode=mode,
                topic_hint=(
                    str(llm_result.get("topic_hint"))
                    if isinstance(llm_result.get("topic_hint"), str)
                    else None
                ),
                key_points=(
                    [str(item) for item in llm_result.get("key_points", [])]
                    if isinstance(llm_result.get("key_points"), list)
                    else []
                ),
            )
            if not compact_result.get("success"):
                logger.warning("memory compact failed: %s", compact_result.get("error"))
                break
            compacted_batches += 1

        if compacted_batches <= 0:
            return {}
        return {
            "memory_consolidated_count": int(state.get("memory_consolidated_count", 0)) + compacted_batches,
        }

    def _to_events(self, batch: list[MemoryItem], fallback_mode: str) -> list[dict[str, Any]]:
        """转换压缩输入：提取角色、文本、topic、时间等字段。"""
        ordered = sorted(batch, key=lambda item: item.timestamp)
        events: list[dict[str, Any]] = []
        for item in ordered:
            raw_turn = item.metadata.get("turn")
            role = "unknown"
            text = item.content
            topic = item.metadata.get("topic")
            mode = item.metadata.get("mode")
            if isinstance(raw_turn, dict):
                role = str(raw_turn.get("role") or role)
                text = str(raw_turn.get("text") or text)
                topic = raw_turn.get("topic", topic)
                mode = raw_turn.get("mode", mode)
            events.append(
                {
                    "role": role,
                    "text": text,
                    "topic": topic if isinstance(topic, str) else "",
                    "mode": mode if isinstance(mode, str) else fallback_mode,
                    "ts": item.timestamp,
                }
            )
        return events
