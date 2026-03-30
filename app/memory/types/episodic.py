from __future__ import annotations

from typing import Any

from app.memory.base import MemoryItem
from app.memory.types.persistent_base import PersistentMemoryBase


class EpisodicMemory(PersistentMemoryBase):
    """情景记忆：强调事件时间与上下文回溯。"""

    memory_type = "episodic"

    def _score(self, item: MemoryItem, vec_score: float, query: str, **kwargs: Any) -> float:
        """情景评分：(向量0.8 + 近因0.2) * 重要性权重。"""
        recency = self._recency_score(item.timestamp)
        base_relevance = float(vec_score) * 0.8 + recency * 0.2
        importance_weight = 0.8 + item.importance * 0.4
        return base_relevance * importance_weight
