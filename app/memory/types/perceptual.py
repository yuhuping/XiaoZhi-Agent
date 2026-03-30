from __future__ import annotations

from typing import Any

from app.memory.base import MemoryItem
from app.memory.types.persistent_base import PersistentMemoryBase


class PerceptualMemory(PersistentMemoryBase):
    """感知记忆：支持模态字段和时间加权检索。"""

    memory_type = "perceptual"

    def add(self, item: MemoryItem) -> str:
        """添加感知记忆：默认补齐modality。"""
        metadata = dict(item.metadata)
        if "modality" not in metadata:
            metadata["modality"] = "text"
        item.metadata = metadata
        return super().add(item)

    def retrieve(
        self,
        user_id: str,
        query: str,
        limit: int,
        min_importance: float,
        **kwargs: Any,
    ):
        """检索感知记忆：允许按目标模态过滤。"""
        target_modality = kwargs.get("target_modality")
        all_results = super().retrieve(
            user_id=user_id,
            query=query,
            limit=max(limit * 2, 10),
            min_importance=min_importance,
            **kwargs,
        )
        if not isinstance(target_modality, str) or not target_modality.strip():
            return all_results[: max(1, limit)]
        filtered = [
            result
            for result in all_results
            if str(result.item.metadata.get("modality") or "").strip() == target_modality.strip()
        ]
        return filtered[: max(1, limit)]

    def _score(self, item: MemoryItem, vec_score: float, query: str, **kwargs: Any) -> float:
        """感知评分：(向量0.8 + 近因0.2) * 重要性权重。"""
        recency = self._recency_score(item.timestamp)
        base_relevance = float(vec_score) * 0.8 + recency * 0.2
        importance_weight = 0.8 + item.importance * 0.4
        return base_relevance * importance_weight
