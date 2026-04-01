from __future__ import annotations

from datetime import timedelta
from typing import Any

from app.memory.base import MemoryItem, MemorySearchResult, parse_iso_or_now, utc_now_iso
from app.memory.embedding import HashingTextEmbedder
from app.memory.storage import SQLiteMemoryStore
from app.memory.vector_store import LocalVectorStore


class PersistentMemoryBase:
    """持久记忆基类：封装SQLite+向量索引的通用逻辑。"""

    memory_type: str = "episodic"

    def __init__(
        self,
        store: SQLiteMemoryStore,
        vector_store: LocalVectorStore,
        embedder: HashingTextEmbedder,
    ) -> None:
        self.store = store
        self.vector_store = vector_store
        self.embedder = embedder

    def add(self, item: MemoryItem) -> str:
        """添加记忆：写数据库并更新向量索引。"""
        self.store.upsert_item(item)
        self.vector_store.upsert(
            user_id=item.user_id,
            memory_type=self.memory_type,
            memory_id=item.id,
            vector=self.embedder.encode_text(item.content),
        )
        return item.id

    def remove(self, user_id: str, memory_id: str) -> bool:
        """删除记忆：同步删除数据库和向量索引。"""
        target = self.store.get_item(memory_id)
        ok = self.store.delete_item(memory_id)
        if target and ok:
            self.vector_store.remove(user_id=user_id, memory_type=self.memory_type, memory_id=memory_id)
        return ok

    def update(self, memory_id: str, patch: dict[str, Any]) -> bool:
        """更新记忆：内容变化时刷新向量。"""
        ok = self.store.update_item(memory_id, patch)
        if not ok:
            return False
        target = self.store.get_item(memory_id)
        if not target:
            return False
        self.vector_store.upsert(
            user_id=target.user_id,
            memory_type=self.memory_type,
            memory_id=target.id,
            vector=self.embedder.encode_text(target.content),
        )
        return True

    def retrieve(
        self,
        user_id: str,
        query: str,
        limit: int,
        min_importance: float,
        **kwargs: Any,
    ) -> list[MemorySearchResult]:
        """检索记忆：向量召回 + 子类评分。"""
        query_vec = self.embedder.encode_text(query)
        hits = self.vector_store.search(
            user_id=user_id,
            memory_type=self.memory_type,
            query_vector=query_vec,
            top_k=max(limit * 5, 20),
            min_score=0.0,
        )

        results: list[MemorySearchResult] = []
        for memory_id, vec_score in hits:
            item = self.store.get_item(memory_id)
            if not item or item.memory_type != self.memory_type:
                continue
            if item.importance < min_importance:
                continue
            score = self._score(item=item, vec_score=vec_score, query=query, **kwargs)
            if score > 0:
                results.append(MemorySearchResult(item=item, score=score))

        results.sort(key=lambda x: x.score, reverse=True)
        return results[: max(1, limit)]

    def list_for_consolidate(self, user_id: str, threshold: float, limit: int = 20) -> list[MemoryItem]:
        """提取高重要性候选。"""
        items = self.store.list_items(user_id=user_id, memory_type=self.memory_type, limit=max(100, limit * 2))
        candidates = [item for item in items if item.importance >= threshold]
        candidates.sort(key=lambda x: (x.importance, x.timestamp), reverse=True)
        return candidates[: max(1, limit)]

    def forget(self, user_id: str, strategy: str, threshold: float, max_age_days: int) -> int:
        """遗忘：按策略过滤并软删除。"""
        items = self.store.list_items(user_id=user_id, memory_type=self.memory_type, limit=10_000)
        if not items:
            return 0

        remove_ids: list[str] = []
        now = parse_iso_or_now(utc_now_iso())
        age_limit = timedelta(days=max(1, int(max_age_days)))

        if strategy == "importance_based":
            remove_ids = [item.id for item in items if item.importance < threshold]
        elif strategy == "time_based":
            remove_ids = [
                item.id
                for item in items
                if now - parse_iso_or_now(item.timestamp) > age_limit
            ]
        elif strategy == "capacity_based":
            keep = max(1, int(len(items) * max(0.2, min(1.0, threshold))))
            ordered = sorted(items, key=lambda x: (x.importance, x.timestamp), reverse=True)
            remove_ids = [item.id for item in ordered[keep:]]

        removed = 0
        for memory_id in remove_ids:
            if self.remove(user_id=user_id, memory_id=memory_id):
                removed += 1
        return removed

    def _score(self, item: MemoryItem, vec_score: float, query: str, **kwargs: Any) -> float:
        """评分函数：由子类实现。"""
        raise NotImplementedError

    def _recency_score(self, timestamp: str, min_value: float = 0.1) -> float:
        """时间近因性分数：指数衰减。"""
        now = parse_iso_or_now(utc_now_iso())
        age_hours = max(0.0, (now - parse_iso_or_now(timestamp)).total_seconds() / 3600.0)
        score = float(2.718281828 ** (-0.1 * age_hours / 24.0))
        return max(min_value, score)
