from __future__ import annotations

from collections import Counter
from typing import Any

from app.memory.base import (
    MemoryConfig,
    MemoryItem,
    MemorySearchResult,
    MemoryType,
    parse_iso_or_now,
    utc_now_iso,
)
from app.memory.embedding import HashingTextEmbedder
from app.memory.storage import SQLiteMemoryStore
from app.memory.types import EpisodicMemory, PerceptualMemory, SemanticMemory, WorkingMemory
from app.memory.vector_store import LocalVectorStore


class MemoryManager:
    """记忆管理器：统一调度四类记忆。"""

    def __init__(self, config: MemoryConfig) -> None:
        self.config = config
        self.embedder = HashingTextEmbedder(dim=config.vector_dim)
        self.store = SQLiteMemoryStore(db_path=config.db_path)
        self.vector_store = LocalVectorStore(index_dir=config.index_dir, vector_dim=config.vector_dim)

        self.working = WorkingMemory(config=config, embedder=self.embedder)
        self.episodic = EpisodicMemory(store=self.store, vector_store=self.vector_store, embedder=self.embedder)
        self.semantic = SemanticMemory(store=self.store, vector_store=self.vector_store, embedder=self.embedder)
        self.perceptual = PerceptualMemory(store=self.store, vector_store=self.vector_store, embedder=self.embedder)

    def add_memory(
        self,
        user_id: str,
        session_id: str,
        content: str,
        memory_type: MemoryType = "working",
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """添加记忆：根据类型分发到对应子模块。"""
        item = MemoryItem.create(
            user_id=user_id,
            session_id=session_id,
            memory_type=memory_type,
            content=content,
            importance=importance,
            metadata=metadata or {},
        )
        target = self._target(memory_type)
        return target.add(item)

    def retrieve_memories(
        self,
        user_id: str,
        session_id: str,
        query: str,
        limit: int = 5,
        memory_types: list[MemoryType] | None = None,
        min_importance: float | None = None,
        target_modality: str | None = None,
    ) -> list[MemorySearchResult]:
        """检索记忆：支持多类型混合排序。"""
        types = memory_types or ["working", "episodic", "semantic", "perceptual"]
        threshold = self.config.default_min_importance if min_importance is None else float(min_importance)
        all_results: list[MemorySearchResult] = []

        for memory_type in types:
            if memory_type == "working":
                all_results.extend(
                    self.working.retrieve(
                        user_id=user_id,
                        session_id=session_id,
                        query=query,
                        limit=max(1, limit),
                        min_importance=threshold,
                    )
                )
            elif memory_type == "episodic":
                all_results.extend(
                    self.episodic.retrieve(
                        user_id=user_id,
                        query=query,
                        limit=max(1, limit),
                        min_importance=threshold,
                    )
                )
            elif memory_type == "semantic":
                all_results.extend(
                    self.semantic.retrieve(
                        user_id=user_id,
                        query=query,
                        limit=max(1, limit),
                        min_importance=threshold,
                    )
                )
            elif memory_type == "perceptual":
                all_results.extend(
                    self.perceptual.retrieve(
                        user_id=user_id,
                        query=query,
                        limit=max(1, limit),
                        min_importance=threshold,
                        target_modality=target_modality,
                    )
                )

        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results[: max(1, limit)]

    def update_memory(self, memory_id: str, patch: dict[str, Any]) -> bool:
        """更新记忆：自动分发到对应类型模块。"""
        target = self.store.get_item(memory_id)
        if not target:
            return False
        if target.memory_type == "working":
            # 工作记忆直接替换。
            self.working.remove(target.user_id, target.session_id, target.id)
            content = str(patch.get("content") or target.content)
            importance = float(patch.get("importance", target.importance))
            metadata = dict(target.metadata)
            if isinstance(patch.get("metadata"), dict):
                metadata.update(patch["metadata"])
            new_item = MemoryItem.create(
                user_id=target.user_id,
                session_id=target.session_id,
                memory_type="working",
                content=content,
                importance=importance,
                metadata=metadata,
            )
            new_item.id = target.id
            self.working.add(new_item)
            return True

        return self._target(target.memory_type).update(memory_id, patch)

    def remove_memory(self, user_id: str, session_id: str, memory_id: str) -> bool:
        """删除记忆：按实际类型删除。"""
        target = self.store.get_item(memory_id)
        if target and target.memory_type != "working":
            return self._target(target.memory_type).remove(user_id=user_id, memory_id=memory_id)
        return self.working.remove(user_id=user_id, session_id=session_id, memory_id=memory_id)

    def forget_memories(
        self,
        user_id: str,
        strategy: str = "importance_based",
        threshold: float = 0.1,
        max_age_days: int = 30,
        memory_types: list[MemoryType] | None = None,
    ) -> int:
        """遗忘记忆：跨类型执行策略。"""
        types = memory_types or ["working", "episodic", "semantic", "perceptual"]
        removed = 0
        for memory_type in types:
            if memory_type == "working":
                removed += self.working.forget(
                    user_id=user_id,
                    strategy=strategy,
                    threshold=threshold,
                    max_age_days=max_age_days,
                )
            else:
                removed += self._target(memory_type).forget(
                    user_id=user_id,
                    strategy=strategy,
                    threshold=threshold,
                    max_age_days=max_age_days,
                )
        return removed

    def consolidate_memories(
        self,
        user_id: str,
        from_type: MemoryType = "working",
        to_type: MemoryType = "episodic",
        importance_threshold: float = 0.7,
    ) -> int:
        """整合记忆：将高重要性记忆提升到更长期层级。"""
        if from_type == to_type:
            return 0

        if from_type == "working":
            candidates = self.working.take_for_consolidate(user_id=user_id, threshold=importance_threshold)
        else:
            candidates = self._target(from_type).list_for_consolidate(
                user_id=user_id,
                threshold=importance_threshold,
            )

        moved = 0
        for item in candidates:
            metadata = dict(item.metadata)
            metadata["consolidated_from"] = from_type
            metadata["consolidated_at"] = utc_now_iso()
            new_item = MemoryItem.create(
                user_id=item.user_id,
                session_id=item.session_id,
                memory_type=to_type,
                content=item.content,
                importance=max(item.importance, importance_threshold),
                metadata=metadata,
            )
            self._target(to_type).add(new_item)
            moved += 1
        return moved

    def clear_all(self, user_id: str) -> int:
        """清空用户全部记忆。"""
        removed_working = self.working.clear_user(user_id)
        removed_persistent = self.store.clear_user(user_id)
        self.vector_store.clear_user(user_id)
        return removed_working + removed_persistent

    def get_stats(self, user_id: str) -> dict[str, Any]:
        """统计信息：返回各类型数量和总数。"""
        counts = self.store.count_by_type(user_id=user_id)
        counts["working"] = self.working.count_user(user_id=user_id)
        total = sum(counts.values())
        return {"counts": counts, "total": total}

    def get_summary(self, user_id: str, session_id: str, limit: int = 8) -> dict[str, Any]:
        """摘要信息：返回近期主题和记忆摘要。"""
        profile = self.get_profile_snapshot(user_id=user_id, limit=limit)
        session = self.get_session_snapshot(user_id=user_id, session_id=session_id, limit=limit)
        return {
            "profile": profile,
            "session": session,
            "stats": self.get_stats(user_id),
        }

    def get_session_snapshot(self, user_id: str, session_id: str, limit: int = 6) -> dict[str, Any]:
        """会话快照：读取工作记忆窗口。"""
        return self.working.get_snapshot(user_id=user_id, session_id=session_id, limit=limit)

    def get_profile_snapshot(self, user_id: str, limit: int = 30) -> dict[str, Any]:
        """画像快照：从长期记忆抽取偏好和历史主题。"""
        episodic_items = self.store.list_items(user_id=user_id, memory_type="episodic", limit=max(limit, 60))
        semantic_items = self.store.list_items(user_id=user_id, memory_type="semantic", limit=max(limit, 60))
        return {
            "Memory": {
                "memory_summaries": [item.content[:500] for item in episodic_items[:limit]] + 
                [item.content[:500] for item in semantic_items[:limit]],
            },
        }

    def get_memory_bundle(self, user_id: str, session_id: str, limit: int = 6) -> dict[str, Any]:
        """记忆包：一次返回session+profile。"""
        return {
            "session": self.get_session_snapshot(user_id=user_id, session_id=session_id, limit=limit),
            "profile": self.get_profile_snapshot(user_id=user_id, limit=max(10, limit * 3)),
            "tool_success": True,
        }

    def get_episodic_compact_candidates(self, user_id: str, batch_size: int = 10) -> list[MemoryItem]:
        """提取压缩候选：返回最旧的未压缩episodic+semantic批次。"""
        candidate_types = ("episodic", "semantic")
        all_items: list[MemoryItem] = []
        for memory_type in candidate_types:
            all_items.extend(
                self.store.list_items(
                    user_id=user_id,
                    memory_type=memory_type,
                    limit=10_000,
                )
            )
        if not all_items:
            return []
        # 重要变量：按时间正序压缩，避免持续挤压最新上下文。
        ordered = sorted(all_items, key=lambda item: parse_iso_or_now(item.timestamp))
        raw_items = [item for item in ordered if not bool(item.metadata.get("compacted"))]
        return raw_items[: max(1, int(batch_size))]

    def compact_episodic_batch(
        self,
        user_id: str,
        source_items: list[MemoryItem],
        summary_text: str,
        mode: str,
        topic_hint: str | None = None,
        key_points: list[str] | None = None,
    ) -> dict[str, Any]:
        """执行一批压缩：写入1条摘要并删除原始条目。"""
        content = (summary_text or "").strip()
        if not content:
            return {"success": False, "error": "empty summary text"}

        allowed_types = {"episodic", "semantic"}
        batch = [
            item
            for item in source_items
            if item.user_id == user_id and item.memory_type in allowed_types
        ]
        if len(batch) < 10:
            return {"success": False, "error": "source batch is smaller than 10"}

        ordered = sorted(batch, key=lambda item: parse_iso_or_now(item.timestamp))
        source_ids = [item.id for item in ordered]
        source_types: list[str] = []
        for item in ordered:
            if item.memory_type not in source_types:
                source_types.append(item.memory_type)

        topics: list[str] = []
        for item in ordered:
            raw_topic = item.metadata.get("topic")
            topic = raw_topic.strip() if isinstance(raw_topic, str) else ""
            if topic and topic not in topics:
                topics.append(topic)

        normalized_points: list[str] = []
        for point in key_points or []:
            text = str(point).strip()
            if text and text not in normalized_points:
                normalized_points.append(text)

        # 中文注释: 摘要条目保留压缩来源，便于后续检索和调试。
        summary_metadata: dict[str, Any] = {
            "compacted": True,
            "source_count": len(ordered),
            "source_memory_ids": source_ids,
            "source_memory_types": source_types,
            "range_start_ts": ordered[0].timestamp,
            "range_end_ts": ordered[-1].timestamp,
            "topics": topics[:10],
            "mode": mode,
            "key_points": normalized_points[:8],
            "created_at": utc_now_iso(),
        }
        if isinstance(topic_hint, str) and topic_hint.strip():
            summary_metadata["topic"] = topic_hint.strip()

        summary_item = MemoryItem.create(
            user_id=user_id,
            session_id=ordered[-1].session_id,
            memory_type="episodic",
            content=content,
            importance=max(0.5, max(item.importance for item in ordered)),
            metadata=summary_metadata,
        )
        summary_id = self.episodic.add(summary_item)

        removed = 0
        for item in ordered:
            self.vector_store.remove(user_id=user_id, memory_type=item.memory_type, memory_id=item.id)
            if self.store.hard_delete_item(item.id):
                removed += 1

        return {
            "success": True,
            "summary_memory_id": summary_id,
            "removed_count": removed,
            "source_count": len(ordered),
        }

    def _target(self, memory_type: MemoryType):
        """获取类型实例。"""
        if memory_type == "working":
            return self.working
        if memory_type == "episodic":
            return self.episodic
        if memory_type == "semantic":
            return self.semantic
        return self.perceptual
