from __future__ import annotations

from typing import Any

from app.memory.base import MemoryType
from app.memory.manager import MemoryManager


class MemoryTool:
    """记忆工具：统一 action 入口。"""

    def __init__(self, manager: MemoryManager) -> None:
        self.manager = manager

    def execute(self, action: str, **kwargs: Any) -> dict[str, Any]:
        """执行记忆动作：返回统一结构。"""
        try:
            if action == "add":
                return {"success": True, "data": self._add(**kwargs), "error": None}
            if action == "search":
                return {"success": True, "data": self._search(**kwargs), "error": None}
            if action == "summary":
                return {"success": True, "data": self._summary(**kwargs), "error": None}
            if action == "stats":
                return {"success": True, "data": self._stats(**kwargs), "error": None}
            if action == "update":
                return {"success": True, "data": self._update(**kwargs), "error": None}
            if action == "remove":
                return {"success": True, "data": self._remove(**kwargs), "error": None}
            if action == "forget":
                return {"success": True, "data": self._forget(**kwargs), "error": None}
            if action == "consolidate":
                return {"success": True, "data": self._consolidate(**kwargs), "error": None}
            if action == "clear_all":
                return {"success": True, "data": self._clear_all(**kwargs), "error": None}
            if action == "read_bundle":
                return {"success": True, "data": self._read_bundle(**kwargs), "error": None}
            if action == "read_session":
                return {"success": True, "data": self._read_session(**kwargs), "error": None}
            if action == "read_profile":
                return {"success": True, "data": self._read_profile(**kwargs), "error": None}
            return {"success": False, "data": {}, "error": f"unsupported action: {action}"}
        except Exception as exc:
            return {"success": False, "data": {}, "error": str(exc)}

    def _add(
        self,
        user_id: str,
        session_id: str,
        content: str,
        memory_type: MemoryType = "working",
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """添加记忆。"""
        memory_id = self.manager.add_memory(
            user_id=user_id,
            session_id=session_id,
            content=content,
            memory_type=memory_type,
            importance=importance,
            metadata=metadata,
        )
        return {"memory_id": memory_id, "memory_type": memory_type}

    def _search(
        self,
        user_id: str,
        session_id: str,
        query: str,
        limit: int = 5,
        memory_types: list[MemoryType] | None = None,
        min_importance: float = 0.1,
        target_modality: str | None = None,
    ) -> dict[str, Any]:
        """搜索记忆。"""
        results = self.manager.retrieve_memories(
            user_id=user_id,
            session_id=session_id,
            query=query,
            limit=limit,
            memory_types=memory_types,
            min_importance=min_importance,
            target_modality=target_modality,
        )
        return {
            "query": query,
            "results": [
                {
                    "memory_id": result.item.id,
                    "memory_type": result.item.memory_type,
                    "score": round(float(result.score), 4),
                    "content": result.item.content,
                    "importance": result.item.importance,
                    "metadata": result.item.metadata,
                    "timestamp": result.item.timestamp,
                }
                for result in results
            ],
        }

    def _summary(self, user_id: str, session_id: str, limit: int = 8) -> dict[str, Any]:
        """获取摘要。"""
        return self.manager.get_summary(user_id=user_id, session_id=session_id, limit=limit)

    def _stats(self, user_id: str) -> dict[str, Any]:
        """获取统计。"""
        return self.manager.get_stats(user_id=user_id)

    def _update(self, memory_id: str, patch: dict[str, Any]) -> dict[str, Any]:
        """更新记忆。"""
        ok = self.manager.update_memory(memory_id=memory_id, patch=patch)
        return {"updated": ok}

    def _remove(self, user_id: str, session_id: str, memory_id: str) -> dict[str, Any]:
        """删除记忆。"""
        ok = self.manager.remove_memory(user_id=user_id, session_id=session_id, memory_id=memory_id)
        return {"removed": ok}

    def _forget(
        self,
        user_id: str,
        strategy: str = "importance_based",
        threshold: float = 0.1,
        max_age_days: int = 30,
        memory_types: list[MemoryType] | None = None,
    ) -> dict[str, Any]:
        """遗忘记忆。"""
        count = self.manager.forget_memories(
            user_id=user_id,
            strategy=strategy,
            threshold=threshold,
            max_age_days=max_age_days,
            memory_types=memory_types,
        )
        return {"forgotten": count, "strategy": strategy}

    def _consolidate(
        self,
        user_id: str,
        from_type: MemoryType = "working",
        to_type: MemoryType = "episodic",
        importance_threshold: float = 0.7,
    ) -> dict[str, Any]:
        """整合记忆。"""
        count = self.manager.consolidate_memories(
            user_id=user_id,
            from_type=from_type,
            to_type=to_type,
            importance_threshold=importance_threshold,
        )
        return {
            "consolidated": count,
            "from_type": from_type,
            "to_type": to_type,
            "importance_threshold": importance_threshold,
        }

    def _clear_all(self, user_id: str) -> dict[str, Any]:
        """清空记忆。"""
        count = self.manager.clear_all(user_id=user_id)
        return {"cleared": count}

    def _read_bundle(self, user_id: str, session_id: str, limit: int = 6) -> dict[str, Any]:
        """读取记忆包。"""
        return self.manager.get_memory_bundle(user_id=user_id, session_id=session_id, limit=limit)

    def _read_session(self, user_id: str, session_id: str, limit: int = 6) -> dict[str, Any]:
        """读取会话记忆。"""
        return self.manager.get_session_snapshot(user_id=user_id, session_id=session_id, limit=limit)

    def _read_profile(self, user_id: str, limit: int = 30) -> dict[str, Any]:
        """读取画像记忆。"""
        return self.manager.get_profile_snapshot(user_id=user_id, limit=limit)
