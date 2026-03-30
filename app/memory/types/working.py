from __future__ import annotations

from collections import defaultdict
from datetime import timedelta
from typing import Any

from app.memory.base import MemoryConfig, MemoryItem, MemorySearchResult, parse_iso_or_now, utc_now_iso
from app.memory.embedding import HashingTextEmbedder


class WorkingMemory:
    """工作记忆：会话内短期记忆，支持TTL和容量管理。"""

    def __init__(self, config: MemoryConfig, embedder: HashingTextEmbedder) -> None:
        self.config = config
        self.embedder = embedder
        # 重要变量：按 user+session 保存短期记忆。
        self._session_memories: dict[str, list[MemoryItem]] = defaultdict(list)

    def add(self, item: MemoryItem) -> str:
        """添加工作记忆：写入前先做清理。"""
        key = self._key(item.user_id, item.session_id)
        bucket = self._session_memories[key]
        self._expire(bucket)
        bucket.append(item)
        if len(bucket) > self.config.working_memory_capacity:
            bucket.sort(key=lambda x: (x.importance, x.timestamp))
            bucket[:] = bucket[-self.config.working_memory_capacity :]
        return item.id

    def retrieve(
        self,
        user_id: str,
        session_id: str,
        query: str,
        limit: int,
        min_importance: float,
    ) -> list[MemorySearchResult]:
        """检索工作记忆：混合相似度、时间衰减、重要性。"""
        key = self._key(user_id, session_id)
        bucket = self._session_memories.get(key, [])
        self._expire(bucket)
        if not bucket:
            return []

        query_vec = self.embedder.encode_text(query)
        now = parse_iso_or_now(utc_now_iso())
        results: list[MemorySearchResult] = []
        for item in bucket:
            if item.importance < min_importance:
                continue
            vec = self.embedder.encode_text(item.content)
            similarity = float(vec @ query_vec)
            keyword_score = self._keyword_score(query, item.content)
            base_relevance = similarity * 0.7 + keyword_score * 0.3 if similarity > 0 else keyword_score

            age_minutes = max(0.0, (now - parse_iso_or_now(item.timestamp)).total_seconds() / 60.0)
            time_decay = max(0.1, 1.0 - age_minutes / max(1.0, float(self.config.working_memory_ttl_minutes * 2)))
            importance_weight = 0.8 + item.importance * 0.4
            final_score = base_relevance * time_decay * importance_weight
            if final_score > 0:
                results.append(MemorySearchResult(item=item, score=final_score))

        results.sort(key=lambda x: x.score, reverse=True)
        return results[: max(1, limit)]

    def get_snapshot(self, user_id: str, session_id: str, limit: int = 6) -> dict[str, Any]:
        """会话快照：给 state_update 和 prompt 使用。"""
        key = self._key(user_id, session_id)
        bucket = self._session_memories.get(key, [])
        self._expire(bucket)
        if not bucket:
            return {"recent_turns": [], "last_topic": None, "last_agent_question": None}

        recent = bucket[-max(1, limit) :]
        turns = [item.metadata.get("turn") for item in recent if isinstance(item.metadata.get("turn"), dict)]

        last_topic = None
        last_agent_question = None
        for item in reversed(recent):
            topic = item.metadata.get("topic")
            if last_topic is None and isinstance(topic, str) and topic.strip():
                last_topic = topic.strip()
            question = item.metadata.get("asked_question")
            role = item.metadata.get("role")
            if role == "assistant" and isinstance(question, str) and question.strip():
                last_agent_question = question.strip()
                break

        return {
            "recent_turns": turns,
            "last_topic": last_topic,
            "last_agent_question": last_agent_question,
        }

    def remove(self, user_id: str, session_id: str, memory_id: str) -> bool:
        """删除指定工作记忆。"""
        key = self._key(user_id, session_id)
        bucket = self._session_memories.get(key, [])
        old_len = len(bucket)
        bucket[:] = [item for item in bucket if item.id != memory_id]
        return len(bucket) != old_len

    def clear_user(self, user_id: str) -> int:
        """清空用户所有工作记忆。"""
        keys = [k for k in self._session_memories if k.startswith(f"{user_id}::")]
        removed = 0
        for key in keys:
            removed += len(self._session_memories.get(key, []))
            self._session_memories.pop(key, None)
        return removed

    def count_user(self, user_id: str) -> int:
        """统计用户工作记忆数量。"""
        total = 0
        keys = [k for k in self._session_memories if k.startswith(f"{user_id}::")]
        for key in keys:
            bucket = self._session_memories.get(key, [])
            self._expire(bucket)
            total += len(bucket)
        return total

    def forget(
        self,
        user_id: str,
        strategy: str,
        threshold: float,
        max_age_days: int,
    ) -> int:
        """遗忘工作记忆：支持重要性、时间、容量策略。"""
        removed = 0
        now = parse_iso_or_now(utc_now_iso())
        ttl_delta = timedelta(days=max(1, max_age_days))
        user_keys = [k for k in self._session_memories if k.startswith(f"{user_id}::")]
        for key in user_keys:
            bucket = self._session_memories.get(key, [])
            self._expire(bucket)
            before = len(bucket)
            if strategy == "importance_based":
                bucket[:] = [item for item in bucket if item.importance >= threshold]
            elif strategy == "time_based":
                bucket[:] = [
                    item
                    for item in bucket
                    if now - parse_iso_or_now(item.timestamp) <= ttl_delta
                ]
            elif strategy == "capacity_based":
                keep = max(1, int(self.config.working_memory_capacity * max(0.2, min(1.0, threshold))))
                bucket.sort(key=lambda x: (x.importance, x.timestamp))
                bucket[:] = bucket[-keep:]
            removed += before - len(bucket)
        return removed

    def take_for_consolidate(self, user_id: str, threshold: float, limit: int = 20) -> list[MemoryItem]:
        """根据重要性阈值threshold提取可整合记忆：用于 working -> episodic。"""
        candidates: list[MemoryItem] = []
        keys = [k for k in self._session_memories if k.startswith(f"{user_id}::")]
        for key in keys:
            bucket = self._session_memories.get(key, [])
            self._expire(bucket)
            candidates.extend([item for item in bucket if item.importance >= threshold])
        candidates.sort(key=lambda x: (x.importance, x.timestamp), reverse=True)
        return candidates[: max(1, limit)]

    def _expire(self, bucket: list[MemoryItem]) -> None:
        """TTL过期清理。"""
        now = parse_iso_or_now(utc_now_iso())
        max_age = timedelta(minutes=max(1, self.config.working_memory_ttl_minutes))
        bucket[:] = [item for item in bucket if now - parse_iso_or_now(item.timestamp) <= max_age]

    def _key(self, user_id: str, session_id: str) -> str:
        """生成会话键。"""
        return f"{user_id}::{session_id}"

    def _keyword_score(self, query: str, content: str) -> float:
        """关键词匹配分。"""
        query_tokens = set(self.embedder._tokenize(query))
        content_tokens = set(self.embedder._tokenize(content))
        if not query_tokens or not content_tokens:
            return 0.0
        return len(query_tokens & content_tokens) / max(1, len(query_tokens))
