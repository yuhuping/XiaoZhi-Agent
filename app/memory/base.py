from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

MemoryType = Literal["working", "episodic", "semantic", "perceptual"]


@dataclass
class MemoryItem:
    """记忆项：统一的记忆数据结构。"""

    id: str
    user_id: str
    session_id: str
    memory_type: MemoryType
    content: str
    importance: float = 0.5
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)
    last_accessed: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    access_count: int = 0

    @classmethod
    def create(
        cls,
        user_id: str,
        session_id: str,
        memory_type: MemoryType,
        content: str,
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> "MemoryItem":
        """创建记忆项：自动生成ID和时间戳。"""
        return cls(
            id=str(uuid4()),
            user_id=user_id,
            session_id=session_id,
            memory_type=memory_type,
            content=content,
            importance=max(0.0, min(1.0, float(importance))),
            metadata=metadata or {},
        )


@dataclass(frozen=True)
class MemorySearchResult:
    """检索结果：包含记忆项与综合分数。"""

    item: MemoryItem
    score: float


@dataclass(frozen=True)
class MemoryConfig:
    """记忆配置：控制容量、TTL、阈值和本地存储路径。"""

    db_path: str
    index_dir: str
    working_memory_capacity: int = 50
    working_memory_ttl_minutes: int = 60
    default_search_limit: int = 5
    default_min_importance: float = 0.1
    consolidate_working_threshold: float = 0.7
    consolidate_episodic_threshold: float = 0.8
    forget_default_max_age_days: int = 30
    vector_dim: int = 256


def utc_now_iso() -> str:
    """返回UTC时间字符串。"""
    return datetime.now(timezone.utc).isoformat()


def parse_iso_or_now(value: str | None) -> datetime:
    """解析ISO时间，失败时回退到当前UTC时间。"""
    if not value:
        return datetime.now(timezone.utc)
    try:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed
    except Exception:
        return datetime.now(timezone.utc)
