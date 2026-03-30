from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from app.memory.base import MemoryItem


class SQLiteMemoryStore:
    """SQLite记忆存储：负责持久化、更新、过滤查询。"""

    def __init__(self, db_path: str) -> None:
        # 重要变量：数据库文件路径。
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_tables()

    def _connect(self) -> sqlite3.Connection:
        """创建数据库连接：统一设置 row_factory。"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_tables(self) -> None:
        """初始化表结构：memory_items + schema_version。"""
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_items (
                    memory_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    importance REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    last_accessed TEXT NOT NULL,
                    access_count INTEGER NOT NULL,
                    archived INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY
                )
                """
            )
            conn.execute("DELETE FROM schema_version")
            conn.execute("INSERT INTO schema_version(version) VALUES (1)")
            conn.commit()
        finally:
            conn.close()

    def upsert_item(self, item: MemoryItem) -> None:
        """写入或覆盖记忆项。"""
        conn = self._connect()
        try:
            with self._lock:
                conn.execute(
                    """
                    INSERT INTO memory_items (
                        memory_id,user_id,session_id,memory_type,content,importance,
                        timestamp,metadata,last_accessed,access_count,archived
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                    ON CONFLICT(memory_id) DO UPDATE SET
                        user_id=excluded.user_id,
                        session_id=excluded.session_id,
                        memory_type=excluded.memory_type,
                        content=excluded.content,
                        importance=excluded.importance,
                        timestamp=excluded.timestamp,
                        metadata=excluded.metadata,
                        last_accessed=excluded.last_accessed,
                        access_count=excluded.access_count,
                        archived=0
                    """,
                    (
                        item.id,
                        item.user_id,
                        item.session_id,
                        item.memory_type,
                        item.content,
                        float(item.importance),
                        item.timestamp,
                        json.dumps(item.metadata, ensure_ascii=False),
                        item.last_accessed,
                        int(item.access_count),
                    ),
                )
                conn.commit()
        finally:
            conn.close()

    def get_item(self, memory_id: str) -> MemoryItem | None:
        """按ID读取记忆项。"""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM memory_items WHERE memory_id=? AND archived=0",
                (memory_id,),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        return self._row_to_item(row)

    def update_item(self, memory_id: str, patch: dict[str, Any]) -> bool:
        """更新记忆项：仅更新提供的字段。"""
        old = self.get_item(memory_id)
        if old is None:
            return False
        metadata = dict(old.metadata)
        if isinstance(patch.get("metadata"), dict):
            metadata.update(patch["metadata"])
        updated = MemoryItem(
            id=old.id,
            user_id=str(patch.get("user_id") or old.user_id),
            session_id=str(patch.get("session_id") or old.session_id),
            memory_type=str(patch.get("memory_type") or old.memory_type),  # type: ignore[arg-type]
            content=str(patch.get("content") or old.content),
            importance=max(0.0, min(1.0, float(patch.get("importance", old.importance)))),
            timestamp=str(patch.get("timestamp") or old.timestamp),
            metadata=metadata,
            last_accessed=str(patch.get("last_accessed") or old.last_accessed),
            access_count=int(patch.get("access_count", old.access_count)),
        )
        self.upsert_item(updated)
        return True

    def delete_item(self, memory_id: str) -> bool:
        """软删除记忆项。"""
        conn = self._connect()
        try:
            with self._lock:
                cursor = conn.execute(
                    "UPDATE memory_items SET archived=1 WHERE memory_id=?",
                    (memory_id,),
                )
                conn.commit()
                return int(cursor.rowcount) > 0
        finally:
            conn.close()

    def hard_delete_item(self, memory_id: str) -> bool:
        """硬删除记忆项：物理移除数据库记录。"""
        conn = self._connect()
        try:
            with self._lock:
                cursor = conn.execute(
                    "DELETE FROM memory_items WHERE memory_id=?",
                    (memory_id,),
                )
                conn.commit()
                return int(cursor.rowcount) > 0
        finally:
            conn.close()

    def clear_user(self, user_id: str) -> int:
        """清空用户记忆。"""
        conn = self._connect()
        try:
            with self._lock:
                cursor = conn.execute(
                    "UPDATE memory_items SET archived=1 WHERE user_id=?",
                    (user_id,),
                )
                conn.commit()
                return int(cursor.rowcount)
        finally:
            conn.close()

    def list_items(
        self,
        user_id: str,
        memory_type: str | None = None,
        session_id: str | None = None,
        limit: int | None = None,
    ) -> list[MemoryItem]:
        """筛选读取记忆项。"""
        sql = "SELECT * FROM memory_items WHERE user_id=? AND archived=0"
        args: list[Any] = [user_id]
        if memory_type:
            sql += " AND memory_type=?"
            args.append(memory_type)
        if session_id:
            sql += " AND session_id=?"
            args.append(session_id)
        sql += " ORDER BY timestamp DESC"
        if limit:
            sql += " LIMIT ?"
            args.append(max(1, int(limit)))

        conn = self._connect()
        try:
            rows = conn.execute(sql, tuple(args)).fetchall()
        finally:
            conn.close()
        return [self._row_to_item(row) for row in rows]

    def count_by_type(self, user_id: str) -> dict[str, int]:
        """统计各类型记忆数量。"""
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT memory_type, COUNT(*) AS cnt
                FROM memory_items
                WHERE user_id=? AND archived=0
                GROUP BY memory_type
                """,
                (user_id,),
            ).fetchall()
        finally:
            conn.close()
        result = {"working": 0, "episodic": 0, "semantic": 0, "perceptual": 0}
        for row in rows:
            key = str(row["memory_type"])
            if key in result:
                result[key] = int(row["cnt"])
        return result

    def _row_to_item(self, row: sqlite3.Row) -> MemoryItem:
        """行数据转记忆对象。"""
        metadata = self._load_json(row["metadata"])
        return MemoryItem(
            id=str(row["memory_id"]),
            user_id=str(row["user_id"]),
            session_id=str(row["session_id"]),
            memory_type=str(row["memory_type"]),  # type: ignore[arg-type]
            content=str(row["content"]),
            importance=float(row["importance"]),
            timestamp=str(row["timestamp"]),
            metadata=metadata,
            last_accessed=str(row["last_accessed"]),
            access_count=int(row["access_count"]),
        )

    def _load_json(self, raw: object) -> dict[str, Any]:
        """解析JSON字段。"""
        if not isinstance(raw, str) or not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}
