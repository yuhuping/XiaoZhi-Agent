from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ProfileMemoryStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_table()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_table(self) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS profile_memory (
                    profile_id TEXT PRIMARY KEY,
                    age_band TEXT,
                    preferred_topics TEXT,
                    repeated_mistakes TEXT,
                    interaction_preferences TEXT,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def read_profile(self, profile_id: str) -> dict[str, Any]:
        conn = self._connect()
        try:
            with self._lock:
                row = conn.execute(
                    "SELECT * FROM profile_memory WHERE profile_id = ?",
                    (profile_id,),
                ).fetchone()
        finally:
            conn.close()
        if row is None:
            return {}
        memory_payload = self._load_json_field(row["interaction_preferences"], {})
        return {
            "profile_id": row["profile_id"],
            "age_band": row["age_band"] or "",
            "preferred_topics": self._load_json_field(row["preferred_topics"], []),
            "repeated_mistakes": self._load_json_field(row["repeated_mistakes"], []),
            "Memory": memory_payload,
            "updated_at": row["updated_at"],
        }

    def write_profile(self, profile_id: str, patch: dict[str, Any]) -> None:
        existing = self.read_profile(profile_id)
        incoming_memory = patch.get("Memory")
        if not isinstance(incoming_memory, dict):
            # backward compatibility for old callers
            fallback = patch.get("interaction_preferences")
            incoming_memory = fallback if isinstance(fallback, dict) else {}
        merged = {
            "age_band": patch.get("age_band") or existing.get("age_band") or "",
            "preferred_topics": self._merge_string_list(
                existing.get("preferred_topics", []),
                patch.get("preferred_topics", []),
            ),
            "repeated_mistakes": self._merge_string_list(
                existing.get("repeated_mistakes", []),
                patch.get("repeated_mistakes", []),
            ),
            "Memory": self._merge_memory(
                existing.get("Memory", {}),
                incoming_memory,
            ),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        conn = self._connect()
        try:
            with self._lock:
                conn.execute(
                    """
                    INSERT INTO profile_memory (
                        profile_id,
                        age_band,
                        preferred_topics,
                        repeated_mistakes,
                        interaction_preferences,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(profile_id) DO UPDATE SET
                        age_band = excluded.age_band,
                        preferred_topics = excluded.preferred_topics,
                        repeated_mistakes = excluded.repeated_mistakes,
                        interaction_preferences = excluded.interaction_preferences,
                        updated_at = excluded.updated_at
                    """,
                    (
                        profile_id,
                        merged["age_band"],
                        json.dumps(merged["preferred_topics"], ensure_ascii=False),
                        json.dumps(merged["repeated_mistakes"], ensure_ascii=False),
                        json.dumps(merged["Memory"], ensure_ascii=False),
                        merged["updated_at"],
                    ),
                )
                conn.commit()
        finally:
            conn.close()

    def _merge_memory(self, current: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
        merged = {**current, **incoming}

        current_history = current.get("topic_history", [])
        incoming_history = incoming.get("topic_history", [])
        replace_history = bool(incoming.get("_replace_topic_history"))
        if replace_history:
            merged["topic_history"] = self._normalize_topic_events(incoming_history, limit=30)
        else:
            merged["topic_history"] = self._merge_topic_events(
                current_history=current_history,
                incoming_history=incoming_history,
                limit=30,
            )

        merged["memory_summaries"] = self._merge_summary_items(
            current_items=current.get("memory_summaries", []),
            incoming_items=incoming.get("memory_summaries", []),
            limit=5,
        )
        merged.pop("_replace_topic_history", None)
        return merged

    def _load_json_field(self, raw: str | None, default: Any) -> Any:
        if not raw:
            return default
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return default
        return parsed if isinstance(parsed, type(default)) else default

    def _merge_string_list(self, current: list[str], incoming: list[str]) -> list[str]:
        merged = [*current]
        for value in incoming:
            if isinstance(value, str):
                normalized = value.strip()
                if normalized and normalized not in merged:
                    merged.append(normalized)
        return merged[-10:]

    def _normalize_topic_events(self, events: Any, limit: int) -> list[dict[str, str]]:
        if not isinstance(events, list):
            return []
        normalized: list[dict[str, str]] = []
        for item in events:
            if not isinstance(item, dict):
                continue
            topic = item.get("topic")
            ts = item.get("ts")
            mode = item.get("mode")
            if not isinstance(topic, str) or not topic.strip():
                continue
            normalized.append(
                {
                    "topic": topic.strip(),
                    "mode": mode.strip() if isinstance(mode, str) else "",
                    "ts": ts.strip() if isinstance(ts, str) else "",
                }
            )
        return normalized[-limit:]

    def _merge_topic_events(
        self,
        current_history: Any,
        incoming_history: Any,
        limit: int,
    ) -> list[dict[str, str]]:
        merged = self._normalize_topic_events(current_history, limit=10_000)
        merged.extend(self._normalize_topic_events(incoming_history, limit=10_000))
        return merged[-limit:]

    def _merge_summary_items(self, current_items: Any, incoming_items: Any, limit: int) -> list[dict[str, Any]]:
        if not isinstance(current_items, list):
            current_items = []
        if not isinstance(incoming_items, list):
            incoming_items = []
        merged: list[dict[str, Any]] = []
        for item in [*current_items, *incoming_items]:
            if isinstance(item, dict):
                merged.append(item)
        return merged[-limit:]
