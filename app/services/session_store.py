from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from app.agent.state import ConversationTurn


@dataclass
class SessionRecord:
    turns: list[ConversationTurn] = field(default_factory=list)
    last_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class SessionStore:
    def __init__(self, max_turns: int = 12, ttl_minutes: int = 30) -> None:
        self.max_turns = max_turns
        self.ttl = timedelta(minutes=ttl_minutes)
        self._sessions: dict[str, SessionRecord] = {}

    def get_history(self, session_id: str) -> list[ConversationTurn]:
        self._cleanup()
        record = self._sessions.get(session_id)
        if not record:
            return []
        record.last_seen = datetime.now(timezone.utc)
        return [*record.turns]

    def append_turns(self, session_id: str, turns: list[ConversationTurn]) -> None:
        self._cleanup()
        record = self._sessions.setdefault(session_id, SessionRecord())
        record.turns.extend(turns)
        if len(record.turns) > self.max_turns:
            record.turns = record.turns[-self.max_turns :]
        record.last_seen = datetime.now(timezone.utc)

    def get_snapshot(self, session_id: str, limit: int = 6) -> dict[str, object]:
        history = self.get_history(session_id)
        if not history:
            return {
                "recent_turns": [],
                "last_topic": None,
                "last_agent_question": None,
            }

        recent_turns = history[-limit:]
        last_topic = None
        last_agent_question = None
        for turn in reversed(history):
            if last_topic is None and turn.get("topic"):
                last_topic = turn.get("topic")
            if turn.get("role") == "assistant" and turn.get("asked_question"):
                last_agent_question = turn.get("asked_question")
                break
        return {
            "recent_turns": recent_turns,
            "last_topic": last_topic,
            "last_agent_question": last_agent_question,
        }

    def _cleanup(self) -> None:
        now = datetime.now(timezone.utc)
        expired = [
            session_id
            for session_id, record in self._sessions.items()
            if now - record.last_seen > self.ttl
        ]
        for session_id in expired:
            self._sessions.pop(session_id, None)
