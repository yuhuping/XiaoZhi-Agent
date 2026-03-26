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

    def _cleanup(self) -> None:
        now = datetime.now(timezone.utc)
        expired = [
            session_id
            for session_id, record in self._sessions.items()
            if now - record.last_seen > self.ttl
        ]
        for session_id in expired:
            self._sessions.pop(session_id, None)
