# bot/session.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List
from uuid import uuid4

@dataclass
class SessionMemory:
    """
    Ephemeral, per-run memory for this CLI/app session.
    - Holds reply_lang and the latest search/suggestion 'hits' for follow-ups.
    - New session_id each run -> fresh memory.
    """
    user_id: str
    reply_lang: str = "en"
    session_id: str = field(default_factory=lambda: str(uuid4()))
    last_hits: List[Dict[str, Any]] = field(default_factory=list)

    def remember_hits(self, hits: List[Dict[str, Any]]) -> None:
        self.last_hits = list(hits or [])

    def get_hits(self) -> List[Dict[str, Any]]:
        return list(self.last_hits or [])
