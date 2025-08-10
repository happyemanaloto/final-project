# kusina_memory.py
from __future__ import annotations
import json, threading
from pathlib import Path
from typing import Any, Dict, Optional

class MemoryStore:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self._data: Dict[str, Dict[str, Any]] = {}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                self._data = {}

    def _save(self):
        self.path.write_text(json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8")

    def get(self, user_id: str) -> Dict[str, Any]:
        with self._lock:
            return self._data.get(user_id, {}).copy()

    def update(self, user_id: str, patch: Dict[str, Any]):
        with self._lock:
            cur = self._data.get(user_id, {})
            cur.update({k: v for k, v in patch.items() if v is not None})
            self._data[user_id] = cur
            self._save()
