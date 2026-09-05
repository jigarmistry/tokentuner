"""
SQLite backend.

Stdlib-only persistence. Its real use is offline work - evaluation runs, batch
scripts, a CLI replayed against the same corpus a dozen times - where the same
prompts recur across separate processes and there is no server to talk to.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from typing import Any, Optional

from .base import Store


# A TTL of zero or less means "no expiry". Expressed as a far-future
# timestamp so the expiry comparison stays a single unconditional SQL predicate.
_NEVER = 4102444800.0  # 2100-01-01


def _expiry(now: float, ttl_seconds: int) -> float:
    return now + ttl_seconds if ttl_seconds > 0 else _NEVER


class SQLiteStore(Store):
    persistent = True
    name = "sqlite"

    def __init__(self, path: str, max_entries: int = 20000) -> None:
        if not path:
            raise ValueError("SQLiteStore needs a path")
        self._path = path
        self._max = max(1, int(max_entries))
        self._lock = threading.Lock()
        # check_same_thread=False plus our own lock: the connection is shared,
        # and every write below happens inside that lock.
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS cache ("
            " key TEXT PRIMARY KEY,"
            " value TEXT NOT NULL,"
            " expires_at REAL NOT NULL,"
            " created_at REAL NOT NULL)"
        )
        self._db.execute("CREATE INDEX IF NOT EXISTS cache_expires ON cache(expires_at)")
        self._db.commit()

    def get(self, key: str) -> Optional[Any]:
        try:
            with self._lock:
                row = self._db.execute(
                    "SELECT value, expires_at FROM cache WHERE key = ?", (key,)
                ).fetchone()
        except Exception:  # noqa: BLE001
            return None
        if not row:
            return None
        value, expires_at = row
        if expires_at and expires_at <= time.time():
            self.delete(key)
            return None
        try:
            return json.loads(value)
        except Exception:  # noqa: BLE001
            return None

    def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        now = time.time()
        try:
            with self._lock:
                self._db.execute(
                    "INSERT OR REPLACE INTO cache (key, value, expires_at, created_at)"
                    " VALUES (?, ?, ?, ?)",
                    (key, json.dumps(value, default=str), _expiry(now, ttl_seconds), now),
                )
                self._db.execute("DELETE FROM cache WHERE expires_at <= ?", (now,))
                # Bound the file. Oldest-first is a reasonable proxy for LRU
                # without paying for an access-time write on every read.
                self._db.execute(
                    "DELETE FROM cache WHERE key IN ("
                    " SELECT key FROM cache ORDER BY created_at DESC LIMIT -1 OFFSET ?)",
                    (self._max,),
                )
                self._db.commit()
        except Exception:  # noqa: BLE001
            pass

    def delete(self, key: str) -> None:
        try:
            with self._lock:
                self._db.execute("DELETE FROM cache WHERE key = ?", (key,))
                self._db.commit()
        except Exception:  # noqa: BLE001
            pass

    def clear(self) -> None:
        try:
            with self._lock:
                self._db.execute("DELETE FROM cache")
                self._db.commit()
        except Exception:  # noqa: BLE001
            pass

    def stats(self) -> dict:
        try:
            with self._lock:
                n = self._db.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
        except Exception:  # noqa: BLE001
            n = -1
        return {"store": self.name, "persistent": True, "entries": n, "path": self._path}
