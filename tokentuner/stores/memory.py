"""
In-process LRU with per-entry expiry.

The default backend, and the only one that is safe with no further decisions:
it dies with the worker, so nothing outlives the process, and it needs no
infrastructure. In a multi-worker deployment each worker keeps its own copy,
which lowers the hit rate but never produces a wrong hit.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any, Optional

from .base import Store


class MemoryStore(Store):
    persistent = False
    name = "memory"

    def __init__(self, max_entries: int = 2000) -> None:
        self._max = max(1, int(max_entries))
        self._data: "OrderedDict[str, tuple]" = OrderedDict()
        self._lock = threading.Lock()
        self._evictions = 0
        self._expiries = 0

    def get(self, key: str) -> Optional[Any]:
        now = time.time()
        with self._lock:
            row = self._data.get(key)
            if row is None:
                return None
            expires_at, value = row
            if expires_at and expires_at <= now:
                del self._data[key]
                self._expiries += 1
                return None
            # Touch: this is the L in LRU.
            self._data.move_to_end(key)
            return value

    def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        expires_at = time.time() + ttl_seconds if ttl_seconds > 0 else 0.0
        with self._lock:
            self._data[key] = (expires_at, value)
            self._data.move_to_end(key)
            while len(self._data) > self._max:
                self._data.popitem(last=False)
                self._evictions += 1

    def delete(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def stats(self) -> dict:
        with self._lock:
            return {
                "store": self.name,
                "persistent": False,
                "entries": len(self._data),
                "max_entries": self._max,
                "evictions": self._evictions,
                "expiries": self._expiries,
            }
