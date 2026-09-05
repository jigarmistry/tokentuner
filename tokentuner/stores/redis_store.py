"""
Redis backend.

The right choice when several workers or several machines should share hits.
Values are stored as JSON rather than pickled: a cache is a place other tools
will want to read, and a pickle in a shared store is a remote-code-execution
surface waiting for someone to write to it.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from .base import Store


class RedisStore(Store):
    persistent = True
    name = "redis"

    def __init__(self, url: str, namespace: str = "tt", client: Any = None) -> None:
        if client is not None:
            self._r = client
        else:
            if not url:
                raise ValueError("RedisStore needs a url")
            import redis  # imported lazily: optional extra

            self._r = redis.Redis.from_url(url, socket_timeout=2)
        self._ns = namespace

    def _k(self, key: str) -> str:
        return f"{self._ns}:{key}"

    def get(self, key: str) -> Optional[Any]:
        try:
            raw = self._r.get(self._k(key))
        except Exception:  # noqa: BLE001
            return None
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except Exception:  # noqa: BLE001
            return None

    def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        try:
            self._r.set(self._k(key), json.dumps(value, default=str),
                        ex=int(ttl_seconds) if ttl_seconds > 0 else None)
        except Exception:  # noqa: BLE001
            pass

    def delete(self, key: str) -> None:
        try:
            self._r.delete(self._k(key))
        except Exception:  # noqa: BLE001
            pass

    def clear(self) -> None:
        try:
            for key in self._r.scan_iter(match=f"{self._ns}:*", count=500):
                self._r.delete(key)
        except Exception:  # noqa: BLE001
            pass

    def stats(self) -> dict:
        return {"store": self.name, "persistent": True, "namespace": self._ns}
