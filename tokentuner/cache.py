"""
The response cache.

The rules about *when not to cache* are the interesting part of this file. A
cache that stores everything is easy to write and produces wrong answers:

  * High temperature means the caller asked for variety. Returning a previous
    answer removes exactly the thing they asked for, and it does so invisibly -
    a "regenerate" button that returns the same draft word for word looks like
    a model quirk, not like a cache.

  * Failed and escalated-away answers are never stored. Caching a failure turns
    one bad minute into an hour of them.

  * Responses to prompts that can carry personal data stay in process memory
    unless persistence is explicitly enabled. A cache entry is a second copy of
    that data, held somewhere nobody wrote a retention policy for.

  * Every entry carries the fingerprint of the inputs it was built from, and a
    hit is only served when that fingerprint still matches. A digest collision
    is unlikely; silently answering the wrong prompt is bad enough that the
    check is worth its cost.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from .config import TunerConfig, get_config
from .keys import call_key
from .ledger import Ledger, default_ledger
from .stores import Store, build_store
from .stores.base import NullStore


class CacheDecision:
    """Why a call was or was not cacheable. Returned rather than logged so the
    caller can put the reason on its own telemetry."""

    __slots__ = ("cacheable", "reason")

    def __init__(self, cacheable: bool, reason: str = "") -> None:
        self.cacheable = cacheable
        self.reason = reason

    def __bool__(self) -> bool:
        return self.cacheable


class ResponseCache:
    def __init__(self, config: Optional[TunerConfig] = None,
                 store: Optional[Store] = None,
                 ledger: Optional[Ledger] = None) -> None:
        self._config = config or get_config()
        self._store = store if store is not None else build_store(self._config)
        self._ledger = ledger or default_ledger()

    def use_ledger(self, ledger: Ledger) -> None:
        """Adopt someone else's ledger.

        A cache and the tuner holding it must report into the same ledger, or
        the hit counters live in one object and the savings totals in another -
        and the resulting stats look plausible while being wrong.
        """
        self._ledger = ledger

    @property
    def store(self) -> Store:
        return self._store

    @property
    def config(self) -> TunerConfig:
        return self._config

    # -- policy ---------------------------------------------------------------

    def should_read(self, *, task_cacheable: bool, temperature: Optional[float],
                    stream: bool = False, tools: Optional[list] = None) -> CacheDecision:
        cfg = self._config
        if not cfg.enabled or not cfg.cache_enabled:
            return CacheDecision(False, "disabled")
        if isinstance(self._store, NullStore):
            return CacheDecision(False, "no_store")
        if not task_cacheable:
            return CacheDecision(False, "task_opted_out")
        if stream:
            # A streamed answer can be replayed from cache, but only by a caller
            # that knows to do so; the read path here returns whole responses.
            return CacheDecision(False, "stream")
        if tools:
            # Tool-calling turns depend on state the key cannot see - what the
            # tools would return right now. Caching them serves yesterday's data.
            return CacheDecision(False, "tool_call")
        if temperature is not None and float(temperature) > cfg.cache_max_temperature:
            return CacheDecision(False, "temperature")
        return CacheDecision(True)

    def should_write(self, *, decision: CacheDecision, ok: bool,
                     sensitive: bool, value_bytes: int) -> CacheDecision:
        if not decision.cacheable:
            return decision
        if not ok:
            return CacheDecision(False, "not_ok")
        if value_bytes > self._config.cache_max_value_bytes:
            return CacheDecision(False, "too_large")
        if sensitive and self._store.persistent and not self._config.cache_persist_sensitive:
            return CacheDecision(False, "sensitive_persistent_store")
        return CacheDecision(True)

    # -- keys -----------------------------------------------------------------

    def key(self, **kw) -> str:
        kw.setdefault("namespace", self._config.cache_namespace)
        return call_key(**kw)

    # -- read / write ---------------------------------------------------------

    def get(self, key: str, *, task: str = "", model: str = "",
            fingerprint: Optional[str] = None) -> Optional[dict]:
        try:
            entry = self._store.get(key)
        except Exception:  # noqa: BLE001
            entry = None

        if not isinstance(entry, dict):
            self._ledger.miss(task, model)
            return None
        if fingerprint and entry.get("fingerprint") != fingerprint:
            # Same key, different inputs. Refuse rather than guess.
            self._ledger.miss(task, model)
            return None

        self._ledger.hit(
            task,
            entry.get("model") or model,
            tokens_saved=int(entry.get("tokens_in") or 0) + int(entry.get("tokens_out") or 0),
            latency_saved_ms=int(entry.get("latency_ms") or 0),
        )
        return entry

    def put(self, key: str, value: dict, *, task: str = "", model: str = "",
            ttl_seconds: Optional[int] = None, fingerprint: Optional[str] = None) -> None:
        entry = dict(value)
        entry["cached_at"] = time.time()
        entry["task"] = task or entry.get("task")
        entry["model"] = model or entry.get("model")
        if fingerprint:
            entry["fingerprint"] = fingerprint
        ttl = self._config.cache_ttl_seconds if ttl_seconds is None else ttl_seconds
        try:
            self._store.set(key, entry, ttl)
            self._ledger.stored(task, model)
        except Exception:  # noqa: BLE001
            pass

    def invalidate(self, key: str) -> None:
        try:
            self._store.delete(key)
        except Exception:  # noqa: BLE001
            pass

    def clear(self) -> None:
        try:
            self._store.clear()
        except Exception:  # noqa: BLE001
            pass

    def stats(self) -> dict:
        out = {"enabled": self._config.cache_enabled and self._config.enabled}
        try:
            out.update(self._store.stats())
        except Exception:  # noqa: BLE001
            pass
        return out


def value_bytes(value: Any) -> int:
    """Rough serialised size, used only for the too-large check."""
    try:
        import json

        return len(json.dumps(value, default=str))
    except Exception:  # noqa: BLE001
        return 0
