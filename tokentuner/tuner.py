"""
The facade.

One object holding the config, the cache, the two single-flight registries, the
ledger and the prefix tracker, plus the wrapper that applies them in the right
order:

    cache read  ->  de-duplicate  ->  run  ->  cache write

That order is the whole design. Reading the cache first means the cheapest
outcome is checked first. De-duplicating second means the concurrent callers who
missed the cache still collapse into one request. Writing last means only real
answers are stored.

The wrapper never inspects the value it is caching. Callers hand in `to_entry`
and `from_entry`, so the package stays ignorant of whatever result class the
host application uses - which is what lets it move between projects without an
adapter layer for each one.

The one rule that carries: **whatever the runner returns on a miss, `from_entry`
must return the same type on a hit.** A caller cannot tell the two apart, so a
runner that hands back a raw SDK response while `from_entry` hands back a string
works perfectly until the cache warms up, and then breaks everywhere at once.
The reliable way to satisfy it is to have the runner return a small result type
of your own rather than the provider's object.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

from .cache import CacheDecision, ResponseCache, value_bytes
from .config import TunerConfig, get_config
from .flight import AsyncSingleFlight, SingleFlight
from .keys import call_key
from .layout import PrefixTracker, default_tracker
from .ledger import Ledger, default_ledger


class CallMeta:
    """What the tuner did to this call. Hosts put it on their own telemetry."""

    __slots__ = ("cache", "reason", "deduped", "key", "tokens_saved")

    def __init__(self, cache: str = "miss", reason: str = "", deduped: bool = False,
                 key: str = "", tokens_saved: int = 0) -> None:
        self.cache = cache          # "hit" | "miss" | "skip"
        self.reason = reason
        self.deduped = deduped
        self.key = key
        self.tokens_saved = tokens_saved

    @property
    def hit(self) -> bool:
        return self.cache == "hit"

    def as_dict(self) -> dict:
        return {"cache": self.cache, "reason": self.reason, "deduped": self.deduped,
                "tokens_saved": self.tokens_saved}

    def __repr__(self) -> str:
        return f"<CallMeta cache={self.cache}{'/' + self.reason if self.reason else ''} deduped={self.deduped}>"


class Tuner:
    def __init__(self, config: Optional[TunerConfig] = None,
                 cache: Optional[ResponseCache] = None,
                 ledger: Optional[Ledger] = None,
                 tracker: Optional[PrefixTracker] = None) -> None:
        self.config = config or get_config()
        self.ledger = ledger or default_ledger()
        self.cache = cache or ResponseCache(self.config, ledger=self.ledger)
        # One ledger per tuner. A caller who built the cache separately would
        # otherwise split the counters across two objects.
        self.cache.use_ledger(self.ledger)
        self.tracker = tracker or default_tracker()
        self.flight = SingleFlight(self.ledger, self.config.dedupe_wait_seconds)
        self.aflight = AsyncSingleFlight(self.ledger, self.config.dedupe_wait_seconds)

    # -- keys -----------------------------------------------------------------

    def key_for(self, *, model: str, messages, task: str = "",
                temperature: Optional[float] = None, max_tokens: Optional[int] = None,
                schema: Optional[dict] = None, tools: Optional[list] = None,
                want_json: bool = False, tenant: Optional[str] = None,
                key_extra: Optional[dict] = None) -> str:
        return call_key(
            model=model, messages=messages, task=task, temperature=temperature,
            max_tokens=max_tokens, schema=schema, tools=tools, want_json=want_json,
            tenant=tenant, namespace=self.config.cache_namespace, extra=key_extra,
        )

    def _decide(self, *, cacheable: bool, temperature, stream, tools) -> CacheDecision:
        return self.cache.should_read(task_cacheable=cacheable, temperature=temperature,
                                      stream=stream, tools=tools)

    # -- sync -----------------------------------------------------------------

    def run(
        self,
        run: Callable[[], Any],
        *,
        task: str,
        model: str,
        messages,
        # to_entry: result -> a JSON-safe dict, or None to refuse to cache it.
        # from_entry: that dict -> a result of THE SAME TYPE the runner returns.
        to_entry: Callable[[Any], Optional[dict]],
        from_entry: Callable[[dict], Any],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        schema: Optional[dict] = None,
        tools: Optional[list] = None,
        want_json: bool = False,
        tenant: Optional[str] = None,
        sensitive: bool = True,
        cacheable: bool = True,
        ttl_seconds: Optional[int] = None,
        stream: bool = False,
        # Any further request parameter that changes the answer. Without it, a
        # caller-supplied kwarg the tuner does not know about would produce a
        # different response under the same key.
        key_extra: Optional[dict] = None,
    ) -> Tuple[Any, CallMeta]:
        """Run a call through cache -> de-duplicate -> execute -> store.

        Returns (result, CallMeta). The result is the runner's own return value
        on a miss and `from_entry(entry)` on a hit; those two must be the same
        type - see the module docstring.
        """
        if not self.config.enabled:
            return run(), CallMeta("skip", "tuner_disabled")

        decision = self._decide(cacheable=cacheable, temperature=temperature,
                                stream=stream, tools=tools)
        if not decision:
            self.ledger.skipped(task, decision.reason)
            result = run()
            return result, CallMeta("skip", decision.reason)

        key = self.key_for(model=model, messages=messages, task=task,
                           temperature=temperature, max_tokens=max_tokens,
                           schema=schema, tools=tools, want_json=want_json,
                           tenant=tenant, key_extra=key_extra)

        entry = self.cache.get(key, task=task, model=model)
        if entry is not None:
            saved = int(entry.get("tokens_in") or 0) + int(entry.get("tokens_out") or 0)
            return from_entry(entry), CallMeta("hit", key=key, tokens_saved=saved)

        def _execute():
            got = run()
            self._store(key, got, task=task, model=model, to_entry=to_entry,
                        decision=decision, sensitive=sensitive, ttl_seconds=ttl_seconds)
            return got

        if not self.config.dedupe_enabled:
            return _execute(), CallMeta("miss", key=key)

        result, shared = self.flight.run(key, _execute, task=task, model=model)
        return result, CallMeta("miss", deduped=shared, key=key)

    # -- async ----------------------------------------------------------------

    async def arun(
        self,
        run: Callable[[], Any],
        *,
        task: str,
        model: str,
        messages,
        to_entry: Callable[[Any], Optional[dict]],
        from_entry: Callable[[dict], Any],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        schema: Optional[dict] = None,
        tools: Optional[list] = None,
        want_json: bool = False,
        tenant: Optional[str] = None,
        sensitive: bool = True,
        cacheable: bool = True,
        ttl_seconds: Optional[int] = None,
        stream: bool = False,
        # Any further request parameter that changes the answer. Without it, a
        # caller-supplied kwarg the tuner does not know about would produce a
        # different response under the same key.
        key_extra: Optional[dict] = None,
    ) -> Tuple[Any, CallMeta]:
        if not self.config.enabled:
            return await run(), CallMeta("skip", "tuner_disabled")

        decision = self._decide(cacheable=cacheable, temperature=temperature,
                                stream=stream, tools=tools)
        if not decision:
            self.ledger.skipped(task, decision.reason)
            return await run(), CallMeta("skip", decision.reason)

        key = self.key_for(model=model, messages=messages, task=task,
                           temperature=temperature, max_tokens=max_tokens,
                           schema=schema, tools=tools, want_json=want_json,
                           tenant=tenant, key_extra=key_extra)

        entry = self.cache.get(key, task=task, model=model)
        if entry is not None:
            saved = int(entry.get("tokens_in") or 0) + int(entry.get("tokens_out") or 0)
            return from_entry(entry), CallMeta("hit", key=key, tokens_saved=saved)

        async def _execute():
            got = await run()
            self._store(key, got, task=task, model=model, to_entry=to_entry,
                        decision=decision, sensitive=sensitive, ttl_seconds=ttl_seconds)
            return got

        if not self.config.dedupe_enabled:
            return await _execute(), CallMeta("miss", key=key)

        result, shared = await self.aflight.run(key, _execute, task=task, model=model)
        return result, CallMeta("miss", deduped=shared, key=key)

    # -- internals ------------------------------------------------------------

    def _store(self, key: str, result: Any, *, task: str, model: str,
               to_entry: Callable[[Any], Optional[dict]], decision: CacheDecision,
               sensitive: bool, ttl_seconds: Optional[int]) -> None:
        try:
            entry = to_entry(result)
        except Exception:  # noqa: BLE001
            entry = None
        if not entry:
            # `to_entry` returning None is how a caller says "this answer is not
            # worth keeping" - a failure, or an escalated-away result.
            self.ledger.skipped(task, "not_storable")
            return

        write = self.cache.should_write(decision=decision, ok=True, sensitive=sensitive,
                                        value_bytes=value_bytes(entry))
        if not write:
            self.ledger.skipped(task, write.reason)
            return
        self.cache.put(key, entry, task=task, model=model, ttl_seconds=ttl_seconds)

    # -- reporting ------------------------------------------------------------

    def stats(self) -> dict:
        return {
            "config": {
                "enabled": self.config.enabled,
                "cache_enabled": self.config.cache_enabled,
                "cache_store": self.config.cache_store,
                "cache_ttl_seconds": self.config.cache_ttl_seconds,
                "cache_max_temperature": self.config.cache_max_temperature,
                "cache_persist_sensitive": self.config.cache_persist_sensitive,
                "dedupe_enabled": self.config.dedupe_enabled,
                "exact_counting": _exact(),
            },
            "cache": self.cache.stats(),
            "in_flight": {"sync": self.flight.in_flight(), "async": self.aflight.in_flight()},
            "ledger": self.ledger.snapshot(),
            "prefixes": self.tracker.summary(),
        }


def _exact() -> bool:
    from .counting import exact_counting_available

    return exact_counting_available()


_default: Optional[Tuner] = None


def default_tuner() -> Tuner:
    global _default
    if _default is None:
        _default = Tuner()
    return _default


def set_default_tuner(tuner: Optional[Tuner]) -> None:
    """Install a tuner built from the host's own settings, or reset for tests."""
    global _default
    _default = tuner
