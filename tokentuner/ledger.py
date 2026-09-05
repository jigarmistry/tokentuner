"""
The savings ledger.

An optimisation you cannot measure is an optimisation you cannot defend. Every
component in this package reports what it did here, so the question "was any of
this worth it" has an answer made of counters rather than opinions.

Two rules the numbers follow:

  * Saved tokens are *avoided* tokens - what the call would have cost had the
    optimisation not run. For a cache hit that is the whole request; for a trim
    it is what was cut. They are estimates, and are named that way.

  * A cache hit is recorded as a cache hit and never as a successful model
    call. Conflating them corrupts every downstream rate - escalation rate,
    failure rate, latency percentiles - by mixing in calls that never happened.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional


@dataclass
class Event:
    kind: str
    task: str = ""
    model: str = ""
    tokens_saved: int = 0
    calls_saved: int = 0
    latency_saved_ms: int = 0
    detail: dict = field(default_factory=dict)


# Every kind the package emits. Listed explicitly so a typo in a call site
# shows up as an unknown kind rather than as a silently new counter.
KINDS = (
    "cache_hit",
    "cache_miss",
    "cache_store",
    "cache_skip",
    "dedupe_hit",
    "trim",
    "minify",
    "prefix",
    "batch",
    "output_cap",
)


class Ledger:
    def __init__(self, sink: Optional[Callable[[Event], None]] = None) -> None:
        self._lock = threading.Lock()
        self._counts: Dict[str, int] = defaultdict(int)
        self._tokens: Dict[str, int] = defaultdict(int)
        self._calls: Dict[str, int] = defaultdict(int)
        self._latency: Dict[str, int] = defaultdict(int)
        self._by_task: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        # Where a host application hooks its own metrics or usage table in.
        self._sink = sink

    def set_sink(self, sink: Optional[Callable[[Event], None]]) -> None:
        self._sink = sink

    def record(self, event: Event) -> None:
        """Never raises. Accounting must not be able to break a request."""
        try:
            with self._lock:
                self._counts[event.kind] += 1
                self._tokens[event.kind] += int(event.tokens_saved or 0)
                self._calls[event.kind] += int(event.calls_saved or 0)
                self._latency[event.kind] += int(event.latency_saved_ms or 0)
                if event.task:
                    self._by_task[event.task][event.kind] += 1
                    if event.tokens_saved:
                        self._by_task[event.task]["tokens_saved"] += int(event.tokens_saved)
            if self._sink is not None:
                self._sink(event)
        except Exception:  # noqa: BLE001
            pass

    # -- convenience emitters -------------------------------------------------

    def hit(self, task: str, model: str, tokens_saved: int, latency_saved_ms: int = 0) -> None:
        self.record(Event("cache_hit", task, model, tokens_saved=tokens_saved,
                          calls_saved=1, latency_saved_ms=latency_saved_ms))

    def miss(self, task: str, model: str = "") -> None:
        self.record(Event("cache_miss", task, model))

    def stored(self, task: str, model: str = "") -> None:
        self.record(Event("cache_store", task, model))

    def skipped(self, task: str, reason: str) -> None:
        self.record(Event("cache_skip", task, detail={"reason": reason}))

    def deduped(self, task: str, model: str, tokens_saved: int) -> None:
        self.record(Event("dedupe_hit", task, model, tokens_saved=tokens_saved, calls_saved=1))

    def trimmed(self, task: str, tokens_saved: int, section: str = "") -> None:
        if tokens_saved > 0:
            self.record(Event("trim", task, tokens_saved=tokens_saved,
                              detail={"section": section}))

    def minified(self, task: str, tokens_saved: int) -> None:
        if tokens_saved > 0:
            self.record(Event("minify", task, tokens_saved=tokens_saved))

    def prefix(self, task: str, model: str, prefix_tokens: int, cacheable: bool) -> None:
        self.record(Event("prefix", task, model,
                          detail={"prefix_tokens": prefix_tokens, "cacheable": cacheable}))

    def batched(self, task: str, model: str, calls_saved: int, tokens_saved: int) -> None:
        self.record(Event("batch", task, model, calls_saved=calls_saved,
                          tokens_saved=tokens_saved))

    def output_capped(self, task: str, cap: int) -> None:
        self.record(Event("output_cap", task, detail={"max_tokens": cap}))

    # -- reporting ------------------------------------------------------------

    def snapshot(self) -> dict:
        with self._lock:
            counts = dict(self._counts)
            tokens = dict(self._tokens)
            calls = dict(self._calls)
            latency = dict(self._latency)
            by_task = {t: dict(v) for t, v in self._by_task.items()}

        looked_up = counts.get("cache_hit", 0) + counts.get("cache_miss", 0)
        return {
            "counts": {k: counts.get(k, 0) for k in KINDS},
            "cache": {
                "hits": counts.get("cache_hit", 0),
                "misses": counts.get("cache_miss", 0),
                "hit_rate": round(counts.get("cache_hit", 0) / looked_up, 4) if looked_up else 0.0,
                "stores": counts.get("cache_store", 0),
                "skips": counts.get("cache_skip", 0),
            },
            "saved": {
                "calls": sum(calls.values()),
                "tokens_est": sum(tokens.values()),
                "latency_ms_est": sum(latency.values()),
                "by_kind_tokens": {k: tokens.get(k, 0) for k in KINDS if tokens.get(k)},
            },
            "by_task": by_task,
        }

    def reset(self) -> None:
        with self._lock:
            self._counts.clear()
            self._tokens.clear()
            self._calls.clear()
            self._latency.clear()
            self._by_task.clear()


_default = Ledger()


def default_ledger() -> Ledger:
    return _default
