"""
In-flight de-duplication ("single flight").

The cache handles a request that repeats *after* an earlier one finished. This
handles the harder case: N identical requests in the air at the same time,
where the cache is empty for all of them because none has returned yet.

That is the normal shape of a fan-out. Classifying a page of records against
one rubric fires many calls at once, several of which are byte-identical;
without this, every one of them is paid for.

Sync and async are separate implementations rather than one clever one. The
sync path parks threads on an Event; the async path awaits a shared Future.
Bridging those two through a single abstraction would mean either blocking an
event loop or spinning up threads under async code, and both are worse than
eighty lines of duplication.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Callable, Dict, Optional, Tuple

from .ledger import Ledger, default_ledger


class _Slot:
    __slots__ = ("event", "result", "error", "waiters")

    def __init__(self) -> None:
        self.event = threading.Event()
        self.result: Any = None
        self.error: Optional[BaseException] = None
        self.waiters = 0


class SingleFlight:
    """Collapses concurrent identical calls into one execution."""

    def __init__(self, ledger: Optional[Ledger] = None,
                 wait_seconds: float = 120.0) -> None:
        self._lock = threading.Lock()
        self._slots: Dict[str, _Slot] = {}
        self._ledger = ledger or default_ledger()
        self._wait = wait_seconds

    def run(self, key: str, fn: Callable[[], Any], *, task: str = "",
            model: str = "", tokens_hint: int = 0) -> Tuple[Any, bool]:
        """
        Execute `fn`, or wait for the identical call already running.

        Returns (result, was_shared). `was_shared` is True for the callers that
        piggybacked, which is what makes the saving countable.
        """
        with self._lock:
            slot = self._slots.get(key)
            if slot is None:
                slot = _Slot()
                self._slots[key] = slot
                leader = True
            else:
                slot.waiters += 1
                leader = False

        if leader:
            try:
                slot.result = fn()
            except BaseException as e:  # noqa: BLE001 - re-raised below
                slot.error = e
            finally:
                with self._lock:
                    self._slots.pop(key, None)
                slot.event.set()
            if slot.error is not None:
                raise slot.error
            return slot.result, False

        if not slot.event.wait(timeout=self._wait):
            # The leader is wedged. Running it ourselves costs a duplicate call,
            # which is strictly better than inheriting someone else's hang.
            return fn(), False
        if slot.error is not None:
            # Followers do not inherit the leader's failure: a transient error
            # for one caller should not be multiplied across all of them.
            return fn(), False

        self._ledger.deduped(task, model, tokens_hint)
        return slot.result, True

    def in_flight(self) -> int:
        with self._lock:
            return len(self._slots)


class AsyncSingleFlight:
    """The same idea on an event loop.

    Futures are created per running loop. A process with more than one loop
    (a worker thread running its own) would otherwise await a future belonging
    to a loop that will never run it.
    """

    def __init__(self, ledger: Optional[Ledger] = None,
                 wait_seconds: float = 120.0) -> None:
        self._futures: Dict[Tuple[int, str], "asyncio.Future"] = {}
        self._ledger = ledger or default_ledger()
        self._wait = wait_seconds

    def _loop_id(self) -> int:
        try:
            return id(asyncio.get_running_loop())
        except RuntimeError:
            return 0

    async def run(self, key: str, fn: Callable[[], Any], *, task: str = "",
                  model: str = "", tokens_hint: int = 0) -> Tuple[Any, bool]:
        slot_key = (self._loop_id(), key)
        existing = self._futures.get(slot_key)

        if existing is not None:
            try:
                result = await asyncio.wait_for(asyncio.shield(existing), timeout=self._wait)
            except Exception:  # noqa: BLE001 - leader failed, timed out, or was cancelled
                return await fn(), False
            self._ledger.deduped(task, model, tokens_hint)
            return result, True

        loop = asyncio.get_running_loop()
        future: "asyncio.Future" = loop.create_future()
        self._futures[slot_key] = future
        try:
            result = await fn()
        except BaseException as e:  # noqa: BLE001 - re-raised below
            if not future.done():
                future.set_exception(e)
            # Nobody may be awaiting this future; without this the loop logs a
            # "Future exception was never retrieved" warning on collection.
            future.exception()
            raise
        else:
            if not future.done():
                future.set_result(result)
            return result, False
        finally:
            self._futures.pop(slot_key, None)

    def in_flight(self) -> int:
        return len(self._futures)
