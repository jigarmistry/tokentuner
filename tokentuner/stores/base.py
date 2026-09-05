"""The store protocol, and the backend that does nothing."""

from __future__ import annotations

from typing import Any, Optional


class Store:
    """
    Minimal key/value contract.

    `persistent` is not decoration: the cache layer refuses to write responses
    to prompts that can carry personal data into a store that outlives the
    process, unless that has been explicitly switched on.
    """

    persistent: bool = False
    name: str = "base"

    def get(self, key: str) -> Optional[Any]:
        raise NotImplementedError

    def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        raise NotImplementedError

    def delete(self, key: str) -> None:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError

    def stats(self) -> dict:
        return {"store": self.name, "persistent": self.persistent}


class NullStore(Store):
    """Accepts everything, remembers nothing. Used to disable caching without
    threading a conditional through every call site."""

    persistent = False
    name = "null"

    def get(self, key: str) -> Optional[Any]:
        return None

    def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        return None

    def delete(self, key: str) -> None:
        return None

    def clear(self) -> None:
        return None
