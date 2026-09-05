"""
Cache backends.

One tiny protocol, four implementations. The protocol is deliberately smaller
than any real cache client's API so that adding a fifth backend is an afternoon
rather than a project, and so the cache layer above can never depend on a
capability only one backend happens to have.

Every backend is expected to fail soft. A cache that raises when the database
is unreachable has converted an optimisation into an outage.
"""

from __future__ import annotations

from .base import NullStore, Store
from .memory import MemoryStore

__all__ = ["Store", "NullStore", "MemoryStore", "build_store"]


def build_store(config):
    """Construct the backend named in config, falling back to memory.

    Fallback rather than failure is the right call: a misconfigured cache
    should cost cache hits, not availability.
    """
    kind = (config.cache_store or "memory").strip().lower()

    if kind in ("none", "off", "null"):
        return NullStore()
    if kind == "memory":
        return MemoryStore(max_entries=config.cache_max_entries)

    try:
        if kind == "mongo":
            from .mongo import MongoStore

            return MongoStore(
                uri=config.mongo_uri,
                database=config.mongo_database,
                collection=config.mongo_collection,
            )
        if kind == "redis":
            from .redis_store import RedisStore

            return RedisStore(url=config.redis_url, namespace=config.cache_namespace)
        if kind == "sqlite":
            from .sqlite_store import SQLiteStore

            return SQLiteStore(path=config.sqlite_path, max_entries=config.cache_max_entries)
    except Exception as e:  # noqa: BLE001
        print(f"[tokentuner] cache store {kind!r} unavailable ({e}); using memory")
        return MemoryStore(max_entries=config.cache_max_entries)

    print(f"[tokentuner] unknown cache store {kind!r}; using memory")
    return MemoryStore(max_entries=config.cache_max_entries)
