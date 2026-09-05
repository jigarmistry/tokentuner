"""
MongoDB backend.

Uses the synchronous driver on purpose. Most call sites that benefit from a
response cache in a typical service are ordinary blocking functions buried in
business logic; handing them an awaitable would mean rewriting them, which is
exactly the sort of churn this package exists to avoid.

Expiry is Mongo's job via a TTL index, with a local check as well, because the
TTL monitor only runs every 60 seconds and a stale answer served in that window
is still a stale answer.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from .base import Store


class MongoStore(Store):
    persistent = True
    name = "mongo"

    def __init__(self, uri: str, database: str, collection: str = "tokentuner_cache",
                 client: Any = None) -> None:
        if client is not None:
            self._client = client
        else:
            if not uri or not database:
                raise ValueError("MongoStore needs a uri and a database name")
            from pymongo import MongoClient  # imported lazily: optional extra

            self._client = MongoClient(uri, serverSelectionTimeoutMS=2000)
        self._col = self._client[database][collection]
        self._ensure_index()

    def _ensure_index(self) -> None:
        try:
            self._col.create_index("expires_at", expireAfterSeconds=0)
        except Exception as e:  # noqa: BLE001
            print(f"[tokentuner] could not create cache TTL index: {e}")

    def get(self, key: str) -> Optional[Any]:
        try:
            doc = self._col.find_one({"_id": key})
        except Exception:  # noqa: BLE001 - an unreachable cache is a miss
            return None
        if not doc:
            return None
        expires_at = doc.get("expires_at")
        if expires_at is not None and expires_at.timestamp() <= time.time():
            return None
        return doc.get("value")

    def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        from datetime import datetime, timedelta

        doc = {"value": value, "created_at": datetime.utcnow()}
        if ttl_seconds > 0:
            doc["expires_at"] = datetime.utcnow() + timedelta(seconds=ttl_seconds)
        else:
            # No expiry. The TTL index ignores documents without the field, so
            # omitting it is how "keep this" is spelled.
            doc["expires_at"] = None
        try:
            self._col.update_one({"_id": key}, {"$set": doc}, upsert=True)
        except Exception:  # noqa: BLE001
            pass

    def delete(self, key: str) -> None:
        try:
            self._col.delete_one({"_id": key})
        except Exception:  # noqa: BLE001
            pass

    def clear(self) -> None:
        try:
            self._col.delete_many({})
        except Exception:  # noqa: BLE001
            pass

    def stats(self) -> dict:
        try:
            entries = self._col.estimated_document_count()
        except Exception:  # noqa: BLE001
            entries = -1
        return {"store": self.name, "persistent": True, "entries": entries}
