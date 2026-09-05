"""
Configuration.

One frozen dataclass, readable from the environment, with defaults chosen so
that importing this package and doing nothing else is safe: the cache is in
memory only, nothing is written to disk or to a database, and no behaviour
changes until a caller opts in.

The defaults lean conservative on purpose. A tuner that silently persists model
output containing personal data, or that serves a stale answer to a call the
author assumed was fresh, is worse than no tuner at all.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Optional

ENV_PREFIX = "TOKENTUNER_"


def _env(name: str, default: str = "") -> str:
    return os.getenv(ENV_PREFIX + name, default)


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, "") or default)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, "") or default)
    except ValueError:
        return default


@dataclass(frozen=True)
class TunerConfig:
    # ---- master switch -----------------------------------------------------
    # Off flips every optimisation into a pass-through. Keep it reachable from
    # config: when an answer looks wrong in production, "is it the cache?" must
    # be answerable in one restart rather than one deploy.
    enabled: bool = True

    # ---- response cache ----------------------------------------------------
    cache_enabled: bool = True
    # "memory" | "mongo" | "redis" | "sqlite" | "none"
    cache_store: str = "memory"
    cache_ttl_seconds: int = 3600
    cache_max_entries: int = 2000
    # Skip anything larger than this; one enormous entry should not evict the
    # thousand small ones that were actually earning their keep.
    cache_max_value_bytes: int = 256_000
    # Above this temperature the caller asked for variety, and handing back a
    # previous answer quietly takes it away.
    cache_max_temperature: float = 0.35
    # Responses to prompts that can carry personal data. Off means such calls
    # are still cached in process memory (which dies with the worker) but never
    # written to a persistent store.
    cache_persist_sensitive: bool = False
    cache_namespace: str = "tt"

    # ---- in-flight de-duplication -----------------------------------------
    dedupe_enabled: bool = True
    dedupe_wait_seconds: float = 120.0

    # ---- prompt shaping ----------------------------------------------------
    layout_enabled: bool = True
    # Below this, a provider will not cache the prefix, so the marker is noise.
    # 1024 is OpenAI's documented automatic-caching threshold.
    layout_min_prefix_tokens: int = 1024

    budget_enabled: bool = True
    minify_enabled: bool = True

    # ---- token counting ----------------------------------------------------
    # Use tiktoken when importable. The fallback estimator over-counts slightly
    # so that budgets under-fill rather than overflow a context window.
    counting_exact: bool = True

    # ---- persistent store connection --------------------------------------
    mongo_uri: str = ""
    mongo_database: str = ""
    mongo_collection: str = "tokentuner_cache"
    redis_url: str = ""
    sqlite_path: str = ""

    def with_(self, **kw) -> "TunerConfig":
        return replace(self, **kw)


def from_env() -> TunerConfig:
    d = TunerConfig()
    return TunerConfig(
        enabled=_env_bool("ENABLED", d.enabled),
        cache_enabled=_env_bool("CACHE_ENABLED", d.cache_enabled),
        cache_store=_env("CACHE_STORE", d.cache_store).strip().lower() or d.cache_store,
        cache_ttl_seconds=_env_int("CACHE_TTL_SECONDS", d.cache_ttl_seconds),
        cache_max_entries=_env_int("CACHE_MAX_ENTRIES", d.cache_max_entries),
        cache_max_value_bytes=_env_int("CACHE_MAX_VALUE_BYTES", d.cache_max_value_bytes),
        cache_max_temperature=_env_float("CACHE_MAX_TEMPERATURE", d.cache_max_temperature),
        cache_persist_sensitive=_env_bool("CACHE_PERSIST_SENSITIVE", d.cache_persist_sensitive),
        cache_namespace=_env("CACHE_NAMESPACE", d.cache_namespace),
        dedupe_enabled=_env_bool("DEDUPE_ENABLED", d.dedupe_enabled),
        dedupe_wait_seconds=_env_float("DEDUPE_WAIT_SECONDS", d.dedupe_wait_seconds),
        layout_enabled=_env_bool("LAYOUT_ENABLED", d.layout_enabled),
        layout_min_prefix_tokens=_env_int("LAYOUT_MIN_PREFIX_TOKENS", d.layout_min_prefix_tokens),
        budget_enabled=_env_bool("BUDGET_ENABLED", d.budget_enabled),
        minify_enabled=_env_bool("MINIFY_ENABLED", d.minify_enabled),
        counting_exact=_env_bool("COUNTING_EXACT", d.counting_exact),
        mongo_uri=_env("MONGO_URI", d.mongo_uri),
        mongo_database=_env("MONGO_DATABASE", d.mongo_database),
        mongo_collection=_env("MONGO_COLLECTION", d.mongo_collection),
        redis_url=_env("REDIS_URL", d.redis_url),
        sqlite_path=_env("SQLITE_PATH", d.sqlite_path),
    )


_config: Optional[TunerConfig] = None


def get_config() -> TunerConfig:
    global _config
    if _config is None:
        _config = from_env()
    return _config


def configure(config: Optional[TunerConfig] = None, **kw) -> TunerConfig:
    """Install a config. Host applications call this once at startup with
    values from their own settings object rather than the environment."""
    global _config
    base = config if config is not None else get_config()
    _config = base.with_(**kw) if kw else base
    return _config
