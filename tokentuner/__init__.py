"""
tokentuner - cut tokens, cost and latency on LLM calls without changing what
the model is asked.

Nothing here decides which model to use or judges whether an answer is good.
Those are routing and evaluation, they are specific to an application's domain,
and mixing them in is what makes this kind of layer unportable. This package
does one thing: given a call somebody has already decided to make, spend as
little as possible making it.

    from tokentuner import default_tuner, PromptLayout, Budget, minify_rows

    layout = PromptLayout(model=model_id, task="classify")
    layout.static("system", INSTRUCTIONS)
    layout.volatile("user", document_text)

    result, meta = default_tuner().run(
        lambda: client.chat.completions.create(...),
        task="score", model=model_id, messages=layout.messages(),
        to_entry=..., from_entry=...,
    )

Every component also works alone; the facade is a convenience, not a framework.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .batch import (
    EmbeddingBatcher,
    batch_savings,
    chunk,
    index_items,
    split_results,
)
from .budget import (
    Budget,
    Section,
    recommend_max_tokens,
    trim_history,
    trim_text,
)
from .cache import CacheDecision, ResponseCache
from .config import TunerConfig, configure, from_env, get_config
from .counting import (
    count_messages,
    count_request,
    count_text,
    count_tools,
    count_value,
    estimate_tokens,
    exact_counting_available,
    truncate_to_tokens,
)
from .flight import AsyncSingleFlight, SingleFlight
from .keys import call_key, content_key, prefix_key
from .layout import PrefixTracker, PromptLayout, cache_style_for, default_tracker
from .ledger import Event, Ledger, default_ledger
from .minify import (
    compact_json,
    dedupe_rows,
    drop_empty,
    minify_document,
    minify_rows,
    squeeze_text,
    to_kv_lines,
)
from .stores import MemoryStore, NullStore, Store, build_store
from .tuner import CallMeta, Tuner, default_tuner, set_default_tuner

__all__ = [
    "AsyncSingleFlight",
    "Budget",
    "CacheDecision",
    "CallMeta",
    "EmbeddingBatcher",
    "Event",
    "Ledger",
    "MemoryStore",
    "NullStore",
    "PrefixTracker",
    "PromptLayout",
    "ResponseCache",
    "Section",
    "SingleFlight",
    "Store",
    "Tuner",
    "TunerConfig",
    "__version__",
    "batch_savings",
    "build_store",
    "cache_style_for",
    "call_key",
    "chunk",
    "compact_json",
    "configure",
    "content_key",
    "count_messages",
    "count_request",
    "count_text",
    "count_tools",
    "count_value",
    "dedupe_rows",
    "default_ledger",
    "default_tracker",
    "default_tuner",
    "drop_empty",
    "estimate_tokens",
    "exact_counting_available",
    "from_env",
    "get_config",
    "index_items",
    "minify_document",
    "minify_rows",
    "prefix_key",
    "recommend_max_tokens",
    "set_default_tuner",
    "split_results",
    "squeeze_text",
    "to_kv_lines",
    "trim_history",
    "trim_text",
    "truncate_to_tokens",
]
