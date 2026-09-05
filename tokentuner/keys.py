"""
Cache keys.

A cache key has one job: two calls share a key exactly when serving the second
one the first one's answer is correct. Everything here follows from that.

  * The key covers every input that can change the answer - model, messages,
    temperature, schema, tool definitions. Leave one out and you serve the
    wrong answer to a call that differs in precisely that way.

  * The key is a digest, never the prompt. Cache keys end up in logs, metrics
    labels and error messages; a prompt in this system carries CVs and phone
    numbers, and those must not be the thing that leaks through the telemetry.

  * The key is namespaced by tenant. Cross-tenant reuse of a cached answer is
    a data leak between customers, and it is the kind that looks like a cache
    hit rather than like a bug.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Optional

# Bump when the key composition changes, so a deploy cannot serve entries built
# under different rules.
KEY_VERSION = "1"


def _normalize_content(content: Any) -> Any:
    if isinstance(content, str):
        # Leading and trailing whitespace never changes an answer, and prompt
        # builders produce inconsistent amounts of it.
        return content.strip()
    if isinstance(content, list):
        return [_normalize_content(part) for part in content]
    if isinstance(content, dict):
        return {k: _normalize_content(content[k]) for k in sorted(content)}
    return content


def _normalize_message(message: Any) -> dict:
    if not isinstance(message, dict):
        message = {
            "role": getattr(message, "role", None),
            "content": getattr(message, "content", None),
            "tool_calls": [str(c) for c in (getattr(message, "tool_calls", None) or [])] or None,
        }
    out = {}
    for key in ("role", "content", "name", "tool_call_id", "tool_calls"):
        if message.get(key) not in (None, [], ""):
            out[key] = _normalize_content(message[key])
    return out


def normalize_messages(messages: Iterable) -> list:
    return [_normalize_message(m) for m in messages or []]


def _stable(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def call_key(
    *,
    model: str,
    messages: Iterable,
    task: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    schema: Optional[dict] = None,
    tools: Optional[list] = None,
    want_json: bool = False,
    namespace: str = "tt",
    tenant: Optional[str] = None,
    extra: Optional[dict] = None,
) -> str:
    """A hex digest identifying this exact call. Same inputs, same key."""
    payload = {
        "v": KEY_VERSION,
        "model": model,
        "task": task,
        # Rounded because 0.30000000000000004 and 0.3 are the same request, and
        # float formatting differs across the paths that build these.
        "temp": None if temperature is None else round(float(temperature), 4),
        "max_tokens": max_tokens,
        "json": bool(want_json),
        "schema": schema,
        "tools": tools,
        "messages": normalize_messages(messages),
        "extra": extra or None,
    }
    digest = hashlib.blake2b(_stable(payload).encode("utf-8"), digest_size=20).hexdigest()
    scope = tenant or "_"
    # The tenant is hashed too: a key travels into logs and metric labels, and
    # a raw customer id there is an identifier we did not mean to publish.
    scope_digest = hashlib.blake2b(str(scope).encode("utf-8"), digest_size=6).hexdigest()
    return f"{namespace}:{KEY_VERSION}:{scope_digest}:{digest}"


def prefix_key(*, model: str, messages: Iterable, tools: Optional[list] = None,
               namespace: str = "tt") -> str:
    """Identity of a static prompt prefix, for measuring how often a prefix is
    actually reused. Carries no tenant scope - a prefix is by definition the
    part that contains no tenant data."""
    payload = {"v": KEY_VERSION, "model": model,
               "messages": normalize_messages(messages), "tools": tools}
    return f"{namespace}:p:{hashlib.blake2b(_stable(payload).encode('utf-8'), digest_size=12).hexdigest()}"


def content_key(text: str) -> str:
    """Identity of a chunk of text, used to collapse duplicate embedding
    inputs before they turn into duplicate paid requests."""
    return hashlib.blake2b((text or "").strip().encode("utf-8"), digest_size=16).hexdigest()
