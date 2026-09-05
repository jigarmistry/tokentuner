"""
Token counting.

Two implementations behind one function. If `tiktoken` is importable you get
exact counts for OpenAI-family encodings; otherwise you get an estimator.

The estimator deliberately over-counts. Every consumer of this module uses the
number to decide how much to *keep*, so an under-count silently overflows a
context window and a request fails at the provider, while an over-count costs a
few tokens of unused headroom. Those are not symmetric mistakes.

Non-OpenAI tokenisers (Llama, Mistral, Claude) differ from cl100k, typically by
5-20% on English prose. That is well inside the headroom the budgeter leaves,
and it is the reason this module never claims a count is authoritative.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Iterable, Optional

from .config import get_config

# Per-message framing in the OpenAI chat format: role, separators, and the
# priming tokens the server adds before the reply.
_PER_MESSAGE_OVERHEAD = 4
_REPLY_PRIMING = 3

_WORD = re.compile(r"\w+|[^\w\s]")

_ENCODERS: dict = {}
_TIKTOKEN: Any = None
_TIKTOKEN_TRIED = False


def _tiktoken():
    global _TIKTOKEN, _TIKTOKEN_TRIED
    if _TIKTOKEN_TRIED:
        return _TIKTOKEN
    _TIKTOKEN_TRIED = True
    try:
        import tiktoken  # type: ignore

        _TIKTOKEN = tiktoken
    except Exception:  # noqa: BLE001 - absence is a supported configuration
        _TIKTOKEN = None
    return _TIKTOKEN


def exact_counting_available() -> bool:
    return _tiktoken() is not None and get_config().counting_exact


def _encoding_name(model: Optional[str]) -> str:
    """
    Map a model id to an encoding.

    Routed ids arrive namespaced (`vendor/model`); those models do not use an
    OpenAI encoding at all, so the choice here is only ever an approximation
    for them. o200k is the closer of the two for modern tokenisers.
    """
    m = (model or "").lower()
    if "/" in m or "gpt-4o" in m or "gpt-5" in m or m.startswith("o1") or m.startswith("o3"):
        return "o200k_base"
    return "cl100k_base"


def _encoder(model: Optional[str]):
    tk = _tiktoken()
    if tk is None:
        return None
    name = _encoding_name(model)
    if name in _ENCODERS:
        # None is cached deliberately. tiktoken fetches its BPE file over the
        # network on first use, and in a sandboxed container that fails every
        # time - retrying it once per token count would turn a graceful
        # fallback into a request-latency problem.
        return _ENCODERS[name]
    try:
        _ENCODERS[name] = tk.get_encoding(name)
    except Exception as e:  # noqa: BLE001
        print(f"[tokentuner] encoding {name} unavailable ({e}); using the estimator")
        _ENCODERS[name] = None
    return _ENCODERS[name]


def estimate_tokens(text: str) -> int:
    """
    Encoding-free estimate, biased high.

    Two signals, take the larger. Characters/3.6 tracks prose and beats the
    familiar chars/4 rule of thumb on the JSON and code that dominate tool
    results. Word-and-punctuation count times 1.15 tracks text with long runs
    of whitespace, where a character count collapses to nothing.
    """
    if not text:
        return 0
    by_chars = len(text) / 3.6
    by_words = len(_WORD.findall(text)) * 1.15
    return int(math.ceil(max(by_chars, by_words)))


def count_text(text: Optional[str], model: Optional[str] = None) -> int:
    if not text:
        return 0
    if get_config().counting_exact:
        enc = _encoder(model)
        if enc is not None:
            try:
                return len(enc.encode(text, disallowed_special=()))
            except Exception:  # noqa: BLE001
                pass
    return estimate_tokens(text)


def count_value(value: Any, model: Optional[str] = None) -> int:
    """Count anything: a string, or a structure serialised the way it would be
    sent (compactly, which is how the minifier will send it)."""
    if value is None:
        return 0
    if isinstance(value, str):
        return count_text(value, model)
    try:
        return count_text(json.dumps(value, separators=(",", ":"), default=str), model)
    except Exception:  # noqa: BLE001
        return count_text(str(value), model)


def count_message(message: dict, model: Optional[str] = None) -> int:
    """One chat message, including its framing.

    Handles the three content shapes in circulation: a plain string, the
    multipart content array, and a tool-call turn whose payload lives in
    `tool_calls` with `content` set to None.
    """
    total = _PER_MESSAGE_OVERHEAD
    content = message.get("content") if isinstance(message, dict) else None

    if isinstance(content, str):
        total += count_text(content, model)
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    total += count_text(part.get("text"), model)
                else:
                    # Images and files are not text; their cost is provider
                    # specific and not knowable from here.
                    total += 0
            else:
                total += count_value(part, model)
    elif content is not None:
        total += count_value(content, model)

    if isinstance(message, dict):
        for key in ("name", "tool_call_id"):
            if message.get(key):
                total += count_text(str(message[key]), model)
        if message.get("tool_calls"):
            total += count_value(message["tool_calls"], model)
    return total


def _as_dict(message: Any) -> dict:
    """SDK message objects turn up in histories alongside plain dicts."""
    if isinstance(message, dict):
        return message
    out = {}
    for key in ("role", "content", "name", "tool_call_id"):
        value = getattr(message, key, None)
        if value is not None:
            out[key] = value
    calls = getattr(message, "tool_calls", None)
    if calls:
        out["tool_calls"] = [str(c) for c in calls]
    return out


def count_messages(messages: Iterable, model: Optional[str] = None) -> int:
    total = _REPLY_PRIMING
    for message in messages or []:
        total += count_message(_as_dict(message), model)
    return total


def count_tools(tools: Optional[list], model: Optional[str] = None) -> int:
    """Tool definitions are billed as input on every call that carries them,
    which is easy to forget because they never appear in the message list."""
    if not tools:
        return 0
    return count_value(tools, model)


def count_request(messages: Iterable, tools: Optional[list] = None,
                  model: Optional[str] = None) -> int:
    return count_messages(messages, model) + count_tools(tools, model)


def truncate_to_tokens(text: str, max_tokens: int, model: Optional[str] = None,
                       suffix: str = "") -> str:
    """
    Cut text to a token budget, on a token boundary where possible.

    Falls back to a proportional character cut walked down until it fits, which
    terminates because each pass removes at least one character.
    """
    if max_tokens <= 0 or not text:
        return ""
    if count_text(text, model) <= max_tokens:
        return text

    budget = max_tokens - count_text(suffix, model) if suffix else max_tokens
    if budget <= 0:
        return suffix[:0]

    if get_config().counting_exact:
        enc = _encoder(model)
        if enc is not None:
            try:
                return enc.decode(enc.encode(text, disallowed_special=())[:budget]) + suffix
            except Exception:  # noqa: BLE001
                pass

    cut = text
    while cut and count_text(cut, model) > budget:
        ratio = budget / max(count_text(cut, model), 1)
        nxt = int(len(cut) * min(ratio, 0.98))
        cut = cut[: max(nxt, 0)]
    return cut + suffix
