"""
Prompt layout: getting the stable half of a prompt to the front.

Providers discount input tokens that repeat the *beginning* of a request they
have seen recently. OpenAI does it automatically once the shared prefix passes
1024 tokens; Anthropic-family models do it when the request marks where the
reusable part ends. Both need the same thing from us: everything that does not
change goes first, everything that changes goes last.

Prompts written by hand almost never satisfy that, and not for bad reasons -
the natural way to write one is to state the case and then ask the question, so
the volatile document lands above the fixed instructions. The result is a
prompt whose first token differs on every call, and a discount that never
applies. Reordering costs nothing and changes no meaning.

`PromptLayout` is for prompts you are writing or rewriting. `PrefixTracker` is
for the ones you are not: it watches real traffic and reports how much stable
prefix each task actually has, which is how you find out which prompts are
worth rewriting before touching any of them.
"""

from __future__ import annotations

import threading
from typing import Dict, List, Optional, Tuple

from .config import get_config
from .counting import count_messages, count_text
from .ledger import Ledger, default_ledger

# How to mark the end of the reusable prefix for a given family.
#   "none"      - the provider caches automatically; ordering is the whole job
#   "ephemeral" - Anthropic-style cache_control on the last static block
CACHE_STYLES = ("none", "ephemeral")


def cache_style_for(model: Optional[str]) -> str:
    """
    Pick a marker style from a model id.

    Only Anthropic-family models take an explicit marker; sending one to a model
    that does not understand it is at best ignored and at worst a 400, so the
    default is to send nothing and rely on automatic caching.
    """
    m = (model or "").lower()
    if "claude" in m or m.startswith("anthropic/"):
        return "ephemeral"
    return "none"


def _as_blocks(content) -> list:
    if isinstance(content, list):
        return content
    return [{"type": "text", "text": content or ""}]


class PromptLayout:
    """
    Build a prompt as two piles: the part that repeats, and the part that does not.

        layout = PromptLayout(model="claude-3-5-sonnet")
        layout.static("system", INSTRUCTIONS)
        layout.static("system", TOOL_GUIDE)
        layout.volatile("user", document_text)
        messages = layout.messages()

    The static pile is emitted first, in the order it was added, and the volatile
    pile after it. Nothing is reordered *within* a pile, so an author keeps
    control of anything where order carries meaning.
    """

    def __init__(self, model: Optional[str] = None, *, task: str = "",
                 min_prefix_tokens: Optional[int] = None,
                 ledger: Optional[Ledger] = None) -> None:
        self.model = model
        self.task = task
        self._static: List[dict] = []
        self._volatile: List[dict] = []
        cfg = get_config()
        self._min_prefix = (cfg.layout_min_prefix_tokens
                            if min_prefix_tokens is None else min_prefix_tokens)
        self._ledger = ledger or default_ledger()

    def static(self, role: str, content) -> "PromptLayout":
        """Content identical on every call: instructions, schemas, knowledge
        bases, few-shot examples."""
        if content:
            self._static.append({"role": role, "content": content})
        return self

    def volatile(self, role: str, content) -> "PromptLayout":
        """Content that differs per call: the document, the question, history."""
        if content:
            self._volatile.append({"role": role, "content": content})
        return self

    def extend_volatile(self, messages) -> "PromptLayout":
        for m in messages or []:
            self._volatile.append(m)
        return self

    @property
    def prefix_tokens(self) -> int:
        return count_messages(self._static, self.model)

    @property
    def cacheable(self) -> bool:
        """Whether the stable prefix is long enough for a provider to bother
        caching it. Below the threshold the marker is noise."""
        return self.prefix_tokens >= self._min_prefix

    def messages(self, cache_style: Optional[str] = None) -> list:
        style = cache_style if cache_style is not None else cache_style_for(self.model)
        out = [dict(m) for m in self._static]

        if style == "ephemeral" and out and self.cacheable:
            # The marker goes on the last block of the last static message: it
            # means "everything up to here is reusable".
            last = out[-1]
            blocks = [dict(b) if isinstance(b, dict) else b for b in _as_blocks(last["content"])]
            if blocks and isinstance(blocks[-1], dict):
                blocks[-1] = {**blocks[-1], "cache_control": {"type": "ephemeral"}}
                last["content"] = blocks

        out.extend(dict(m) if isinstance(m, dict) else m for m in self._volatile)
        self._ledger.prefix(self.task, self.model or "", self.prefix_tokens, self.cacheable)
        return out

    def report(self) -> dict:
        total = count_messages(self._static + self._volatile, self.model)
        prefix = self.prefix_tokens
        return {
            "task": self.task,
            "model": self.model,
            "prefix_tokens": prefix,
            "total_tokens": total,
            "prefix_ratio": round(prefix / total, 4) if total else 0.0,
            "min_prefix_tokens": self._min_prefix,
            "cacheable": self.cacheable,
            "cache_style": cache_style_for(self.model),
        }


def _common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


class PrefixTracker:
    """
    Measures the stable prefix of prompts that are already in production.

    Keeps one previously-seen serialisation per task and compares each new call
    against it. The longest common leading substring across consecutive calls is
    a good proxy for what a provider would be able to cache, and it needs no
    cooperation from the call site - which is the point, since the call sites
    worth rewriting are the ones nobody has annotated yet.
    """

    def __init__(self, max_tasks: int = 200) -> None:
        self._lock = threading.Lock()
        self._last: Dict[str, str] = {}
        self._stats: Dict[str, dict] = {}
        self._max_tasks = max_tasks

    @staticmethod
    def _serialize(messages) -> str:
        parts = []
        for m in messages or []:
            role = m.get("role") if isinstance(m, dict) else getattr(m, "role", "")
            content = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
            if isinstance(content, list):
                content = "".join(
                    b.get("text", "") for b in content if isinstance(b, dict)
                )
            parts.append(f"<{role}>{content or ''}")
        return "\n".join(parts)

    def observe(self, task: str, messages, model: Optional[str] = None) -> dict:
        blob = self._serialize(messages)
        with self._lock:
            previous = self._last.get(task)
            self._last[task] = blob
            if len(self._last) > self._max_tasks:
                self._last.pop(next(iter(self._last)))

            row = self._stats.setdefault(
                task, {"samples": 0, "compared": 0, "shared_chars": 0, "total_chars": 0}
            )
            row["samples"] += 1
            row["total_chars"] += len(blob)

            if previous is None:
                # Nothing to compare against yet. Counting it as zero shared
                # prefix would understate every task by one sample's worth.
                return {"task": task, "first_sample": True}

            shared = _common_prefix_len(previous, blob)
            row["compared"] += 1
            row["shared_chars"] += shared

        prefix_tokens = count_text(blob[:shared], model)
        total_tokens = count_text(blob, model)
        return {
            "task": task,
            "prefix_tokens": prefix_tokens,
            "total_tokens": total_tokens,
            "prefix_ratio": round(prefix_tokens / total_tokens, 4) if total_tokens else 0.0,
        }

    def summary(self, model: Optional[str] = None) -> list:
        with self._lock:
            rows = {t: dict(v) for t, v in self._stats.items()}
        out = []
        for task, row in rows.items():
            compared = row.get("compared", 0)
            if not compared:
                # One observation tells you nothing about what repeats.
                continue
            avg_shared = row["shared_chars"] / compared
            avg_total = row["total_chars"] / max(row["samples"], 1)
            out.append({
                "task": task,
                "samples": row["samples"],
                "compared": compared,
                # Character counts converted once, at the average, rather than
                # summing per-call token counts we did not keep.
                "avg_prefix_tokens": int(avg_shared / 3.6),
                "avg_total_tokens": int(avg_total / 3.6),
                "prefix_ratio": round(avg_shared / avg_total, 4) if avg_total else 0.0,
            })
        return sorted(out, key=lambda r: r["prefix_ratio"])


_tracker = PrefixTracker()


def default_tracker() -> PrefixTracker:
    return _tracker
