"""
Payload compaction.

This is about the *data* injected into a prompt, never the instructions. The
instructions are the part a human tuned for quality; the data is the part a
serialiser produced, and serialisers optimise for being read by people.

Where the tokens go, in the order they are usually worth reclaiming:

  * Indentation and key quoting. `json.dumps(obj, indent=2)` on a list of rows
    spends a large fraction of its tokens on whitespace and punctuation that
    carry no information the model needs.
  * Null and empty fields. A key whose value is null tells the model the field
    exists and is unknown; usually the absence of the key says the same thing
    for a fraction of the cost. Reversible per call, because occasionally the
    difference matters.
  * Repeated rows. Tool results routinely contain near-duplicates.
  * Long free text inside otherwise small records - a `notes` field with an
    entire email thread in it.

None of this changes what is being asked, which is why it is safe to apply by
default in a way that dropping message history is not.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Iterable, Optional, Sequence

from .counting import count_text, count_value, truncate_to_tokens
from .ledger import Ledger, default_ledger

_BLANK_RUN = re.compile(r"\n{3,}")
_TRAILING_WS = re.compile(r"[ \t]+\n")
_SPACE_RUN = re.compile(r"[ \t]{2,}")

EMPTY = (None, "", [], {}, ())


def squeeze_text(text: str, *, collapse_spaces: bool = True) -> str:
    """
    Remove whitespace that carries no meaning.

    Blank-line runs collapse to one and trailing spaces go. Single newlines are
    preserved: in a document or a transcript the line structure is information,
    and flattening it measurably hurts extraction.
    """
    if not text:
        return ""
    out = _TRAILING_WS.sub("\n", text)
    out = _BLANK_RUN.sub("\n\n", out)
    if collapse_spaces:
        out = _SPACE_RUN.sub(" ", out)
    return out.strip()


def drop_empty(obj: Any, *, keep_false: bool = True) -> Any:
    """
    Recursively remove null and empty values.

    `keep_false` defaults to true because False and 0 are answers, not absences,
    and stripping them turns "this record has zero retries" into "nobody
    counted the retries".
    """
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            cleaned = drop_empty(v, keep_false=keep_false)
            if cleaned in EMPTY and not (keep_false and cleaned in (False, 0)):
                continue
            out[k] = cleaned
        return out
    if isinstance(obj, (list, tuple)):
        items = [drop_empty(v, keep_false=keep_false) for v in obj]
        return [v for v in items
                if v not in EMPTY or (keep_false and v in (False, 0))]
    return obj


def compact_json(obj: Any, *, strip_empty: bool = True) -> str:
    """Serialise with no cosmetic whitespace. The single highest-yield change
    for any prompt that embeds structured data."""
    if strip_empty:
        obj = drop_empty(obj)
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False, default=str)


def to_kv_lines(obj: dict, *, prefix: str = "", sep: str = ": ",
                skip_empty: bool = True) -> str:
    """
    Flatten a record to `key: value` lines.

    Cheaper than JSON for a flat record - no braces, quotes or commas - and
    models read it at least as well. Nested structures fall back to compact
    JSON on the value side rather than exploding into dotted paths, which get
    long enough to lose the saving.
    """
    lines = []
    for key, value in (obj or {}).items():
        if skip_empty and value in EMPTY:
            continue
        if isinstance(value, (dict, list, tuple)):
            rendered = compact_json(value)
        else:
            rendered = str(value)
        lines.append(f"{prefix}{key}{sep}{rendered}")
    return "\n".join(lines)


def dedupe_rows(rows: Sequence[dict], *, key_fields: Optional[Sequence[str]] = None
                ) -> list:
    """Drop duplicate records, preserving order. Identity is the whole record
    unless specific fields are named."""
    seen = set()
    out = []
    for row in rows or []:
        if key_fields:
            identity = tuple(str(row.get(f)) for f in key_fields)
        else:
            try:
                identity = json.dumps(row, sort_keys=True, default=str)
            except Exception:  # noqa: BLE001
                identity = str(row)
        if identity in seen:
            continue
        seen.add(identity)
        out.append(row)
    return out


def cap_fields(rows: Sequence[dict], *, max_field_tokens: int = 200,
               fields: Optional[Sequence[str]] = None,
               model: Optional[str] = None) -> list:
    """Truncate long free-text fields inside otherwise compact records."""
    out = []
    for row in rows or []:
        if not isinstance(row, dict):
            out.append(row)
            continue
        copy = dict(row)
        for key, value in row.items():
            if fields and key not in fields:
                continue
            if isinstance(value, str) and count_text(value, model) > max_field_tokens:
                copy[key] = truncate_to_tokens(value, max_field_tokens, model, suffix="…")
        out.append(copy)
    return out


def minify_rows(rows: Sequence[dict], *, max_rows: Optional[int] = None,
                key_fields: Optional[Sequence[str]] = None,
                max_field_tokens: int = 200, model: Optional[str] = None,
                task: str = "", ledger: Optional[Ledger] = None) -> str:
    """
    The whole treatment for a list of records, as one compact string.

    Row limiting is last and is reported, because it is the only step here that
    removes information rather than formatting. When rows are dropped the
    string says so - a silently truncated list reads to the model as the
    complete answer, and it will summarise it as if it were.
    """
    ledger = ledger or default_ledger()
    before = count_value(rows, model)

    cleaned = dedupe_rows(rows, key_fields=key_fields)
    cleaned = [drop_empty(r) for r in cleaned]
    cleaned = cap_fields(cleaned, max_field_tokens=max_field_tokens, model=model)

    note = ""
    if max_rows is not None and len(cleaned) > max_rows:
        note = f"\n[{len(cleaned) - max_rows} more rows not shown of {len(cleaned)} total]"
        cleaned = cleaned[:max_rows]

    text = compact_json(cleaned) + note
    ledger.minified(task, before - count_text(text, model))
    return text


def minify_document(text: str, *, max_tokens: Optional[int] = None,
                    model: Optional[str] = None, task: str = "",
                    ledger: Optional[Ledger] = None) -> str:
    """Squeeze a document, and optionally cap it keeping both ends."""
    from .budget import trim_text

    ledger = ledger or default_ledger()
    before = count_text(text, model)
    out = squeeze_text(text)
    if max_tokens is not None:
        out = trim_text(out, max_tokens, "head_tail", model)
    ledger.minified(task, before - count_text(out, model))
    return out
