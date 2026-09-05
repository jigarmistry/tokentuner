"""
Request coalescing.

Two different savings, often confused.

**Prompt batching.** Scoring twenty records against one rubric sends the rubric,
the instructions and the schema twenty times. Sent as one request over twenty
records, the shared part is paid once. The saving is the shared prefix
multiplied by the number of items, and on a fan-out with a large brief it is
most of the bill.

The catch is that one bad item can spoil the batch, so `split_results` is
strict about identity: results are matched to inputs by an explicit index the
model is asked to echo, never by position in the returned array. A model that
returns nineteen objects for twenty inputs is common; a caller that assumes
position has now silently attributed each result to the wrong record.

**Embedding batching.** Embedding endpoints take arrays natively, so this is
purely about round trips and about not paying twice for the same text. The
de-duplication matters more than the batching: re-embedding an unchanged
document is the most common avoidable spend in a system that embeds anything.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .counting import count_value
from .keys import content_key
from .ledger import Ledger, default_ledger


def chunk(items: Sequence, *, max_items: int = 10,
          max_tokens: Optional[int] = None,
          sizer: Optional[Callable[[Any], int]] = None,
          model: Optional[str] = None) -> List[list]:
    """
    Split items into groups that respect both a count and a token ceiling.

    An item bigger than the whole token ceiling gets a group to itself rather
    than being dropped or silently truncated - the caller asked for it to be
    processed, and it is their budget that is wrong, not their data.
    """
    sizer = sizer or (lambda x: count_value(x, model))
    groups: List[list] = []
    current: list = []
    current_tokens = 0

    for item in items or []:
        size = sizer(item)
        too_many = len(current) >= max_items
        too_big = max_tokens is not None and current and current_tokens + size > max_tokens
        if too_many or too_big:
            groups.append(current)
            current, current_tokens = [], 0
        current.append(item)
        current_tokens += size

    if current:
        groups.append(current)
    return groups


def index_items(items: Sequence, *, renderer: Callable[[Any], str],
                start: int = 0) -> str:
    """
    Render a group as an indexed block for the prompt.

    The index is what makes the result attributable. It is stated explicitly in
    the text rather than implied by ordering, because ordering is exactly what
    the model will fail to preserve.
    """
    parts = []
    for offset, item in enumerate(items):
        parts.append(f"--- ITEM {start + offset} ---\n{renderer(item)}")
    return "\n\n".join(parts)


def split_results(parsed: Any, count: int, *, index_key: str = "index",
                  results_key: str = "results") -> List[Optional[dict]]:
    """
    Turn a batch answer into a per-item list, `None` where the model did not
    answer for that item.

    Never falls back to positional matching. A missing result is a result the
    caller must handle - re-running that one item is cheap, and attributing an
    answer to the wrong record is not.
    """
    out: List[Optional[dict]] = [None] * count

    rows = parsed
    if isinstance(parsed, dict):
        rows = parsed.get(results_key)
        if rows is None:
            # Tolerate a single-key wrapper under a different name.
            lists = [v for v in parsed.values() if isinstance(v, list)]
            rows = lists[0] if len(lists) == 1 else None
    if not isinstance(rows, list):
        return out

    for row in rows:
        if not isinstance(row, dict):
            continue
        raw = row.get(index_key)
        try:
            index = int(raw)
        except (TypeError, ValueError):
            continue
        if 0 <= index < count and out[index] is None:
            out[index] = row
    return out


def batch_savings(group_size: int, shared_tokens: int) -> int:
    """Tokens avoided by sending one request instead of `group_size`."""
    return max(group_size - 1, 0) * max(shared_tokens, 0)


class EmbeddingBatcher:
    """
    De-duplicate, batch, and re-expand embedding inputs.

        batcher = EmbeddingBatcher(embed_fn=client_embed_many, batch_size=96)
        vectors = batcher.run(texts)          # aligned with `texts`

    `embed_fn` takes a list of strings and returns a list of vectors in the
    same order. Empty inputs map to None without ever reaching the provider.
    """

    def __init__(self, embed_fn: Callable[[List[str]], List[Any]], *,
                 batch_size: int = 96, max_tokens_per_batch: Optional[int] = None,
                 model: Optional[str] = None, task: str = "embeddings",
                 ledger: Optional[Ledger] = None) -> None:
        self._embed = embed_fn
        self._batch_size = max(1, batch_size)
        self._max_tokens = max_tokens_per_batch
        self._model = model
        self._task = task
        self._ledger = ledger or default_ledger()

    def run(self, texts: Sequence[Optional[str]]) -> List[Any]:
        texts = list(texts or [])
        if not texts:
            return []

        # content hash -> first position holding it
        unique: Dict[str, int] = {}
        # position -> content hash, for non-empty inputs
        positions: Dict[int, str] = {}

        for i, text in enumerate(texts):
            cleaned = (text or "").strip()
            if not cleaned:
                continue
            key = content_key(cleaned)
            positions[i] = key
            unique.setdefault(key, i)

        duplicates = len(positions) - len(unique)
        pending = [(key, texts[i]) for key, i in unique.items()]

        vectors: Dict[str, Any] = {}
        for group in chunk(pending, max_items=self._batch_size,
                           max_tokens=self._max_tokens,
                           sizer=lambda kv: count_value(kv[1], self._model)):
            try:
                got = self._embed([kv[1] for kv in group])
            except Exception as e:  # noqa: BLE001
                print(f"[tokentuner] embedding batch failed: {e}")
                got = [None] * len(group)
            for (key, _), vector in zip(group, list(got) + [None] * len(group)):
                vectors[key] = vector

        if duplicates:
            self._ledger.batched(self._task, self._model or "", calls_saved=duplicates,
                                 tokens_saved=sum(
                                     count_value(texts[i], self._model)
                                     for i, key in positions.items()
                                     if unique.get(key) != i
                                 ))

        return [vectors.get(positions[i]) if i in positions else None
                for i in range(len(texts))]
