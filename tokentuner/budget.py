"""
Token budgets: input and output.

Two problems, both of which are usually solved with a magic number in a slice
expression somewhere.

**Input.** Prompts are assembled from parts of wildly different value per
token - a system prompt earns its place, a tool result that dumped fifty rows
when the answer needed three does not. A single global truncation cannot tell
them apart, so it either cuts the instructions or lets the payload run. The fix
is a budget *per section*, with a strategy per section, because what to throw
away differs: the middle of a document is expendable in a way the start of a
system prompt is not.

**Output.** Output tokens cost several times what input tokens do, and an
unbounded `max_tokens` is an open invitation for a model having a bad day to
write two thousand tokens of preamble. The cap should come from what the task
actually produces, which is measurable, rather than from a guess.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

from .counting import count_messages, count_text, truncate_to_tokens
from .ledger import Ledger, default_ledger

ELLIPSIS = "\n...[trimmed]...\n"

# How a section gives up tokens when it does not fit.
#   head       - keep the beginning (documents, instructions)
#   tail       - keep the end (logs, transcripts, anything most recent-first)
#   head_tail  - keep both ends, drop the middle (a report: subject at the top,
#                conclusions at the bottom, filler in between)
STRATEGIES = ("head", "tail", "head_tail")


@dataclass
class Section:
    name: str
    content: str
    # None means "no cap of its own"; it still competes for the shared total.
    max_tokens: Optional[int] = None
    strategy: str = "head"
    # Higher priority sections are trimmed last. Instructions outrank payloads.
    priority: int = 0
    # A section that must survive intact - trimming it changes the task rather
    # than shortening it.
    required: bool = False


def trim_text(text: str, max_tokens: int, strategy: str = "head",
              model: Optional[str] = None) -> str:
    """Cut one blob to a token budget using the named strategy."""
    if max_tokens <= 0:
        return ""
    if count_text(text, model) <= max_tokens:
        return text

    if strategy == "tail":
        # Walk in from the front until the remainder fits.
        cut = text
        while cut and count_text(cut, model) > max_tokens:
            drop = max(int(len(cut) * 0.1), 1)
            cut = cut[drop:]
        return cut

    if strategy == "head_tail":
        marker = count_text(ELLIPSIS, model)
        usable = max(max_tokens - marker, 1)
        head_budget = usable // 2
        tail_budget = usable - head_budget
        head = truncate_to_tokens(text, head_budget, model)
        tail = trim_text(text[len(head):], tail_budget, "tail", model)
        return head + ELLIPSIS + tail

    return truncate_to_tokens(text, max_tokens, model)


@dataclass
class Budget:
    """
    A total input budget, divided across named sections.

    `fit` applies each section's own cap first, then - only if the total is
    still over - takes tokens from the lowest-priority sections until it fits.
    Required sections are never touched: if the budget cannot be met without
    them, the budget was wrong and silently mutilating the instructions would
    hide that.
    """

    total_tokens: int
    model: Optional[str] = None
    task: str = ""
    sections: List[Section] = field(default_factory=list)
    ledger: Ledger = field(default_factory=default_ledger)

    def add(self, name: str, content: str, *, max_tokens: Optional[int] = None,
            strategy: str = "head", priority: int = 0,
            required: bool = False) -> "Budget":
        self.sections.append(Section(name, content or "", max_tokens, strategy,
                                     priority, required))
        return self

    def fit(self) -> dict:
        """Returns {section name: trimmed content}."""
        out = {}
        sizes = {}
        original_total = 0

        for section in self.sections:
            before = count_text(section.content, self.model)
            original_total += before
            content = section.content
            if section.max_tokens is not None and before > section.max_tokens:
                content = trim_text(content, section.max_tokens, section.strategy, self.model)
                self.ledger.trimmed(self.task, before - count_text(content, self.model),
                                    section.name)
            out[section.name] = content
            sizes[section.name] = count_text(content, self.model)

        over = sum(sizes.values()) - self.total_tokens
        if over <= 0:
            return out

        # Lowest priority first; among equals, biggest first - taking from the
        # largest section is what actually moves the number.
        order = sorted(
            (s for s in self.sections if not s.required),
            key=lambda s: (s.priority, -sizes[s.name]),
        )
        for section in order:
            if over <= 0:
                break
            current = sizes[section.name]
            if current <= 0:
                continue
            keep = max(current - over, 0)
            trimmed = trim_text(out[section.name], keep, section.strategy, self.model)
            new_size = count_text(trimmed, self.model)
            self.ledger.trimmed(self.task, current - new_size, section.name)
            out[section.name] = trimmed
            over -= current - new_size
            sizes[section.name] = new_size

        return out

    def report(self) -> dict:
        return {
            "task": self.task,
            "total_budget": self.total_tokens,
            "sections": [
                {"name": s.name, "tokens": count_text(s.content, self.model),
                 "cap": s.max_tokens, "strategy": s.strategy,
                 "priority": s.priority, "required": s.required}
                for s in self.sections
            ],
        }


def trim_history(messages: Sequence, max_tokens: int, *,
                 model: Optional[str] = None, keep_system: bool = True,
                 keep_last: int = 2, task: str = "",
                 ledger: Optional[Ledger] = None) -> list:
    """
    Drop the oldest turns of a conversation until the history fits.

    Whole messages, never partial ones: half a turn is worse than no turn,
    because the model reads it as a complete statement that happens to be wrong.
    The system prompt and the most recent turns are kept regardless - they are
    the two ends the answer actually depends on.

    A tool result is dropped together with the assistant turn that requested it.
    An orphaned `tool` message with no matching `tool_call_id` above it is a
    protocol error at most providers, not merely untidy.
    """
    ledger = ledger or default_ledger()
    msgs = list(messages or [])
    if not msgs:
        return msgs

    if count_messages(msgs, model) <= max_tokens:
        return msgs

    def role_of(m):
        return m.get("role") if isinstance(m, dict) else getattr(m, "role", None)

    def has_tool_calls(m):
        if isinstance(m, dict):
            return bool(m.get("tool_calls"))
        return bool(getattr(m, "tool_calls", None))

    head = []
    body = msgs
    if keep_system and role_of(msgs[0]) == "system":
        head, body = [msgs[0]], msgs[1:]

    tail = body[-keep_last:] if keep_last > 0 else []
    middle = body[: len(body) - len(tail)]
    before = count_messages(msgs, model)

    while middle and count_messages(head + middle + tail, model) > max_tokens:
        middle.pop(0)
        # Having dropped a turn, any tool results it produced are now orphans.
        while middle and role_of(middle[0]) == "tool":
            middle.pop(0)

    result = head + middle + tail

    # The tail itself can still open with an orphaned tool message.
    while len(result) > len(head) and role_of(result[len(head)]) == "tool":
        result.pop(len(head))

    # If a kept assistant turn requested tools whose results were dropped, the
    # request is inconsistent; drop that turn too.
    cleaned = []
    for i, m in enumerate(result):
        if has_tool_calls(m):
            following = result[i + 1] if i + 1 < len(result) else None
            if following is None or role_of(following) != "tool":
                continue
        cleaned.append(m)

    ledger.trimmed(task, before - count_messages(cleaned, model), "history")
    return cleaned


def recommend_max_tokens(observed_output_tokens: Sequence[int], *,
                         percentile: float = 95.0, headroom: float = 1.25,
                         floor: int = 64, ceiling: int = 8192) -> Optional[int]:
    """
    Derive an output cap from what a task has actually produced.

    A percentile with headroom rather than the observed maximum: the maximum is
    one bad generation away from being useless as a bound, and that bad
    generation is precisely what the cap exists to stop.
    """
    values = sorted(int(v) for v in observed_output_tokens if v and v > 0)
    if len(values) < 20:
        # Too few samples to bound anything. Saying so beats a confident guess.
        return None
    index = min(int(math.ceil(percentile / 100.0 * len(values))) - 1, len(values) - 1)
    return max(floor, min(int(values[index] * headroom), ceiling))
