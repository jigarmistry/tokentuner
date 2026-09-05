"""
The analyzer.

Everything else in this package saves tokens on a call that is already
happening. This part answers the prior question: which calls are worth
changing at all.

It reads usage rows - one per model call, in the shape almost every application
already records - and turns them into ranked, specific recommendations. It
takes rows, not a database handle, so it works against Mongo, a JSON export, or
a list built in a test, and the package keeps its promise of no dependencies.

The recommendations are deliberately conservative. Each one names the evidence
and the sample size behind it, because "your escalation rate is 40%" is
actionable and "consider optimising your prompts" is not.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional

from .budget import recommend_max_tokens

# Below this many calls, a rate is noise.
MIN_SAMPLES = 20

# An escalation costs the first attempt in full and then pays again. Above this
# share, the cheap tier is not earning its place.
ESCALATION_WARN = 0.25

# Input:output ratio above which a prompt dominates the bill, and prefix
# caching or payload minification is where the money is.
PROMPT_HEAVY_RATIO = 8.0

# A task whose output is this consistent has a safe, tight cap available.
OUTPUT_STABLE_SPREAD = 2.0


def _row(r: Any, key: str, default=0):
    value = r.get(key, default) if isinstance(r, dict) else getattr(r, key, default)
    return default if value is None else value


def aggregate(rows: Iterable[dict]) -> Dict[str, dict]:
    """Collapse usage rows into per-task statistics."""
    stats: Dict[str, dict] = defaultdict(
        lambda: {
            "task": "", "calls": 0, "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0,
            "latency_ms": 0, "escalations": 0, "failures": 0,
            "outputs": [], "models": defaultdict(int), "tiers": defaultdict(int),
        }
    )
    for r in rows or []:
        task = str(_row(r, "task", "") or "unknown")
        s = stats[task]
        s["task"] = task
        s["calls"] += 1
        s["tokens_in"] += int(_row(r, "tokens_in") or 0)
        s["tokens_out"] += int(_row(r, "tokens_out") or 0)
        s["cost_usd"] += float(_row(r, "cost_usd", 0.0) or 0.0)
        s["latency_ms"] += int(_row(r, "latency_ms") or 0)
        if _row(r, "escalated", False):
            s["escalations"] += 1
        if not _row(r, "ok", True):
            s["failures"] += 1
        out = int(_row(r, "tokens_out") or 0)
        if out > 0:
            s["outputs"].append(out)
        model = _row(r, "model", "") or "?"
        s["models"][model] += 1
        s["tiers"][_row(r, "tier", "") or "?"] += 1
    return dict(stats)


def _finding(severity: str, task: str, kind: str, message: str,
             action: str, evidence: dict) -> dict:
    return {"severity": severity, "task": task, "kind": kind, "message": message,
            "action": action, "evidence": evidence}


def analyze(rows: Iterable[dict], *, task_config: Optional[Dict[str, dict]] = None
            ) -> dict:
    """
    Produce findings from usage rows.

    `task_config` optionally carries what the application believes about each
    task - `{"task.id": {"max_tokens": 512, "tier": "small"}}` - which lets the
    analyzer notice the difference between what a task is configured to do and
    what it actually does.
    """
    stats = aggregate(rows)
    task_config = task_config or {}
    findings: List[dict] = []

    total_cost = sum(s["cost_usd"] for s in stats.values()) or 0.0
    total_calls = sum(s["calls"] for s in stats.values()) or 0

    for task, s in stats.items():
        calls = s["calls"]
        cfg = task_config.get(task, {})

        # -- mis-tiered: the cheap answer keeps being thrown away ------------
        if calls >= MIN_SAMPLES:
            rate = s["escalations"] / calls
            if rate >= ESCALATION_WARN:
                findings.append(_finding(
                    "high" if rate >= 0.4 else "medium", task, "mis_tiered",
                    f"{rate:.0%} of calls escalate a tier, so the first attempt is "
                    f"paid for and discarded.",
                    "Move this task up a tier, or tighten the prompt so the cheap "
                    "model can satisfy the validator.",
                    {"calls": calls, "escalation_rate": round(rate, 4),
                     "configured_tier": cfg.get("tier")},
                ))

        # -- prompt-heavy: the input is the bill ------------------------------
        if s["tokens_out"] > 0 and calls >= MIN_SAMPLES:
            ratio = s["tokens_in"] / max(s["tokens_out"], 1)
            if ratio >= PROMPT_HEAVY_RATIO:
                avg_in = s["tokens_in"] // calls
                findings.append(_finding(
                    "medium", task, "prompt_heavy",
                    f"Input is {ratio:.0f}x output ({avg_in} input tokens per call). "
                    f"The prompt, not the answer, is what this task costs.",
                    "Put the stable half of the prompt first so the provider can "
                    "cache the prefix, and compact any embedded records.",
                    {"avg_tokens_in": avg_in,
                     "avg_tokens_out": s["tokens_out"] // calls,
                     "ratio": round(ratio, 1)},
                ))

        # -- output cap: derivable from what it actually writes ---------------
        rec = recommend_max_tokens(s["outputs"])
        if rec and not cfg.get("max_tokens"):
            spread = max(s["outputs"]) / max(min(s["outputs"]), 1)
            severity = "medium" if spread <= OUTPUT_STABLE_SPREAD else "low"
            findings.append(_finding(
                severity, task, "no_output_cap",
                f"No max_tokens is set; observed p95 output is "
                f"{sorted(s['outputs'])[int(0.95 * len(s['outputs'])) - 1]} tokens.",
                f"Set max_tokens={rec} to bound a runaway generation.",
                {"recommended_max_tokens": rec, "samples": len(s["outputs"]),
                 "max_observed": max(s["outputs"])},
            ))
        elif rec and cfg.get("max_tokens") and cfg["max_tokens"] > rec * 2:
            findings.append(_finding(
                "low", task, "loose_output_cap",
                f"max_tokens={cfg['max_tokens']} is more than twice the "
                f"recommended {rec}.",
                f"Tighten to max_tokens={rec}.",
                {"configured": cfg["max_tokens"], "recommended": rec},
            ))

        # -- failures ---------------------------------------------------------
        if calls >= MIN_SAMPLES:
            fail_rate = s["failures"] / calls
            if fail_rate >= 0.1:
                findings.append(_finding(
                    "high", task, "failing",
                    f"{fail_rate:.0%} of calls fail outright - tokens spent for "
                    f"no answer.",
                    "Fix the failure before tuning anything else; a failing call "
                    "is 100% waste.",
                    {"failure_rate": round(fail_rate, 4), "calls": calls},
                ))

        # -- cost concentration ----------------------------------------------
        if total_cost > 0 and s["cost_usd"] / total_cost >= 0.4 and calls >= MIN_SAMPLES:
            findings.append(_finding(
                "medium", task, "cost_concentration",
                f"This one task is {s['cost_usd'] / total_cost:.0%} of total spend.",
                "Tune here first; savings anywhere else are rounding errors "
                "next to it.",
                {"cost_usd": round(s["cost_usd"], 4),
                 "share": round(s["cost_usd"] / total_cost, 4)},
            ))

    order = {"high": 0, "medium": 1, "low": 2}
    findings.sort(key=lambda f: (order.get(f["severity"], 3), f["task"]))

    summary = []
    for task, s in sorted(stats.items(), key=lambda kv: -kv[1]["cost_usd"]):
        calls = max(s["calls"], 1)
        summary.append({
            "task": task,
            "calls": s["calls"],
            "cost_usd": round(s["cost_usd"], 4),
            "avg_tokens_in": s["tokens_in"] // calls,
            "avg_tokens_out": s["tokens_out"] // calls,
            "avg_latency_ms": s["latency_ms"] // calls,
            "escalation_rate": round(s["escalations"] / calls, 4),
            "failure_rate": round(s["failures"] / calls, 4),
            "models": dict(s["models"]),
        })

    return {
        "totals": {
            "calls": total_calls,
            "cost_usd": round(total_cost, 4),
            "tokens_in": sum(s["tokens_in"] for s in stats.values()),
            "tokens_out": sum(s["tokens_out"] for s in stats.values()),
            "tasks": len(stats),
        },
        "findings": findings,
        "by_task": summary,
    }


def render_text(result: dict) -> str:
    """Human-readable report. Severity first, because the top of this list is
    where the money is."""
    t = result["totals"]
    lines = [
        "tokentuner report",
        "=" * 60,
        f"{t['calls']} calls across {t['tasks']} tasks | "
        f"${t['cost_usd']} | {t['tokens_in']:,} in / {t['tokens_out']:,} out",
        "",
    ]

    findings = result["findings"]
    if not findings:
        lines.append("No findings. Either everything is tuned, or there is not "
                     "enough traffic yet to tell.")
    else:
        lines.append(f"FINDINGS ({len(findings)})")
        lines.append("-" * 60)
        for f in findings:
            lines.append(f"[{f['severity'].upper():<6}] {f['task']}  ({f['kind']})")
            lines.append(f"          {f['message']}")
            lines.append(f"     -->  {f['action']}")
            lines.append("")

    lines.append("BY TASK (most expensive first)")
    lines.append("-" * 60)
    header = f"{'task':<32}{'calls':>7}{'$':>10}{'in':>8}{'out':>7}{'esc':>7}"
    lines.append(header)
    for row in result["by_task"]:
        lines.append(
            f"{row['task'][:31]:<32}{row['calls']:>7}{row['cost_usd']:>10.4f}"
            f"{row['avg_tokens_in']:>8}{row['avg_tokens_out']:>7}"
            f"{row['escalation_rate']:>7.0%}"
        )
    return "\n".join(lines)


def _load_rows(args) -> List[dict]:
    if args.json:
        with open(args.json) as fh:
            data = json.load(fh)
        return data.get("rows", data) if isinstance(data, dict) else data

    if args.mongo_uri:
        from datetime import datetime, timedelta

        from pymongo import MongoClient

        client = MongoClient(args.mongo_uri, serverSelectionTimeoutMS=5000)
        col = client[args.database][args.collection]
        since = datetime.utcnow() - timedelta(days=args.days)
        return list(col.find({"created_at": {"$gte": since}}))

    raise SystemExit("Give either --json PATH or --mongo-uri URI")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="tokentuner",
        description="Turn LLM usage rows into ranked tuning recommendations.",
    )
    p.add_argument("--json", help="Path to a JSON file of usage rows")
    p.add_argument("--mongo-uri", help="MongoDB connection string")
    p.add_argument("--database", default="", help="Mongo database name")
    p.add_argument("--collection", default="llm_usage", help="Mongo collection name")
    p.add_argument("--days", type=int, default=30, help="Window in days (Mongo only)")
    p.add_argument("--task-config", help="JSON file of per-task config to compare against")
    p.add_argument("--format", choices=("text", "json"), default="text")
    args = p.parse_args(argv)

    rows = _load_rows(args)
    task_config = {}
    if args.task_config:
        with open(args.task_config) as fh:
            task_config = json.load(fh)

    result = analyze(rows, task_config=task_config)
    if args.format == "json":
        print(json.dumps(result, indent=2, default=str))
    else:
        print(render_text(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
