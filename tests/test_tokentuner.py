"""
tokentuner contract tests. No network, no database, no pytest:

    python packages/tokentuner/tests/test_tokentuner.py

Each test pins a way this package could quietly do damage rather than good. A
caching layer fails silently by construction - the wrong answer arrives fast and
well-formed - so these check the refusals as hard as the savings.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tokentuner as tt  # noqa: E402
from tokentuner.config import TunerConfig  # noqa: E402
from tokentuner.ledger import Ledger  # noqa: E402
from tokentuner.stores import MemoryStore  # noqa: E402
from tokentuner.stores.sqlite_store import SQLiteStore  # noqa: E402

CHECKS = []


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


def _tuner(**cfg) -> tt.Tuner:
    config = TunerConfig(**cfg)
    return tt.Tuner(config=config,
                    cache=tt.ResponseCache(config, store=MemoryStore(), ledger=Ledger()),
                    ledger=Ledger())


ENTRY = dict(to_entry=lambda r: r, from_entry=lambda e: e)
CALL = dict(task="t", model="m", messages=[{"role": "user", "content": "q"}],
            temperature=0.1, **ENTRY)


# ---------------------------------------------------------------- counting


@check("token estimates never undercount, which is what would overflow a context")
def _c1():
    for text in ("hello world", "{}" * 200, "a b c\n" * 100, "x" * 5000):
        assert tt.estimate_tokens(text) >= len(text) / 4.0 * 0.9, text[:20]


@check("truncation respects a token budget")
def _c2():
    text = "word " * 500
    cut = tt.truncate_to_tokens(text, 50)
    assert tt.count_text(cut) <= 50, tt.count_text(cut)
    assert cut  # and it is not simply empty


@check("tool definitions are counted; they are billed but never appear in messages")
def _c3():
    tools = [{"type": "function", "function": {"name": "x", "parameters": {"a": 1}}}]
    assert tt.count_tools(tools) > 0
    assert tt.count_request([], tools) > tt.count_request([], None)


# ---------------------------------------------------------------- keys


@check("cache keys separate tenants")
def _k1():
    args = dict(model="m", messages=[{"role": "user", "content": "hi"}])
    assert tt.call_key(tenant="a", **args) != tt.call_key(tenant="b", **args)


@check("cache keys cover every input that can change the answer")
def _k2():
    base = dict(model="m", messages=[{"role": "user", "content": "hi"}], tenant="a")
    key = tt.call_key(**base)
    for changed in (
        dict(base, model="other"),
        dict(base, temperature=0.9),
        dict(base, max_tokens=100),
        dict(base, schema={"type": "object"}),
        dict(base, tools=[{"name": "x"}]),
        dict(base, want_json=True),
        dict(base, messages=[{"role": "user", "content": "different"}]),
    ):
        assert tt.call_key(**changed) != key, changed


@check("a cache key never contains the prompt or the tenant id")
def _k3():
    key = tt.call_key(model="m", tenant="acme-corp",
                      messages=[{"role": "user", "content": "Jane Doe, 07700 900000"}])
    assert "Jane" not in key and "900000" not in key and "acme" not in key


# ---------------------------------------------------------------- cache policy


@check("a second identical call is served from cache")
def _p1():
    t = _tuner()
    calls = []
    r1, m1 = t.run(lambda: (calls.append(1), {"text": "a"})[1], **CALL)
    r2, m2 = t.run(lambda: (calls.append(1), {"text": "b"})[1], **CALL)
    assert len(calls) == 1 and m1.cache == "miss" and m2.cache == "hit"
    assert r2["text"] == "a"


@check("a hot temperature is never cached - the caller asked for variety")
def _p2():
    t = _tuner()
    calls = []
    for _ in range(2):
        t.run(lambda: (calls.append(1), {"text": "x"})[1], **dict(CALL, temperature=0.9))
    assert len(calls) == 2


@check("a tool-calling turn is never cached - it depends on data the key cannot see")
def _p3():
    t = _tuner()
    calls = []
    for _ in range(2):
        _, meta = t.run(lambda: (calls.append(1), {"text": "x"})[1],
                        **dict(CALL, tools=[{"name": "get_jobs"}]))
    assert len(calls) == 2 and meta.reason == "tool_call"


@check("a failed answer is not stored")
def _p4():
    t = _tuner()
    calls = []

    def run():
        calls.append(1)
        return {"ok": False}

    # to_entry returning None is how a caller refuses to store a result.
    args = dict(CALL, to_entry=lambda r: None if not r.get("ok") else r)
    t.run(run, **args)
    t.run(run, **args)
    assert len(calls) == 2


@check("personal data is not written to a persistent store unless enabled")
def _p5(tmp=None):
    import tempfile, os

    path = os.path.join(tempfile.mkdtemp(), "c.db")
    config = TunerConfig(cache_persist_sensitive=False)
    t = tt.Tuner(config=config,
                 cache=tt.ResponseCache(config, store=SQLiteStore(path), ledger=Ledger()),
                 ledger=Ledger())
    calls = []
    for _ in range(2):
        t.run(lambda: (calls.append(1), {"text": "personal"})[1],
              **dict(CALL, sensitive=True))
    assert len(calls) == 2, "sensitive response was persisted"

    # ...and the same call is cached once persistence is explicitly allowed.
    config2 = TunerConfig(cache_persist_sensitive=True)
    t2 = tt.Tuner(config=config2,
                  cache=tt.ResponseCache(config2, store=SQLiteStore(path), ledger=Ledger()),
                  ledger=Ledger())
    calls2 = []
    for _ in range(2):
        t2.run(lambda: (calls2.append(1), {"text": "personal"})[1],
               **dict(CALL, sensitive=True))
    assert len(calls2) == 1, "opt-in persistence did not take effect"


@check("an oversized response is not allowed to evict everything else")
def _p6():
    t = _tuner(cache_max_value_bytes=100)
    calls = []
    for _ in range(2):
        t.run(lambda: (calls.append(1), {"text": "x" * 5000})[1], **CALL)
    assert len(calls) == 2


@check("the master switch turns every optimisation off")
def _p7():
    t = _tuner(enabled=False)
    calls = []
    for _ in range(2):
        _, meta = t.run(lambda: (calls.append(1), {"text": "x"})[1], **CALL)
    assert len(calls) == 2 and meta.reason == "tuner_disabled"


# ---------------------------------------------------------------- dedupe


@check("concurrent identical calls collapse into one request")
def _d1():
    t = _tuner()
    calls = []

    def slow():
        calls.append(1)
        time.sleep(0.15)
        return {"text": "a"}

    results = []
    threads = [threading.Thread(target=lambda: results.append(t.run(slow, **CALL)))
               for _ in range(6)]
    [th.start() for th in threads]
    [th.join() for th in threads]
    assert len(calls) == 1, f"expected 1 upstream call, got {len(calls)}"
    assert all(r["text"] == "a" for r, _ in results)


@check("a follower does not inherit the leader's failure")
def _d2():
    t = _tuner()
    attempts = []

    def boom():
        attempts.append(1)
        raise RuntimeError("upstream down")

    errors = []

    def call():
        try:
            t.run(boom, **CALL)
        except RuntimeError as e:
            errors.append(e)

    threads = [threading.Thread(target=call) for _ in range(3)]
    [th.start() for th in threads]
    [th.join() for th in threads]
    assert len(errors) == 3, "every caller must see the failure"
    assert len(attempts) == 3, "each caller retries rather than inheriting"


@check("async concurrent identical calls collapse into one request")
def _d3():
    t = _tuner()

    async def body():
        calls = []

        async def slow():
            calls.append(1)
            await asyncio.sleep(0.1)
            return {"text": "a"}

        out = await asyncio.gather(*[t.arun(slow, **CALL) for _ in range(5)])
        assert len(calls) == 1, len(calls)
        assert sum(1 for _, m in out if m.deduped) == 4

    asyncio.run(body())


# ---------------------------------------------------------------- layout


@check("static content is emitted before volatile content")
def _l1():
    layout = tt.PromptLayout(model="gpt-4o", task="t", min_prefix_tokens=1)
    layout.volatile("user", "the document")
    layout.static("system", "the instructions")
    messages = layout.messages()
    assert messages[0]["content"] == "the instructions"
    assert messages[1]["content"] == "the document"


@check("a cache marker is only sent to a model that understands it")
def _l2():
    for model, marked in (("anthropic/claude-3.5-sonnet", True), ("gpt-4o", False)):
        layout = tt.PromptLayout(model=model, min_prefix_tokens=1)
        layout.static("system", "S" * 100).volatile("user", "V")
        content = layout.messages()[0]["content"]
        has_marker = isinstance(content, list) and "cache_control" in content[-1]
        assert has_marker is marked, model


@check("a prefix below the provider threshold is not marked")
def _l3():
    layout = tt.PromptLayout(model="anthropic/claude-3.5-sonnet", min_prefix_tokens=1024)
    layout.static("system", "short").volatile("user", "V")
    assert not layout.cacheable
    assert isinstance(layout.messages()[0]["content"], str)


# ---------------------------------------------------------------- budget


@check("a required section is never trimmed")
def _b1():
    budget = tt.Budget(total_tokens=50, task="t")
    budget.add("rules", "RULE " * 40, priority=10, required=True)
    budget.add("payload", "DATA " * 400, priority=0)
    out = budget.fit()
    assert out["rules"] == "RULE " * 40
    assert tt.count_text(out["payload"]) < tt.count_text("DATA " * 400)


@check("history trimming drops whole turns and never orphans a tool result")
def _b2():
    history = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "old " * 200},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
        {"role": "tool", "tool_call_id": "1", "content": "result " * 200},
        {"role": "user", "content": "latest"},
        {"role": "assistant", "content": "final"},
    ]
    kept = tt.trim_history(history, 40, keep_last=2)
    roles = [m["role"] for m in kept]
    assert roles[0] == "system"
    assert "tool" not in roles, roles
    assert kept[-1]["content"] == "final"


@check("an output cap needs enough samples to be a bound rather than a guess")
def _b3():
    assert tt.recommend_max_tokens([100] * 5) is None
    rec = tt.recommend_max_tokens([100] * 40)
    assert rec and rec >= 100


# ---------------------------------------------------------------- minify


@check("compaction preserves meaning while removing formatting")
def _m1():
    import json

    rows = [{"name": "A", "email": None, "score": 0, "active": False}]
    pretty = json.dumps(rows, indent=2)
    tight = tt.minify_rows(rows)
    assert tt.count_text(tight) < tt.count_text(pretty)
    assert "false" in tight and '"score":0' in tight, tight


@check("dropped rows are announced, never silently removed")
def _m2():
    rows = [{"i": i} for i in range(50)]
    out = tt.minify_rows(rows, max_rows=5)
    assert "more rows not shown" in out


# ---------------------------------------------------------------- batching


@check("batch results are matched by index, never by position")
def _t1():
    parsed = {"results": [{"index": 2, "score": 30}, {"index": 0, "score": 10}]}
    out = tt.split_results(parsed, 3)
    assert out[0]["score"] == 10 and out[1] is None and out[2]["score"] == 30


@check("a result with no index is discarded rather than guessed at")
def _t2():
    out = tt.split_results({"results": [{"score": 10}, {"score": 20}]}, 2)
    assert out == [None, None]


@check("duplicate embedding inputs are paid for once")
def _t3():
    seen = []

    def embed(batch):
        seen.extend(batch)
        return [f"v:{b}" for b in batch]

    out = tt.EmbeddingBatcher(embed, batch_size=10, ledger=Ledger()).run(
        ["a", "b", "a", "", None, "b"]
    )
    assert sorted(seen) == ["a", "b"], seen
    assert out == ["v:a", "v:b", "v:a", None, None, "v:b"]


# ---------------------------------------------------------------- ledger


@check("a cache hit is counted as a hit, not as a successful model call")
def _g1():
    t = _tuner()
    t.run(lambda: {"text": "a", "tokens_in": 900, "tokens_out": 50}, **CALL)
    t.run(lambda: {"text": "a", "tokens_in": 900, "tokens_out": 50}, **CALL)
    snap = t.ledger.snapshot()
    assert snap["cache"]["hits"] == 1 and snap["cache"]["misses"] == 1
    assert snap["saved"]["tokens_est"] == 950


# ---------------------------------------------------------------- analyzer


@check("the analyzer stays quiet below a usable sample size")
def _r1():
    from tokentuner.report import analyze

    rows = [{"task": "t", "tokens_in": 9000, "tokens_out": 10, "escalated": True,
             "ok": True, "cost_usd": 1.0}] * 5
    kinds = {f["kind"] for f in analyze(rows)["findings"]}
    assert "mis_tiered" not in kinds and "prompt_heavy" not in kinds


@check("the analyzer flags a mis-tiered task once there is evidence")
def _r2():
    from tokentuner.report import analyze

    rows = [{"task": "t", "tokens_in": 100, "tokens_out": 100, "escalated": i % 2 == 0,
             "ok": True, "cost_usd": 0.01} for i in range(40)]
    kinds = {f["kind"] for f in analyze(rows)["findings"]}
    assert "mis_tiered" in kinds


def main() -> int:
    failures = 0
    for name, fn in CHECKS:
        try:
            fn()
            print(f"  ok    {name}")
        except AssertionError as e:
            failures += 1
            print(f"  FAIL  {name}\n          {e}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"  ERROR {name}\n          {type(e).__name__}: {e}")
    print(f"\n{len(CHECKS) - failures}/{len(CHECKS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
