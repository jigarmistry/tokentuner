"""
Minimal end-to-end example: dropping tokentuner in front of a plain OpenAI call.

Run:  PYTHONPATH=. python examples/openai_basic.py   (from packages/tokentuner/)
"""
from dataclasses import asdict, dataclass
from typing import Optional

from tokentuner import PromptLayout, Tuner, TunerConfig, minify_document

# ---------------------------------------------------------------- 1. setup --
# Build one tuner at startup from your own settings. Module level, done once.
tuner = Tuner(config=TunerConfig(
    cache_ttl_seconds=3600,
    cache_store="memory",        # or "sqlite" / "redis" / "mongo"
    cache_max_temperature=0.35,
))


# ------------------------------------------------- 2. one result type -------
# THE RULE: whatever your runner returns on a miss, `from_entry` must return the
# same type on a hit. Callers cannot tell the two apart, so if the runner hands
# back a raw SDK response and from_entry hands back a string, every call site
# breaks the first time the cache warms up.
#
# The easy way to satisfy that is a small result type of your own.

@dataclass
class Answer:
    text: str
    tokens_in: int = 0
    tokens_out: int = 0
    cached: bool = False


def to_entry(answer: Answer) -> Optional[dict]:
    """What to store, or None to refuse this answer."""
    if not answer.text:
        return None                                  # nothing useful -> no store
    return asdict(answer)


def from_entry(entry: dict) -> Answer:
    """Rebuild the same type. Tokens are zeroed: a hit never paid for them."""
    return Answer(text=entry["text"], tokens_in=0, tokens_out=0, cached=True)


# ------------------------------------------------------------ 3. call site --
SYSTEM = """You are a precise analyst. Output only valid JSON.
Return {"sentiment": "positive"|"neutral"|"negative", "reason": "one sentence"}."""


def analyze(client, document: str, *, tenant: str) -> Answer:
    # Static instructions first, volatile document last, so the provider has a
    # reusable prefix. Documents go through minify_document, not text[:8000].
    layout = PromptLayout(model="gpt-4o-mini", task="analyze")
    layout.static("system", SYSTEM)
    layout.volatile("user", minify_document(document, max_tokens=2000))
    messages = layout.messages()

    def call() -> Answer:
        resp = client.chat.completions.create(
            model="gpt-4o-mini", messages=messages, temperature=0.1,
            response_format={"type": "json_object"},
        )
        usage = getattr(resp, "usage", None)
        return Answer(
            text=resp.choices[0].message.content,
            tokens_in=getattr(usage, "prompt_tokens", 0) if usage else 0,
            tokens_out=getattr(usage, "completion_tokens", 0) if usage else 0,
        )

    answer, meta = tuner.run(
        call,
        task="analyze",
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.1,
        want_json=True,
        tenant=tenant,          # scopes the key; omit only in single-tenant apps
        sensitive=True,         # prompt may carry personal data
        to_entry=to_entry,
        from_entry=from_entry,
    )
    print(f"cache={meta.cache:<5} deduped={meta.deduped}  -> {answer.text}")
    return answer


# ------------------------------------------------------------- 4. it works --
if __name__ == "__main__":
    import types

    calls = []

    class FakeClient:
        """Stands in for `openai.OpenAI()` so this example runs offline."""

        def __init__(self):
            self.chat = types.SimpleNamespace(completions=self)

        def create(self, **kw):
            calls.append(kw)
            msg = types.SimpleNamespace(content='{"sentiment":"positive","reason":"ok"}')
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=msg)],
                usage=types.SimpleNamespace(prompt_tokens=850, completion_tokens=20),
            )

    client = FakeClient()
    doc = "Great service.\n\n\n\n   Really happy with it.   \n\n\n"

    print("first call  : ", end=""); analyze(client, doc, tenant="acme")
    print("same again  : ", end=""); analyze(client, doc, tenant="acme")
    print("other tenant: ", end=""); analyze(client, doc, tenant="globex")

    print(f"\nupstream API calls: {len(calls)}  (3 call sites, 1 served from cache)")
    saved = tuner.ledger.snapshot()["saved"]
    print(f"tokens avoided: {saved['tokens_est']}, calls avoided: {saved['calls']}")
    print(f"hit rate: {tuner.ledger.snapshot()['cache']['hit_rate']:.0%}")
