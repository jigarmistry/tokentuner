# Changelog

Consumers pin a tag (`pip install "git+https://github.com/jigarmistry/tokentuner.git@v0.1.0"`),
so every release that changes behaviour is listed here.

## v0.1.0

First release.

- `ResponseCache` — content-addressed, tenant-salted, TTL'd, with backends for
  memory (default), SQLite, Mongo and Redis.
- `SingleFlight` / `AsyncSingleFlight` — collapse identical calls that are in
  flight at the same moment.
- `PromptLayout` / `PrefixTracker` — static-before-volatile ordering, provider
  cache markers, and measurement of how much prefix real traffic repeats.
- `Budget`, `trim_history`, `trim_text` — token-aware trimming per section;
  whole conversation turns, never orphaned tool results.
- `minify_rows`, `minify_document`, `squeeze_text` — payload compaction that
  never touches instructions.
- `EmbeddingBatcher`, `chunk`, `split_results` — de-duplicated embedding
  batching, and batch results matched by echoed index rather than position.
- `Ledger` — counts what each mechanism actually saved.
- `report.analyze` plus a `tokentuner` CLI — ranked tuning recommendations from
  usage rows.
