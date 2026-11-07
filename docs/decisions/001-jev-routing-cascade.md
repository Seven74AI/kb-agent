# 001. Jev Routing Cascade for Content-Type Detection

## Status
Accepted — 2026-11-04

## Context
Content-type routing was fully deterministic: URL regex for the platform, then
scrape-and-parse for subtypes (threads video/text via `video_versions`,
substack text/image/video via `window.__preloads` + SSR markers, mega via
`megadl --print-names` + extension heuristics). This was fragile — 44% accuracy
on ambiguous/hard cases — and depended on brittle markers and live scraping.

We benchmarked alternatives on 395 real KB URLs (URL → content_type) plus a
synthetic hard-case set:

| Router | Clean | Hard/ambiguous |
|---|---|---|
| regex + scrape (prior) | 100% | 44% |
| Laya zero-shot (self-hosted) | 75% | 56% |
| DeepSeek v4-pro (frontier LLM) | 100% | 78% |
| **Jev 1.14 (decision model)** | 100% | **100%** |

Jev is TypeSafe's dedicated decision model, called via the OpenRouter decisions
API. On the 395 real URLs it scored 389/395 (98.5%); the 6 misses are dataset
label noise (Mega local files inconsistently tagged `web`), so effective
accuracy is ~100%. Cost ~$0.00002/call, ~200 ms latency, no reasoning tokens.

## Decision
Adopt a cascade: keep the deterministic URL-regex platform detector as the
fast, free first pass; route the ambiguous tail through Jev.

- `detect_platform` (regex) — unchanged. Clean platforms (`youtube`,
  `instagram_reel`, `instagram_post`, `bunny_stream`, `wikispooks`, `book`)
  return as-is.
- `detect_content_subtype` — for `threads`/`substack`/`mega`/`web`, fetch the
  content signal and classify with Jev (`agent/router.py::route_jev`).
- Fallback — if Jev fails (no `OPENROUTER_API_KEY`, API error, timeout), the
  legacy scrape-and-parse functions run unchanged.

## Consequences

**Positive:** ambiguous-tail accuracy ~100% (was 44%); cheap (~$0.0002/month
at current volume); fast (~200 ms); isolated in `agent/router.py`.

**Negative:** the ingest path depends on a hosted API (OpenRouter) and an
`OPENROUTER_API_KEY`. Mitigated by the regex fast path (most URLs never call
Jev) and the legacy fallback (Jev outage degrades to the old behavior, not a
hard failure).

**Rejected alternatives:**
- Self-hosted Laya — 56% zero-shot; would need fine-tuning on our data to even
  approach Jev, uncertain + more moving parts.
- DeepSeek v4-pro — 78% on hard cases, slower and more expensive than Jev.
- Pplx-Decider-27B — private HF repo, ~49 GB VRAM, A100-only; not runnable at
  our scale.
