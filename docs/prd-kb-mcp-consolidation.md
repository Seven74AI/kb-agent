# PRD — Consolidate KB onto kb-agent + expose as stateless MCP

## Problem Statement

The KB ingestion workflow is split across two overlapping systems: the Hermes-based
pipeline (fragile regex/scrape routing, manual URL submission, driven via skills +
kanban) and the kb-agent (18 pipelines + Jev routing, but a separate empty A/B vault and
no standard agent interface). Consequences: manual URL submission; no way for remote
agents (Cursor cloud, Claude) to drive the KB; fragmented automation; two vaults, so
consolidation is deferred indefinitely.

## Solution

Consolidate on the kb-agent and expose it as a standard tool surface:

1. Refactor into a core/adapter split — pure domain logic in a `core` layer, thin Flask
   (human dashboard) + MCP (agent tools) adapters over it.
2. Ship a stateless MCP server (spec 2026-07-28, streamable HTTP) exposing 11 tools,
   gated by Cloudflare Access service tokens (per-client).
3. Point the kb-agent at the main vault; retire the separate A/B vault.
4. Keep the Hermes pipeline alive only for a note-quality A/B comparison window; retire
   it after validation.

## User Stories

1. As a remote agent (Cursor cloud / Claude), I want to submit a URL via MCP, so I can
   ingest content without a human pasting it into a form.
2. As an agent, I want to detect a URL's content type without ingesting, so I can
   classify before committing to a pipeline.
3. As an agent, I want to list tasks and get task status, so I can see queue state and
   progress.
4. As an agent, I want to resume or retry a paused/failed task, so I can drive the
   PAUSE/RETRY model without the dashboard.
5. As an agent, I want to search the vault and read a note, so I can answer questions
   from the KB.
6. As an agent, I want to ingest a whole playlist/collection, so I can bulk-ingest
   without N manual calls.
7. As an agent, I want to bulk-import a list of URLs, so I can batch-ingest in one call.
8. As an agent, I want to watch a recurring source, so new content is auto-ingested
   without re-triggering.
9. As the operator (Matt), I want the web dashboard to keep working, so I can watch
   progress and resume paused tasks by hand.
10. As Matt, I want new notes to land in my existing vault, so my KB history stays in
    one place.
11. As Matt, I want to compare kb-agent notes against Hermes notes on the same URLs, so
    I can validate quality before retiring Hermes.
12. As Matt, I want per-client access tokens, so I can revoke one agent without
    affecting others.

## Implementation Decisions

- **Architecture — core/adapter split.** Extract enqueue/detect/status/resume/search/
  note logic into a pure `core` layer (no Flask/MCP imports). `web.py` (Flask: webhook
  + dashboard) and `mcp_server.py` (MCP: tools) become thin adapters over it. New
  interface = new adapter; core unchanged.
- **MCP spec — 2026-07-28, stateless.** No handshake/session; each request
  self-contained. Streamable HTTP transport; requests carry `MCP-Protocol-Version`,
  `Mcp-Method`, `Mcp-Name` headers. Build on the current Python SDK (FastMCP 4.0). Do
  NOT adopt the deprecated legacy HTTP+SSE transport or Roots/Sampling/Logging.
- **Tool surface — 11 tools.** Core: `submit_url`, `detect_type`, `list_tasks`,
  `get_task`, `resume_task`, `retry_task`, `search_notes`, `get_note`. Bulk:
  `ingest_playlist`, `bulk_import`, `watch_source`.
- **Stateless + tasks.** `submit_url` returns a task handle; `list_tasks`/`get_task`
  poll it. Long operations (transcription) run in the existing async consumer, not a
  held-open session.
- **watch_source** is a new subsystem: a scheduler loop + per-source state (URL,
  cadence, last-seen) + diff → auto-submit. Generalizes the existing TDAH daily-check
  cron.
- **Security — Cloudflare Access service tokens.** `mcp.sevend.io` added to the
  existing `sevend-io-stream` tunnel; own Access app with a Service Token policy (one
  token per consumer: Cursor cloud, Mac). Clients send `CF-Access-Client-Id` +
  `CF-Access-Client-Secret` headers. HTTP-layer auth, orthogonal to MCP-native OAuth.
- **Vault — main vault.** Point the kb-agent at the production vault
  (`Seven74AI/obsidian-vault`); retire the separate A/B vault. Structure already
  identical, so this is a config pointer, not a migration.
- **Benchmark — note-quality A/B.** Keep the Hermes pipeline + its separate vault alive
  for a comparison window: feed the same URL set to both systems, compare notes
  (quality-gate metrics + manual review), retire Hermes only when kb-agent matches/beats
  it.

## Testing Decisions

- **One seam — the core layer.** Test pure functions directly (external behavior, not
  internals).
- `detect_type` → the existing 395-URL benchmark set.
- `submit_url` / `list_tasks` / `get_task` / `resume_task` / `retry_task` → a temp
  SQLite.
- `search_notes` / `get_note` → a fixture vault.
- `ingest_playlist` / `bulk_import` → verify N× `submit_url` expansion.
- `watch_source` → verify polling/diff logic.
- **Adapters smoke-tested** — MCP exposes 11 tools, each routes to the right core
  function. Flask webhook + dashboard unchanged (existing tests).
- **Prior art** — the existing 297-test pytest suite (SQLite + fixture approach).

## Out of Scope

- Hermes pipeline retirement — after the A/B benchmark, as a separate decision.
- MCP-native OAuth / CIMD — gated at HTTP layer with CF Access service tokens.
- Horizontal scaling / multi-instance deployment.
- Per-tool CF Access authorization (one token per consumer initially).

## Further Notes

- Migration order: core extraction → MCP adapter → CF Access gate → vault pointer →
  benchmark window → (later) Hermes retirement.
