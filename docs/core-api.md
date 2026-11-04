# Core API contract

Pure domain layer for kb-agent (`core/` package). Adapters:

- **Flask** — `agent/webhook.py` (human dashboard + `POST /submit`, `POST /resume`)
- **MCP** — `mcp_server.py` / `agent.mcp` (stateless streamable HTTP; see PRD)

**Constraints:** No Flask, FastMCP, or MCP SDK imports inside `core/`. Side effects (SQLite, grep, curl, LLM) are injected or isolated in module functions that adapters wire to production dependencies.

## Module map

| Module | MCP tool(s) | Responsibility |
|--------|-------------|----------------|
| `core/detect.py` | `detect_type` | Regex platform detection; optional subtype via injected detector |
| `core/submit.py` | `submit_url` | Enqueue URL (dedup → health → subtype → steps → task row) |
| `core/status.py` | `list_tasks`, `get_task` | Read task queue + optional log enrichment |
| `core/resume.py` | `resume_task`, `retry_task` | Bulk resume by pause reason; single-task retry |
| `core/notes.py` | `search_notes`, `get_note` | Vault grep search + note load |
| `core/pipelines.py` | (internal) | Content-type → ordered pipeline step names |
| `core/util.py` | (internal) | `slugify` |
| `core/types.py` | — | Dataclasses / enums returned to adapters |
| `core/watch_source.py` | `watch_source` | Register/list/poll recurring page watches; diff + auto-submit |

Bulk tools (`ingest_playlist`, `bulk_import`) will compose `submit_url` in later tickets.

**`watch_source`** (`agent/core/watch_source.py`): register a listing page (URL + regex +
`{item}` URL template + cadence), poll on a scheduler loop, diff against `last_seen_json`,
and call `submit_url` for new items. Cron one-shot: `scripts/watch_sources_poll.py`.
The consumer runs the in-process scheduler by default; set `watch_scheduler_enabled: false`
(config.yaml / `KB_WATCH_SCHEDULER_ENABLED=false`) when cron drives polling instead, so the
two pollers don't enqueue duplicate tasks.

## Functions

### `detect_platform(url: str) -> str`

Deterministic regex classifier. No network I/O.

**Returns:** `youtube`, `threads`, `instagram_reel`, `instagram_post`, `substack`, `mega`, `bunny_stream`, `wikispooks`, `book`, or `web`.

### `detect_type(url, *, subtype_detector=None) -> DetectTypeResult`

- Always runs `detect_platform`.
- If `subtype_detector` is provided, `content_type = subtype_detector(url, platform)` (today: `agent.detection.detect_content_subtype`).
- If omitted, `content_type == platform` (cheap probe for agents that only need platform).

### `submit_url(conn, url, *, vault_path, deps) -> SubmitResult`

**Dependencies (`SubmitDependencies`):**

| Field | Current implementation |
|-------|------------------------|
| `dedup_check` | `agent.detection.dedup_check` |
| `preflight_health` | `agent.detection.preflight_health` |
| `detect_content_subtype` | `agent.detection.detect_content_subtype` |
| `create_task` | `agent.db.create_task` |

**Outcomes (`SubmitOutcome`):**

| Outcome | HTTP (today) | Fields |
|---------|--------------|--------|
| `SKIPPED` | 200 | `reason` (vault dedup) |
| `PAUSED` | 200 | `task_id`, `reason` (health check) |
| `CREATED` | 201 | `task_id`, `content_type`, `slug` |

### `list_tasks(conn, *, limit=100, logs_conn=None, now=None) -> TaskListResult`

Returns tasks (newest first), per-status counts, and optional `last_log` / `elapsed` / `staleness` when `logs_conn` is set.

### `get_task(conn, task_id, *, logs_conn=None, now=None) -> TaskRecord | None`

Single task + steps; same optional enrichment as `list_tasks`.

### `resume_task(conn, reason_pattern) -> ResumeResult`

SQL LIKE match on `tasks.error` for `status='paused'`. Reason normalization via `normalize_resume_reason_pattern` (Flask query param parity).

### `retry_task(conn, task_id) -> RetryResult`

Re-queue one `paused` or `failed` task; reset failed/paused/running steps to `pending`.

### `search_notes(vault_path, query, *, limit=20) -> NoteSearchResult`

Grep under `{vault}/Knowledge base/**/*.md`.

### `get_note(vault_path, slug_or_path) -> NoteContent | None`

Resolve slug across standard KB folders or load by vault-relative path.

## Pure vs I/O-bound inventory

| Function | Pure? | External deps |
|----------|-------|---------------|
| `detect_platform` | yes | — |
| `detect_type` (no detector) | yes | — |
| `detect_content_subtype` | no | curl, megadl, Jev (`agent.detection`) |
| `preflight_health` | no | cookie files, curl, `which megadl` |
| `dedup_check` | no | grep vault |
| `slugify` | yes | — |
| `resolve_pipeline_steps` | yes | — |
| `submit_url` | orchestration | via `SubmitDependencies` |
| `list_tasks` / `get_task` | no | SQLite |
| `format_elapsed` / `compute_staleness` | yes* | *needs timestamps from DB |
| `resume_task` / `retry_task` | no | SQLite |
| `search_notes` / `get_note` | no | grep / filesystem |

## Consumer boundary

The async **consumer** (`agent/consumer.py`) executes pipeline steps; it does not expose MCP tools. Future core extraction may add `core/execution.py` for claim/step orchestration. This ticket covers **agent-facing** submit/detect/status/resume/search/note only.
