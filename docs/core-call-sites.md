# Core migration — call-site inventory

Every location that implements (or partially implements) submit/detect/status/resume/search/note behavior and must switch to `core` during adapter refactors.

## Submit / enqueue

| Location | What it does | Target core API |
|----------|--------------|-----------------|
| `agent/webhook.py` — `submit()` | Full enqueue flow | `core.submit.submit_url` + `SubmitDependencies` wired to `agent.detection` / `agent.db` |
| `agent/webhook.py` — via `plan_submission` | Step names from pipeline registry | `agent.pipelines.get_pipeline_names` (single registry) |
| `agent/webhook.py` — `slugify()` | URL → slug | `core.util.slugify` (duplicate) |
| `run.py` | Does not submit URLs | — |
| `scripts/tdah_daily_check.py` | Hermes kanban tickets, not kb-agent queue | Out of scope (separate automation) |

**Pure helpers to relocate from webhook:**

- `slugify` → `core.util.slugify` ✓ (skeleton)
- Step resolution → `agent.pipelines.get_pipeline_names` via `core.submit.plan_submission` ✓

**Injected deps (stay in `agent` until moved):**

- `detect_platform` — `agent.detection`; regex copy in `core.detect` ✓
- `detect_content_subtype`, `preflight_health`, `dedup_check` — `agent.detection`

## Detect

| Location | What it does | Target core API |
|----------|--------------|-----------------|
| `agent/webhook.py` — `submit()` | `detect_platform`, `detect_content_subtype` | `core.detect.detect_type` + injected subtype |
| `agent/detection.py` | Platform regex, subtype fetch, health, dedup | Split: pure regex → `core.detect`; I/O helpers remain or move to `core/health.py` later |
| `agent/router.py` | Jev subtype routing | Called from `detect_content_subtype` (unchanged) |
| `agent/pipelines/threads.py` — `threads_detect` step | Runtime re-detection inside pipeline | May call `core.detect` after migration |
| `tests/test_router.py` | Jev routing tests | Keep; add `tests/test_core.py` for core detect |

## Status (list / get)

| Location | What it does | Target core API |
|----------|--------------|-----------------|
| `agent/webhook.py` — `dashboard()` | `get_all_tasks`, `get_task_steps`, counts, staleness | `core.status.list_tasks` |
| `agent/webhook.py` — `_format_elapsed`, `_compute_staleness` | Presentation helpers | `core.status.format_elapsed`, `compute_staleness` ✓ |
| `agent/db.py` — `get_all_tasks`, `get_task_steps`, `get_last_log_line` | SQLite reads | Used by core status layer (adapter passes `conn`) |
| `agent/consumer.py` | `claim_task`, `get_next_step`, step state updates | Execution path — not MCP status tools (future extraction) |

**Gap:** No HTTP route for single-task GET; MCP `get_task` will use `core.status.get_task` directly.

## Resume / retry

| Location | What it does | Target core API |
|----------|--------------|-----------------|
| `agent/webhook.py` — `resume()` | Reason pattern + `resume_tasks_by_reason` | `core.resume.resume_task` |
| `agent/webhook.py` — dashboard forms | `POST /resume?reason=` | Flask adapter only |
| `agent/db.py` — `resume_tasks_by_reason` | SQL update paused → pending | Duplicated in `core.resume.resume_task` ✓ |
| `agent/consumer.py` | In-step RETRY backoff, `pause_task`, `fail_task` | Operator **retry_task** (MCP) → `core.resume.retry_task` ✓ (new) |

## Search / read note

| Location | What it does | Target core API |
|----------|--------------|-----------------|
| `agent/detection.py` — `dedup_check` | grep `source_url:` in vault | Related to submit dedup; not full-text search |
| `agent/pipelines/see_also.py` — `_search_vault` | Keyword grep for wikilinks | Pipeline-internal; MCP `search_notes` → `core.notes.search_notes` |
| `agent/pipelines/see_also.py` — `step_see_also` | Read note + LLM | Synthesis pipeline; `get_note` is read-only for agents |
| Hermes / Obsidian plugins | N/A | — |

**New surface (no Flask route):**

- `core.notes.search_notes`
- `core.notes.get_note`

## Consumer-only (documented, not in MCP core tools)

| Location | Feature | Notes |
|----------|---------|-------|
| `agent/consumer.py` — `run()`, `_execute_task` | Task execution, retry delays | Stays in agent until `core/execution` ticket |
| `agent/consumer.py` — `_recover_orphans` | Crash recovery | Duplicated partially in `run.py` |
| `run.py` — `_recover_orphans` | Startup orphan recovery | Consolidate later |

## Test call sites

| Test module | Covers | After migration |
|-------------|--------|-----------------|
| `tests/test_webhook.py` | Flask routes, slugify, staleness | Import helpers from `core`; route tests unchanged |
| `tests/test_core.py` | Core API + import guard | Primary seam tests per PRD |
| `tests/test_consumer.py` | Step retry/pause | Unchanged (execution layer) |

## Migration order (recommended)

1. Wire `agent/webhook.submit` → `core.submit.submit_url` (deps from agent).
2. Wire `dashboard` → `core.status.list_tasks`.
3. Wire `resume` → `core.resume.resume_task`.
4. Point `agent.detection.detect_platform` at `core.detect.detect_platform` (delete duplicate regex).
5. Keep `slugify` in core only (webhook uses `plan_submission` for step names).
6. Add MCP adapter calling the same core functions.
