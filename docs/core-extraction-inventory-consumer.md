# Core extraction inventory — consumer paths

Inventory of **non-HTTP** logic for the six MCP/core features defined in
[`docs/prd-kb-mcp-consolidation.md`](prd-kb-mcp-consolidation.md): **submit**,
**detect**, **status**, **resume**, **search**, **note**. Companion doc (HTTP
adapters): `docs/core-extraction-inventory-http.md` (when present).

This ticket does **not** change production code. It records what must move into a
future `core/` layer so Flask, MCP, schedulers, and the async consumer call the
same functions.

---

## Scope: what counts as “consumer”

| Surface | Role |
|--------|------|
| `run.py` | Process entry: DB init, orphan recovery, starts background consumer thread |
| `agent/consumer.py` | Async queue worker: claim tasks, run pipeline steps, update task/step status, step retries |
| `agent/pipelines/*.py` | Step functions invoked **only** by the consumer (after a task exists) |
| `agent/quality.py` | Quality gates run inside pipeline steps |
| `agent/archival.py` | MinIO / Git steps run inside pipelines |
| `agent/pipelines/see_also.py` | Vault search during `see_also` step |
| `scripts/tdah_daily_check.py` | Cron-style scheduled job (legacy Hermes kanban path; future `watch_source`) |
| `scripts/rotate_logs.py` | Scheduled log archival (FTS on **logs**, not vault notes) |
| `scripts/batch_transcribe.py`, `modal/kb_transcribe.py` | Batch transcription helpers (no submit/detect/status API) |

**Out of scope for this doc (HTTP adapters):** `agent/webhook.py` (`POST /submit`,
`POST /resume`, `GET /dashboard`), though those paths duplicate orchestration that
consumer-side code will eventually share via `core`.

**Purity legend**

- **Pure** — deterministic, no I/O (suitable for unit tests without mocks).
- **IO** — filesystem, subprocess, network, SQLite, LLM.
- **Mixed** — mostly IO with small pure helpers.

---

## Target core API (reference)

From the PRD, adapters should call:

| Feature | Planned core entrypoints |
|---------|-------------------------|
| submit | `submit_url` (+ bulk helpers later) |
| detect | `detect_type` (platform + subtype + optional preflight/dedup) |
| status | `list_tasks`, `get_task` |
| resume | `resume_task`, `retry_task` |
| search | `search_notes` |
| note | `get_note` |

---

## 1. Submit (enqueue)

**Consumer:** does **not** accept new URLs. Tasks are created on the HTTP
`POST /submit` path (`agent/webhook.py` → `agent/db.create_task`). The consumer
only **claims** existing `pending` rows.

### 1.1 Indirect / planned consumer submit paths

| File | Symbol | Purity | Dependencies | Must switch to core |
|------|--------|--------|--------------|---------------------|
| `agent/pipelines/books.py` | `step_classify_content` | IO | Mega download dir, `_pdf_preflight`, JSON classification | When **delegate** creates child tasks, call `core.submit_url` / shared `enqueue_task` instead of inline `create_task` (not implemented yet; see FIXME on `delegate` step) |
| `agent/pipelines/books.py` | `Step(name="delegate", …)` | — | Re-runs `step_classify_content` | Replace with dispatcher that calls core enqueue per classified book |
| `scripts/tdah_daily_check.py` | `main()` | IO | curl, ffmpeg, Modal, MinIO, **Hermes kanban** | Future `watch_source` should call `core.submit_url` (Bunny/TDAH URLs) instead of Hermes tickets |

### 1.2 Shared enqueue primitive (used by HTTP; consumer will reuse)

| File | Symbol | Purity | Dependencies | Must switch to core |
|------|--------|--------|--------------|---------------------|
| `agent/db.py` | `create_task` | IO | SQLite `agent.db`, step name list | **`core.submit_url`** (or `core.enqueue_task`) — single implementation; webhook + mega delegate + watch cron |
| `agent/core/submit.py` | `plan_submission` → `get_pipeline_names` | Mixed | `agent.pipelines` registry | **`core.pipeline_steps_for(content_type)`** — already unified via registry; core submit re-exports |
| `agent/webhook.py` | `slugify` | Pure | — | **`core.slugify_url`** |
| `agent/detection.py` | `dedup_check` | IO | `grep` on vault | Part of **`core.submit_url`** pre-checks (also used on HTTP submit only) |
| `agent/detection.py` | `preflight_health` | IO | cookies, curl, `which megadl` | Part of **`core.submit_url`** (HTTP only) |
| `agent/detection.py` | `detect_platform`, `detect_content_subtype` | Mixed | `agent/router.route_jev`, legacy scrapers | Part of **`core.detect_type`** then submit |

### 1.3 Consumer call sites (claim only)

| File | Symbol | Line of responsibility | Must switch to core |
|------|--------|------------------------|---------------------|
| `agent/consumer.py` | `Consumer.run` → `claim_task` | Poll loop | No submit; stays adapter — but **`claim_task`** may move to `core` for MCP status symmetry |
| `agent/db.py` | `claim_task` | Atomic `pending` → `in_progress` | **`core.claim_next_task`** (internal worker API) |

---

## 2. Detect (content type / subtype)

**Ingress detect** (`detect_platform`, `detect_content_subtype`, Jev, preflight) runs
on **HTTP submit** only. **Consumer-side detect** re-classifies or validates during
pipeline execution.

### 2.1 Module: `agent/detection.py` (shared library; not invoked by consumer)

| Symbol | Purity | Dependencies | Consumer relevance |
|--------|--------|--------------|-------------------|
| `detect_platform` | Pure | regex `PLATFORM_PATTERNS` | Used by HTTP submit; consumer tasks already store `content_type` |
| `detect_content_subtype` | Mixed | `_fetch_content`, `route_jev`, legacy `_detect_*` | HTTP submit only |
| `_fetch_content`, `_detect_threads_subtype`, `_detect_substack_subtype`, `_detect_mega_subtype`, `_classify_mega_filenames` | IO / Mixed | curl, megadl | Fallback when Jev fails; HTTP path |
| `preflight_health`, `_check_*` | IO | cookies, Firecrawl, megadl | HTTP submit only |
| `dedup_check` | IO | vault grep | HTTP submit only |

**Must switch to core:** entire module → **`core.detect_type`** (+ optional
`core.preflight`, `core.dedup` helpers).

### 2.2 Pipeline steps (consumer-executed)

| File | Symbol | Purity | Dependencies | Must switch to core |
|------|--------|--------|--------------|---------------------|
| `agent/pipelines/threads.py` | `step_threads_detect` | IO | `_curl_threads`, `_has_video_versions`, `_is_login_wall` | **`core.detect_threads_post_type`** or fold into `detect_type` with live HTML signal |
| `agent/pipelines/threads.py` | `_has_video_versions`, `_is_login_wall`, `_curl_threads` | Mixed | cookies, curl | Helpers for detect step |
| `agent/pipelines/books.py` | `step_classify_content` | IO | filesystem, `_pdf_preflight`, `quality_score` | **`core.classify_mega_books`** (post-download, consumer-only) |
| `agent/pipelines/books.py` | `_pdf_preflight`, `quality_score`, `_quality_score` | Mixed | pymupdf | Book/PDF tier detection inside consumer |
| `agent/pipelines/youtube.py` | `step_extract_metadata` (bot branch) | IO | yt-dlp stderr heuristics | Consumer re-validates cookies/IP at run time — overlap with **`core.preflight`** |
| `agent/pipelines/youtube.py` | `_validate_cookies` | IO | cookie file | Duplicate of detection preflight — consolidate in core |
| `agent/pipelines/instagram.py` | `_is_music_only_reel` | Pure | char counts on segments | Post-transcription **content** detect (not URL type) |
| `agent/pipelines/instagram.py` | `step_ig_reel_merge` | Mixed | calls music-only detect | Keep detect helper in core domain |
| `agent/pipelines/web.py` | `step_substack_video_pause` | Pure | — | Explicit **subtype guard** — pauses unsupported substack video |
| `agent/pipelines/web.py` | Substack preloads / SSR helpers in extract/cleanup | IO | regex on HTML | Substack media detect during extract (see `step_firecrawl_extract` / cleanup) |
| `agent/pipelines/bunny_stream.py` | `step_extract_metadata`, UUID regex helpers | IO | curl, `BUNNY_UUID_RE` | Terragrandir / Bunny UUID detection at runtime |
| `agent/pipelines/books.py` | `_detect_boundaries_llm` | IO | LLM | **Chunk boundary** detect (note synthesis), not URL detect |

### 2.3 Router (Jev)

| File | Symbol | Purity | Dependencies | Must switch to core |
|------|--------|--------|--------------|---------------------|
| `agent/router.py` | `route_jev` | IO | OpenRouter HTTP | Called from `detect_content_subtype` (HTTP). **`core.detect_type`** owns Jev + fallback |

### 2.4 Scheduled / legacy detect

| File | Symbol | Purity | Dependencies | Must switch to core |
|------|--------|--------|--------------|---------------------|
| `scripts/tdah_daily_check.py` | slug discovery via regex on terragrandir HTML | IO | urllib | **`watch_source` diff/detect** → then `core.submit_url` |

---

## 3. Status (task / step visibility)

**Consumer** is the source of truth for **mutating** task/step state during
execution. **Reading** queue state for humans/agents is HTTP (`dashboard`).

### 3.1 Writes (consumer path)

| File | Symbol | Purity | Dependencies | Must switch to core |
|------|--------|--------|--------------|---------------------|
| `agent/db.py` | `claim_task` | IO | SQLite | Worker internal; expose read APIs separately |
| `agent/db.py` | `get_next_step` | IO | SQLite | Worker internal |
| `agent/db.py` | `update_step` | IO | SQLite | Worker internal |
| `agent/db.py` | `complete_task` | IO | SQLite | Worker internal |
| `agent/db.py` | `fail_task` | IO | SQLite | Worker internal |
| `agent/db.py` | `pause_task` | IO | SQLite | Worker internal |
| `agent/consumer.py` | `Consumer._execute_task` | IO | all DB fns above, `StepError` handling | Orchestration stays in consumer; **state transitions** → `core` task service |
| `agent/consumer.py` | `Consumer._recover_orphans` | IO | SQL + `/proc` PID check | Crash recovery — **`core.recover_in_progress_tasks`** |
| `run.py` | `_recover_orphans` | IO | duplicates consumer orphan logic | **Single core implementation** (dual path) |

### 3.2 Reads (HTTP; MCP `list_tasks` / `get_task`)

| File | Symbol | Used by consumer? | Must switch to core |
|------|--------|-------------------|---------------------|
| `agent/db.py` | `get_all_tasks` | No (dashboard) | **`core.list_tasks`** |
| `agent/db.py` | `get_task_steps` | No (dashboard) | **`core.get_task`** |
| `agent/db.py` | `get_last_log_line` | No (dashboard) | **`core.get_task_logs`** or adapter-only |
| `agent/streaming.py` | `log_progress`, `stream_to_logs` | Yes (steps) | Logging adapter; optional **`core.append_step_log`** |
| `agent/db.py` | `archive_llm_call` | Yes (LLM steps) | Archive adapter |

### 3.3 FTS schema (not queried in consumer code paths)

| File | Symbol | Notes |
|------|--------|-------|
| `agent/db.py` | `tasks_fts`, `logs_fts` | Defined in schema; **no Python queries** in repo yet. Future **`core.search_tasks`** / log search |
| `scripts/rotate_logs.py` | `rebuild_fts` | Maintains `logs_fts` after archival |

---

## 4. Resume / retry

Three distinct mechanisms:

1. **Operator resume** — paused → pending (HTTP).
2. **Step retry** — transient failures inside consumer loop.
3. **Crash resume** — orphan `in_progress` tasks/steps.

### 4.1 Operator resume (HTTP only)

| File | Symbol | Purity | Dependencies | Must switch to core |
|------|--------|--------|--------------|---------------------|
| `agent/db.py` | `resume_tasks_by_reason` | IO | SQLite `LIKE` on `error` | **`core.resume_task`** / bulk variant |
| `agent/webhook.py` | `resume()` route | IO | calls above | Thin adapter |

Consumer does **not** call `resume_tasks_by_reason`; it only picks up tasks once
status is `pending` again.

### 4.2 Step retry (consumer)

| File | Symbol | Purity | Dependencies | Must switch to core |
|------|--------|--------|--------------|---------------------|
| `agent/consumer.py` | `Consumer._execute_task` retry branch | IO | `config.max_retries`, `config.retry_delays`, `update_step` | **`core.retry_task_step`** policy or keep in worker; MCP **`retry_task`** should reset task/step to pending consistently |
| `agent/exceptions.py` | `ErrorType.RETRY`, `StepError` | Pure | — | Shared domain types in **`core.errors`** |
| `agent/quality.py` | `quality_gate_retry_loop` | IO | LLM re-read | Quality **synthesis** retry (not queue retry) — document under note pipeline |

### 4.3 Crash resume (consumer + run.py)

| File | Symbol | Must switch to core |
|------|--------|---------------------|
| `agent/consumer.py` | `Consumer._recover_orphans`, `_extract_pid_from_output`, `_is_process_alive` | **`core.recover_orphans(agent_conn)`** |
| `run.py` | `_recover_orphans`, `_is_process_alive` | Remove duplicate; call core once at startup |

---

## 5. Search (vault / notes)

MCP **`search_notes`** is vault-oriented. Consumer executes search during **See
Also** (and HTTP submit uses vault grep for dedup — see submit section).

### 5.1 Vault grep search (consumer)

| File | Symbol | Purity | Dependencies | Must switch to core |
|------|--------|--------|--------------|---------------------|
| `agent/pipelines/see_also.py` | `_search_vault` | IO | `grep -rl` on `Knowledge base/` | **`core.search_notes`** (keyword / grep backend) |
| `agent/pipelines/see_also.py` | `_extract_keywords` | Pure | heuristics on note text | **`core.note_search_keywords`** |
| `agent/pipelines/see_also.py` | `step_see_also` | IO | search + LLM | Step stays pipeline; call **`core.search_notes`** + **`core.get_note`** for candidates |
| `agent/detection.py` | `dedup_check` | IO | grep `source_url:` | **`core.search_notes`** or dedicated **`core.vault_has_source_url`** |

### 5.2 Log FTS (not vault notes)

| File | Symbol | Must switch to core |
|------|--------|---------------------|
| `scripts/rotate_logs.py` | `rebuild_fts` | Optional **`core.rebuild_log_index`** — separate from `search_notes` |

---

## 6. Note (read / write vault markdown)

MCP **`get_note`** is read-by-slug/path. Consumer **writes** notes in synthesis
steps and **reads** them in quality, See Also, MinIO, and Git steps.

### 6.1 Path resolution (duplicated across pipelines)

| File | Symbol / pattern | Vault subfolder |
|------|------------------|-----------------|
| `agent/pipelines/see_also.py` | `CONTENT_TYPE_FOLDERS`, `_get_folder` | maps `content_type` → folder |
| `agent/archival.py` | `CONTENT_TYPE_TO_VAULT_FOLDER` | duplicate mapping for git push |
| `agent/pipelines/bunny_stream.py` | `_note_path` | `Knowledge base/{slug}.md` (no subfolder) |
| `agent/pipelines/*.py` | inline `config.vault_path / "Knowledge base" / …` | per-type folders |

**Must switch to core:** **`core.note_path(content_type, slug)`** (single map).

### 6.2 Write (synthesis — consumer steps)

| File | Symbol | Must switch to core |
|------|--------|---------------------|
| `agent/pipelines/web.py` | `step_synthesize_web` | Pipeline stays; optional **`core.write_note`** helper |
| `agent/pipelines/youtube.py` | `step_synthesize_youtube` | Same |
| `agent/pipelines/books.py` | `step_synthesize_book`, `_chunked_book_synthesis` | Same |
| `agent/pipelines/threads.py` | `step_threads_text_synthesize`, `step_threads_video_synthesize` | Same |
| `agent/pipelines/instagram.py` | `step_ig_reel_synthesize`, `step_ig_carousel_synthesize` | Same |
| `agent/pipelines/bunny_stream.py` | `step_synthesize_note` | Same |
| `agent/pipelines/mega_video.py` | reuses `step_synthesize_youtube` | Same |
| `agent/pipelines/wikispooks.py` | reuses `step_synthesize_web` | Same |

### 6.3 Read / mutate (consumer steps)

| File | Symbol | Purpose | Must switch to core |
|------|--------|---------|---------------------|
| `agent/quality.py` | `quality_gate_web`, `quality_gate_book`, `quality_gate_youtube` | Read note for metrics | Gates in **`core.validate_note`** |
| `agent/quality.py` | `quality_gate_retry_loop` | Read/write note on LLM retry | Same |
| `agent/pipelines/see_also.py` | `step_see_also` | Read note + candidates | **`core.get_note`**, **`core.patch_note_section`** |
| `agent/pipelines/web.py` | `step_minio_upload` | Read note for frontmatter | **`core.get_note`** |
| `agent/pipelines/youtube.py` | `step_minio_upload`, `_update_note_source_files` | Read/write frontmatter | Same |
| `agent/pipelines/bunny_stream.py` | `_update_note_source_files` | Read/write frontmatter | Same |
| `agent/pipelines/instagram.py` | `step_ig_minio_upload` | Read/write frontmatter | Same |
| `agent/pipelines/threads.py` | `step_threads_minio_upload` | Read/write frontmatter | Same |
| `agent/archival.py` | `step_git_push` | Read note path for git add | **`core.get_note_path`** |

### 6.4 Templates (note content shape)

| Path | Used by (consumer) |
|------|---------------------|
| `templates/deep-treatment-template.md` | `web.py`, wikispooks |
| `templates/youtube-note-template.md`, `templates/resume-prompt.md` | `youtube.py`, mega_video |
| `templates/book-note-template.md` | `books.py` |
| `templates/tdah-summit-note-template.md` | `bunny_stream.py` |

Template loading should live in **`core.note_templates`**.

---

## 7. Consumer orchestration (cross-cutting)

| File | Symbol | Role | Core extraction |
|------|--------|------|-----------------|
| `agent/consumer.py` | `Consumer.startup` | DB init, import pipelines, `register_step`, orphan recovery | Keep as adapter; call **`core.init_worker`** |
| `agent/consumer.py` | `STEP_REGISTRY`, `register_step` | Global step name → fn | Prefer **`agent/pipelines` only** (already resolved per `content_type` in `_run_step`) |
| `agent/consumer.py` | `Consumer._run_step` | Resolve step from pipeline registry | Unchanged adapter |
| `agent/pipelines/__init__.py` | `PIPELINES`, `get_pipeline`, `Step` | Pipeline definitions | **`core.pipelines`** metadata vs executable steps split TBD |
| `run.py` | `_run_consumer` | Thread + event loop | Process adapter |

---

## 8. Call-site checklist — must switch to `core`

Every location below should become a thin wrapper over the planned core API (no
duplicated business logic across HTTP, MCP, cron, or consumer).

### submit / enqueue

- [ ] `agent/webhook.py` — `submit()` orchestration
- [ ] `agent/db.py` — `create_task`
- [ ] `agent/core/submit.py` — `plan_submission` (step names via registry), `slugify` in core
- [ ] `agent/pipelines/books.py` — future delegate → child tasks
- [ ] `scripts/tdah_daily_check.py` — replace Hermes kanban with `submit_url`

### detect

- [ ] `agent/detection.py` — all public detect/preflight/dedup exports
- [ ] `agent/router.py` — `route_jev`
- [ ] `agent/pipelines/threads.py` — `step_threads_detect` (+ helpers)
- [ ] `agent/pipelines/books.py` — `step_classify_content`, PDF preflight
- [ ] `agent/pipelines/youtube.py` — `_validate_cookies`, metadata bot detect
- [ ] `agent/pipelines/instagram.py` — music-only detect
- [ ] `agent/pipelines/web.py` — `step_substack_video_pause`, substack HTML detect
- [ ] `agent/pipelines/bunny_stream.py` — Bunny UUID / metadata detect

### status

- [ ] `agent/db.py` — `claim_task`, `get_next_step`, `update_step`, `complete_task`, `fail_task`, `pause_task`
- [ ] `agent/db.py` — `get_all_tasks`, `get_task_steps`, `get_last_log_line`
- [ ] `agent/consumer.py` — `_recover_orphans` (and dedupe with `run.py`)
- [ ] `run.py` — `_recover_orphans`

### resume / retry

- [ ] `agent/db.py` — `resume_tasks_by_reason`
- [ ] `agent/webhook.py` — `resume()` route
- [ ] `agent/consumer.py` — step retry policy in `_execute_task` (align with MCP `retry_task`)

### search

- [ ] `agent/pipelines/see_also.py` — `_search_vault`, `_extract_keywords`
- [ ] `agent/detection.py` — `dedup_check`

### note

- [ ] `agent/pipelines/see_also.py` — `CONTENT_TYPE_FOLDERS` + read/write in `step_see_also`
- [ ] `agent/archival.py` — `CONTENT_TYPE_TO_VAULT_FOLDER`, `step_git_push`
- [ ] All `step_synthesize_*` / `step_*_synthesize` in pipelines (write)
- [ ] All quality / minio / density helpers that `read_text` / `write_text` notes
- [ ] `agent/quality.py` — all `quality_gate_*` and `quality_gate_retry_loop`

---

## 9. Gaps and duplicates (extraction risks)

1. **Dual orphan recovery** — `run.py::_recover_orphans` and
   `Consumer._recover_orphans` overlap; consumer version is stricter (resets task
   with no running step). Consolidate in core before MCP exposes resume/status.

2. **Pipeline step lists** — HTTP submit uses `plan_submission` →
   `get_pipeline_names` from the `agent/pipelines` registry (no separate webhook dict).

3. **Dual vault folder maps** — `see_also.CONTENT_TYPE_FOLDERS` vs
   `archival.CONTENT_TYPE_TO_VAULT_FOLDER`.

4. **No consumer submit** — bulk/mega delegate and `watch_source` will add
   consumer-side enqueue; plan for **`core.submit_url`** first.

5. **FTS unused for tasks/notes** — vault search is grep-only in consumer; MCP
   `search_notes` may need richer indexing later.

---

## 10. Related tests (guard behavior during extraction)

| Area | Test module |
|------|-------------|
| Consumer loop, orphans, retries | `tests/test_consumer.py` |
| DB resume / tasks | `tests/test_webhook.py` |
| Detect / Jev | `tests/test_router.py`, pipeline detect tests |
| See Also / vault | Covered indirectly via pipeline tests |
| Log FTS rotation | `tests/test_rotate_logs.py` |

When extracting `core`, port these to target **`core.*`** directly per PRD testing
decisions.

---

*Generated for kanban ticket: Inventory consumer extraction points.*
