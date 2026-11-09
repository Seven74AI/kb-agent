# KB Agent — Context & Glossary

A custom agent that replaces layers A (ingestion), B (orchestration), and D (synthesis)
of the Hermes-based knowledge base pipeline. Layers C (extraction) and E (archival)
remain unchanged — same tools, same scripts, same MinIO/Obsidian infrastructure.

---

## Architecture Overview

```
URL in (web form)
    │
    ▼
Flask webhook ──► SQLite agent.db (tasks)
    │
    ▼
Python consumer loop (async)
    ├── content_type='video'  → BoundedSemaphore(1), sequential
    └── content_type='text'   → parallel (N consumers)

              │
              ▼
         Pipeline steps (Python code, decorated for logging)
              │
              ├── Mechanical (Python functions)
              │     download, ffmpeg, diarize, transcribe, MinIO, git
              │     child process crash → consumer catches, RETRY
              │     consumer crash → orphan check on restart, resume step
              │
              └── Synthesis (LLM prompt-only calls)
                    source text embedded in prompt, note returned as text
                    │
                    ▼
              Quality gates (orchestrator)
                    ├── Programmatic: quote count, line count, coverage, size
                    ├── Structural review: LLM checks Content > Critique, completeness
                    └── Max 3 rounds, full re-read each retry
                    │
                    ▼
              See Also call (separate LLM call)
                    │
                    ▼
              Git push → Obsidian vault
```

---

## Core Decisions

### Scope
- **Rebuilding:** Ingestion (A), Orchestration (B), Synthesis (D)
- **Keeping:** Extraction tools (C — yt-dlp, faster-whisper, pyannote, Firecrawl, curl)
  and Archival (E — MinIO, Obsidian, Git)
- Same VPS, same scripts, same MinIO instance, same Firecrawl instance.

### Stack
- **Language:** Python, from scratch. No agent framework (LangChain, CrewAI, etc.).
- **LLM gateway:** Edgee (provider routing + prompt compression). Compression tested
  empirically on 2-3 books before committing. LLM backend is swappable — architecture
  does not depend on Edgee.
- **Web layer:** Flask (cosmetic — saves ~50 lines of boilerplate vs raw `http.server`).
  Two routes: POST /submit, GET /dashboard. Flask runs sync in main thread; the asyncio
  consumer runs in a background thread. No conflict at this scale (5 hits/day).
- **Database:** SQLite (WAL mode). Raw `sqlite3` module — built-in, zero dependencies.
  Blocking calls (~1ms) are invisible next to 8h diarization. Two files: `agent.db`
  (tasks + steps + config) and `logs.db` (step outputs + LLM archives). FTS5 for search.
- **Config:** YAML for structured config (paths, concurrency, thresholds, retry),
  `.env` for secrets (API keys). Same pattern as Hermes.
- **Auth:** Bearer `KB_API_TOKEN` (env or `config.yaml` `api_token`) on the webhook
  machine routes (`submit`) and MCP. `GET /dashboard` is NOT token-gated — it is
  protected by Cloudflare Access (email) at the `kba.sevend.io` hostname. The
  dashboard's own mutating forms (`resume`, `upload-cookies`, `remove-ingest`)
  accept EITHER the bearer token (curl/MCP) OR a verified Cloudflare Access JWT
  (`Cf-Access-Jwt-Assertion`, signature + `iss`/`aud` checked against the team's
  JWKS) PLUS a signed CSRF token the dashboard page embeds — the JWT is the
  browser's identity, the CSRF token blocks cross-site forgery. Access config:
  `CF_ACCESS_TEAM` (default `floral-surf-abe5`) + `CF_ACCESS_AUD` (the app's AUD
  tag). Webhook and MCP bind loopback by default (see `config.yaml`).

### Ingestion — Full Detection & Pre-Flight Model

Sourced from `knowledge-base/references/detection-model.md` (2026-06-18).

#### URL Detection Flow

```
URL submitted
  │
  ├── Platform detection (URL regex, deterministic)
  │     youtube.com/watch, youtu.be/, youtube.com/shorts,
  │       m.youtube.com, youtube.com/embed, music.youtube.com → youtube
  │     threads.net/@, threads.com/@                         → threads
  │     instagram.com/reel/                                  → instagram_reel
  │     instagram.com/p/                                     → instagram_post
  │     substack.com/                                        → substack
  │     mega.nz/                                             → mega
  │     player.mediadelivery.net/, terragrandir.com/         → bunny_stream
  │     wikispooks.org/                                      → wikispooks
  │     .epub, .pdf, .mobi, .azw3 (direct upload)           → book
  │     everything else                                      → web
  │
  ├── Dedup check: grep vault for source_url → skip if exists
  │
  ├── Pre-flight health (platform-specific)
  │     youtube:       Tier 1 (grep LOGIN_INFO+SAPISID+3PSID ≥3)
  │                    Tier 2 (curl oembed, IP fingerprinting)
  │                    Tier 3 (yt-dlp version, nightly vs stable)
  │     instagram:     grep -c sessionid ≥1
  │     threads:       cookies file exists + captions test on known post
  │     substack:      curl Firecrawl /scrape health
  │     web:           curl Firecrawl /scrape health
  │     → any fail = PAUSE with reason, do NOT create task
  │
  └── Content subtype (platforms with variants + regex misses)
        threads/substack/mega/web → Jev (OpenRouter decisions API, ~$0.00002/call)
           with fetched content signal; falls back to legacy scrape on Jev failure:
             threads:   curl + grep video_versions (media_type=19 handled)
             substack:  curl preloads JSON + SSR HTML fallback → TEXT/IMAGE/VIDEO
             mega:      megadl --print-names + extension heuristics
        instagram_post: always TEXT/IMAGE (carousel alt text extraction)
                        no video-in-carousel detection in current system
        See docs/decisions/001-jev-routing-cascade.md
```

#### Mega Link Detection

```
mega.nz link submitted
  │
  ├── mega.nz/file/  → check file metadata:
  │     ├── .zip archive → BOOK BATCH (see Book flow)
  │     ├── .mp4/.mkv/.webm → MEGA VIDEO (researcher-videos pipeline)
  │     └── unknown → download metadata, classify
  │
  └── mega.nz/folder/ → list contents → detect per file
```

#### Book Flow (Mega Archive)

```
Mega.nz/.zip link → megadl download → unzip to clean directory
  │
  ├── List all books (.epub, .pdf, .mobi, .azw3)
  │
  ├── Pre-flight per PDF (THREE TIER):
  │     Tier 1: pymupdf char count < 500 → SCANNED → queue OCR, do NOT ticket
  │     Tier 2: 500+ chars, quality_score(16 pages) < 80 → DEGRADED OCR
  │              (pre-19th-century PDFs: long-s→f, garbled headers)
  │              → queue for fresh OCR, do NOT ticket
  │     Tier 3: 500+ chars, quality_score(16p) ≥ 80 → GOOD TEXT → ticket
  │     epub/mobi/azw3 → always ticket (extractable text)
  │
  ├── Create one task per ticketed book
  │     Digital: extract → read → note → MinIO → git
  │     Scanned: marker-pdf 1 page at a time → OCR → concatenate → note
  │
  └── Scanned tasks chained AFTER all digital tasks
```

#### Post-Extraction Detection

- **Music-only Reels:** Detected after transcription — segments < 50 chars total.
  Use caption/metadata via Googlebot UA as primary content.
  Annotate: `⚠️ Music-only Reel — analysis based on caption text`.

#### Known Limitation

- **Substack video download:** Detection works, but download is NOT fully solved.
  Mux CDN requires signed JWT token. See `substack-video-investigation.md`.

### Queue & State Machine
- **Queue:** SQLite `tasks` table. Consumer polls every 5s.
- **Atomic claim:** `UPDATE ... WHERE id = (SELECT ... LIMIT 1) RETURNING *`. One
  query, no race condition. SQLite WAL mode.
- **Steps:** `task_steps` table with `step_order`, `status`, `output`. Crash → resume
  at the next `pending` step. No work lost.
- **Pipeline definitions:** Python code — lists of `Step` objects with `name`, `fn`,
  `on_error`. Decorated for auto-logging. No JSON in SQL, no YAML.
  Adding a new source = write a Python list + drop a template.
- **No Phase A/B split:** One task per URL, all steps sequential.

### Consumer Startup & Crash Recovery
- **First run:** Auto-create schema (`CREATE TABLE IF NOT EXISTS`).
- **Crash restart with orphan tasks:** Find `status='in_progress'` tasks. For each:
  find the `running` step. If a subprocess is still alive (diarize, yt-dlp, ffmpeg) →
  wait for it to finish. If no subprocess → re-run the step.
- **Child process crash:** Consumer catches exit code → RETRY with backoff.
  No orphan check needed (parent is alive).

### Concurrency
- **Single async process** (`asyncio`). Consumer runs in a background thread with its
  own event loop. Flask runs sync in the main thread — they don't share loop state.
- **Video:** `asyncio.BoundedSemaphore(1)` — one at a time (pyannote RAM). Sequential.
  `BoundedSemaphore` chosen over `Semaphore`: raises `ValueError` if `release()` is
  called too many times, catching bugs instead of silently breaking the limit.
- **Text:** Parallel, N consumers (configurable). No semaphore.
- **Semaphore:** A counter limiting concurrent access. BoundedSemaphore(1) = one at a
  time, with bug detection on release mismatch.

### Error Model
Three buckets — no per-step complexity:

| Bucket | Meaning | System action |
|--------|---------|---------------|
| **RETRY** | Transient, auto-recoverable | 5s → 15s → 60s → 3min → PAUSE after 5th |
| **PAUSE** | Needs operator (cookies, IP flag, persistent infra failure) | Store in queue with reason. Operator clicks "Resume" on dashboard. |
| **FAIL** | URL unfixable (deleted, 404, unsupported) | Mark failed, move to next. |

- **Skip queue:** Replaced by SQL `WHERE status='paused'`. No flat files.
- **Relaunch:** Web dashboard button per pause reason. `UPDATE tasks SET
  status='pending', retries=0 WHERE status='paused' AND reason LIKE '%cookies%'`.
  System re-validates cookies before processing.

### Pipeline — Mechanical vs LLM
- **Mechanical (Python only):** Cookie validation, metadata extraction, yt-dlp download,
  ffmpeg audio extraction, pyannote diarization, faster-whisper transcription,
  diarization-transcription merge, speaker identification, chapter generation,
  Firecrawl extraction, markdown cleanup, ePub extraction, MinIO upload, Git push.
  All deterministic shell commands or Python functions.
- **LLM synthesis calls:** YouTube two-pass resume (concepts → note),
  deep-treatment note, book note with chapter summaries, fact-checking.

### LLM Interface
- **Model:** Prompt-only interface — no tool calling. The LLM receives source text
  inline in the user prompt and returns synthesized notes as text. Source retrieval
  (Firecrawl, yt-dlp, ePub extraction) is done by Python before the LLM call. No
  terminal, no browser, no process management.
- **Source delivery:** Source text is embedded directly in the user prompt by Python.
  For content ≤50K tokens, the full source is sent in a single prompt. For large
  content, Python-side progressive chunking is used:
  chunk 1 → partial note → chunk 2 + previous note → append → ... → final
  polish call. Context stays clean (current chunk ~50K + growing note ~20K).
- **Chunk boundaries:** A separate, cheap LLM call (also prompt-only) determines
  semantic boundaries (not regex). This boundary-detection LLM receives a
  structural sample (beginning + end + section headers) and returns byte offsets.
  Fallback: if no boundary at 50K → expand to 100K. If still none → boundary LLM
  picks best boundary even if imperfect. The synthesis LLM receives chunks
  progressively, building the note as it goes. 10K token overlap between chunks.
- **Template:** Python injects template + quality rules into system prompt. Template
  varies by content type (YouTube resume-prompt, book-note-template, deep-treatment).

### Quality Gates (Orchestrator)
Two tiers, run AFTER synthesis, BEFORE git push:

**Tier 1 — Programmatic (Python, instant):**
- Book: line_count ≥ 100, quote_count ≥ 4, size ≥ 8K chars, fact_check_count ≥ 2,
  chapter_coverage (every source chapter → note section)
- YouTube: concept_count ≥ 3, key_points ≥ 8, density ≥ 100 chars/min
- Article: quote_count ≥ 4, size ≥ 5K chars, fact_check_count ≥ 2

**Tier 2 — Structural review (cheap LLM call):**
- Is the dominant section actually dominant (Content > Critique)?
- Are any template sections empty or placeholder?
- Does the note feel rushed or incomplete?
- Does NOT fact-check claims. Does NOT verify accuracy.

**Retry loop:** Max 3 rounds. Each retry = full source re-read. Orchestrator tells LLM
exactly what failed. LLM re-reads source, writes fresh note. "Never push v1" enforced
programmatically — if gates pass, push immediately.

**What we DON'T check:** No hallucination detection (LLM can't fact-check another LLM).
Accept factual content at face value, same as Hermes.

### See Also
- **NOT part of the main synthesis call.** Separate post-synthesis flow:
  1. LLM writes note (no See Also section).
  2. Orchestrator searches vault for related notes.
  3. Separate LLM call: "Here's the note. Here are candidate related notes. Add See Also."
- Eliminates ghost wikilinks — every candidate comes from vault search.

### Visibility
- **Level D — full logs.** Every step output stored in `logs.db`. Every LLM prompt and
  response archived. FTS5 searchable.
- **Website dashboard:** Queue status (pending, in_progress, done, paused, failed).
  Per-task step detail with elapsed time and output. Polling with configurable
  interval (5s / 30s / 1min), default 1min.
- **No push notifications.** Website is the single source of truth.

### Log Rotation
- Monthly archive to MinIO: dump `logs.db` rows older than 30 days → JSON → gzip →
  `mc cp kb-agent-archive/logs/YYYY-MM.json.gz`.
- After upload: `DELETE` old rows + `VACUUM`.
- Retention: indefinite on MinIO, 30 days active in DB.

### Process Management & Monitoring

- **No timeouts — anywhere.** Step durations are unpredictable: a 24h video means 24h
  diarization. A blanket timeout will inevitably kill legitimate work. Steps take as
  long as they take. The only exception is HTTP calls — `requests`/`httpx` have
  built-in timeouts (default ~30s), which is fine for cookie validation, Firecrawl, etc.
- **Streaming progress via stdout:** Consumer reads `proc.stdout.readline()` in a loop,
  writing each line to `logs.db` in real time. The canonical scripts (`diarize.py`,
  `transcribe.py`) already output timestamped progress with `flush=True`:
  - diarize: `[HH:MM:SS] Progress: {N} segments, {X}min elapsed` (every 300 segments)
  - transcribe: `[HH:MM:SS] Progress: {X}min / ~{Y}min ({Z}min elapsed, {W}x realtime)`
    (every 5min of audio)
  - yt-dlp/ffmpeg: built-in progress lines
- **Hang detection via staleness:** Dashboard shows the last progress line + timestamp.
  If a step hasn't produced output in hours, the operator sees the staleness and
  intervenes. No heuristic — visible staleness, human judgment.
  `proc.returncode is None` confirms the process is still alive.
- **`communicate()` NOT used.** It buffers all output until the process exits — useless
  for monitoring. `readline()` streams live.
- **Hard crash (OOM, SIGKILL):** `PR_SET_PDEATHSIG` in canonical scripts — kernel kills
  child when parent dies. No orphans. One line: `libc.prctl(1, signal.SIGKILL)`.
- **Graceful shutdown:** Consumer catches SIGTERM/SIGINT, waits for running steps to
  finish (or kills children via `PR_SET_PDEATHSIG`), then exits.
- **Python docs:** `asyncio.Process.wait()` has no timeout (confirmed via Context7).
  `proc.returncode` = `None` while alive. `PR_SET_PDEATHSIG` via `man 2 prctl`.

## Infrastructure Isolation

Both Hermes and KB Agent run on the same VPS but are fully isolated:

| Resource | Hermes | KB Agent |
|---|---|---|
| MinIO bucket | `knowledge-base/` | `kb-agent-archive/` |
| Obsidian vault | `OBSIDIAN_VAULT_PATH` / `KB_VAULT_PATH` | `/root/obsidian-vault/` (shared main vault) |
| Cookies | `/root/.hermes/cookies/` | `/root/kb-agent/cookies/` |
| Config | `~/.hermes/` | `/root/kb-agent/` |
| Firecrawl | `localhost:3002` (shared, stateless) | Same |

Cookie upload: web dashboard drag-and-drop. System applies `chmod 600` (owner
read/write only) and re-validates paused tasks for that platform.

## Deployment

- **Development:** tmux session for interactive debugging
- **Production:** systemd service once stable
- **Single VPS, single Python process**

## Book Processing
- Mega.nz link → `megadl` download → unzip → auto-create one task per book.
  No user confirmation step. Same extraction pipeline as current system
  (ebooklib + BeautifulSoup for ePub, pymupdf for PDF, marker-pdf for scanned).
- Temp path isolation: `/tmp/book_{slug}.epub`, `/tmp/book_{slug}_full.txt`.
  No shared `/tmp/book.epub` — parallel book tasks each get unique paths.

## Vault

- **Main production vault.** KB Agent writes to the same Obsidian Git repo as Hermes:
  `Seven74AI/obsidian-vault` at `/root/obsidian-vault/` (override with `KB_VAULT_PATH`
  or `OBSIDIAN_VAULT_PATH`). The separate `/root/kb-agent-vault/` A/B repo is retired —
  config rejects that path so new notes cannot land there.
- **Shared structure.** Same folder conventions (`articles/`, `videos/`, `books/`, etc.)
  and frontmatter (YAML: `source_url`, `content_type`, `date`, `tags`, MinIO links).

## Testing / Rollout
- **Two independent systems.** No automated mirroring. No same-URL-to-both.
- Operator chooses which system to use for any given URL.
- Hermes KB pipeline remains active until KB Agent is trusted.
- No formal cutover plan — retire Hermes KB pipeline when ready.

---

## Glossary

- **Task:** One URL to process. Lives in `agent.db` → `tasks` table. Has a `content_type`
  and a `status`. Moves through `task_steps`.
- **Task step:** One discrete pipeline operation (e.g., `download_video`, `diarize`,
  `synthesize_note`). Defined as a Python `Step` object. Has its own status, output,
  and error handling.
- **Consumer:** The async loop that atomically claims `status='pending'` tasks with
  `RETURNING` and executes steps. One process, concurrent via `asyncio`.
- **Pipeline:** An ordered list of `Step` objects for a content type. Defined in Python
  code. Extensible by writing a new list.
- **Content type:** Classification determined by a detector function. Examples:
  `youtube`, `threads_text`, `threads_video`, `instagram_reel`, `instagram_post`,
  `substack`, `book`, `web`.
- **Semaphore:** An `asyncio` counter that limits concurrent access to a resource.
  `BoundedSemaphore(1)` ensures only one video pipeline runs at a time, and raises
  `ValueError` if released more times than acquired (catches bugs).
- **Chunk:** A ~50K-token segment of source text, split at LLM-determined semantic
  boundaries. Used for large content where the note is built progressively.
- **Passe 1 / Passe 2:** YouTube two-pass synthesis. Passe 1 extracts concepts from
  the transcript. Passe 2 writes the full note using those concepts.
- **Deep treatment:** The mandatory note depth standard — ≥4 claims with direct quotes,
  section-by-section analysis as the dominant section, context, critical analysis,
  nuances, confidence level.
- **PAUSE:** Error state where a task waits for operator intervention. Not failed,
  not retrying. Resumed via "Resume" button on dashboard.
- **See Also:** The wikilinks section at the bottom of a note. Generated in a
  separate LLM call after the main note is written, using vault search results as
  candidates.
- **LLM synthesis:** Prompt-only LLM calls used for note generation. Source text
  is embedded directly in the prompt by Python; the LLM returns synthesized notes
  as text. No tool calling — Python handles all file I/O and source retrieval.
  No terminal, no browser, no process management. The LLM reads, thinks, writes.
- **RETURNING:** SQLite clause that makes UPDATE/INSERT/DELETE return the affected row.
  Enables atomic claim + read in one query. Eliminates race conditions on task claiming.
- **Orphan process:** A subprocess (diarize.py, yt-dlp) that survived after the
  consumer process crashed. Detected on restart by checking if the PID is still alive.

---

**All 15 fuzzy spots resolved.** Ready for implementation planning.

**Context7 MCP:** Configured (6 tools: query_docs, resolve_library_id, read_resource, list/get prompts + resources). GitHub MCP enabled (26 tools) for PR/issues/review.
