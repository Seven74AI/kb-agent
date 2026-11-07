# Core extraction inventory — Flask webhook layer

Inventory of **submit**, **detect**, **status**, **resume**, **search**, and **note** behavior as implemented in the Flask webhook (`agent/webhook.py`) and symbols it calls directly. Intended input for the core/adapter split described in `docs/prd-kb-mcp-consolidation.md`.

**Scope:** `agent/webhook.py` routes and module-level helpers, plus `agent/detection.py` / `agent/db.py` entry points invoked from the webhook (not consumer or pipeline code).

**Purity:** *Pure* = deterministic from inputs, no I/O or globals. *Impure* = I/O, globals, side effects, or Flask request/response coupling.

---

## Feature inventory

| Feature | File | Symbol | Purity | Dependencies | Call Sites (webhook layer) |
|--------|------|--------|--------|--------------|----------------------------|
| submit | `agent/webhook.py` | `submit` | Impure | Flask `request.form`, `jsonify`; `config` (vault path); SQLite via `_get_agent_conn`; `core.submit.plan_submission`; `agent.db.create_task` via `submit_plan` | `@app.route("/submit")` — route entry |
| submit | `agent/core/submit.py` | `plan_submission` | Mixed | `agent.detection.*`; `agent.pipelines.get_pipeline_names`; `slugify` in core | `submit()` — detect + step list after subtype detection |
| submit | `agent/core/submit.py` | `submit_plan` | Impure | SQLite `create_task` | `submit()` — persists planned steps |
| submit | `agent/webhook.py` | `_get_agent_conn` | Impure | `config.kb_agent_dir`; SQLite; `AGENT_SCHEMA` | `submit()` (paused + pending paths), `resume()`, `dashboard()` |
| submit | `agent/webhook.py` | `_agent_db_path` | Impure | `config.kb_agent_dir` | `_get_agent_conn()` |
| submit | `agent/db.py` | `create_task` | Impure | SQLite connection; UUID | `submit()` — creates task + `task_steps` rows |
| detect | `agent/webhook.py` | `submit` | Impure | (orchestrates detect; see submit row) | Same route; steps 1–5 before/at task creation |
| detect | `agent/detection.py` | `detect_platform` | Pure | stdlib `re`; `PLATFORM_PATTERNS` | `submit()` step 1 |
| detect | `agent/detection.py` | `dedup_check` | Impure | Filesystem vault (`config.vault_path`); subprocess `grep` | `submit()` step 2 |
| detect | `agent/detection.py` | `preflight_health` | Impure | Platform string; cookie files; subprocess `curl`/`which`; localhost Firecrawl | `submit()` step 3 |
| detect | `agent/detection.py` | `detect_content_subtype` | Impure | URL + platform; may call `_fetch_content` (curl/megadl), `route_jev`, legacy subtype helpers | `submit()` steps 4 and paused-path step 3 |
| status | `agent/webhook.py` | `dashboard` | Impure | Flask `render_template_string`; SQLite; `config`; datetime | `@app.route("/dashboard")` |
| status | `agent/webhook.py` | `index` | Impure | Flask `redirect`, `url_for` | `@app.route("/")` → `dashboard` |
| status | `agent/webhook.py` | `_format_elapsed` | Pure | `datetime` | `dashboard()` — per-task elapsed string |
| status | `agent/webhook.py` | `_compute_staleness` | Impure | SQLite `logs.db` (`step_logs` query) | `dashboard()` — hang/staleness display |
| status | `agent/webhook.py` | `_get_logs_conn` | Impure | `config.kb_agent_dir`; SQLite; `LOGS_SCHEMA` | `dashboard()` |
| status | `agent/webhook.py` | `_logs_db_path` | Impure | `config.kb_agent_dir` | `_get_logs_conn()` |
| status | `agent/db.py` | `get_all_tasks` | Impure | SQLite | `dashboard()` |
| status | `agent/db.py` | `get_task_steps` | Impure | SQLite | `dashboard()` |
| status | `agent/db.py` | `get_last_log_line` | Impure | SQLite (`logs.db`) | `dashboard()` |
| status | `agent/webhook.py` | `submit` | Impure | Flask `jsonify` — response fields `status`, `reason`, `task_id`, etc. | JSON status on create/skip/pause (not a dedicated status API) |
| status | `agent/webhook.py` | `resume` | Impure | Flask `jsonify` — `status`, `resumed` | JSON acknowledgment after resume |
| status | `agent/webhook.py` | `DASHBOARD_HTML` | Pure (template string) | Jinja placeholders; embedded forms POST `/resume`, `/upload-cookies` | Rendered by `dashboard()` |
| status | `agent/webhook.py` | `PAUSE_REASON_LABELS` | Pure (data) | None | Passed into dashboard template for resume buttons |
| resume | `agent/webhook.py` | `resume` | Impure | Flask `request.args`, `jsonify`; SQLite via `_get_agent_conn` | `@app.route("/resume")`; also targeted by dashboard HTML forms |
| resume | `agent/db.py` | `resume_tasks_by_reason` | Impure | SQLite (`UPDATE tasks … LIKE ?`) | `resume()` |
| search | — | *(no webhook implementation)* | — | Planned core: vault / `tasks_fts` / `logs_fts` (FTS5 defined in `agent/db.py` schemas, unused by webhook) | No routes, helpers, or DB queries for vault search in `agent/webhook.py` |
| note | — | *(no webhook implementation)* | — | Note read/write lives in async consumer + `agent/pipelines/*` (e.g. `synthesize_note` steps); webhook enqueues step names from `get_pipeline_names` | No `get_note` / vault markdown read in webhook; dashboard shows `last_log` text, not note body |

---

## Flask coupling points (webhook module)

All locations where domain logic is tied to Flask or HTTP, to be thinned in a core/adapter split.

| Location | Coupling |
|----------|----------|
| Module init | `app = Flask(__name__)`; `config = load_config()` global |
| `index` | `redirect(url_for("dashboard"))` |
| `submit` | `request.form.get("url")`; HTTP status codes 400/200/201; `jsonify(...)` |
| `resume` | `request.args.get("reason")`; `jsonify(...)` |
| `upload_cookies` | `request.form`, `request.files`; `jsonify(...)` — operator utility, not one of the six MCP core tools |
| `dashboard` | `render_template_string(DASHBOARD_HTML, ...)` — HTML UI |
| `DASHBOARD_HTML` | Inline `<form method="POST" action="/resume?reason=…">` and `/upload-cookies` |
| `if __name__ == "__main__"` | `app.run(host=config.webhook_host, port=config.webhook_port)` |

**Not Flask but adapter-adjacent:** `_get_agent_conn` / `_get_logs_conn` duplicate connection bootstrapping that core should own; `config` module global read throughout.

---

## Gaps vs planned MCP core tools

| Planned core capability | Webhook coverage |
|-------------------------|-------------------------|
| `submit_url` | `POST /submit` (+ inline detect/dedup/health) |
| `detect_type` | Embedded only inside `submit`; no standalone detect endpoint |
| `list_tasks` / `get_task` | Partial — `GET /dashboard` lists up to 100 tasks in HTML; no JSON task API or single-task GET |
| `resume_task` | `POST /resume` by reason pattern (batch), not per `task_id` |
| `retry_task` | Not in webhook (consumer/DB only) |
| `search_notes` | Not in webhook |
| `get_note` | Not in webhook |

---

## Related files (out of webhook scope)

| Area | Location |
|------|----------|
| Async queue consumer | `agent/consumer.py` |
| Note synthesis & vault writes | `agent/pipelines/*.py` |
| Vault See Also search | `agent/pipelines/see_also.py` |
| Detection implementation detail | `agent/detection.py` (full module) |
| Task persistence | `agent/db.py` |
