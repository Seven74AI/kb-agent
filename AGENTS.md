# AGENTS.md

## Workflow (canonical: Seven74AI/agent-workflow)

- Non-negotiable: secrets never in context; never ship without a risk gate; read
  `docs/decisions/` before proposing a new primitive or surface.
- Track work: GitHub Issues + the "Development" board. `ready-for-agent` =
  implement unattended; `needs-info` / `ready-for-human` / `wontfix` = stop.
- ADRs: `docs/decisions/` — steering veto list.
- Process detail: `Seven74AI/agent-workflow` (`docs/domains/`, `docs/decisions/`).

## Repo gotchas

- Python 3.11: `/usr/local/lib/hermes-agent/venv/bin/python3`.
- yt-dlp: `--remote-components ejs:github --no-update`.
- Cookies: `/root/kb-agent/cookies` (uploaded via kba.sevend.io `/upload-cookies`).
- MinIO bucket: `knowledge-base` (the only bucket).
- Services: `kb-agent.service` (consumer) + `kb-mcp.service` (MCP, port 8002).

## Agent skills

### Issue tracker

GitHub Issues on `Seven74AI/kb-agent`. See `docs/agents/issue-tracker.md`.

### Triage labels

Default canonical labels (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: glossary is `CONTEXT.md` (repo root), ADRs are `docs/decisions/`. See `docs/agents/domain.md`.
