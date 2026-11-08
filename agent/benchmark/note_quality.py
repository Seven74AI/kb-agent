"""Note-quality A/B comparison — quality-gate metrics and side-by-side review."""

from __future__ import annotations

import html
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.config import Config
from agent.core.slugify import slugify
from agent.redact import url_for_display
from agent.quality import (
    count_fact_checks,
    quality_gate_book,
    quality_gate_web,
    quality_gate_youtube,
)

BOOK_CONTENT_TYPES = frozenset({
    "book",
    "book_epub",
    "book_pdf",
    "book_mobi",
    "book_scanned",
    "book_degraded",
    "mega",
    "book_batch",
})
VIDEO_CONTENT_TYPES = frozenset(
    {
        "youtube",
        "instagram_reel",
        "instagram_post",
        "threads_video",
        "threads_text",
        "bunny_stream",
    }
)


def load_url_set(path: Path) -> list[dict[str, str]]:
    """Load benchmark URLs from a JSONL file (``url`` required, ``content_type`` optional)."""
    rows: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        row = json.loads(line)
        url = (row.get("url") or "").strip()
        if url:
            rows.append(row)
    return rows


def _safe_slug(url: str, *, content_type: str | None = None) -> str:
    return slugify(url, content_type=content_type)


def resolve_note_in_dir(
    url: str,
    notes_dir: Path,
    *,
    content_type: str | None = None,
) -> Path | None:
    """Find a staged note file for a URL under ``notes_dir/{slug}.md``."""
    if not notes_dir.is_dir():
        return None
    slug = _safe_slug(url, content_type=content_type)
    for name in (f"{slug}.md", f"{url.split('/')[-1]}.md"):
        candidate = notes_dir / name
        if candidate.is_file():
            return candidate
    # Fall back: grep frontmatter in directory
    vault_source_url = url_for_display(url)
    for path in sorted(notes_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8", errors="replace")
        if (
            f"source_url: {vault_source_url}" in text
            or f'source_url: "{vault_source_url}"' in text
        ):
            return path
    return None


def find_notes_in_vault(url: str, vault_path: Path) -> list[Path]:
    """Return all vault note paths whose frontmatter contains ``source_url: {url}``.

    Uses a fixed-string grep (``-F``) so URL regex metacharacters are matched
    literally. Duplicate ``source_url`` entries yield multiple results, which
    :func:`find_note_in_vault` treats as ambiguous.
    """
    knowledge = vault_path / "Knowledge base"
    if not knowledge.is_dir():
        return []
    vault_source_url = url_for_display(url)
    try:
        result = subprocess.run(
            ["grep", "-rlF", f"source_url: {vault_source_url}", str(knowledge)],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    matches: list[Path] = []
    for line in result.stdout.strip().splitlines():
        path = Path(line.strip())
        if path.is_file():
            matches.append(path)
    return matches


def find_note_in_vault(url: str, vault_path: Path) -> Path | None:
    """Locate the unique vault note for a URL, or ``None`` if missing/ambiguous.

    If more than one note carries the same ``source_url`` the match is ambiguous
    and this returns ``None`` rather than silently picking one — a benchmark
    should flag ambiguity instead of comparing an arbitrary file.
    """
    matches = find_notes_in_vault(url, vault_path)
    if len(matches) == 1:
        return matches[0]
    return None


def _gate_for_content_type(content_type: str):
    if content_type in BOOK_CONTENT_TYPES:
        return quality_gate_book
    if content_type in VIDEO_CONTENT_TYPES:
        return quality_gate_youtube
    return quality_gate_web


def _programmatic_metrics(note_path: Path, content_type: str, config: Config) -> dict[str, int]:
    content = note_path.read_text(encoding="utf-8")
    lines = content.split("\n")
    metrics: dict[str, int] = {
        "char_count": len(content),
        "line_count": len(lines),
        "quote_count": sum(1 for line in lines if line.strip().startswith("> ")),
    }
    if content_type in BOOK_CONTENT_TYPES:
        metrics["non_empty_line_count"] = len([l for l in lines if l.strip()])
    elif content_type in VIDEO_CONTENT_TYPES:
        concept_count = 0
        key_points = 0
        in_concepts = in_points = False
        for line in lines:
            if line.startswith("## ") and "concept" in line.lower():
                in_concepts = True
                in_points = False
                continue
            if line.startswith("## ") and "point" in line.lower():
                in_points = True
                in_concepts = False
                continue
            if line.startswith("## "):
                in_concepts = in_points = False
            if in_concepts and line.startswith("### "):
                concept_count += 1
            if in_points and line.strip().startswith(("- ", "* ", "1. ")):
                key_points += 1
        metrics["concept_count"] = concept_count
        metrics["key_points_count"] = key_points
    else:
        metrics["fact_check_count"] = count_fact_checks(lines)
    return metrics


def evaluate_note(
    note_path: Path | None,
    *,
    content_type: str,
    config: Config,
) -> dict[str, Any]:
    """Run the appropriate quality gate and return pass/fail, failures, and metrics."""
    if note_path is None or not note_path.is_file():
        return {
            "note_path": None,
            "passed": False,
            "failures": ["Note not found"],
            "metrics": {},
            "content": "",
        }

    gate = _gate_for_content_type(content_type)
    passed, failures = gate(note_path, config, vault_path=config.vault_path)
    content = note_path.read_text(encoding="utf-8")
    return {
        "note_path": str(note_path),
        "passed": passed,
        "failures": failures,
        "metrics": _programmatic_metrics(note_path, content_type, config),
        "content": content,
    }


def compare_url_pair(
    url: str,
    kb_note: Path | None,
    hermes_note: Path | None,
    *,
    content_type: str,
    config: Config,
) -> dict[str, Any]:
    """Compare kb-agent and Hermes notes for one URL."""
    return {
        "url": url,
        "content_type": content_type,
        "slug": _safe_slug(url, content_type=content_type),
        "kb_agent": evaluate_note(kb_note, content_type=content_type, config=config),
        "hermes": evaluate_note(hermes_note, content_type=content_type, config=config),
    }


def compare_from_staging(
    url_rows: list[dict],
    *,
    kb_agent_dir: Path,
    hermes_dir: Path,
    config: Config,
) -> list[dict[str, Any]]:
    """Build comparison entries from staged note directories."""
    entries: list[dict[str, Any]] = []
    for row in url_rows:
        url = row["url"]
        content_type = row.get("content_type") or "web"
        kb_path = resolve_note_in_dir(url, kb_agent_dir, content_type=content_type)
        hermes_path = resolve_note_in_dir(url, hermes_dir, content_type=content_type)
        entries.append(
            compare_url_pair(
                url,
                kb_path,
                hermes_path,
                content_type=content_type,
                config=config,
            )
        )
    return entries


def compare_from_vaults(
    url_rows: list[dict],
    *,
    kb_vault: Path,
    hermes_vault: Path,
    config: Config,
) -> list[dict[str, Any]]:
    """Compare notes resolved independently from two vault roots."""
    entries: list[dict[str, Any]] = []
    for row in url_rows:
        url = row["url"]
        content_type = row.get("content_type") or "web"
        kb_path = find_note_in_vault(url, kb_vault)
        hermes_path = find_note_in_vault(url, hermes_vault)
        entries.append(
            compare_url_pair(
                url,
                kb_path,
                hermes_path,
                content_type=content_type,
                config=config,
            )
        )
    return entries


def _summary(entries: list[dict[str, Any]]) -> dict[str, int]:
    kb_pass = sum(1 for e in entries if e["kb_agent"]["passed"])
    hermes_pass = sum(1 for e in entries if e["hermes"]["passed"])
    both = sum(
        1 for e in entries if e["kb_agent"]["passed"] and e["hermes"]["passed"]
    )
    return {
        "url_count": len(entries),
        "kb_agent_pass": kb_pass,
        "hermes_pass": hermes_pass,
        "both_pass": both,
    }


def build_review_html(entries: list[dict[str, Any]]) -> str:
    """Side-by-side HTML for manual note review."""
    parts = [
        "<!DOCTYPE html>",
        "<html lang='en'><head><meta charset='utf-8'>",
        "<title>KB Agent vs Hermes — note quality A/B</title>",
        "<style>",
        "body{font-family:system-ui,sans-serif;margin:1rem;background:#111;color:#eee;}",
        "h1{font-size:1.25rem;} .url{color:#8cf;margin:1.5rem 0 0.5rem;}",
        ".metrics{font-size:0.85rem;color:#aaa;margin-bottom:0.5rem;}",
        ".fail{color:#f88;} .pass{color:#8f8;}",
        ".grid{display:grid;grid-template-columns:1fr 1fr;gap:1rem;}",
        ".col h3{margin:0 0 0.5rem;font-size:0.95rem;}",
        "pre{background:#1a1a1a;border:1px solid #333;padding:0.75rem;",
        "overflow:auto;max-height:70vh;font-size:0.75rem;white-space:pre-wrap;}",
        "@media(max-width:900px){.grid{grid-template-columns:1fr;}}",
        "</style></head><body>",
        "<h1>Note quality A/B — kb-agent vs Hermes</h1>",
        "<p>Quality-gate metrics per note; scroll each column for side-by-side review.</p>",
    ]
    for entry in entries:
        url = html.escape(entry["url"])
        ct = html.escape(entry.get("content_type") or "web")
        parts.append(f"<section><h2 class='url'>{url} <small>({ct})</small></h2>")
        for side, label in (("kb_agent", "KB Agent"), ("hermes", "Hermes")):
            data = entry[side]
            status = "pass" if data["passed"] else "fail"
            metrics = data.get("metrics") or {}
            metric_bits = ", ".join(f"{k}={v}" for k, v in sorted(metrics.items()))
            failures = data.get("failures") or []
            fail_html = ""
            if failures:
                fail_html = (
                    "<div class='fail'>"
                    + html.escape("; ".join(failures))
                    + "</div>"
                )
            parts.append(
                f"<div class='metrics'><strong>{label}</strong> "
                f"<span class='{status}'>{status.upper()}</span> — "
                f"{html.escape(metric_bits)}{fail_html}</div>"
            )
        kb_body = html.escape(entry["kb_agent"].get("content") or "(missing)")
        hermes_body = html.escape(entry["hermes"].get("content") or "(missing)")
        parts.append(
            "<div class='grid'>"
            f"<div class='col'><h3>KB Agent</h3><pre>{kb_body}</pre></div>"
            f"<div class='col'><h3>Hermes</h3><pre>{hermes_body}</pre></div>"
            "</div></section>"
        )
    parts.append("</body></html>")
    return "\n".join(parts)


def write_comparison_report(
    entries: list[dict[str, Any]],
    out_dir: Path,
) -> dict[str, Path]:
    """Write ``comparison.json`` and ``review.html`` under ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": _summary(entries),
        "entries": [
            {
                **e,
                "kb_agent": {**e["kb_agent"], "content": None},
                "hermes": {**e["hermes"], "content": None},
            }
            for e in entries
        ],
    }
    json_path = out_dir / "comparison.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    html_path = out_dir / "review.html"
    html_path.write_text(build_review_html(entries), encoding="utf-8")
    return {"json": json_path, "html": html_path}
