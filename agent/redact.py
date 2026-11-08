"""Shared redaction helpers for outward-facing URL and response surfaces.

Leaf module: importable without loading :mod:`agent.core` package exports.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

from agent.notify import redact_credential_shapes

_ALLOWED_SOURCE_URL_SCHEMES = frozenset({"http", "https"})

_HTTP_URL_IN_TEXT = re.compile(r"https?://[^\s\"'<>)\]]+", re.IGNORECASE)


def _http_netloc_for_display(parsed) -> str:
    """Rebuild netloc with hostname/port only (no user:pass@ credentials)."""
    if parsed.hostname:
        host = parsed.hostname
        if ":" in host:
            host = f"[{host}]"
        if parsed.port is not None:
            return f"{host}:{parsed.port}"
        return host
    netloc = parsed.netloc
    if "@" in netloc:
        netloc = netloc.rsplit("@", 1)[-1]
    return netloc


def redact_llm_archive_text_content(text: str) -> str:
    """Scrub LLM archive bodies: display-safe URLs survive path/credential sanitization."""
    placeholders: list[str] = []

    def shield(match: re.Match[str]) -> str:
        placeholders.append(url_for_display(match.group(0)))
        return f"__KB_ARCHIVE_URL_{len(placeholders) - 1}__"

    shielded = _HTTP_URL_IN_TEXT.sub(shield, text or "")
    from agent.notify import sanitize_visible_message

    cleaned = sanitize_visible_message(shielded)
    for index, safe in enumerate(placeholders):
        cleaned = cleaned.replace(f"__KB_ARCHIVE_URL_{index}__", safe)
    return cleaned


def redact_http_urls_in_text(text: str) -> str:
    """Replace http(s) URLs in free text with display-safe forms (no credentials/query/fragment)."""

    def repl(match: re.Match[str]) -> str:
        return url_for_display(match.group(0))

    return _HTTP_URL_IN_TEXT.sub(repl, text or "")


def url_for_display(url: str) -> str:
    """Return a URL safe for dashboard/MCP/JSON/logs: no credentials, query, or fragment on http(s).

    Credential-shaped segments in the remaining path are redacted. Local filesystem
    paths are reduced to their basename (same policy as task source_url display).
    """
    if not url:
        return url
    stripped = url.strip()
    parsed = urlparse(stripped)
    if parsed.scheme in _ALLOWED_SOURCE_URL_SCHEMES:
        path_only = urlunparse(
            (parsed.scheme, _http_netloc_for_display(parsed), parsed.path, "", "", ""),
        )
        return redact_credential_shapes(path_only)
    if parsed.scheme == "file":
        name = Path(parsed.path).name
        return name if name else "<local-path>"
    if parsed.scheme == "" and stripped.startswith("/"):
        name = Path(stripped).name
        return name if name else "<local-path>"
    return redact_credential_shapes(stripped)


_FRONTMATTER_SOURCE_URL = re.compile(
    r"^(?P<indent>[ \t]*)source_url:\s*(?P<value>.+?)\s*$",
    re.MULTILINE,
)


def _yaml_scalar_unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def prepare_vault_note_for_write(content: str) -> str:
    """Apply body URL redaction and frontmatter scrub before persisting a vault note."""
    return scrub_vault_note_content(redact_http_urls_in_text(content))


def write_vault_note(note_path: Path, content: str) -> None:
    """Write a vault note with outbound URL redaction applied to body and frontmatter."""
    note_path = Path(note_path)
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text(prepare_vault_note_for_write(content), encoding="utf-8")


def scrub_vault_note_content(content: str) -> str:
    """Return note markdown with display-safe ``source_url`` in YAML frontmatter."""
    if not content.startswith("---"):
        return _redact_frontmatter_source_url_block(content)
    close = content.find("\n---", 3)
    if close < 0:
        return _redact_frontmatter_source_url_block(content)
    head = content[: close + 1]
    tail = content[close + 1 :]
    return _redact_frontmatter_source_url_block(head) + tail


def _redact_frontmatter_source_url_block(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        raw = _yaml_scalar_unquote(match.group("value"))
        safe = url_for_display(raw)
        return f"{match.group('indent')}source_url: {safe}"

    return _FRONTMATTER_SOURCE_URL.sub(repl, text, count=1)


def _redact_segment_text_fields(segments: Any) -> None:
    if not isinstance(segments, list):
        return
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        text = seg.get("text")
        if isinstance(text, str):
            seg["text"] = redact_http_urls_in_text(text)


def redact_merged_transcript_payload(data: dict) -> dict:
    """Return a copy of merged transcript JSON with segment ``text`` URLs redacted."""
    out = copy.deepcopy(data)
    _redact_segment_text_fields(out.get("segments"))
    nested = out.get("transcript")
    if isinstance(nested, dict):
        _redact_segment_text_fields(nested.get("segments"))
    return out


def write_merged_transcript_json(
    path: Path,
    data: dict,
    *,
    indent: int | None = 2,
) -> None:
    """Persist merged transcript JSON with segment text redacted for archival upload."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    redacted = redact_merged_transcript_payload(data)
    payload = json.dumps(redacted, indent=indent, ensure_ascii=False)
    path.write_text(payload, encoding="utf-8")


def ensure_merged_transcript_file_redacted(path: Path) -> None:
    """Rewrite an on-disk merged transcript so segment text is redacted before upload."""
    path = Path(path)
    if not path.is_file():
        return
    try:
        raw = path.read_text(encoding="utf-8")
        parsed = json.loads(raw)
    except (OSError, ValueError):
        return
    compact = raw.lstrip().startswith("{") and "\n" not in raw.strip()
    write_merged_transcript_json(path, parsed, indent=None if compact else 2)


def redact_outward_url_field(result: dict) -> dict:
    """Copy a submit/status payload, redacting ``url`` when present."""
    if "url" not in result:
        return result
    out = dict(result)
    out["url"] = url_for_display(out["url"])
    return out
