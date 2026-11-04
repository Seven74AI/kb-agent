"""Pure helpers for watched-source content diff and dedup.

Compare freshly fetched source bytes against persisted ``last_content_hash``
without mutating registry state. Content fingerprints use SHA-256 over
:func:`normalize_source_content`; optional HTTP ETag values compare after
:func:`normalize_etag` when no body is available (for example 304 responses).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from enum import StrEnum


class ContentChangeStatus(StrEnum):
    NEW = "NEW"
    UNCHANGED = "UNCHANGED"


@dataclass(frozen=True, slots=True)
class ContentChangeResult:
    status: ContentChangeStatus
    content_hash: str | None


def normalize_etag(etag: str) -> str:
    """Normalize an HTTP ETag for stable comparison."""
    value = (etag or "").strip()
    if value.startswith("W/"):
        value = value[2:].strip()
    if len(value) >= 2 and value[0] == value[-1] == '"':
        value = value[1:-1]
    return value


def normalize_source_content(raw: str | bytes) -> str:
    """Normalize fetched source bytes for stable fingerprinting."""
    if isinstance(raw, bytes):
        text = raw.decode("utf-8", errors="replace")
    else:
        text = raw or ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _strip_http_envelope(text)

    lines: list[str] = []
    blank_pending = False
    for line in text.split("\n"):
        collapsed = re.sub(r"[ \t]+", " ", line.strip())
        if not collapsed:
            if lines and not blank_pending:
                lines.append("")
            blank_pending = True
            continue
        blank_pending = False
        lines.append(collapsed)

    while lines and lines[0] == "":
        lines.pop(0)
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def content_fingerprint(raw: str | bytes) -> str:
    """Return ``sha256:<hex>`` for normalized *raw* content."""
    normalized = normalize_source_content(raw)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def check_content_change(
    *,
    last_content_hash: str | None,
    last_seen: str | None = None,
    content: str | bytes | None = None,
    etag: str | None = None,
) -> ContentChangeResult:
    """Classify whether fetched source material is new relative to stored state.

    ``last_seen`` is accepted for API symmetry with the sources registry but is
    not used for comparison — only ``last_content_hash`` (or its absence) drives
    the diff outcome.
    """
    del last_seen  # persisted timestamp; callers update via sources.update_source_state

    if content is not None and etag is not None:
        raise ValueError("provide content or etag, not both")

    if content is None and etag is None:
        raise ValueError("content or etag required")

    if last_content_hash is None:
        if content is not None:
            return ContentChangeResult(
                status=ContentChangeStatus.NEW,
                content_hash=content_fingerprint(content),
            )
        normalized = normalize_etag(etag or "")
        return ContentChangeResult(
            status=ContentChangeStatus.NEW,
            content_hash=normalized,
        )

    if content is not None:
        fingerprint = content_fingerprint(content)
        if fingerprint == last_content_hash:
            return ContentChangeResult(
                status=ContentChangeStatus.UNCHANGED,
                content_hash=fingerprint,
            )
        return ContentChangeResult(
            status=ContentChangeStatus.NEW,
            content_hash=fingerprint,
        )

    normalized = normalize_etag(etag or "")
    if normalized == normalize_etag(last_content_hash):
        return ContentChangeResult(
            status=ContentChangeStatus.UNCHANGED,
            content_hash=None,
        )
    return ContentChangeResult(
        status=ContentChangeStatus.NEW,
        content_hash=normalized,
    )


def _strip_http_envelope(text: str) -> str:
    if text.startswith("HTTP/"):
        sep = text.find("\n\n")
        if sep != -1:
            return text[sep + 2 :]
    first_line = text.split("\n", 1)[0]
    if re.match(r"^[\w-]+:\s", first_line):
        sep = text.find("\n\n")
        if sep != -1:
            return text[sep + 2 :]
    return text
