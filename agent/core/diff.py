"""Pure helpers for watched-source content diff and dedup.

Compare freshly fetched source bytes against persisted ``last_content_hash``
without mutating registry state. Content fingerprints use SHA-256 over
:func:`normalize_source_content`; optional HTTP ETag values compare after
:func:`normalize_etag` when no body is available (for example 304 responses).

Hash families
-------------
``last_content_hash`` must stay in a single family per source:

* body polling stores ``sha256:<hex>`` (see :func:`content_fingerprint`);
* ETag polling stores the bare normalized ETag (see :func:`normalize_etag`).

Mixing families (a body ``sha256:`` hash later polled with ``etag``, or vice
versa) always compares unequal and reports ``NEW``. Use one mode consistently
per source.

``normalize_content`` / ``sha256_fingerprint``
----------------------------------------------
:func:`normalize_content` prepares arbitrary text for stable hashing:

* Accept ``str`` as-is; decode ``bytes`` as UTF-8 with ``errors="replace"``
  (invalid bytes become U+FFFD, matching :func:`normalize_source_content`).
* Normalize CRLF and lone CR to LF.
* Strip trailing whitespace on each line.
* Collapse three or more consecutive LF characters to two.
* Drop volatile header lines that start exactly (column 0) with ``Date:``,
  ``Last-Modified:``, ``ETag:``, ``Server:``, ``X-Cache:``, or ``X-Cache-Hits:``.
  This is intentional and applies to *any* line, including body text — a
  ``Date:`` line in prose is treated as a volatile header and dropped, so do not
  use this helper on content where such lines are meaningful.
* Strip leading and trailing blank lines so a trailing newline does not change
  the result (``"alpha\\nbeta"`` and ``"alpha\\nbeta\\n"`` normalize identically).

:func:`sha256_fingerprint` returns the lowercase hex SHA-256 digest (bare hex,
no ``sha256:`` prefix) of ``normalize_content(content)`` encoded as UTF-8 bytes.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from enum import StrEnum


_VOLATILE_HEADER_PREFIXES: tuple[str, ...] = (
    "Date:",
    "Last-Modified:",
    "ETag:",
    "Server:",
    "X-Cache:",
    "X-Cache-Hits:",
)


class ContentChangeStatus(StrEnum):
    NEW = "NEW"
    UNCHANGED = "UNCHANGED"


@dataclass(frozen=True, slots=True)
class ContentChangeResult:
    """Outcome of a diff against persisted ``last_content_hash``.

    ``content_hash`` is the value to persist on ``NEW``. It is ``None`` on an
    ``UNCHANGED`` ETag match — do not overwrite the stored hash in that case;
    the caller should still update ``last_seen``.
    """

    status: ContentChangeStatus
    content_hash: str | None


def normalize_content(content: str | bytes) -> str:
    """Normalize text for stable SHA-256 fingerprinting (see module docstring)."""
    if isinstance(content, bytes):
        text = content.decode("utf-8", errors="replace")
    else:
        text = content

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    filtered: list[str] = []
    for line in text.split("\n"):
        if line.startswith(_VOLATILE_HEADER_PREFIXES):
            continue
        filtered.append(line)

    # EOF newline stability: strip leading/trailing blank lines so a trailing
    # newline does not change the fingerprint (mirrors normalize_source_content).
    while filtered and filtered[0] == "":
        filtered.pop(0)
    while filtered and filtered[-1] == "":
        filtered.pop()
    return "\n".join(filtered)


def sha256_fingerprint(content: str | bytes) -> str:
    """Return lowercase hex SHA-256 of :func:`normalize_content` UTF-8 bytes."""
    normalized = normalize_content(content)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def normalize_etag(etag: str) -> str:
    """Normalize an HTTP ETag for stable comparison."""
    value = (etag or "").strip()
    if value.startswith("W/"):
        value = value[2:].strip()
    if len(value) >= 2 and value[0] == value[-1] == '"':
        value = value[1:-1]
    return value


def normalize_source_content(raw: str | bytes) -> str:
    """Normalize fetched source bytes for stable fingerprinting.

    Accepts a decoded response body (the usual case for ``urllib``/``requests``)
    or a full raw HTTP message when a status line is present. Header stripping
    only applies to a genuine ``HTTP/...`` status line, so a body that merely
    begins with a ``word: value`` line is left intact and fingerprints stably.
    """
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

    Hash families
    -------------
    ``last_content_hash`` must stay in one family per source: ``sha256:<hex>``
    from :func:`content_fingerprint` when polling with ``content``, or a bare
    normalized ETag when polling with ``etag``. Mixing families compares unequal
    and always yields ``NEW``.

    Persisting the result
    ---------------------
    On ``NEW``, ``content_hash`` is the value to persist via
    ``update_source_state``. On ``UNCHANGED`` with ``etag``, ``content_hash``
    is ``None``: skip the hash update but still update ``last_seen`` so the
    source is recorded as polled.
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
    """Strip a raw HTTP status line + headers when a status line is present.

    Only a genuine ``HTTP/...`` status line triggers stripping. A body that
    merely begins with a ``word: value`` line is returned unchanged, so body-only
    content (the usual case for ``urllib``/``requests``) fingerprints stably.
    """
    if text.startswith("HTTP/"):
        sep = text.find("\n\n")
        if sep != -1:
            return text[sep + 2 :]
    return text
