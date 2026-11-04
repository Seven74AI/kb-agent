"""Content normalization and fingerprinting for stable source diffs.

Normalization (``normalize_content``)
-------------------------------------
Applied in order:

1. Treat *content* as UTF-8 text (already decoded ``str``).
2. Normalize line endings: ``CRLF`` and lone ``CR`` become ``LF``.
3. Drop volatile header lines whose names are ``Date``, ``Last-Modified``,
   ``ETag``, ``Server``, ``X-Cache``, or ``X-Cache-Hits`` (case-insensitive),
   matching ``name:`` at the start of a line (optional leading whitespace).
4. Strip trailing whitespace on each remaining line.
5. Collapse runs of three or more consecutive ``LF`` characters to exactly two.

``sha256_fingerprint`` returns the lowercase hexadecimal SHA-256 digest of the
UTF-8 encoding of the normalized string. Helpers are pure: no I/O and no
mutation of caller state.
"""

from __future__ import annotations

import hashlib
import re

_VOLATILE_HEADER = re.compile(
    r"^\s*(?:Date|Last-Modified|ETag|Server|X-Cache|X-Cache-Hits)\s*:",
    re.IGNORECASE,
)
_EXCESS_NEWLINES = re.compile(r"\n{3,}")


def normalize_content(content: str) -> str:
    """Return normalized text suitable for stable content comparison."""
    text = (content or "").replace("\r\n", "\n").replace("\r", "\n")
    lines: list[str] = []
    for line in text.split("\n"):
        if _VOLATILE_HEADER.match(line):
            continue
        lines.append(line.rstrip())
    while lines and lines[-1] == "":
        lines.pop()
    normalized = "\n".join(lines)
    return _EXCESS_NEWLINES.sub("\n\n", normalized)


def sha256_fingerprint(content: str) -> str:
    """Return lowercase hex SHA-256 of UTF-8 bytes of normalized *content*."""
    normalized = normalize_content(content)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
