"""Shared redaction helpers for outward-facing URL and response surfaces.

Leaf module: importable without loading :mod:`agent.core` package exports.
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse, urlunparse

from agent.notify import redact_credential_shapes

_ALLOWED_SOURCE_URL_SCHEMES = frozenset({"http", "https"})


def url_for_display(url: str) -> str:
    """Return a URL safe for dashboard/MCP/JSON/logs: no query or fragment on http(s).

    Credential-shaped segments in the remaining path are redacted. Local filesystem
    paths are reduced to their basename (same policy as task source_url display).
    """
    if not url:
        return url
    stripped = url.strip()
    parsed = urlparse(stripped)
    if parsed.scheme in _ALLOWED_SOURCE_URL_SCHEMES:
        path_only = urlunparse(
            (parsed.scheme, parsed.netloc, parsed.path, "", "", ""),
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


def redact_outward_url_field(result: dict) -> dict:
    """Copy a submit/status payload, redacting ``url`` when present."""
    if "url" not in result:
        return result
    out = dict(result)
    out["url"] = url_for_display(out["url"])
    return out
