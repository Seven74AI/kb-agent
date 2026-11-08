"""Shared redaction helpers for outward-facing URL and response surfaces."""

from __future__ import annotations

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


def redact_outward_url_field(result: dict) -> dict:
    """Copy a submit/status payload, redacting ``url`` when present."""
    if "url" not in result:
        return result
    out = dict(result)
    out["url"] = url_for_display(out["url"])
    return out
