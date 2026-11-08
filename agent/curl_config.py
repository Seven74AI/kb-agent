"""Helpers for building curl --config sidecars without directive injection."""

from __future__ import annotations


def escape_curl_config_quoted_string(value: str) -> str:
    """Escape ``value`` for use inside a curl config double-quoted string.

    Invariant: interpolated bytes must not terminate the quoted string or inject
    new config lines (quotes, backslashes, CR/LF).
    """
    return (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\r", "\\r")
        .replace("\n", "\\n")
    )


def curl_config_quoted_assignment(key: str, raw_value: str) -> str:
    """Format ``key = "escaped-value"`` for curl --config files."""
    return f'{key} = "{escape_curl_config_quoted_string(raw_value)}"'


def curl_config_header_line(header: str) -> str:
    """Format a curl config ``header`` directive with a single header field."""
    return curl_config_quoted_assignment("header", header)
