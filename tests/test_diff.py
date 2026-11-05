"""Tests for pure source-content diff / dedup helpers."""

from __future__ import annotations

import pytest

from agent.core.diff import (
    ContentChangeStatus,
    check_content_change,
    content_fingerprint,
    normalize_content,
    normalize_etag,
    normalize_source_content,
    sha256_fingerprint,
)


def test_normalize_content_line_endings_and_trailing_whitespace():
    a = "line one\r\nline two  \r\n"
    b = "line one\nline two\n"
    assert normalize_content(a) == normalize_content(b)
    assert sha256_fingerprint(a) == sha256_fingerprint(b)


def test_normalize_content_collapses_excess_blank_lines():
    a = "alpha\n\n\n\nbeta"
    b = "alpha\n\nbeta"
    assert normalize_content(a) == normalize_content(b)
    assert sha256_fingerprint(a) == sha256_fingerprint(b)


def test_normalize_content_strips_volatile_headers():
    with_headers = (
        "Date: Wed, 05 Nov 2026 00:00:00 GMT\n"
        "ETag: \"abc\"\n"
        "Server: nginx\n"
        "Body line\n"
    )
    bare = "Body line\n"
    assert normalize_content(with_headers) == normalize_content(bare)
    assert sha256_fingerprint(with_headers) == sha256_fingerprint(bare)


def test_normalize_content_keeps_non_volatile_colon_lines():
    body = "Note: important\n\nActual body"
    assert normalize_content(body) == body


def test_normalize_content_strips_remaining_volatile_headers():
    with_headers = (
        "Last-Modified: Tue, 04 Nov 2026 00:00:00 GMT\n"
        "X-Cache: HIT\n"
        "X-Cache-Hits: 7\n"
        "Body line"
    )
    assert normalize_content(with_headers) == "Body line"


def test_normalize_content_keeps_indented_header_like_lines():
    # Matching is exact (no leading whitespace): an indented line is body
    # content, not a volatile header, so it must survive normalization.
    body = " Date: 2026-01-01\n\nActual body"
    assert normalize_content(body) == body


def test_normalize_content_decodes_invalid_utf8_bytes_with_replacement():
    assert normalize_content(b"bad \xff bytes") == "bad \ufffd bytes"
    assert sha256_fingerprint(b"bad \xff bytes") == sha256_fingerprint("bad \ufffd bytes")


def test_sha256_fingerprint_differs_on_substantive_change():
    assert sha256_fingerprint("version one") != sha256_fingerprint("version two")


def test_sha256_fingerprint_is_deterministic():
    payload = "stable payload"
    assert sha256_fingerprint(payload) == sha256_fingerprint(payload)
    assert sha256_fingerprint(payload) == sha256_fingerprint(payload.encode("utf-8"))


def test_content_fingerprint_is_stable():
    body = "Hello, watched source."
    assert content_fingerprint(body) == content_fingerprint(body.encode("utf-8"))


def test_normalize_source_content_strips_http_headers():
    raw = (
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/html\r\n"
        "X-Noise: ignore-me\r\n"
        "\r\n"
        "  same   body  \n"
    )
    bare = "  same   body  \n"
    assert normalize_source_content(raw) == normalize_source_content(bare)


def test_normalize_source_content_collapses_whitespace():
    a = "line one\n\n\nline two   with   gaps"
    b = "line one\n\nline two with gaps"
    assert normalize_source_content(a) == normalize_source_content(b)


def test_normalize_source_content_keeps_word_value_leading_line():
    # A body that starts with a `word: value` line must not be treated as
    # HTTP headers: the leading line is real content, not an envelope.
    body = "Note: important\n\nActual body"
    assert normalize_source_content(body) == body


def test_check_content_change_first_seen_with_content():
    result = check_content_change(
        last_content_hash=None,
        last_seen=None,
        content="fresh payload",
    )
    assert result.status == ContentChangeStatus.NEW
    assert result.content_hash == content_fingerprint("fresh payload")


def test_check_content_change_first_seen_with_etag():
    result = check_content_change(
        last_content_hash=None,
        last_seen=None,
        etag='W/"abc123"',
    )
    assert result.status == ContentChangeStatus.NEW
    assert result.content_hash == normalize_etag('W/"abc123"')


def test_check_content_change_unchanged_content():
    content = "stable text"
    stored = content_fingerprint(content)
    result = check_content_change(
        last_content_hash=stored,
        last_seen="2026-11-04T12:00:00+00:00",
        content=content,
    )
    assert result.status == ContentChangeStatus.UNCHANGED
    assert result.content_hash == stored


def test_check_content_change_detects_changed_content():
    stored = content_fingerprint("version one")
    result = check_content_change(
        last_content_hash=stored,
        last_seen="2026-11-04T12:00:00+00:00",
        content="version two",
    )
    assert result.status == ContentChangeStatus.NEW
    assert result.content_hash == content_fingerprint("version two")


def test_check_content_change_missing_previous_hash():
    result = check_content_change(
        last_content_hash=None,
        last_seen="2026-11-04T12:00:00+00:00",
        content="only hash matters for diff",
    )
    assert result.status == ContentChangeStatus.NEW


def test_check_content_change_equivalent_after_normalization():
    stored = content_fingerprint("alpha\n\nbeta")
    noisy = (
        "HTTP/1.1 200 OK\nContent-Length: 99\n\n"
        "alpha\n\n\nbeta   \n"
    )
    result = check_content_change(
        last_content_hash=stored,
        last_seen="2026-11-04T12:00:00+00:00",
        content=noisy,
    )
    assert result.status == ContentChangeStatus.UNCHANGED


def test_check_content_change_etag_unchanged():
    etag = 'W/"abc123"'
    stored = normalize_etag(etag)
    result = check_content_change(
        last_content_hash=stored,
        last_seen="2026-11-04T12:00:00+00:00",
        etag='W/"abc123"',
    )
    assert result.status == ContentChangeStatus.UNCHANGED
    assert result.content_hash is None


def test_check_content_change_etag_changed():
    result = check_content_change(
        last_content_hash=normalize_etag('W/"old"'),
        last_seen="2026-11-04T12:00:00+00:00",
        etag='W/"new"',
    )
    assert result.status == ContentChangeStatus.NEW
    assert result.content_hash == normalize_etag('W/"new"')


def test_check_content_change_mixed_hash_families_reports_new():
    # Documented contract: mixing a body `sha256:` hash with an ETag poll always
    # compares unequal and reports NEW — never silently match across families.
    stored_body_hash = content_fingerprint("body content")
    result = check_content_change(
        last_content_hash=stored_body_hash,
        last_seen="2026-11-04T12:00:00+00:00",
        etag='W/"abc123"',
    )
    assert result.status == ContentChangeStatus.NEW


def test_check_content_change_requires_content_or_etag():
    with pytest.raises(ValueError, match="content or etag"):
        check_content_change(last_content_hash=None, last_seen=None)


def test_check_content_change_rejects_both_content_and_etag():
    with pytest.raises(ValueError, match="not both"):
        check_content_change(
            last_content_hash=None,
            content="body",
            etag='"x"',
        )


def test_poll_contract_caller_persists_hash_without_module_mutation():
    """Simulate a poller: diff helper is pure; caller owns persisted state."""
    last_content_hash: str | None = None
    last_seen: str | None = None

    result = check_content_change(
        last_content_hash=last_content_hash,
        last_seen=last_seen,
        content="body v1",
    )
    assert result.status == ContentChangeStatus.NEW
    last_seen = "2026-11-04T12:00:00+00:00"
    last_content_hash = result.content_hash

    result = check_content_change(
        last_content_hash=last_content_hash,
        last_seen=last_seen,
        content="body v1",
    )
    assert result.status == ContentChangeStatus.UNCHANGED

    result = check_content_change(
        last_content_hash=last_content_hash,
        last_seen=last_seen,
        content="body v2",
    )
    assert result.status == ContentChangeStatus.NEW
    assert result.content_hash == content_fingerprint("body v2")
    last_content_hash = result.content_hash
    assert last_content_hash == content_fingerprint("body v2")
