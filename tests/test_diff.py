"""Tests for pure source-content diff / dedup helpers."""

from __future__ import annotations

import pytest

from agent.core.diff import (
    ContentChangeStatus,
    check_content_change,
    content_fingerprint,
    normalize_etag,
    normalize_source_content,
)


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


def test_check_content_change_first_seen_with_content():
    result = check_content_change(
        last_content_hash=None,
        last_seen=None,
        content="fresh payload",
    )
    assert result.status == ContentChangeStatus.NEW
    assert result.content_hash == content_fingerprint("fresh payload")


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
