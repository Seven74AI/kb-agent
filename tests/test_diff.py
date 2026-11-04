"""Tests for content normalization and SHA-256 fingerprint helpers."""

from __future__ import annotations

import hashlib

import pytest

from agent.core.diff import normalize_content, sha256_fingerprint


def test_normalize_content_line_endings():
    assert normalize_content("a\r\nb\rc") == "a\nb\nc"


def test_normalize_content_strips_trailing_whitespace_per_line():
    assert normalize_content("hello   \nworld\t\t") == "hello\nworld"


def test_normalize_content_collapses_three_or_more_blank_lines():
    assert normalize_content("a\n\n\n\nb") == "a\n\nb"
    assert normalize_content("a\n\n\nb") == "a\n\nb"


def test_normalize_content_drops_volatile_header_lines():
    raw = (
        "Date: Wed, 04 Nov 2026 12:00:00 GMT\n"
        "Keep-Me: value\n"
        "Last-Modified: Tue, 03 Nov 2026 11:00:00 GMT\n"
        "ETag: \"abc123\"\n"
        "Server: nginx\n"
        "X-Cache: HIT\n"
        "X-Cache-Hits: 42\n"
        "body line\n"
    )
    assert normalize_content(raw) == "Keep-Me: value\nbody line"


def test_fingerprint_stable_across_line_endings_and_trailing_space():
    variants = [
        "alpha\nbeta",
        "alpha\r\nbeta",
        "alpha\rbeta",
        "alpha  \nbeta\t",
    ]
    fingerprints = {sha256_fingerprint(text) for text in variants}
    assert len(fingerprints) == 1


def test_fingerprint_differs_for_substantive_change():
    assert sha256_fingerprint("version one") != sha256_fingerprint("version two")


def test_sha256_fingerprint_is_lowercase_hex_of_normalized_utf8():
    content = "hello\nworld"
    expected = hashlib.sha256(normalize_content(content).encode("utf-8")).hexdigest()
    assert sha256_fingerprint(content) == expected
    assert sha256_fingerprint(content) == sha256_fingerprint(content).lower()
    assert len(sha256_fingerprint(content)) == 64


def test_normalize_content_accepts_utf8_decoded_text():
    text = "café — naïve"
    assert normalize_content(text) == text


@pytest.mark.parametrize(
    "header_line",
    [
        "Date: now",
        "date: now",
        "Last-Modified: then",
        "ETag: x",
        "Server: apache",
        "X-Cache: MISS",
        "X-Cache-Hits: 0",
    ],
)
def test_normalize_content_drops_volatile_headers_case_insensitive(header_line: str):
    assert normalize_content(f"{header_line}\ncontent") == "content"
