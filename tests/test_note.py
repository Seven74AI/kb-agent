"""Tests for agent.core.note — slug sanitization + path traversal."""

from agent.core.note import get_note

TOKENIZED = "https://cdn.example.com/video.mp4?token=secret&expires=999"
MEGA_WITH_KEY = "https://mega.nz/file/abc123#decryptionkey"


def test_get_note_no_path_traversal(tmp_path):
    vault = tmp_path / "vault"
    (vault / "Knowledge base" / "articles").mkdir(parents=True)
    (vault / "secret.md").write_text("TOP SECRET", encoding="utf-8")

    # A traversal slug must never resolve outside Knowledge base.
    assert get_note("../../secret", vault, content_type="web") is None
    assert get_note("../secret", vault) is None


def test_get_note_finds_sanitized_slug(tmp_path):
    vault = tmp_path / "vault"
    (vault / "Knowledge base" / "articles").mkdir(parents=True)
    (vault / "Knowledge base" / "articles" / "hello-world.md").write_text("# Hi", encoding="utf-8")

    n = get_note("hello-world", vault, content_type="web")
    assert n is not None
    assert n["slug"] == "hello-world"
    assert "# Hi" in n["content"]


def test_get_note_redacts_frontmatter_source_url(tmp_path):
    vault = tmp_path / "vault"
    articles = vault / "Knowledge base" / "articles"
    articles.mkdir(parents=True)
    raw_note = f"""---
date: 2026-01-01
source: example.com
source_url: {TOKENIZED}
confidence: plausible
tags: []
---

# Title
"""
    (articles / "token-note.md").write_text(raw_note, encoding="utf-8")

    note = get_note("token-note", vault, content_type="web")
    assert note is not None
    assert "token=secret" not in note["content"]
    assert "source_url: https://cdn.example.com/video.mp4" in note["content"]

    mega_note = f"""---
source_url: {MEGA_WITH_KEY}
---
# Mega
"""
    (articles / "mega-note.md").write_text(mega_note, encoding="utf-8")
    mega = get_note("mega-note", vault, content_type="web")
    assert mega is not None
    assert "decryptionkey" not in mega["content"]
    assert "#" not in mega["content"].split("---")[1]
