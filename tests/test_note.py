"""Tests for agent.core.note — slug sanitization + path traversal."""

from agent.core.note import get_note


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
