"""Regression: public docs must not fingerprint production Tailscale topology."""

from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
PRD = WORKSPACE / "docs" / "prd-tigris-storage-migration.md"
CONTEXT = WORKSPACE / "CONTEXT.md"

_PRODUCTION_TAILSCALE_MARKERS = (
    "tail5c02a1.ts.net",
    "vmi3304846",
)


def test_prd_tigris_migration_has_no_concrete_tailscale_host():
    text = PRD.read_text(encoding="utf-8")
    for marker in _PRODUCTION_TAILSCALE_MARKERS:
        assert marker not in text, f"docs leak production host marker: {marker!r}"


def test_context_auth_documents_kb_api_token_not_tailscale_only():
    text = CONTEXT.read_text(encoding="utf-8")
    assert "Tailscale-only. No login on web dashboard." not in text
    assert "KB_API_TOKEN" in text
