"""Regression: Tier-2 structural review must not masquerade as pass on LLM errors."""

from agent.quality import _structural_review


class _FailingLLM:
    def call(self, **kwargs):
        raise RuntimeError("provider down api_key=supersecret-token-value")


def test_structural_review_llm_error_is_not_pass():
    result = _structural_review("# Note\n\nBody.", _FailingLLM())
    assert result.strip().lower() != "pass"
    assert not result.strip().lower().startswith("pass")
    assert "supersecret-token-value" not in result
    assert result.lower().startswith("review_error")


def test_structural_review_success_unchanged():
    class _OkLLM:
        def call(self, **kwargs):
            class R:
                content = "pass"
            return R()

    assert _structural_review("note", _OkLLM()).strip().lower() == "pass"
