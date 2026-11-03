"""Tests for web pipeline quality gate — fact_check_count validation.

Validates that quality_gate_web checks for ≥2 fact checks
under ### Factual Issues within ## Critical Analysis.
"""

import sys
from pathlib import Path

import pytest

# Add kb-agent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.config import Config
from agent.quality import quality_gate_web


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def config(tmp_path):
    """Create a Config with temp paths for testing."""
    return Config(
        kb_agent_dir=Path(__file__).resolve().parents[1],
        vault_path=tmp_path / "vault",
        temp_dir=tmp_path / "tmp",
        cookie_dir=tmp_path / "cookies",
        article_min_quotes=2,
        article_min_size=100,
        article_min_fact_checks=2,
    )


def _write_web_note(note_path: Path, critical_analysis_section: str) -> None:
    """Write a complete web note with a configurable Critical Analysis section."""
    # Section-by-Section needs to be LONGER than Critical Analysis
    long_section = [
        "## Section-by-Section Analysis",
        "### Section 1: Introduction",
        "The article opens by establishing the context and background for the "
        "discussion that follows. The author frames the debate carefully, "
        "acknowledging multiple perspectives before advancing their own thesis. "
        "This approach builds credibility with readers who may be skeptical. " * 3,
        "",
        "### Section 2: Main argument",
        "The core argument is presented with evidence and examples drawn from "
        "multiple disciplines. The author weaves together empirical data, "
        "historical analysis, and theoretical frameworks to support each claim. "
        "This interdisciplinary approach strengthens the overall thesis. " * 3,
        "",
        "### Section 3: Supporting evidence",
        "Additional data points reinforce the central thesis with quantitative "
        "metrics and qualitative case studies. The evidence is well-sourced "
        "and the methodology is transparent, allowing readers to evaluate "
        "the claims independently. " * 3,
        "",
        "### Section 4: Implications",
        "The author explores what this means for the broader field, outlining "
        "practical applications and policy recommendations. The implications "
        "are far-reaching and merit serious consideration by practitioners. " * 3,
        "",
        "### Section 5: Limitations acknowledged",
        "To their credit, the author acknowledges several limitations of the "
        "analysis, including data gaps and methodological constraints. This "
        "honesty adds to the credibility of the overall argument. " * 3,
        "",
        "### Section 6: Future directions",
        "The article concludes with suggestions for future research and areas "
        "where additional work is needed. These recommendations are specific "
        "and actionable, providing a roadmap for follow-up studies. " * 3,
    ]
    note_content = "\n".join([
        "## Summary",
        "This is a comprehensive article summary that captures the central thesis.",
        "",
        "## Key Claims / Points",
        '> "Direct quote one from the article"',
        '> "Direct quote two with evidence"',
        '> "Direct quote three"',
        "### 1. First claim",
        "Analysis of the first claim with supporting context. " * 3,
        "",
        "### 2. Second claim",
        "Analysis of the second claim with depth. " * 3,
        "",
        *long_section,
        "",
        "## Context",
        "Written by an expert in the field with institutional backing and a "
        "strong publication record. The timing of this article is significant "
        "given recent developments in the field. " * 2,
        "",
        critical_analysis_section,
        "",
        "## Nuances",
        "Some nuanced points about where the article falls short. The author "
        "tends to overstate certain claims while underplaying counterarguments. "
        "The tone is generally measured but occasionally polemical. " * 2,
        "",
        "## Reliability",
        "plausible — the claims are generally well-supported by the evidence "
        "presented, though some conclusions go beyond what the data can "
        "fully sustain. The methodology is sound but the interpretation "
        "is occasionally stretched.",
        "",
        "## Sources",
        "- Author. *Title* (2024)",
        "- Supporting reference for cross-verification",
    ])
    note_path.write_text(note_content)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestWebQualityGateFactChecks:
    """Tests for fact_check_count in quality_gate_web."""

    def test_passes_with_two_fact_checks(self, config):
        """Quality gate passes when ≥2 fact check bullets exist."""
        note_dir = config.vault_path / "Knowledge base" / "articles"
        note_dir.mkdir(parents=True, exist_ok=True)
        note_path = note_dir / "test_article.md"

        _write_web_note(note_path, "\n".join([
            "## Critical Analysis",
            "The article makes some strong claims but has issues.",
            "",
            "### Methodological Issues",
            "- The sample size is small (n=30).",
            "",
            "### Factual Issues",
            "- Claim X on page 2 contradicts research by Smith (2020).",
            "- The timeline on page 5 is off by two years.",
            "",
            "### What Remains Interesting / Valid",
            "- The core argument stands despite these issues.",
        ]))

        passed, failures = quality_gate_web(note_path, config)
        assert passed is True, f"Expected pass, got failures: {failures}"

    def test_fails_with_one_fact_check(self, config):
        """Quality gate fails when < 2 fact check bullets exist."""
        note_dir = config.vault_path / "Knowledge base" / "articles"
        note_dir.mkdir(parents=True, exist_ok=True)
        note_path = note_dir / "test_article.md"

        _write_web_note(note_path, "\n".join([
            "## Critical Analysis",
            "The article has factual issues.",
            "",
            "### Factual Issues",
            "- Only one fact check — the author misquoted a source.",
            "",
            "### What Remains Interesting / Valid",
            "- The core argument stands.",
        ]))

        passed, failures = quality_gate_web(note_path, config)
        assert passed is False
        assert any("Too few fact checks" in f for f in failures)

    def test_fails_with_zero_fact_checks_no_factual_issues_section(self, config):
        """Quality gate fails when no ### Factual Issues section exists."""
        note_dir = config.vault_path / "Knowledge base" / "articles"
        note_dir.mkdir(parents=True, exist_ok=True)
        note_path = note_dir / "test_article.md"

        _write_web_note(note_path, "\n".join([
            "## Critical Analysis",
            "The article's arguments are evaluated for logical consistency.",
            "",
            "### Methodological Issues",
            "- The study lacks a control group.",
            "",
            "### What Remains Interesting / Valid",
            "- The core argument stands.",
        ]))

        passed, failures = quality_gate_web(note_path, config)
        assert passed is False
        assert any("Too few fact checks" in f for f in failures)

    def test_fails_with_no_critical_analysis_section(self, config):
        """Quality gate fails when ## Critical Analysis is missing entirely."""
        note_dir = config.vault_path / "Knowledge base" / "articles"
        note_dir.mkdir(parents=True, exist_ok=True)
        note_path = note_dir / "test_article.md"

        # Note without Critical Analysis at all
        note_content = "\n".join([
            "## Summary",
            "This is a valid article with enough content to pass other checks. " * 3,
            "",
            "## Key Claims / Points",
            '> "Direct quote one"',
            '> "Direct quote two"',
            "",
            "## Section-by-Section Analysis",
            "### Section 1",
            "Detailed analysis of section one. " * 5,
            "",
            "### Section 2",
            "Detailed analysis of section two. " * 5,
            "",
            "## Context",
            "Written by an expert.",
            "",
            "## Nuances",
            "Some limitations.",
            "",
            "## Reliability",
            "plausible",
            "",
            "## Sources",
            "Author. *Title* (2024)",
        ])
        note_path.write_text(note_content)

        passed, failures = quality_gate_web(note_path, config)
        assert passed is False
        assert any("Too few fact checks" in f for f in failures)

    def test_fact_check_paragraph_fallback(self, config):
        """When ### Factual Issues has no bullets, non-empty paragraphs count."""
        note_dir = config.vault_path / "Knowledge base" / "articles"
        note_dir.mkdir(parents=True, exist_ok=True)
        note_path = note_dir / "test_article.md"

        _write_web_note(note_path, "\n".join([
            "## Critical Analysis",
            "The article has notable factual problems.",
            "",
            "### Factual Issues",
            "The author claims X on page 2 but this is contradicted by Y.",
            "",
            "Another issue is the misrepresentation of Z on page 5.",
            "",
            "### What Remains Interesting / Valid",
            "- The core argument stands.",
        ]))

        passed, failures = quality_gate_web(note_path, config)
        # Two non-empty paragraphs → count = 2 → should pass
        assert passed is True, f"Expected pass with paragraph fallback, got failures: {failures}"
