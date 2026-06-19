"""Tests for the book pipeline — Mega.nz, ePub/PDF, 3-tier pre-flight,
chapter-level synthesis, quality gates, MinIO, Git."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest

# Add kb-agent to path
sys.path.insert(0, "/root/kb-agent")

from agent.config import Config
from agent.detection import _classify_mega_filenames
from agent.pipelines.books import (
    _pdf_preflight,
    quality_score,
    step_extract_epub,
    step_extract_pdf,
    step_classify_content,
    step_book_quality_gate,
    step_synthesize_book,
    step_megadl_download,
    step_book_minio_upload,
    step_book_git_push,
    _build_book_prompt,
    _today,
    BOOK_EXTENSIONS,
    PDF_MIN_CHARS_SCANNED,
    PDF_QUALITY_SCORE_THRESHOLD,
    CHARS_PER_TOKEN,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def config(tmp_path):
    """Create a Config with temp paths for testing."""
    return Config(
        kb_agent_dir=Path("/root/kb-agent"),
        vault_path=tmp_path / "vault",
        temp_dir=tmp_path / "tmp",
        cookie_dir=tmp_path / "cookies",
        book_min_lines=10,
        book_min_quotes=2,
        book_min_size=100,
    )


@pytest.fixture
def book_task():
    """Create a sample book task dict."""
    return {
        "id": "test-book-abc123",
        "source_url": "/tmp/test_book.epub",
        "content_type": "book",
        "slug": "test_book_slug",
    }


@pytest.fixture
def mega_task():
    """Create a sample Mega task dict."""
    return {
        "id": "test-mega-abc123",
        "source_url": "https://mega.nz/file/abc123#key",
        "content_type": "mega",
        "slug": "mega_nz_file_abc123",
    }


# ── Detection Tests ───────────────────────────────────────────────────────────

class TestMegaFilenamesClassification:
    """Tests for _classify_mega_filenames."""

    def test_single_epub_returns_book(self):
        """Single .epub file → 'book'."""
        result = _classify_mega_filenames(["my-book.epub"])
        assert result == "book"

    def test_single_pdf_returns_book(self):
        """Single .pdf file → 'book'."""
        result = _classify_mega_filenames(["document.pdf"])
        assert result == "book"

    def test_single_zip_returns_book_batch(self):
        """Single .zip archive → 'book_batch'."""
        result = _classify_mega_filenames(["books-pack.zip"])
        assert result == "book_batch"

    def test_single_rar_returns_book_batch(self):
        """Single .rar archive → 'book_batch'."""
        result = _classify_mega_filenames(["collection.rar"])
        assert result == "book_batch"

    def test_majority_epub_returns_book_batch(self):
        """Multiple books → 'book_batch'."""
        result = _classify_mega_filenames([
            "book1.epub", "book2.epub", "book3.pdf", "notes.txt"
        ])
        assert result == "book_batch"

    def test_majority_video_returns_mega_video(self):
        """Multiple videos → 'mega_video'."""
        result = _classify_mega_filenames([
            "movie.mp4", "trailer.mkv", "readme.txt"
        ])
        assert result == "mega_video"

    def test_single_video_returns_mega_video(self):
        """Single .mp4 → 'mega_video'."""
        result = _classify_mega_filenames(["lecture.mp4"])
        assert result == "mega_video"

    def test_unknown_extension_returns_unknown(self):
        """Unknown extension → 'unknown'."""
        result = _classify_mega_filenames(["data.bin", "stuff.dat"])
        assert result == "unknown"

    def test_empty_list_returns_unknown(self):
        """Empty list → 'unknown'."""
        result = _classify_mega_filenames([])
        assert result == "unknown"

    def test_archive_priority_over_books(self):
        """Archive + books → 'book_batch' (archive takes priority)."""
        result = _classify_mega_filenames([
            "archive.zip", "book1.epub", "book2.pdf"
        ])
        assert result == "book_batch"

    def test_mixed_books_and_videos_returns_unknown(self):
        """Mixed books and videos with no majority → 'unknown'."""
        result = _classify_mega_filenames([
            "book.epub", "movie.mp4"
        ])
        assert result == "unknown"

    def test_case_insensitive_extensions(self):
        """Extensions are case-insensitive."""
        result = _classify_mega_filenames(["Book.EPUB"])
        assert result == "book"


# ── Quality Score Tests ───────────────────────────────────────────────────────

class TestQualityScore:
    """Tests for quality_score and _pdf_preflight functions."""

    def test_quality_score_returns_int(self):
        """quality_score returns an integer 0-100."""
        score = quality_score("/nonexistent/path.pdf")
        assert isinstance(score, int)
        assert 0 <= score <= 100

    def test_quality_score_handles_missing_file(self):
        """quality_score handles missing file gracefully."""
        score = quality_score("/tmp/definitely_not_a_real_file_12345.pdf")
        assert isinstance(score, int)
        assert 0 <= score <= 100

    def test_quality_score_handles_non_pdf(self):
        """quality_score handles non-PDF files gracefully."""
        score = quality_score("/tmp/not_a_pdf.txt")
        assert isinstance(score, int)
        assert 0 <= score <= 100

    @patch("agent.pipelines.books.pymupdf")
    def test_pdf_preflight_scanned_detection(self, mock_pymupdf):
        """PDF with < 500 chars → 'scanned'."""
        # Mock a document with very little text
        mock_doc = MagicMock()
        mock_doc.page_count = 10
        mock_doc.__getitem__.return_value.get_text.return_value = ""  # 0 chars per page
        mock_pymupdf.open.return_value = mock_doc

        classification, score = _pdf_preflight(Path("/fake/scanned.pdf"))
        assert classification == "scanned"
        assert score == 0

    @patch("agent.pipelines.books.pymupdf")
    def test_pdf_preflight_clean_text(self, mock_pymupdf):
        """PDF with high-quality text → 'text' with high score."""
        mock_doc = MagicMock()
        mock_doc.page_count = 20
        # Return clean, normal text for each page
        clean_text = (
            "The quick brown fox jumps over the lazy dog. "
            "This is a perfectly normal sentence with standard "
            "English text and common punctuation marks. "
            * 5  # Enough chars to pass tier 1
        )
        mock_doc.__getitem__.return_value.get_text.return_value = clean_text
        mock_pymupdf.open.return_value = mock_doc

        classification, score = _pdf_preflight(Path("/fake/clean.pdf"))
        assert classification == "text"
        assert score >= PDF_QUALITY_SCORE_THRESHOLD

    @patch("agent.pipelines.books.pymupdf")
    def test_pdf_preflight_degraded_ocr(self, mock_pymupdf):
        """PDF with degraded OCR → 'degraded' with low score."""
        mock_doc = MagicMock()
        mock_doc.page_count = 16
        # Simulate degraded OCR with long-s patterns
        degraded_text = (
            "The fafe and fubject of this fection is the firft "
            "attempt at defcribing the fubject. %J. C  A  "
            * 20  # Enough chars but garbled
        )
        mock_doc.__getitem__.return_value.get_text.return_value = degraded_text
        mock_pymupdf.open.return_value = mock_doc

        classification, score = _pdf_preflight(Path("/fake/degraded.pdf"))
        assert classification == "degraded"
        assert score < PDF_QUALITY_SCORE_THRESHOLD


# ── ePub Extraction Tests ─────────────────────────────────────────────────────

class TestEpubExtraction:
    """Tests for step_extract_epub."""

    @patch("agent.pipelines.books.ITEM_DOCUMENT", 9)
    @patch("agent.pipelines.books.epub")
    @patch("agent.pipelines.books.BeautifulSoup")
    def test_extract_epub_basic(self, mock_bs, mock_epub, config, book_task):
        """Basic ePub extraction with mocked ebooklib."""
        # Create a dummy epub file so the file-existence check passes
        epub_path = Path(config.temp_dir) / "test_book.epub"
        epub_path.parent.mkdir(parents=True, exist_ok=True)
        epub_path.write_text("dummy epub content")
        book_task["source_url"] = str(epub_path)
        # Mock the book
        mock_book = MagicMock()
        mock_epub.read_epub.return_value = mock_book
        mock_epub.ITEM_DOCUMENT = 9

        # Create mock document items with unique soups
        soups = []
        items = []
        chapters = [
            "Chapter 1: Introduction\nThis is the first chapter with enough text to pass the threshold.",
            "Chapter 2: Methods\nThis is the second chapter with detailed methodology description.",
            "Chapter 3: Results\nThis is the third chapter presenting the key findings and analysis.",
        ]
        for body_text in chapters:
            soup = MagicMock()
            soup.get_text.return_value = body_text
            soups.append(soup)
            item = MagicMock()
            item.get_body_content.return_value = b"<p>test</p>"
            items.append(item)
        mock_bs.side_effect = soups
        mock_book.get_items_of_type.return_value = items

        # Mock metadata
        mock_book.get_metadata.return_value = [("Test Book",)]

        # Ensure temp dir exists
        Path(config.temp_dir).mkdir(parents=True, exist_ok=True)

        result = step_extract_epub(book_task, config)
        data = json.loads(result)

        assert data["chapter_count"] == 3
        assert data["total_chars"] > 0
        assert "full_text_path" in data

        # Verify file was written
        full_text_path = Path(data["full_text_path"])
        assert full_text_path.exists()
        content = full_text_path.read_text()
        assert "CHAPTER 0" in content
        assert "CHAPTER 1" in content

    @patch("agent.pipelines.books.ITEM_DOCUMENT", 9)
    @patch("agent.pipelines.books.epub")
    @patch("agent.pipelines.books.BeautifulSoup")
    def test_extract_epub_preserves_order(self, mock_bs, mock_epub, config, book_task):
        """ePub extraction preserves chapter order (not size-sorted)."""
        # Create dummy epub file
        epub_path = Path(config.temp_dir) / "test_book.epub"
        epub_path.parent.mkdir(parents=True, exist_ok=True)
        epub_path.write_text("dummy epub content")
        book_task["source_url"] = str(epub_path)
        mock_book = MagicMock()
        mock_epub.read_epub.return_value = mock_book
        mock_epub.ITEM_DOCUMENT = 9

        # Chapter 1 is long, Chapter 2 is short
        soups = []
        items = []
        for body_text in ["A" * 500, "B"]:
            soup = MagicMock()
            soup.get_text.return_value = body_text
            soups.append(soup)
            item = MagicMock()
            item.get_body_content.return_value = b"<p>test</p>"
            items.append(item)
        mock_bs.side_effect = soups
        mock_book.get_items_of_type.return_value = items
        mock_book.get_metadata.return_value = [("Test",)]

        Path(config.temp_dir).mkdir(parents=True, exist_ok=True)

        result = step_extract_epub(book_task, config)
        full_text = Path(json.loads(result)["full_text_path"]).read_text()

        # Chapter 0 (A) should come before Chapter 1 (B)
        idx_a = full_text.index("CHAPTER 0")
        idx_b = full_text.index("CHAPTER 1")
        assert idx_a < idx_b, "Chapter order must be preserved"

    @patch("agent.pipelines.books.ITEM_DOCUMENT", 9)
    @patch("agent.pipelines.books.epub")
    @patch("agent.pipelines.books.BeautifulSoup")
    def test_extract_epub_skips_tiny_fragments(self, mock_bs, mock_epub, config, book_task):
        """Tiny fragments (< 50 chars) are skipped."""
        # Create dummy epub file
        epub_path = Path(config.temp_dir) / "test_book.epub"
        epub_path.parent.mkdir(parents=True, exist_ok=True)
        epub_path.write_text("dummy epub content")
        book_task["source_url"] = str(epub_path)
        mock_book = MagicMock()
        mock_epub.read_epub.return_value = mock_book
        mock_epub.ITEM_DOCUMENT = 9

        soups = []
        items = []
        for body_text in ["Tiny", "A" * 100]:
            soup = MagicMock()
            soup.get_text.return_value = body_text
            soups.append(soup)
            item = MagicMock()
            item.get_body_content.return_value = b"<p>test</p>"
            items.append(item)
        mock_bs.side_effect = soups
        mock_book.get_items_of_type.return_value = items
        mock_book.get_metadata.return_value = [("Test",)]

        Path(config.temp_dir).mkdir(parents=True, exist_ok=True)

        result = step_extract_epub(book_task, config)
        data = json.loads(result)
        assert data["chapter_count"] == 1  # Only the 100-char one


# ── PDF Extraction Tests ──────────────────────────────────────────────────────

class TestPdfExtraction:
    """Tests for step_extract_pdf."""

    @patch("agent.pipelines.books.pymupdf")
    def test_extract_pdf_basic(self, mock_pymupdf, config, book_task):
        """Basic PDF extraction with mocked pymupdf."""
        # Create dummy pdf file
        pdf_path = Path(config.temp_dir) / "test_book.pdf"
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_path.write_text("dummy pdf content")
        book_task["source_url"] = str(pdf_path)

        # pymupdf.open is called twice (preflight + extraction), so return new doc each time
        text_per_page = (
            "Page content with enough characters to pass "
            "the scanned detection threshold of 500 chars. "
            * 5
        )

        def make_doc():
            mock_doc = MagicMock()
            mock_doc.page_count = 3
            # Make doc iterable over pages
            pages = []
            for _ in range(3):
                page = MagicMock()
                page.get_text.return_value = text_per_page
                pages.append(page)
            mock_doc.__iter__.return_value = iter(pages)
            mock_doc.__getitem__.return_value = pages[0]
            return mock_doc

        mock_pymupdf.open.side_effect = [make_doc(), make_doc()]

        Path(config.temp_dir).mkdir(parents=True, exist_ok=True)

        result = step_extract_pdf(book_task, config)
        data = json.loads(result)

        assert data["page_count"] == 3
        assert data["total_chars"] > 0
        assert "quality_score" in data

    @patch("agent.pipelines.books.pymupdf")
    def test_extract_pdf_scanned_raises(self, mock_pymupdf, config, book_task):
        """Scanned PDF raises StepError."""
        # Create dummy pdf file
        pdf_path = Path(config.temp_dir) / "test_book.pdf"
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_path.write_text("dummy pdf content")
        book_task["source_url"] = str(pdf_path)

        # Return a doc with 0 text → scanned
        mock_doc = MagicMock()
        mock_doc.page_count = 1
        mock_doc.__getitem__.return_value.get_text.return_value = ""  # 0 chars
        mock_pymupdf.open.return_value = mock_doc

        Path(config.temp_dir).mkdir(parents=True, exist_ok=True)

        from agent.exceptions import StepError
        with pytest.raises(StepError) as excinfo:
            step_extract_pdf(book_task, config)
        assert "needs OCR" in str(excinfo.value)


# ── Book Quality Gate Tests ───────────────────────────────────────────────────

class TestBookQualityGate:
    """Tests for step_book_quality_gate."""

    def test_quality_gate_passes_for_valid_note(self, config, book_task):
        """Quality gate passes for a valid book note."""
        note_dir = config.vault_path / "Knowledge base" / "books"
        note_dir.mkdir(parents=True, exist_ok=True)
        note_path = note_dir / f"{book_task['slug']}.md"

        # Build a long enough book note without repetition
        # (repetition with *3 breaks section indexing)
        chapters = []
        for i in range(1, 11):
            chapters.append(f"### Chapter {i}")
            chapters.append(f"This is a detailed summary of chapter {i}. " * 10)

        note_content = "\n".join([
            "## Summary",
            "This is a comprehensive book summary that captures the central thesis.",
            "",
            "## Key Claims",
            '> "Direct quote one from the book" — Chapter 1',
            '> "Direct quote two proving a key point" — Chapter 2',
            '> "Direct quote three with evidence" — Chapter 5',
            '> "Direct quote four with analysis" — Chapter 8',
            "",
            "## Chapter Summaries",
            *chapters,
            "",
            "## Who Is the Author",
            "A detailed author biography with credentials and background information. " * 5,
            "",
            "## Critical Analysis",
            "This section evaluates the book critically but briefly. " * 5,
            "",
            "## Nuances",
            "Some nuanced points about where the book falls short.",
            "",
            "## Reliability",
            "plausible",
            "",
            "## Sources",
            "Author. *Title* (2024)",
        ])
        note_path.write_text(note_content)

        result = step_book_quality_gate(book_task, config)
        data = json.loads(result)
        assert data["passed"] is True

    def test_quality_gate_fails_too_few_quotes(self, config, book_task):
        """Quality gate fails when quote count < required."""
        note_dir = config.vault_path / "Knowledge base" / "books"
        note_dir.mkdir(parents=True, exist_ok=True)
        note_path = note_dir / f"{book_task['slug']}.md"

        note_content = "\n".join([
            "No quotes here.",
            "Just text.",
        ] * 50)  # Enough lines but no quotes
        note_path.write_text(note_content)

        from agent.exceptions import StepError
        with pytest.raises(StepError) as excinfo:
            step_book_quality_gate(book_task, config)
        assert "Too few quotes" in str(excinfo.value)

    def test_quality_gate_fails_too_short(self, config, book_task):
        """Quality gate fails when note is too short."""
        note_dir = config.vault_path / "Knowledge base" / "books"
        note_dir.mkdir(parents=True, exist_ok=True)
        note_path = note_dir / f"{book_task['slug']}.md"
        note_path.write_text("Too short.")

        from agent.exceptions import StepError
        with pytest.raises(StepError):
            step_book_quality_gate(book_task, config)

    def test_quality_gate_note_not_found(self, config, book_task):
        """Quality gate fails when note doesn't exist."""
        from agent.exceptions import StepError
        with pytest.raises(StepError):
            step_book_quality_gate(book_task, config)


# ── Synthesis Prompt Tests ────────────────────────────────────────────────────

class TestBookSynthesis:
    """Tests for book synthesis functions."""

    def test_build_book_prompt_includes_title_author(self):
        """_build_book_prompt includes title and author."""
        prompt = _build_book_prompt("Some text.", "My Book", "Jane Doe")
        assert "My Book" in prompt
        assert "Jane Doe" in prompt
        assert "Some text." in prompt

    def test_build_book_prompt_truncates_long_text(self):
        """_build_book_prompt truncates text > 180K chars."""
        long_text = "A" * 250_000
        prompt = _build_book_prompt(long_text, "Title", "Author")
        assert len(prompt) < 210_000  # Should be truncated
        assert "[... content truncated" in prompt

    def test_today_returns_date_string(self):
        """_today returns a YYYY-MM-DD date string."""
        result = _today()
        import re
        assert re.match(r'^\d{4}-\d{2}-\d{2}$', result)


# ── Content Classification Tests ──────────────────────────────────────────────

class TestContentClassification:
    """Tests for step_classify_content."""

    def test_classify_content_with_epubs(self, config, mega_task):
        """Classify content with ePub files."""
        dl_dir = Path(config.temp_dir) / f"mega_{mega_task['slug']}"
        dl_dir.mkdir(parents=True, exist_ok=True)

        # Create dummy book files
        (dl_dir / "book1.epub").write_text("dummy epub content")
        (dl_dir / "book2.epub").write_text("dummy epub content")

        result = step_classify_content(mega_task, config)
        data = json.loads(result)

        assert data["total_files"] == 2
        assert data["clean_text"] == 2
        assert data["scanned"] == 0
        for book in data["books"]:
            assert book["classification"] == "text"
            assert book["pipeline"] == "book_epub"

    @patch("agent.pipelines.books.pymupdf")
    def test_classify_content_with_pdfs(self, mock_pymupdf, config, mega_task):
        """Classify content with PDF files."""
        dl_dir = Path(config.temp_dir) / f"mega_{mega_task['slug']}"
        dl_dir.mkdir(parents=True, exist_ok=True)

        (dl_dir / "clean.pdf").write_text("dummy pdf content")

        # Mock because classify_content calls _pdf_preflight internally
        mock_doc = MagicMock()
        mock_doc.page_count = 20
        mock_doc.__getitem__.return_value.get_text.return_value = (
            "The quick brown fox jumps over the lazy dog. "
            * 10
        )
        mock_pymupdf.open.return_value = mock_doc

        result = step_classify_content(mega_task, config)
        data = json.loads(result)

        assert data["total_files"] == 1
        for book in data["books"]:
            assert book["extension"] == ".pdf"
            assert "classification" in book
            assert "quality_score" in book


# ── Pipeline Registration Tests ───────────────────────────────────────────────

class TestPipelineRegistration:
    """Tests that pipelines are correctly registered."""

    def test_book_pipeline_registered(self):
        """The 'book' pipeline is registered."""
        from agent.pipelines import get_pipeline
        pipeline = get_pipeline("book")
        assert pipeline is not None
        step_names = [s.name for s in pipeline]
        assert "extract_text" in step_names
        assert "synthesize_note" in step_names
        assert "quality_gate" in step_names
        assert "minio_upload" in step_names
        assert "git_push" in step_names

    def test_book_pdf_pipeline_registered(self):
        """The 'book_pdf' pipeline is registered."""
        from agent.pipelines import get_pipeline
        pipeline = get_pipeline("book_pdf")
        assert pipeline is not None

    def test_mega_pipeline_registered(self):
        """The 'mega' pipeline is registered."""
        from agent.pipelines import get_pipeline
        pipeline = get_pipeline("mega")
        assert pipeline is not None
        step_names = [s.name for s in pipeline]
        assert "megadl_download" in step_names
        assert "classify_content" in step_names

    def test_book_batch_pipeline_registered(self):
        """The 'book_batch' pipeline is registered."""
        from agent.pipelines import get_pipeline
        pipeline = get_pipeline("book_batch")
        assert pipeline is not None


# ── Book Constants Tests ──────────────────────────────────────────────────────

class TestConstants:
    """Tests for books module constants."""

    def test_book_extensions_includes_common(self):
        """BOOK_EXTENSIONS includes epub, pdf, mobi, azw3."""
        assert ".epub" in BOOK_EXTENSIONS
        assert ".pdf" in BOOK_EXTENSIONS
        assert ".mobi" in BOOK_EXTENSIONS
        assert ".azw3" in BOOK_EXTENSIONS

    def test_scanned_threshold_reasonable(self):
        """PDF_MIN_CHARS_SCANNED is a reasonable value."""
        assert PDF_MIN_CHARS_SCANNED > 0
        assert PDF_MIN_CHARS_SCANNED < 10000

    def test_quality_threshold_reasonable(self):
        """PDF_QUALITY_SCORE_THRESHOLD is 0-100."""
        assert 0 <= PDF_QUALITY_SCORE_THRESHOLD <= 100
