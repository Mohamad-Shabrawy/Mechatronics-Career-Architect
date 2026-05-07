"""
test_report_generator.py — Unit Tests for src/report_generator.py

These tests verify generate_career_roadmap_pdf() behavior for specific inputs —
valid combinations, error inputs, and structural checks.

Key behaviors tested:
  - Valid inputs → PDF bytes with %PDF header returned
  - fit_score_result error dict → {"error": "INVALID_INPUT", ...}
  - skills_gap_result error dict → {"error": "INVALID_INPUT", ...}
  - Empty chart_image_bytes → {"error": "INVALID_INPUT", ...}
  - missing_skills == [] → PDF still generated (report still works with 100% coverage)
  - No files are written to disk — all in-memory

Spec reference: specs/005-skills-gap-analytics/tasks.md (T019)
"""

import os
import struct
import zlib

import pytest

from src.model_io import CANONICAL_NICHES
from src.report_generator import generate_career_roadmap_pdf


# ── Minimal valid PNG (stdlib-only, no PIL/Pillow needed) ─────────────────────
# Builds a 1×1 RGB PNG that reportlab's ImageReader can parse correctly.
# Using hand-crafted bytes literal caused "broken data stream" errors in
# reportlab because the truncated literal was missing required chunk data.

def _make_minimal_png() -> bytes:
    """Build a valid 1×1 white-pixel PNG using only stdlib (struct + zlib)."""
    def _chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    sig = b"\x89PNG\r\n\x1a\n"
    # IHDR: width=1, height=1, bit_depth=8, color_type=2 (RGB), compress/filter/interlace=0
    ihdr = _chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    # IDAT: one scanline — filter byte 0x00 + RGB bytes for white (255,255,255)
    idat = _chunk(b"IDAT", zlib.compress(b"\x00\xff\xff\xff"))
    iend = _chunk(b"IEND", b"")
    return sig + ihdr + idat + iend


_MINIMAL_PNG: bytes = _make_minimal_png()


@pytest.fixture
def valid_fit_score_result():
    """A valid FitScoreResult with all 6 niches ranked."""
    return {
        "ranked_niches": [
            {
                "niche":            niche,
                "composite_score":  0.5 + i * 0.05,
                "cv_component":     0.5,
                "intent_component": 0.5,
                "penalty":          0.0,
                "rank":             i + 1,
            }
            for i, niche in enumerate(CANONICAL_NICHES)
        ],
        "pivot_applied": False,
        "fallback_used": False,
    }


@pytest.fixture
def valid_skills_gap_result():
    """A valid SkillsGapResult for the first canonical niche."""
    return {
        "target_niche":        CANONICAL_NICHES[0],
        "present_skills":      ["PLC Programming", "SCADA Development"],
        "missing_skills":      ["HMI Design", "Fieldbus Protocols", "Functional Safety"],
        "coverage_percentage": 40.0,
        "benchmark_size":      10,
    }


@pytest.fixture
def minimal_png():
    """Minimal valid PNG bytes for embedding in the PDF."""
    return _MINIMAL_PNG


# ── Tests: valid inputs ────────────────────────────────────────────────────────

class TestValidInputs:
    """Tests for correctly formed inputs."""

    def test_returns_dict(
        self, valid_fit_score_result, valid_skills_gap_result, minimal_png
    ):
        """generate_career_roadmap_pdf() must always return a dict."""
        result = generate_career_roadmap_pdf(
            valid_fit_score_result, valid_skills_gap_result, minimal_png
        )
        assert isinstance(result, dict), (
            f"generate_career_roadmap_pdf() must return dict, got {type(result)}"
        )

    def test_pdf_bytes_starts_with_pdf_header(
        self, valid_fit_score_result, valid_skills_gap_result, minimal_png
    ):
        """
        pdf_bytes must start with b'%PDF' — the standard PDF file header.
        This verifies we produced a real PDF, not empty bytes or garbage.
        """
        result = generate_career_roadmap_pdf(
            valid_fit_score_result, valid_skills_gap_result, minimal_png
        )
        assert "error" not in result, f"Unexpected error: {result}"
        pdf_bytes = result.get("pdf_bytes")
        assert isinstance(pdf_bytes, bytes), (
            f"pdf_bytes must be bytes, got {type(pdf_bytes)}"
        )
        assert pdf_bytes[:4] == b"%PDF", (
            "pdf_bytes must start with '%PDF' (standard PDF header)"
        )

    def test_pdf_bytes_is_non_empty(
        self, valid_fit_score_result, valid_skills_gap_result, minimal_png
    ):
        """pdf_bytes must be non-empty."""
        result = generate_career_roadmap_pdf(
            valid_fit_score_result, valid_skills_gap_result, minimal_png
        )
        assert "error" not in result
        assert len(result["pdf_bytes"]) > 100, "pdf_bytes must be a substantial PDF"

    def test_filename_matches_expected_pattern(
        self, valid_fit_score_result, valid_skills_gap_result, minimal_png
    ):
        """
        filename must start with 'career_roadmap_' and end with '.pdf'.
        This is the suggested download filename for the HTTP response.
        """
        result = generate_career_roadmap_pdf(
            valid_fit_score_result, valid_skills_gap_result, minimal_png
        )
        assert "error" not in result
        filename = result.get("filename", "")
        assert filename.startswith("career_roadmap_"), (
            f"filename must start with 'career_roadmap_', got: {filename!r}"
        )
        assert filename.endswith(".pdf"), (
            f"filename must end with '.pdf', got: {filename!r}"
        )

    def test_generated_at_is_iso_string(
        self, valid_fit_score_result, valid_skills_gap_result, minimal_png
    ):
        """generated_at must be a non-empty ISO 8601 string."""
        result = generate_career_roadmap_pdf(
            valid_fit_score_result, valid_skills_gap_result, minimal_png
        )
        assert "error" not in result
        generated_at = result.get("generated_at", "")
        assert isinstance(generated_at, str) and len(generated_at) > 0, (
            f"generated_at must be a non-empty string, got {generated_at!r}"
        )
        # ISO 8601 strings contain 'T' between date and time
        assert "T" in generated_at, (
            f"generated_at should be ISO 8601 (contains 'T'), got: {generated_at!r}"
        )


# ── Tests: no files written to disk ───────────────────────────────────────────

class TestNoFilesOnDisk:
    """PDF must be generated entirely in-memory — no disk I/O."""

    def test_no_new_files_written_to_current_directory(
        self, valid_fit_score_result, valid_skills_gap_result, minimal_png, tmp_path
    ):
        """
        generate_career_roadmap_pdf() must not write any files to disk.
        We verify by checking that no new .pdf files appear in a temp directory.
        """
        before_files = set(os.listdir(tmp_path))
        os.chdir(tmp_path)  # Temporarily change to tmp_path to catch accidental writes
        try:
            result = generate_career_roadmap_pdf(
                valid_fit_score_result, valid_skills_gap_result, minimal_png
            )
            assert "error" not in result
        finally:
            import pathlib
            os.chdir(pathlib.Path(__file__).parent.parent.parent)

        after_files = set(os.listdir(tmp_path))
        new_files = after_files - before_files
        pdf_files = [f for f in new_files if f.endswith(".pdf")]
        assert not pdf_files, (
            f"generate_career_roadmap_pdf() wrote PDF files to disk: {pdf_files}. "
            "All output must be returned in-memory only."
        )


# ── Tests: error inputs ────────────────────────────────────────────────────────

class TestErrorInputs:
    """Tests that invalid inputs return the correct error code."""

    def test_fit_score_result_error_dict_returns_invalid_input(
        self, valid_skills_gap_result, minimal_png
    ):
        """
        If fit_score_result is itself an error dict, must return
        {"error": "INVALID_INPUT"}.
        """
        result = generate_career_roadmap_pdf(
            {"error": "SOME_ERROR", "message": "failed"},
            valid_skills_gap_result,
            minimal_png,
        )
        assert result.get("error") == "INVALID_INPUT", (
            f"Expected INVALID_INPUT for error fit_score_result, got: {result}"
        )

    def test_skills_gap_result_error_dict_returns_invalid_input(
        self, valid_fit_score_result, minimal_png
    ):
        """
        If skills_gap_result is itself an error dict, must return
        {"error": "INVALID_INPUT"}.
        """
        result = generate_career_roadmap_pdf(
            valid_fit_score_result,
            {"error": "GAP_ERROR", "message": "failed"},
            minimal_png,
        )
        assert result.get("error") == "INVALID_INPUT", (
            f"Expected INVALID_INPUT for error skills_gap_result, got: {result}"
        )

    def test_empty_chart_image_bytes_returns_invalid_input(
        self, valid_fit_score_result, valid_skills_gap_result
    ):
        """
        Empty bytes for chart_image_bytes must return {"error": "INVALID_INPUT"}.
        """
        result = generate_career_roadmap_pdf(
            valid_fit_score_result,
            valid_skills_gap_result,
            b"",
        )
        assert result.get("error") == "INVALID_INPUT", (
            f"Expected INVALID_INPUT for empty chart_image_bytes, got: {result}"
        )

    def test_error_dict_has_message_key(
        self, valid_fit_score_result, valid_skills_gap_result
    ):
        """Error dicts must include a non-empty 'message' key."""
        result = generate_career_roadmap_pdf(
            valid_fit_score_result, valid_skills_gap_result, b""
        )
        assert "message" in result, "Error dict must have 'message' key"
        assert isinstance(result["message"], str) and len(result["message"]) > 0


# ── Tests: edge cases ─────────────────────────────────────────────────────────

class TestEdgeCases:
    """Edge cases that must still produce a valid PDF."""

    def test_empty_missing_skills_still_generates_pdf(
        self, valid_fit_score_result, minimal_png
    ):
        """
        A skills_gap_result with missing_skills == [] (100% coverage) must
        still generate a valid PDF. The report should note no missing skills.
        """
        gap_result_no_missing = {
            "target_niche":        CANONICAL_NICHES[0],
            "present_skills":      [f"Skill {i}" for i in range(10)],
            "missing_skills":      [],
            "coverage_percentage": 100.0,
            "benchmark_size":      10,
        }
        result = generate_career_roadmap_pdf(
            valid_fit_score_result, gap_result_no_missing, minimal_png
        )
        assert "error" not in result, (
            f"Empty missing_skills should not cause error, got: {result}"
        )
        assert result.get("pdf_bytes", b"")[:4] == b"%PDF", (
            "Should produce valid PDF even with no missing skills"
        )

    def test_large_missing_skills_list_still_generates_pdf(
        self, valid_fit_score_result, minimal_png
    ):
        """
        A gap result with all 10 skills missing must still generate a valid PDF.
        """
        gap_result_all_missing = {
            "target_niche":        CANONICAL_NICHES[0],
            "present_skills":      [],
            "missing_skills":      [f"Missing Skill {i}" for i in range(10)],
            "coverage_percentage": 0.0,
            "benchmark_size":      10,
        }
        result = generate_career_roadmap_pdf(
            valid_fit_score_result, gap_result_all_missing, minimal_png
        )
        assert "error" not in result, f"Unexpected error: {result}"
        assert result.get("pdf_bytes", b"")[:4] == b"%PDF"
