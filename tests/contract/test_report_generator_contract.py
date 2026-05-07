"""
test_report_generator_contract.py — Contract Tests for generate_career_roadmap_pdf()

These tests verify the behavioral CONTRACT of generate_career_roadmap_pdf():
the shape, types, and invariants of every valid return value.

Contract: specs/005-skills-gap-analytics/contracts/
"""

import struct
import zlib

import pytest

from src.model_io import CANONICAL_NICHES
from src.report_generator import generate_career_roadmap_pdf


# ── Minimal valid PNG (stdlib-only, no PIL/Pillow needed) ─────────────────────

def _make_minimal_png() -> bytes:
    """Build a valid 1×1 white-pixel PNG using only stdlib (struct + zlib)."""
    def _chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = _chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    idat = _chunk(b"IDAT", zlib.compress(b"\x00\xff\xff\xff"))
    iend = _chunk(b"IEND", b"")
    return sig + ihdr + idat + iend


_MINIMAL_PNG: bytes = _make_minimal_png()


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def valid_fit_score_result():
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
    return {
        "target_niche":        CANONICAL_NICHES[0],
        "present_skills":      ["PLC Programming"],
        "missing_skills":      ["SCADA", "HMI Design"],
        "coverage_percentage": 33.3,
        "benchmark_size":      10,
    }


# ── Contract tests ────────────────────────────────────────────────────────────

class TestGenerateCareerRoadmapPdfContract:
    """
    Contract: generate_career_roadmap_pdf() must return a dict with the correct
    keys, types, and value invariants on every valid call.
    """

    def test_returns_dict(
        self, valid_fit_score_result, valid_skills_gap_result
    ):
        """Return value must always be a dict."""
        result = generate_career_roadmap_pdf(
            valid_fit_score_result, valid_skills_gap_result, _MINIMAL_PNG
        )
        assert isinstance(result, dict)

    def test_success_result_has_all_required_keys(
        self, valid_fit_score_result, valid_skills_gap_result
    ):
        """
        A successful result must contain: pdf_bytes, filename, generated_at.
        """
        result = generate_career_roadmap_pdf(
            valid_fit_score_result, valid_skills_gap_result, _MINIMAL_PNG
        )
        assert "error" not in result, f"Unexpected error: {result}"
        required_keys = {"pdf_bytes", "filename", "generated_at"}
        missing = required_keys - set(result.keys())
        assert not missing, f"Result missing required keys: {missing}"

    def test_pdf_bytes_is_bytes(
        self, valid_fit_score_result, valid_skills_gap_result
    ):
        """pdf_bytes must be a bytes object."""
        result = generate_career_roadmap_pdf(
            valid_fit_score_result, valid_skills_gap_result, _MINIMAL_PNG
        )
        assert "error" not in result
        assert isinstance(result["pdf_bytes"], bytes), (
            f"pdf_bytes must be bytes, got {type(result['pdf_bytes'])}"
        )

    def test_pdf_bytes_starts_with_pdf_header(
        self, valid_fit_score_result, valid_skills_gap_result
    ):
        """pdf_bytes must start with b'%PDF' — the standard PDF signature."""
        result = generate_career_roadmap_pdf(
            valid_fit_score_result, valid_skills_gap_result, _MINIMAL_PNG
        )
        assert "error" not in result
        assert result["pdf_bytes"][:4] == b"%PDF", (
            "pdf_bytes must start with '%PDF'"
        )

    def test_filename_is_string(
        self, valid_fit_score_result, valid_skills_gap_result
    ):
        """filename must be a non-empty string."""
        result = generate_career_roadmap_pdf(
            valid_fit_score_result, valid_skills_gap_result, _MINIMAL_PNG
        )
        assert "error" not in result
        filename = result["filename"]
        assert isinstance(filename, str) and len(filename) > 0, (
            f"filename must be a non-empty string, got {filename!r}"
        )

    def test_filename_matches_pattern(
        self, valid_fit_score_result, valid_skills_gap_result
    ):
        """filename must match the pattern 'career_roadmap_*.pdf'."""
        result = generate_career_roadmap_pdf(
            valid_fit_score_result, valid_skills_gap_result, _MINIMAL_PNG
        )
        assert "error" not in result
        filename = result["filename"]
        assert filename.startswith("career_roadmap_"), (
            f"filename must start with 'career_roadmap_', got {filename!r}"
        )
        assert filename.endswith(".pdf"), (
            f"filename must end with '.pdf', got {filename!r}"
        )

    def test_generated_at_is_string(
        self, valid_fit_score_result, valid_skills_gap_result
    ):
        """generated_at must be a non-empty string (ISO 8601 UTC timestamp)."""
        result = generate_career_roadmap_pdf(
            valid_fit_score_result, valid_skills_gap_result, _MINIMAL_PNG
        )
        assert "error" not in result
        generated_at = result["generated_at"]
        assert isinstance(generated_at, str) and len(generated_at) > 0, (
            f"generated_at must be a non-empty string, got {generated_at!r}"
        )

    def test_never_raises_on_valid_input(
        self, valid_fit_score_result, valid_skills_gap_result
    ):
        """generate_career_roadmap_pdf() must never raise on valid inputs."""
        try:
            result = generate_career_roadmap_pdf(
                valid_fit_score_result, valid_skills_gap_result, _MINIMAL_PNG
            )
            assert isinstance(result, dict)
        except Exception as e:
            pytest.fail(
                f"generate_career_roadmap_pdf() raised {type(e).__name__}: {e}"
            )

    def test_error_dict_has_error_and_message_keys(
        self, valid_fit_score_result, valid_skills_gap_result
    ):
        """Error dicts must have both 'error' and 'message' string keys."""
        result = generate_career_roadmap_pdf(
            valid_fit_score_result, valid_skills_gap_result, b""
        )
        assert "error" in result
        assert "message" in result
        assert isinstance(result["error"], str)
        assert isinstance(result["message"], str)
