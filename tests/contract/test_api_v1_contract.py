"""
test_api_v1_contract.py — API Contract Tests for POST /api/v1/analyze

These tests verify the HTTP contract for the analysis endpoint:
  - correct status codes for validation failures
  - correct JSON error shapes
  - correct success response shape

All pipeline work is mocked out via invoke_with_timeout so we're
testing the HTTP layer only — not the ML pipeline.

Written BEFORE implementation (TDD — must FAIL until routes.py exists).
"""

import base64
import io
from unittest import mock

import pytest

# We import create_app lazily inside each test so that import errors give
# clearer failure messages during the "failing" TDD phase.

# ── Shared mock pipeline result ───────────────────────────────────────────────
# This is what invoke_with_timeout returns when the pipeline succeeds.
# We use a fixed value so every success test sees the same predictable data.
MOCK_PIPELINE_RESULT = {
    "fit_score_result": {
        "ranked_niches": [
            {"rank": 1, "niche": "robotics",               "composite_score": 0.87},
            {"rank": 2, "niche": "embedded_systems",        "composite_score": 0.75},
            {"rank": 3, "niche": "industrial_automation",   "composite_score": 0.65},
            {"rank": 4, "niche": "automotive",              "composite_score": 0.55},
            {"rank": 5, "niche": "mechanical_design",       "composite_score": 0.45},
            {"rank": 6, "niche": "technical_management",    "composite_score": 0.35},
        ],
        "pivot_applied": False,
        "pivot_explanation": None,
    },
    "skills_gap_result": {
        "target_niche":       "robotics",
        "present_skills":     ["ROS", "Python"],
        "missing_skills":     ["C++"],
        "coverage_percentage": 70.0,
        "benchmark_size":     10,
    },
    "chart_result": {"chart_image_bytes": b"PNG_DATA"},
    "career_roadmap": {
        "pdf_bytes": b"%PDF-1.4 mock",
        "filename":  "career_roadmap_20260423.pdf",
    },
}


def _make_pdf_bytes(size: int = 128) -> bytes:
    """Build the bare minimum bytes that look like a PDF to our validator."""
    # A real tiny PDF header — enough for content_type + .pdf extension checks
    header = b"%PDF-1.4\n"
    padding = b"x" * max(0, size - len(header))
    return header + padding


def _get_client():
    """Create a Flask test client from the app factory."""
    from src.app import create_app
    app = create_app()
    app.config["TESTING"] = True
    # No real model artifact needed — pipeline is mocked
    app.config["MODEL_ARTIFACT"] = None
    return app.test_client()


# ── Validation failure tests ──────────────────────────────────────────────────

class TestB001MissingFile:
    """POST with no file at all must return 400 VALIDATION_ERROR."""

    def test_B001_missing_file(self):
        client = _get_client()
        # Send a multipart POST with no cv_file field
        response = client.post(
            "/api/v1/analyze",
            data={"work_environment": "0"},
            content_type="multipart/form-data",
        )
        assert response.status_code == 400
        body = response.get_json()
        assert body["error"] is True
        assert "CV" in body["message"] or "upload" in body["message"].lower()
        assert body["code"] == "VALIDATION_ERROR"


class TestB002NonPdfFile:
    """POST with a .docx file must return 400 with PDF-specific message."""

    def test_B002_non_pdf_file(self):
        client = _get_client()
        fake_docx = (io.BytesIO(b"PK\x03\x04fake docx content"), "resume.docx")
        response = client.post(
            "/api/v1/analyze",
            data={"cv_file": fake_docx},
            content_type="multipart/form-data",
        )
        assert response.status_code == 400
        body = response.get_json()
        assert body["error"] is True
        assert "PDF" in body["message"]
        assert body["code"] == "VALIDATION_ERROR"


class TestB003OversizedFile:
    """PDF exceeding 10 MB must return 400 size error.

    We test this at two levels:
      1. validate_cv_upload rejects a file whose byte count exceeds 10 MB
         (validated directly via the unit test in test_form_validation.py).
      2. Flask's MAX_CONTENT_LENGTH guard raises 413 when the entire request
         body is too large — our 413 handler converts this to a 400-shape JSON
         response so the front end receives the same error shape either way.

    Testing with a true 11 MB multipart body in the test client requires werkzeug
    to spool the body through a temp file, which can fail when the OS temp
    partition is low on space. We therefore test the 413 path by temporarily
    lowering MAX_CONTENT_LENGTH to 1 KB — functionally equivalent because the
    handler we're exercising is the same error handler either way.
    """

    def test_B003_oversized_file(self):
        from src.app import create_app
        app = create_app()
        app.config["TESTING"] = True
        app.config["MODEL_ARTIFACT"] = None
        # Lower the limit so even a small body triggers 413 — avoids writing
        # a 11 MB temp file to disk in environments with limited temp space.
        app.config["MAX_CONTENT_LENGTH"] = 1024  # 1 KB limit for this test
        client = app.test_client()

        # A body that exceeds the 1 KB MAX_CONTENT_LENGTH we set above.
        # The content is still PDF-typed so the size check is reached.
        over_limit_pdf = (io.BytesIO(b"%PDF-1.4\n" + b"x" * 2048), "big.pdf")
        response = client.post(
            "/api/v1/analyze",
            data={"cv_file": over_limit_pdf},
            content_type="multipart/form-data",
        )
        # Flask raises 413 → our errorhandler converts it to 413 JSON.
        # The front end checks for both 400 and 413 and shows the same message.
        assert response.status_code in (400, 413)
        body = response.get_json()
        assert body["error"] is True
        assert "10MB" in body["message"] or "10 MB" in body["message"]
        assert body["code"] == "VALIDATION_ERROR"


class TestB004PreferenceOutOfRange:
    """POST with work_environment='3' (out of [-2,2]) must return 422."""

    def test_B004_preference_out_of_range(self):
        client = _get_client()
        pdf_bytes = _make_pdf_bytes()
        response = client.post(
            "/api/v1/analyze",
            data={
                "cv_file": (io.BytesIO(pdf_bytes), "cv.pdf"),
                "work_environment": "3",   # invalid — range is -2..2
            },
            content_type="multipart/form-data",
        )
        assert response.status_code == 422
        body = response.get_json()
        assert body["error"] is True
        assert body["code"] == "VALIDATION_ERROR"


# ── Success tests (pipeline mocked) ──────────────────────────────────────────

class TestB008SuccessRankedNiches:
    """POST with a valid PDF and mocked pipeline → 200, ranked_niches length ≥ 3."""

    def test_B008_success_ranked_niches(self):
        client = _get_client()
        pdf_bytes = _make_pdf_bytes()

        with mock.patch(
            "src.api.v1.routes.invoke_with_timeout",
            return_value=MOCK_PIPELINE_RESULT,
        ):
            response = client.post(
                "/api/v1/analyze",
                data={"cv_file": (io.BytesIO(pdf_bytes), "cv.pdf")},
                content_type="multipart/form-data",
            )

        assert response.status_code == 200
        body = response.get_json()
        assert "ranked_niches" in body
        assert len(body["ranked_niches"]) >= 3


class TestB011PdfB64NonEmpty:
    """POST with valid PDF, mocked pipeline → 200, pdf_b64 is a non-empty base64 string."""

    def test_B011_pdf_b64_nonempty(self):
        client = _get_client()
        pdf_bytes = _make_pdf_bytes()

        with mock.patch(
            "src.api.v1.routes.invoke_with_timeout",
            return_value=MOCK_PIPELINE_RESULT,
        ):
            response = client.post(
                "/api/v1/analyze",
                data={"cv_file": (io.BytesIO(pdf_bytes), "cv.pdf")},
                content_type="multipart/form-data",
            )

        assert response.status_code == 200
        body = response.get_json()
        assert "pdf_b64" in body
        pdf_b64 = body["pdf_b64"]
        assert isinstance(pdf_b64, str) and len(pdf_b64) > 0
        # Must be valid base64 — atob() on the client side requires this
        decoded = base64.b64decode(pdf_b64)
        assert len(decoded) > 0
        # B-011: decoded bytes must begin with the PDF magic header (contract requirement)
        assert decoded[:4] == b"%PDF", (
            f"pdf_b64 decoded bytes must start with %PDF, got {decoded[:8]!r}"
        )
