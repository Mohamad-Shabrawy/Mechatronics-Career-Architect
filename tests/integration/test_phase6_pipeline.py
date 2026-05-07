"""
test_phase6_pipeline.py — Integration tests for Phase 6 API response shape

These tests verify the JSON response shape returned by POST /api/v1/analyze
for different pipeline scenarios:
  - normal ranked niches (top 3 returned)
  - null chart (chart_image_b64 is null when chart unavailable)
  - pivot applied (pivot_applied + pivot_explanation in response)
  - all-equal scores (all_equal_scores=True, all 6 niches returned)

The pipeline itself is mocked — we're testing that the route handler
assembles the correct JSON view from the pipeline output.

Written BEFORE implementation (TDD — must FAIL until routes.py exists).
"""

import base64
import io
from unittest import mock

import pytest


def _get_client():
    from src.app import create_app
    app = create_app()
    app.config["TESTING"] = True
    app.config["MODEL_ARTIFACT"] = None
    return app.test_client()


def _make_pdf_bytes() -> bytes:
    return b"%PDF-1.4\n" + b"x" * 256


def _post_with_mock(client, pipeline_result):
    """POST a valid PDF and return the response with a mocked pipeline."""
    with mock.patch(
        "src.api.v1.routes.invoke_with_timeout",
        return_value=pipeline_result,
    ):
        response = client.post(
            "/api/v1/analyze",
            data={"cv_file": (io.BytesIO(_make_pdf_bytes()), "cv.pdf")},
            content_type="multipart/form-data",
        )
    return response


class TestRankedNichesShape:
    """Normal pipeline result → top 3 niches with score_pct in response."""

    def test_ranked_niches_len_3(self):
        client = _get_client()
        result = {
            "fit_score_result": {
                "ranked_niches": [
                    {"rank": 1, "niche": "robotics",             "composite_score": 0.87},
                    {"rank": 2, "niche": "embedded_systems",     "composite_score": 0.75},
                    {"rank": 3, "niche": "industrial_automation", "composite_score": 0.65},
                    {"rank": 4, "niche": "automotive",           "composite_score": 0.55},
                    {"rank": 5, "niche": "mechanical_design",    "composite_score": 0.45},
                    {"rank": 6, "niche": "technical_management", "composite_score": 0.35},
                ],
                "pivot_applied": False,
                "pivot_explanation": None,
            },
            "skills_gap_result": {
                "target_niche": "robotics",
                "present_skills": ["ROS"],
                "missing_skills": ["C++"],
                "coverage_percentage": 70.0,
                "benchmark_size": 10,
            },
            "chart_result": {"chart_image_bytes": b"PNG_DATA"},
            "career_roadmap": {"pdf_bytes": b"%PDF-1.4 mock", "filename": "roadmap.pdf"},
        }
        response = _post_with_mock(client, result)
        assert response.status_code == 200
        body = response.get_json()
        assert len(body["ranked_niches"]) == 3

    def test_niche_has_score_pct(self):
        """Each ranked niche entry should have a score_pct (int 0-100)."""
        client = _get_client()
        result = {
            "fit_score_result": {
                "ranked_niches": [
                    {"rank": 1, "niche": "robotics",             "composite_score": 0.87},
                    {"rank": 2, "niche": "embedded_systems",     "composite_score": 0.75},
                    {"rank": 3, "niche": "industrial_automation", "composite_score": 0.65},
                    {"rank": 4, "niche": "automotive",           "composite_score": 0.55},
                    {"rank": 5, "niche": "mechanical_design",    "composite_score": 0.45},
                    {"rank": 6, "niche": "technical_management", "composite_score": 0.35},
                ],
                "pivot_applied": False,
                "pivot_explanation": None,
            },
            "skills_gap_result": {
                "target_niche": "robotics",
                "present_skills": ["ROS"],
                "missing_skills": [],
                "coverage_percentage": 100.0,
                "benchmark_size": 10,
            },
            "chart_result": {"chart_image_bytes": b"PNG_DATA"},
            "career_roadmap": {"pdf_bytes": b"%PDF-1.4 mock", "filename": "roadmap.pdf"},
        }
        response = _post_with_mock(client, result)
        body = response.get_json()
        first = body["ranked_niches"][0]
        assert "score_pct" in first
        assert first["score_pct"] == 87   # round(0.87 * 100)


class TestNullChartResponse:
    """When chart_image_bytes is None, chart_image_b64 must be null in response."""

    def test_null_chart_image_b64(self):
        client = _get_client()
        result = {
            "fit_score_result": {
                "ranked_niches": [
                    {"rank": 1, "niche": "robotics",             "composite_score": 0.80},
                    {"rank": 2, "niche": "embedded_systems",     "composite_score": 0.70},
                    {"rank": 3, "niche": "industrial_automation", "composite_score": 0.60},
                    {"rank": 4, "niche": "automotive",           "composite_score": 0.50},
                    {"rank": 5, "niche": "mechanical_design",    "composite_score": 0.40},
                    {"rank": 6, "niche": "technical_management", "composite_score": 0.30},
                ],
                "pivot_applied": False,
                "pivot_explanation": None,
            },
            "skills_gap_result": {
                "target_niche": "robotics",
                "present_skills": [],
                "missing_skills": [],
                "coverage_percentage": 0.0,
                "benchmark_size": 10,
            },
            # chart_image_bytes is None — chart rendering failed
            "chart_result": {"chart_image_bytes": None},
            "career_roadmap": {"pdf_bytes": b"%PDF-1.4 mock", "filename": "roadmap.pdf"},
        }
        response = _post_with_mock(client, result)
        assert response.status_code == 200
        body = response.get_json()
        assert body["chart_image_b64"] is None


class TestPivotApplied:
    """When pivot_applied=True, pivot_explanation must be present in response."""

    def test_pivot_fields_in_response(self):
        client = _get_client()
        result = {
            "fit_score_result": {
                "ranked_niches": [
                    {"rank": 1, "niche": "robotics",             "composite_score": 0.80},
                    {"rank": 2, "niche": "embedded_systems",     "composite_score": 0.70},
                    {"rank": 3, "niche": "industrial_automation", "composite_score": 0.60},
                    {"rank": 4, "niche": "automotive",           "composite_score": 0.50},
                    {"rank": 5, "niche": "mechanical_design",    "composite_score": 0.40},
                    {"rank": 6, "niche": "technical_management", "composite_score": 0.30},
                ],
                "pivot_applied": True,
                "pivot_explanation": "Preference pivot applied based on work environment.",
            },
            "skills_gap_result": {
                "target_niche": "robotics",
                "present_skills": [],
                "missing_skills": [],
                "coverage_percentage": 0.0,
                "benchmark_size": 10,
            },
            "chart_result": {"chart_image_bytes": b"PNG_DATA"},
            "career_roadmap": {"pdf_bytes": b"%PDF-1.4 mock", "filename": "roadmap.pdf"},
        }
        response = _post_with_mock(client, result)
        assert response.status_code == 200
        body = response.get_json()
        assert body["pivot_applied"] is True
        assert body["pivot_explanation"] is not None
        assert len(body["pivot_explanation"]) > 0


class TestAllEqualScores:
    """When all niches have identical scores, all 6 are returned and all_equal_scores=True."""

    def test_all_equal_scores_returns_six_niches(self):
        client = _get_client()
        equal_score = 0.5
        result = {
            "fit_score_result": {
                "ranked_niches": [
                    {"rank": 1, "niche": "robotics",             "composite_score": equal_score},
                    {"rank": 2, "niche": "embedded_systems",     "composite_score": equal_score},
                    {"rank": 3, "niche": "industrial_automation", "composite_score": equal_score},
                    {"rank": 4, "niche": "automotive",           "composite_score": equal_score},
                    {"rank": 5, "niche": "mechanical_design",    "composite_score": equal_score},
                    {"rank": 6, "niche": "technical_management", "composite_score": equal_score},
                ],
                "pivot_applied": False,
                "pivot_explanation": None,
            },
            "skills_gap_result": {
                "target_niche": "robotics",
                "present_skills": [],
                "missing_skills": [],
                "coverage_percentage": 0.0,
                "benchmark_size": 10,
            },
            "chart_result": {"chart_image_bytes": b"PNG_DATA"},
            "career_roadmap": {"pdf_bytes": b"%PDF-1.4 mock", "filename": "roadmap.pdf"},
        }
        response = _post_with_mock(client, result)
        assert response.status_code == 200
        body = response.get_json()
        assert body["all_equal_scores"] is True
        assert len(body["ranked_niches"]) == 6


class TestTimeoutResponse:
    """When the pipeline exceeds its timeout, the route must return 408."""

    def test_timeout_returns_408(self):
        client = _get_client()
        with mock.patch(
            "src.api.v1.routes.invoke_with_timeout",
            side_effect=TimeoutError("Analysis is taking too long — please try again"),
        ):
            response = client.post(
                "/api/v1/analyze",
                data={"cv_file": (io.BytesIO(_make_pdf_bytes()), "cv.pdf")},
                content_type="multipart/form-data",
            )
        assert response.status_code == 408
        body = response.get_json()
        assert body["error"] is True
        assert body["code"] == "TIMEOUT"

    def test_timeout_message_in_response(self):
        client = _get_client()
        with mock.patch(
            "src.api.v1.routes.invoke_with_timeout",
            side_effect=TimeoutError("Analysis is taking too long — please try again"),
        ):
            response = client.post(
                "/api/v1/analyze",
                data={"cv_file": (io.BytesIO(_make_pdf_bytes()), "cv.pdf")},
                content_type="multipart/form-data",
            )
        body = response.get_json()
        assert "too long" in body["message"].lower() or "timeout" in body["message"].lower()


class TestPipelineErrorResponse:
    """An unexpected exception in the pipeline must return 500 with a safe generic message."""

    def test_exception_returns_500(self):
        client = _get_client()
        with mock.patch(
            "src.api.v1.routes.invoke_with_timeout",
            side_effect=RuntimeError("internal sklearn error details"),
        ):
            response = client.post(
                "/api/v1/analyze",
                data={"cv_file": (io.BytesIO(_make_pdf_bytes()), "cv.pdf")},
                content_type="multipart/form-data",
            )
        assert response.status_code == 500
        body = response.get_json()
        assert body["error"] is True
        assert body["code"] == "PIPELINE_ERROR"

    def test_internal_details_not_exposed(self):
        """Raw exception text must never reach the client (spec B-006)."""
        client = _get_client()
        with mock.patch(
            "src.api.v1.routes.invoke_with_timeout",
            side_effect=RuntimeError("internal sklearn error details"),
        ):
            response = client.post(
                "/api/v1/analyze",
                data={"cv_file": (io.BytesIO(_make_pdf_bytes()), "cv.pdf")},
                content_type="multipart/form-data",
            )
        body = response.get_json()
        assert "sklearn" not in body["message"]
        assert "internal" not in body["message"].lower()


class TestHealthEndpoint:
    """GET /api/v1/health must return 200 with status and model_loaded fields."""

    def test_health_returns_200(self):
        client = _get_client()
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_health_response_shape(self):
        client = _get_client()
        body = client.get("/api/v1/health").get_json()
        assert "status" in body
        assert "model_loaded" in body
        assert body["status"] == "ok"

    def test_health_model_loaded_false_when_no_artifact(self):
        # _get_client() sets MODEL_ARTIFACT=None — model_loaded must reflect that.
        client = _get_client()
        body = client.get("/api/v1/health").get_json()
        assert body["model_loaded"] is False
