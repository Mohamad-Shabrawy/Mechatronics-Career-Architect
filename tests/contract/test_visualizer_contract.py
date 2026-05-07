"""
test_visualizer_contract.py — Contract Tests for generate_radar_chart()

These tests verify the behavioral CONTRACT of generate_radar_chart():
the shape, types, and invariants of every valid return value.

Contract: specs/005-skills-gap-analytics/contracts/
"""

import pytest

from src.model_io import CANONICAL_NICHES
from src.visualizer import generate_radar_chart


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def valid_fit_score_result():
    """A valid FitScoreResult with all 6 canonical niches."""
    return {
        "ranked_niches": [
            {
                "niche":            niche,
                "composite_score":  0.3 + i * 0.1,
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


# ── Contract tests ────────────────────────────────────────────────────────────

class TestGenerateRadarChartContract:
    """
    Contract: generate_radar_chart() must return a dict with the correct keys,
    types, and value invariants on every valid call.
    """

    def test_returns_dict(self, valid_fit_score_result):
        """Return value must always be a dict."""
        result = generate_radar_chart(valid_fit_score_result)
        assert isinstance(result, dict)

    def test_success_result_has_all_required_keys(self, valid_fit_score_result):
        """
        A successful result must contain:
        chart_image_bytes, niche_labels, scores, chart_format.
        """
        result = generate_radar_chart(valid_fit_score_result)
        assert "error" not in result, f"Unexpected error: {result}"
        required_keys = {"chart_image_bytes", "niche_labels", "scores", "chart_format"}
        missing = required_keys - set(result.keys())
        assert not missing, f"Result missing required keys: {missing}"

    def test_chart_image_bytes_is_bytes(self, valid_fit_score_result):
        """chart_image_bytes must be a bytes object."""
        result = generate_radar_chart(valid_fit_score_result)
        assert "error" not in result
        assert isinstance(result["chart_image_bytes"], bytes), (
            f"chart_image_bytes must be bytes, got {type(result['chart_image_bytes'])}"
        )

    def test_chart_image_bytes_is_valid_png(self, valid_fit_score_result):
        """chart_image_bytes must start with the PNG magic header \\x89PNG."""
        result = generate_radar_chart(valid_fit_score_result)
        assert "error" not in result
        assert result["chart_image_bytes"][:4] == b"\x89PNG", (
            "chart_image_bytes must be a valid PNG (starts with \\x89PNG)"
        )

    def test_niche_labels_is_list_of_6_strings(self, valid_fit_score_result):
        """niche_labels must be a list of exactly 6 strings."""
        result = generate_radar_chart(valid_fit_score_result)
        assert "error" not in result
        labels = result["niche_labels"]
        assert isinstance(labels, list), f"niche_labels must be list, got {type(labels)}"
        assert len(labels) == 6, f"niche_labels must have 6 entries, got {len(labels)}"
        for lbl in labels:
            assert isinstance(lbl, str), f"Each niche label must be str, got {type(lbl)}"

    def test_scores_is_list_of_6_floats(self, valid_fit_score_result):
        """scores must be a list of exactly 6 numeric values."""
        result = generate_radar_chart(valid_fit_score_result)
        assert "error" not in result
        scores = result["scores"]
        assert isinstance(scores, list), f"scores must be list, got {type(scores)}"
        assert len(scores) == 6, f"scores must have 6 entries, got {len(scores)}"
        for s in scores:
            assert isinstance(s, (int, float)), (
                f"Each score must be numeric, got {type(s)}"
            )

    def test_chart_format_is_png_string(self, valid_fit_score_result):
        """chart_format must be the string 'png'."""
        result = generate_radar_chart(valid_fit_score_result)
        assert "error" not in result
        assert result["chart_format"] == "png", (
            f"chart_format must be 'png', got {result['chart_format']!r}"
        )

    def test_niche_labels_matches_canonical_niches_order(self, valid_fit_score_result):
        """niche_labels must equal CANONICAL_NICHES (same order, same values)."""
        result = generate_radar_chart(valid_fit_score_result)
        assert "error" not in result
        assert result["niche_labels"] == CANONICAL_NICHES

    def test_never_raises_on_valid_input(self, valid_fit_score_result):
        """generate_radar_chart() must never raise on a valid input."""
        try:
            result = generate_radar_chart(valid_fit_score_result)
            assert isinstance(result, dict)
        except Exception as e:
            pytest.fail(
                f"generate_radar_chart() raised {type(e).__name__}: {e} on valid input"
            )

    def test_error_dict_has_error_and_message_keys(self):
        """Error dicts must have both 'error' (str) and 'message' (str) keys."""
        result = generate_radar_chart({"error": "upstream_error"})
        assert "error" in result
        assert "message" in result
        assert isinstance(result["error"], str)
        assert isinstance(result["message"], str)
