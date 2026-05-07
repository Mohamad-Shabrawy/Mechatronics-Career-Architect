"""
test_visualizer.py — Unit Tests for src/visualizer.py

These tests verify generate_radar_chart() behavior for specific inputs —
valid FitScoreResult, error inputs, and content correctness.

Key behaviors tested:
  - Valid 6-niche FitScoreResult → PNG bytes returned
  - FitScoreResult with "error" key → {"error": "INVALID_FIT_RESULT", ...}
  - ranked_niches list length != 6 → error dict
  - FitScoreResult missing a niche → error dict
  - niche_labels in result matches CANONICAL_NICHES
  - scores in result match composite_score values from input

Spec reference: specs/005-skills-gap-analytics/tasks.md (T015)
"""

import pytest

from src.model_io import CANONICAL_NICHES
from src.visualizer import generate_radar_chart


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_fit_score_result():
    """
    A valid FitScoreResult with all 6 canonical niches, each with a distinct
    composite_score. Scores are assigned in CANONICAL_NICHES order for
    predictable assertions.
    """
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
            for i, niche in enumerate(sorted(CANONICAL_NICHES))
        ],
        "pivot_applied":  False,
        "fallback_used":  False,
    }


# ── Tests: valid input ─────────────────────────────────────────────────────────

class TestValidInput:
    """Tests for a correctly formed FitScoreResult."""

    def test_returns_dict(self, sample_fit_score_result):
        """generate_radar_chart() must always return a dict."""
        result = generate_radar_chart(sample_fit_score_result)
        assert isinstance(result, dict), (
            f"generate_radar_chart() must return dict, got {type(result)}"
        )

    def test_returns_png_bytes(self, sample_fit_score_result):
        """
        The chart_image_bytes field must be non-empty bytes that start with
        the PNG magic bytes \x89PNG (the standard PNG file signature).
        """
        result = generate_radar_chart(sample_fit_score_result)
        assert "error" not in result, f"Unexpected error: {result}"
        chart_bytes = result.get("chart_image_bytes")
        assert isinstance(chart_bytes, bytes), (
            f"chart_image_bytes must be bytes, got {type(chart_bytes)}"
        )
        assert len(chart_bytes) > 0, "chart_image_bytes must not be empty"
        assert chart_bytes[:4] == b"\x89PNG", (
            "chart_image_bytes must start with PNG magic bytes \\x89PNG"
        )

    def test_chart_format_is_png(self, sample_fit_score_result):
        """chart_format must always be 'png'."""
        result = generate_radar_chart(sample_fit_score_result)
        assert "error" not in result
        assert result.get("chart_format") == "png", (
            f"chart_format must be 'png', got {result.get('chart_format')!r}"
        )

    def test_niche_labels_matches_canonical_niches(self, sample_fit_score_result):
        """
        niche_labels in the result must match CANONICAL_NICHES exactly —
        same elements, same order. Consistent ordering makes charts comparable.
        """
        result = generate_radar_chart(sample_fit_score_result)
        assert "error" not in result
        assert result.get("niche_labels") == CANONICAL_NICHES, (
            f"niche_labels must be CANONICAL_NICHES, got {result.get('niche_labels')}"
        )

    def test_scores_match_composite_scores_from_input(self, sample_fit_score_result):
        """
        scores in the result must correspond to the composite_score values from
        ranked_niches, in CANONICAL_NICHES order.
        """
        result = generate_radar_chart(sample_fit_score_result)
        assert "error" not in result

        # Build expected score list in CANONICAL_NICHES order
        score_map = {
            e["niche"]: e["composite_score"]
            for e in sample_fit_score_result["ranked_niches"]
        }
        expected_scores = [score_map[n] for n in CANONICAL_NICHES]

        assert result.get("scores") == expected_scores, (
            f"scores must match composite_scores in CANONICAL_NICHES order. "
            f"Expected {expected_scores}, got {result.get('scores')}"
        )

    def test_niche_labels_has_6_entries(self, sample_fit_score_result):
        """niche_labels must have exactly 6 entries (one per canonical niche)."""
        result = generate_radar_chart(sample_fit_score_result)
        assert "error" not in result
        labels = result.get("niche_labels", [])
        assert len(labels) == 6, (
            f"niche_labels must have 6 entries, got {len(labels)}"
        )

    def test_scores_has_6_entries(self, sample_fit_score_result):
        """scores must have exactly 6 entries (one per canonical niche)."""
        result = generate_radar_chart(sample_fit_score_result)
        assert "error" not in result
        scores = result.get("scores", [])
        assert len(scores) == 6, f"scores must have 6 entries, got {len(scores)}"


# ── Tests: error inputs ────────────────────────────────────────────────────────

class TestErrorInputs:
    """Tests that invalid inputs return the correct error code."""

    def test_error_dict_input_returns_invalid_fit_result(self):
        """
        A FitScoreResult that is itself an error dict must return
        {"error": "INVALID_FIT_RESULT"}.
        """
        error_input = {"error": "SOME_ERROR", "message": "something went wrong"}
        result = generate_radar_chart(error_input)
        assert result.get("error") == "INVALID_FIT_RESULT", (
            f"Expected INVALID_FIT_RESULT for error-dict input, got: {result}"
        )

    def test_ranked_niches_missing_returns_invalid_fit_result(self):
        """
        A dict without 'ranked_niches' key must return {"error": "INVALID_FIT_RESULT"}.
        """
        result = generate_radar_chart({"no_ranked_niches": True})
        assert result.get("error") == "INVALID_FIT_RESULT", (
            f"Expected INVALID_FIT_RESULT for missing ranked_niches, got: {result}"
        )

    def test_ranked_niches_wrong_length_returns_invalid_fit_result(self):
        """
        A ranked_niches list with length != 6 must return {"error": "INVALID_FIT_RESULT"}.
        """
        result = generate_radar_chart({
            "ranked_niches": [
                {"niche": n, "composite_score": 0.5}
                for n in CANONICAL_NICHES[:5]   # Only 5 — missing one
            ]
        })
        assert result.get("error") == "INVALID_FIT_RESULT", (
            f"Expected INVALID_FIT_RESULT for 5-entry ranked_niches, got: {result}"
        )

    def test_missing_niche_in_ranked_niches_returns_invalid_fit_result(self):
        """
        A ranked_niches list where a canonical niche is replaced with an unknown
        niche name must return {"error": "INVALID_FIT_RESULT"}.
        """
        # Replace one canonical niche with an invalid name
        bad_ranked = [
            {"niche": n if n != "robotics" else "unknown_niche", "composite_score": 0.5}
            for n in CANONICAL_NICHES
        ]
        result = generate_radar_chart({"ranked_niches": bad_ranked})
        assert result.get("error") == "INVALID_FIT_RESULT", (
            f"Expected INVALID_FIT_RESULT for missing canonical niche, got: {result}"
        )

    def test_non_dict_input_returns_invalid_fit_result(self):
        """
        A non-dict input must return {"error": "INVALID_FIT_RESULT"}.
        """
        result = generate_radar_chart("not a dict")
        assert result.get("error") == "INVALID_FIT_RESULT"

    def test_error_result_has_message_key(self):
        """Every error dict must include a non-empty 'message' key."""
        result = generate_radar_chart({})
        assert "message" in result, "Error dict must have 'message' key"
        assert isinstance(result["message"], str) and len(result["message"]) > 0


# ── Tests: never raises ────────────────────────────────────────────────────────

class TestNeverRaises:
    """generate_radar_chart() must never raise — all failures are error dicts."""

    def test_does_not_raise_on_bad_inputs(self):
        """Various malformed inputs must not cause exceptions."""
        bad_inputs = [
            None,
            42,
            "string",
            [],
            {},
            {"ranked_niches": None},
            {"ranked_niches": []},
            {"error": "x"},
        ]
        for inp in bad_inputs:
            try:
                result = generate_radar_chart(inp)
                assert isinstance(result, dict), (
                    f"Must return dict for input {inp!r}, got {type(result)}"
                )
            except Exception as e:
                pytest.fail(
                    f"generate_radar_chart() raised {type(e).__name__}: {e} "
                    f"for input {inp!r}"
                )
