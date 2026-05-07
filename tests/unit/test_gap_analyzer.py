"""
test_gap_analyzer.py — Unit Tests for src/gap_analyzer.py

These tests verify analyze_skills_gap() behavior for specific input conditions —
edge cases, error paths, and content correctness — rather than just the output shape
(which is covered by the contract tests).

Key behaviors tested:
  - All-zero feature vector → all 10 top skills land in missing_skills
  - Feature vector with all top-10 dims active → present_skills covers all 10
  - Invalid niche name → {"error": "INVALID_NICHE", ...}
  - Vector length != 40 → {"error": "INVALID_VECTOR", ...}
  - Artifact missing niche_top_skills → {"error": "MISSING_BENCHMARK", ...}
  - All labels returned are strings, not raw integer dimension indices

Spec reference: specs/005-skills-gap-analytics/tasks.md (T011)
"""

import pytest

from src.gap_analyzer import analyze_skills_gap
from src.model_io import CANONICAL_NICHES
from src.feature_vector import VECTOR_INDEX_ORDER


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def minimal_artifact():
    """
    A minimal model artifact dict with niche_top_skills for testing.
    Does not include a real trained model — gap_analyzer only needs niche_top_skills.

    Each niche gets a unique set of 10 dimension indices derived from its position
    in CANONICAL_NICHES, ensuring the test can distinguish per-niche behavior.
    """
    niche_top_skills = {}
    for i, niche in enumerate(CANONICAL_NICHES):
        # Give each niche distinct indices by starting at a niche-specific offset.
        # Modulo 40 keeps all indices in the valid 0–39 range.
        top10 = [
            (
                (i * 6 + j) % 40,
                f"{VECTOR_INDEX_ORDER[(i * 6 + j) % 40][0]}_"
                f"{VECTOR_INDEX_ORDER[(i * 6 + j) % 40][1]}",
            )
            for j in range(10)
        ]
        niche_top_skills[niche] = top10
    return {"niche_top_skills": niche_top_skills}


@pytest.fixture
def zero_vector():
    """A 40-element all-zero feature vector — user has no CV signal anywhere."""
    return [0] * 40


@pytest.fixture
def full_vector():
    """A 40-element all-ones feature vector — user has signal in every dimension."""
    return [1] * 40


# ── Tests: validation error paths ─────────────────────────────────────────────

class TestValidationErrors:
    """Tests that invalid inputs return the correct error codes."""

    def test_vector_length_not_40_returns_invalid_vector_error(self, minimal_artifact):
        """
        A feature vector shorter than 40 must return {"error": "INVALID_VECTOR"}.
        The gap analyzer cannot operate on a vector that doesn't match the schema.
        """
        result = analyze_skills_gap([0] * 39, "robotics", minimal_artifact)
        assert result.get("error") == "INVALID_VECTOR", (
            f"Expected INVALID_VECTOR for 39-element vector, got: {result}"
        )

    def test_vector_length_too_long_returns_invalid_vector_error(self, minimal_artifact):
        """
        A feature vector longer than 40 must also return {"error": "INVALID_VECTOR"}.
        """
        result = analyze_skills_gap([0] * 41, "robotics", minimal_artifact)
        assert result.get("error") == "INVALID_VECTOR", (
            f"Expected INVALID_VECTOR for 41-element vector, got: {result}"
        )

    def test_invalid_niche_returns_invalid_niche_error(self, zero_vector, minimal_artifact):
        """
        An unrecognised niche name must return {"error": "INVALID_NICHE"}.
        This prevents silent wrong results if a caller uses a free-text niche name.
        """
        result = analyze_skills_gap(zero_vector, "not_a_niche", minimal_artifact)
        assert result.get("error") == "INVALID_NICHE", (
            f"Expected INVALID_NICHE for unknown niche, got: {result}"
        )

    def test_artifact_missing_niche_top_skills_returns_missing_benchmark_error(
        self, zero_vector
    ):
        """
        An artifact without 'niche_top_skills' must return {"error": "MISSING_BENCHMARK"}.
        This is the expected error when calling with a pre-Phase-5 artifact.
        """
        result = analyze_skills_gap(zero_vector, "robotics", {})
        assert result.get("error") == "MISSING_BENCHMARK", (
            f"Expected MISSING_BENCHMARK for artifact without niche_top_skills, got: {result}"
        )

    def test_error_dicts_contain_message_key(self, minimal_artifact):
        """
        Every error dict must include a non-empty 'message' field so the caller
        can display a useful diagnostic to the user.
        """
        results = [
            analyze_skills_gap([0] * 39, "robotics", minimal_artifact),
            analyze_skills_gap([0] * 40, "not_a_niche", minimal_artifact),
            analyze_skills_gap([0] * 40, "robotics", {}),
        ]
        for r in results:
            assert "message" in r, f"Error dict must have 'message' key: {r}"
            assert isinstance(r["message"], str) and len(r["message"]) > 0, (
                f"Error 'message' must be non-empty string: {r}"
            )


# ── Tests: all-zero vector ─────────────────────────────────────────────────────

class TestAllZeroVector:
    """Tests for a user with no CV signal — all skills will be missing."""

    def test_all_zero_vector_all_skills_missing(self, zero_vector, minimal_artifact):
        """
        When the feature vector is all zeros, no dimension is active, so all
        10 benchmark skills for any niche must appear in missing_skills.
        """
        result = analyze_skills_gap(zero_vector, "robotics", minimal_artifact)
        assert "error" not in result, f"Unexpected error: {result}"
        assert len(result["missing_skills"]) == 10, (
            f"All-zero vector should produce 10 missing skills, "
            f"got {len(result['missing_skills'])}"
        )

    def test_all_zero_vector_no_present_skills(self, zero_vector, minimal_artifact):
        """
        With an all-zero vector, present_skills must be empty.
        """
        result = analyze_skills_gap(zero_vector, "robotics", minimal_artifact)
        assert "error" not in result
        assert result["present_skills"] == [], (
            f"All-zero vector should produce empty present_skills, "
            f"got {result['present_skills']}"
        )

    def test_all_zero_vector_coverage_is_zero(self, zero_vector, minimal_artifact):
        """
        All-zero vector → coverage_percentage must be 0.0.
        """
        result = analyze_skills_gap(zero_vector, "robotics", minimal_artifact)
        assert "error" not in result
        assert result["coverage_percentage"] == 0.0, (
            f"All-zero vector should give 0.0% coverage, "
            f"got {result['coverage_percentage']}"
        )


# ── Tests: full vector (all dims active) ─────────────────────────────────────

class TestFullVector:
    """Tests for a user with signal in every CV dimension — nothing will be missing."""

    def test_full_vector_no_missing_skills(self, full_vector, minimal_artifact):
        """
        When every dimension is active (value = 1), all benchmark skills are
        covered and missing_skills must be empty.
        """
        result = analyze_skills_gap(full_vector, "robotics", minimal_artifact)
        assert "error" not in result, f"Unexpected error: {result}"
        assert result["missing_skills"] == [], (
            f"Full vector should produce empty missing_skills, "
            f"got {result['missing_skills']}"
        )

    def test_full_vector_all_skills_present(self, full_vector, minimal_artifact):
        """
        With every dimension active, all 10 benchmark skills must be present.
        """
        result = analyze_skills_gap(full_vector, "robotics", minimal_artifact)
        assert "error" not in result
        assert len(result["present_skills"]) == 10, (
            f"Full vector should produce 10 present skills, "
            f"got {len(result['present_skills'])}"
        )

    def test_full_vector_coverage_is_100(self, full_vector, minimal_artifact):
        """
        Full vector → coverage_percentage must be 100.0.
        """
        result = analyze_skills_gap(full_vector, "robotics", minimal_artifact)
        assert "error" not in result
        assert result["coverage_percentage"] == 100.0, (
            f"Full vector should give 100.0% coverage, "
            f"got {result['coverage_percentage']}"
        )


# ── Tests: label quality ───────────────────────────────────────────────────────

class TestLabelQuality:
    """
    Tests that skill labels in the result are human-readable strings, not raw
    integer indices or bare internal labels.
    """

    def test_all_skill_labels_are_strings(self, zero_vector, minimal_artifact):
        """
        Every label in missing_skills and present_skills must be a string.
        Raw integer dimension indices must never appear in the output.
        """
        result = analyze_skills_gap(zero_vector, "robotics", minimal_artifact)
        assert "error" not in result
        for label in result["missing_skills"] + result["present_skills"]:
            assert isinstance(label, str), (
                f"Skill label must be str, got {type(label)}: {label!r}"
            )

    def test_skill_labels_are_not_raw_integers(self, zero_vector, minimal_artifact):
        """
        Labels must never be bare integers. A label like 4 instead of
        "robotic_framework_skills" would be meaningless to the user.
        """
        result = analyze_skills_gap(zero_vector, "industrial_automation", minimal_artifact)
        assert "error" not in result
        for label in result["missing_skills"] + result["present_skills"]:
            assert not isinstance(label, int), (
                f"Skill label must not be an int, got {label!r}"
            )

    def test_skill_labels_are_non_empty(self, zero_vector, minimal_artifact):
        """
        Every skill label must be a non-empty string — empty strings would display
        as blank bullets in the PDF report.
        """
        result = analyze_skills_gap(zero_vector, "automotive", minimal_artifact)
        assert "error" not in result
        for label in result["missing_skills"] + result["present_skills"]:
            assert len(label) > 0, "Skill label must not be an empty string"

    def test_skills_total_always_10(self, minimal_artifact):
        """
        present + missing must always sum to exactly 10 regardless of vector values.
        This is the fixed benchmark size.
        """
        for vector in ([0] * 40, [1] * 40, [0, 1] * 20):
            result = analyze_skills_gap(vector, "embedded_systems", minimal_artifact)
            assert "error" not in result
            total = len(result["present_skills"]) + len(result["missing_skills"])
            assert total == 10, (
                f"present + missing must equal 10, got {total} for vector {vector[:4]}..."
            )


# ── Tests: all 6 niches work ───────────────────────────────────────────────────

class TestAllNichesWork:
    """Each of the 6 canonical niches must produce a valid result."""

    @pytest.mark.parametrize("niche", CANONICAL_NICHES)
    def test_each_canonical_niche_returns_valid_result(
        self, niche, zero_vector, minimal_artifact
    ):
        """
        analyze_skills_gap() must succeed for every canonical niche name.
        """
        result = analyze_skills_gap(zero_vector, niche, minimal_artifact)
        assert "error" not in result, (
            f"Niche '{niche}' returned error: {result}"
        )
        assert result["target_niche"] == niche
        assert result["benchmark_size"] == 10
