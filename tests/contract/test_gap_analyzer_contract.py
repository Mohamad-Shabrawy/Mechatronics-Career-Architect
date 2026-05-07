"""
test_gap_analyzer_contract.py — Contract Tests for analyze_skills_gap()

These tests verify the behavioral CONTRACT of analyze_skills_gap():
the shape, types, and invariants of every valid return value, regardless
of the specific input values. Any implementation that passes these tests
satisfies the Phase 5 US1 output contract.

Contract: specs/005-skills-gap-analytics/contracts/
"""

import pytest

from src.gap_analyzer import analyze_skills_gap
from src.model_io import CANONICAL_NICHES
from src.feature_vector import VECTOR_INDEX_ORDER


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def artifact_with_top_skills():
    """
    A minimal model artifact dict with niche_top_skills for contract testing.
    Each niche gets 10 (dim_index, label) tuples with indices in range 0–39.
    """
    niche_top_skills = {}
    for i, niche in enumerate(CANONICAL_NICHES):
        top10 = [
            (
                (i * 6 + j) % 40,
                f"{VECTOR_INDEX_ORDER[(i * 6 + j) % 40][0]}_"
                f"{VECTOR_INDEX_ORDER[(i * 6 + j) % 40][1]}",
            )
            for j in range(10)
        ]
        niche_top_skills[niche] = top10
    return {
        "niche_top_skills": niche_top_skills,
        "niche_centroids": {n: [0.0] * 40 for n in CANONICAL_NICHES},
    }


@pytest.fixture
def sample_vector():
    """A 40-element feature vector with mixed zero/nonzero values."""
    v = [0] * 40
    for i in range(0, 40, 3):
        v[i] = 1  # every third dimension is active
    return v


# ── Contract tests ────────────────────────────────────────────────────────────

class TestAnalyzeSkillsGapContract:
    """
    Contract: analyze_skills_gap() must return a dict with the correct keys,
    types, and value invariants on every valid call.
    """

    def test_returns_dict(self, sample_vector, artifact_with_top_skills):
        """Return value must always be a dict."""
        result = analyze_skills_gap(sample_vector, "robotics", artifact_with_top_skills)
        assert isinstance(result, dict), (
            f"analyze_skills_gap() must return dict, got {type(result)}"
        )

    def test_success_result_has_all_required_keys(
        self, sample_vector, artifact_with_top_skills
    ):
        """
        A successful result must contain all 5 required keys:
        target_niche, present_skills, missing_skills, coverage_percentage, benchmark_size.
        """
        result = analyze_skills_gap(sample_vector, "robotics", artifact_with_top_skills)
        assert "error" not in result, f"Unexpected error: {result}"
        required_keys = {
            "target_niche", "present_skills", "missing_skills",
            "coverage_percentage", "benchmark_size",
        }
        missing = required_keys - set(result.keys())
        assert not missing, f"Result is missing required keys: {missing}"

    def test_target_niche_matches_input(self, sample_vector, artifact_with_top_skills):
        """target_niche in the result must match the input target_niche."""
        for niche in CANONICAL_NICHES:
            result = analyze_skills_gap(sample_vector, niche, artifact_with_top_skills)
            assert "error" not in result
            assert result["target_niche"] == niche, (
                f"target_niche should be '{niche}', got '{result['target_niche']}'"
            )

    def test_present_skills_is_list_of_strings(
        self, sample_vector, artifact_with_top_skills
    ):
        """present_skills must be a list of strings."""
        result = analyze_skills_gap(sample_vector, "automotive", artifact_with_top_skills)
        assert "error" not in result
        assert isinstance(result["present_skills"], list), (
            f"present_skills must be list, got {type(result['present_skills'])}"
        )
        for s in result["present_skills"]:
            assert isinstance(s, str), f"Each present skill must be str, got {type(s)}"

    def test_missing_skills_is_list_of_strings(
        self, sample_vector, artifact_with_top_skills
    ):
        """missing_skills must be a list of strings."""
        result = analyze_skills_gap(sample_vector, "automotive", artifact_with_top_skills)
        assert "error" not in result
        assert isinstance(result["missing_skills"], list), (
            f"missing_skills must be list, got {type(result['missing_skills'])}"
        )
        for s in result["missing_skills"]:
            assert isinstance(s, str), f"Each missing skill must be str, got {type(s)}"

    def test_present_plus_missing_equals_10(
        self, sample_vector, artifact_with_top_skills
    ):
        """
        len(present_skills) + len(missing_skills) must always equal 10.
        This is the fixed benchmark size — every dimension is either present or absent.
        """
        for niche in CANONICAL_NICHES:
            result = analyze_skills_gap(sample_vector, niche, artifact_with_top_skills)
            assert "error" not in result
            total = len(result["present_skills"]) + len(result["missing_skills"])
            assert total == 10, (
                f"present + missing must == 10, got {total} for niche '{niche}'"
            )

    def test_benchmark_size_is_10(self, sample_vector, artifact_with_top_skills):
        """benchmark_size must always be 10 (the fixed benchmark size per spec)."""
        result = analyze_skills_gap(sample_vector, "robotics", artifact_with_top_skills)
        assert "error" not in result
        assert result["benchmark_size"] == 10, (
            f"benchmark_size must be 10, got {result['benchmark_size']}"
        )

    def test_coverage_percentage_is_float_in_valid_range(
        self, sample_vector, artifact_with_top_skills
    ):
        """
        coverage_percentage must be a float (or int) in the range [0.0, 100.0].
        It represents the fraction of the 10 benchmark skills the user already has.
        """
        result = analyze_skills_gap(sample_vector, "robotics", artifact_with_top_skills)
        assert "error" not in result
        cov = result["coverage_percentage"]
        assert isinstance(cov, (int, float)), (
            f"coverage_percentage must be numeric, got {type(cov)}"
        )
        assert 0.0 <= cov <= 100.0, (
            f"coverage_percentage must be in [0, 100], got {cov}"
        )

    def test_never_raises_on_any_valid_input(self, artifact_with_top_skills):
        """
        analyze_skills_gap() must never raise an exception — all failures are
        returned as error dicts. We call with a variety of valid and edge-case
        vectors to verify no exception escapes.
        """
        vectors = [
            [0] * 40,
            [1] * 40,
            [0.0] * 40,
            [0, 1] * 20,
            [float(i) for i in range(40)],
        ]
        for v in vectors:
            for niche in CANONICAL_NICHES:
                try:
                    result = analyze_skills_gap(v, niche, artifact_with_top_skills)
                    assert isinstance(result, dict)
                except Exception as e:
                    pytest.fail(
                        f"analyze_skills_gap() raised {type(e).__name__}: {e} "
                        f"for niche='{niche}', vector={v[:4]}..."
                    )

    def test_error_dict_has_error_and_message_keys(self, artifact_with_top_skills):
        """
        When an invalid input is given, the returned error dict must have both
        'error' (a string code) and 'message' (a human-readable explanation).
        """
        # Trigger INVALID_VECTOR
        result = analyze_skills_gap([0] * 5, "robotics", artifact_with_top_skills)
        assert "error" in result, "Error dict must have 'error' key"
        assert "message" in result, "Error dict must have 'message' key"
        assert isinstance(result["error"], str)
        assert isinstance(result["message"], str)
