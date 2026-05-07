"""
test_scorer.py — Unit Tests for src/scorer.py (Phase 4)

These tests cover the internal logic of compute_fit_score() and its helpers:
  - Cosine similarity computation (identical vectors, orthogonal vectors, zero vector)
  - Zero-vector guard: fallback_used=True, cv_component=0.0 for all niches
  - Invalid vector length: error dict returned, no exception raised
  - All 6 niches always present in output
  - composite_score values always in [0.0, 1.0]
  - Neutral preferences produce equal intent_component=0.5 for all niches
  - Strong preferences shift intent scores in the expected direction
  - Missing preference key defaults to neutral (no delta)
  - Extreme stacked preferences clamp to [0.0, 1.0] for intent_component
  - Override penalty demotes conflicted niche (pivot_applied=True)
  - Neutral preferences produce pivot_applied=False with zero penalties
  - Max-conflict penalty sufficient to demote niche by ≥1 rank position
  - SC-002 performance: <50ms per call for 100 randomized inputs
  - SC-005: all 6 niches always present, no exceptions raised

Spec reference: specs/004-hybrid-scoring-logic/tasks.md (T009-T023)
"""

import time

import numpy as np
import pytest

from src.model_io import CANONICAL_NICHES
from src.scorer import (
    compute_fit_score,
    _cosine_similarity,
    _compute_intent_component,
    _apply_override_penalties,
    load_scoring_config,
)


# ── FIXTURES ──────────────────────────────────────────────────────────────────

def _make_artifact_with_centroids(seed: int = 42) -> dict:
    """
    Creates a Phase 4-compatible model artifact with niche_centroids.
    Each niche has a clearly distinct centroid — the centroid for niche i has a
    large value at dimension i*4 and zeros elsewhere. This makes cosine similarity
    tests predictable: a vector with signal at dim 4 will be most similar to
    the robotics centroid (niche index 1, dim 1*4=4).
    """
    from sklearn.ensemble import RandomForestClassifier
    from src.model_io import current_schema_version

    rng = np.random.RandomState(seed)
    X, y, centroids = [], [], {}

    for i, niche in enumerate(CANONICAL_NICHES):
        niche_vecs = []
        for _ in range(20):
            v = [0.0] * 40
            v[i * 4] = float(10 + rng.randint(0, 2))
            X.append(v)
            y.append(niche)
            niche_vecs.append(v)
        centroids[niche] = list(np.array(niche_vecs).mean(axis=0))

    clf = RandomForestClassifier(n_estimators=10, random_state=seed)
    clf.fit(np.array(X), np.array(y))

    return {
        "model":           clf,
        "niche_labels":    [str(c) for c in clf.classes_],
        "training_record": {},
        "schema_version":  current_schema_version(),
        "model_version":   "test_unit_artifact",
        "niche_centroids": centroids,
    }


@pytest.fixture(scope="module")
def artifact():
    """Shared model artifact for all unit tests — built once per test session."""
    return _make_artifact_with_centroids()


@pytest.fixture
def neutral_prefs():
    """All-zero preferences — no bias toward any niche."""
    return {
        "work_environment":  0,
        "system_level":      0,
        "industry_interest": 0,
        "team_scale":        0,
        "travel_tolerance":  0,
    }


def _niche_vector(niche_name: str) -> list[int]:
    """
    Returns a feature vector with strong signal at the dominant dimension for niche_name.
    Dominant dimension for niche at index i in CANONICAL_NICHES = i * 4.
    """
    i = CANONICAL_NICHES.index(niche_name)
    v = [0] * 40
    v[i * 4] = 10
    return v


def _competitive_auto_vector() -> list[int]:
    """
    Returns a feature vector where 'automotive' (dim 12) leads over 'technical_management'
    (dim 20) with neutral preferences — but the advantage is narrow enough that the
    work_environment=2 penalty (0.25) plus the intent shift (-0.10 auto, +0.10 tm) is
    sufficient to demote automotive from rank #1.

    With neutral prefs, automotive cosine ≈ 0.832 vs technical_management ≈ 0.555, so
    automotive ranks #1. With work_environment=2 plus the 0.25 penalty, technical_management's
    intent-boosted composite (≈ 0.573) exceeds automotive's post-penalty composite (≈ 0.409).

    This vector is used only by penalty-demotion tests (SC-004, FR-006) that need to
    verify rank change — not by the "strong signal ranks correct niche" tests.
    """
    # automotive is at index 3 → dim 12; technical_management is at index 5 → dim 20
    auto_idx = CANONICAL_NICHES.index("automotive")
    tm_idx = CANONICAL_NICHES.index("technical_management")
    v = [0] * 40
    v[auto_idx * 4] = 3   # Moderate automotive signal: cosine ≈ 0.832 vs automotive centroid
    v[tm_idx * 4] = 2     # Smaller technical_management signal: cosine ≈ 0.555 vs tm centroid
    return v


def _score_by_niche(result: dict) -> dict[str, float]:
    """Helper: extract {niche: composite_score} dict from a compute_fit_score result."""
    return {ns["niche"]: ns["composite_score"] for ns in result["ranked_niches"]}


def _rank_by_niche(result: dict) -> dict[str, int]:
    """Helper: extract {niche: rank} dict from a compute_fit_score result."""
    return {ns["niche"]: ns["rank"] for ns in result["ranked_niches"]}


# ── TESTS: cosine similarity helper ──────────────────────────────────────────

class TestCosineSimilarity:
    """Unit tests for the _cosine_similarity() private helper."""

    def test_identical_vectors_have_similarity_1(self):
        """
        Cosine similarity of a vector with itself must be exactly 1.0.
        This is the mathematical identity: cos(0°) = 1.
        """
        v = np.array([1.0, 2.0, 3.0] + [0.0] * 37)
        result = _cosine_similarity(v, list(v))
        assert result == pytest.approx(1.0, abs=1e-9), (
            f"Identical vectors must have cosine similarity 1.0, got {result}"
        )

    def test_orthogonal_vectors_have_similarity_0(self):
        """
        Cosine similarity of orthogonal vectors (90° apart) must be 0.0.
        We use vectors that are mathematically orthogonal: dot([1,0,...], [0,1,...]) = 0.
        """
        v1 = np.array([1.0] + [0.0] * 39)
        v2 = [0.0, 1.0] + [0.0] * 38
        result = _cosine_similarity(v1, v2)
        assert result == pytest.approx(0.0, abs=1e-9), (
            f"Orthogonal vectors must have cosine similarity 0.0, got {result}"
        )

    def test_zero_feature_vector_returns_0(self):
        """
        A zero feature vector must return 0.0 (the zero-vector guard from FR-015).
        Dividing by norm(zero) would produce NaN — the guard prevents this.
        """
        zero_fv = np.array([0.0] * 40)
        centroid = [1.0, 0.0, 0.5] + [0.0] * 37
        result = _cosine_similarity(zero_fv, centroid)
        assert result == 0.0, (
            f"Zero feature vector must return cosine similarity 0.0, got {result}"
        )

    def test_zero_centroid_returns_0(self):
        """
        If the centroid is all zeros (niche had no training samples), return 0.0.
        This is the second zero-vector guard case.
        """
        fv = np.array([1.0, 2.0, 3.0] + [0.0] * 37)
        zero_centroid = [0.0] * 40
        result = _cosine_similarity(fv, zero_centroid)
        assert result == 0.0, (
            f"Zero centroid must return cosine similarity 0.0, got {result}"
        )

    def test_result_always_in_0_to_1_range(self):
        """
        Cosine similarity must always be in [0.0, 1.0]. We test with 50 random
        pairs of vectors to catch any floating-point rounding edge cases.
        """
        rng = np.random.RandomState(123)
        for _ in range(50):
            # Use non-negative vectors to ensure cosine is non-negative
            v1 = np.abs(rng.randn(40))
            v2 = list(np.abs(rng.randn(40)))
            sim = _cosine_similarity(v1, v2)
            assert 0.0 <= sim <= 1.0, (
                f"Cosine similarity out of [0, 1] range: {sim}"
            )

    def test_cosine_similarity_is_symmetric(self):
        """
        cos(a, b) should equal cos(b, a). Our implementation uses dot(a,b)/(|a||b|)
        which is symmetric by definition — this test guards against implementation bugs.
        """
        v1 = np.array([3.0, 1.0, 4.0, 1.0, 5.0] + [0.0] * 35)
        v2 = [1.0, 6.0, 1.0, 8.0, 0.0] + [0.0] * 35
        assert _cosine_similarity(v1, v2) == pytest.approx(
            _cosine_similarity(np.array(v2), list(v1)), abs=1e-9
        ), "Cosine similarity must be symmetric"


# ── TESTS: zero-vector input ──────────────────────────────────────────────────

class TestZeroVectorInput:
    """
    Tests the zero-vector guard (FR-015): when the feature vector is all zeros,
    cv_component must be 0.0 for all niches, and fallback_used must be True.
    """

    def test_zero_vector_sets_fallback_used_true(self, artifact, neutral_prefs):
        """Zero feature vector must set fallback_used=True in the result."""
        result = compute_fit_score([0] * 40, artifact, neutral_prefs)
        assert result.get("fallback_used") is True, (
            f"Zero vector must set fallback_used=True, got: {result.get('fallback_used')}"
        )

    def test_zero_vector_all_cv_components_are_zero(self, artifact, neutral_prefs):
        """Zero feature vector must produce cv_component=0.0 for all niches."""
        result = compute_fit_score([0] * 40, artifact, neutral_prefs)
        for ns in result["ranked_niches"]:
            assert ns["cv_component"] == pytest.approx(0.0, abs=1e-9), (
                f"cv_component for '{ns['niche']}' must be 0.0 for zero vector, "
                f"got {ns['cv_component']}"
            )

    def test_zero_vector_with_neutral_prefs_no_error(self, artifact, neutral_prefs):
        """
        Zero vector + neutral prefs must succeed (return a result dict, not an error).
        The 40% intent component alone still drives scoring.
        """
        result = compute_fit_score([0] * 40, artifact, neutral_prefs)
        assert "ranked_niches" in result, (
            f"Zero vector must return ranked_niches, got: {result}"
        )

    def test_zero_vector_returns_all_6_niches(self, artifact, neutral_prefs):
        """Even a zero vector must return all 6 canonical niches (FR-007)."""
        result = compute_fit_score([0] * 40, artifact, neutral_prefs)
        returned_niches = {ns["niche"] for ns in result["ranked_niches"]}
        assert returned_niches == set(CANONICAL_NICHES), (
            f"Zero vector result is missing niches: {set(CANONICAL_NICHES) - returned_niches}"
        )

    def test_nonzero_vector_sets_fallback_used_false(self, artifact, neutral_prefs):
        """A non-zero feature vector must have fallback_used=False."""
        result = compute_fit_score(_niche_vector("robotics"), artifact, neutral_prefs)
        assert result.get("fallback_used") is False, (
            f"Non-zero vector must have fallback_used=False, got: {result.get('fallback_used')}"
        )


# ── TESTS: invalid input handling ────────────────────────────────────────────

class TestInvalidInputHandling:
    """
    Tests that invalid inputs produce INVALID_VECTOR error dicts — not exceptions.
    This verifies FR-009 (never raise) and the INVALID_VECTOR error contract.
    """

    def test_39_element_vector_returns_invalid_vector_error(self, artifact):
        """A 39-element vector must return INVALID_VECTOR error dict."""
        result = compute_fit_score([0] * 39, artifact)
        assert result.get("error") == "INVALID_VECTOR", (
            f"39-element vector must produce INVALID_VECTOR error, got: {result}"
        )

    def test_41_element_vector_returns_invalid_vector_error(self, artifact):
        """A 41-element vector must return INVALID_VECTOR error dict."""
        result = compute_fit_score([0] * 41, artifact)
        assert result.get("error") == "INVALID_VECTOR", (
            f"41-element vector must produce INVALID_VECTOR error, got: {result}"
        )

    def test_empty_vector_returns_invalid_vector_error(self, artifact):
        """An empty list must return INVALID_VECTOR error dict."""
        result = compute_fit_score([], artifact)
        assert result.get("error") == "INVALID_VECTOR"

    def test_none_vector_returns_error_not_exception(self, artifact):
        """None as feature_vector must return an error dict, not raise."""
        result = compute_fit_score(None, artifact)
        assert isinstance(result, dict), "None vector must return a dict"
        assert "error" in result, "None vector must return an error dict"

    def test_error_dict_has_message(self, artifact):
        """All error dicts must include a non-empty 'message' key."""
        result = compute_fit_score([0] * 5, artifact)
        assert "message" in result, "Error dict must have 'message' key"
        assert isinstance(result["message"], str) and result["message"], (
            "Error 'message' must be a non-empty string"
        )

    def test_compute_fit_score_never_raises(self, artifact):
        """
        compute_fit_score() must NEVER raise regardless of input garbage.
        This is the core promise of FR-009 and Behavioral Guarantee #1.
        """
        garbage_inputs = [
            [0] * 39,
            [0] * 41,
            [],
            None,
            "not_a_list",
            42,
            {"not": "a list"},
        ]
        for fv in garbage_inputs:
            try:
                result = compute_fit_score(fv, artifact)
                assert isinstance(result, dict), (
                    f"compute_fit_score must return dict for input {fv!r}, got {type(result)}"
                )
            except Exception as exc:
                pytest.fail(
                    f"compute_fit_score raised {type(exc).__name__} for input {fv!r}: {exc}"
                )


# ── TESTS: output completeness and score ranges ───────────────────────────────

class TestOutputCompletenessAndScoreRanges:
    """Tests that all 6 niches appear and all scores are in valid ranges."""

    def test_all_6_canonical_niches_in_output(self, artifact, neutral_prefs):
        """Every one of the 6 canonical niches must appear in ranked_niches."""
        result = compute_fit_score(_niche_vector("embedded_systems"), artifact, neutral_prefs)
        returned = {ns["niche"] for ns in result["ranked_niches"]}
        missing = set(CANONICAL_NICHES) - returned
        assert not missing, f"ranked_niches is missing niches: {missing}"

    def test_composite_scores_in_0_1_range(self, artifact, neutral_prefs):
        """All composite_score values must be in [0.0, 1.0] after normalization."""
        result = compute_fit_score(_niche_vector("automotive"), artifact, neutral_prefs)
        for ns in result["ranked_niches"]:
            assert 0.0 <= ns["composite_score"] <= 1.0, (
                f"composite_score for '{ns['niche']}' out of range: {ns['composite_score']}"
            )

    def test_top_niche_is_correct_for_strong_signal(self, artifact, neutral_prefs):
        """
        A vector strongly resembling the 'robotics' centroid must rank 'robotics' #1
        when preferences are neutral. This tests the core CV component logic (SC-001).
        """
        result = compute_fit_score(_niche_vector("robotics"), artifact, neutral_prefs)
        top_niche = result["ranked_niches"][0]["niche"]
        assert top_niche == "robotics", (
            f"Robotics signal vector must rank 'robotics' #1, got '{top_niche}'"
        )

    def test_top_composite_score_is_1_0_after_normalization(self, artifact, neutral_prefs):
        """
        After max-normalization, the top niche's composite_score must be exactly 1.0.
        This is the normalization guarantee from R-004.
        """
        result = compute_fit_score(_niche_vector("industrial_automation"), artifact, neutral_prefs)
        top_score = result["ranked_niches"][0]["composite_score"]
        assert top_score == pytest.approx(1.0, abs=1e-6), (
            f"Top niche composite_score must be 1.0 after normalization, got {top_score}"
        )


# ── TESTS: intent component (User Story 2) ────────────────────────────────────

class TestIntentComponent:
    """
    Tests for the intent affinity lookup (US2 — Apply Soft Intent Adjustments).
    """

    def test_neutral_prefs_produce_equal_intent_scores(self, artifact, neutral_prefs):
        """
        When all preferences are neutral (0 or None), every niche must have
        intent_component == 0.5 — the baseline value (R-002).
        """
        # Use zero vector so the 60% CV component is 0 for all niches, making
        # intent_component directly readable in the output (not mixed into composite).
        result = compute_fit_score([0] * 40, artifact, neutral_prefs)
        for ns in result["ranked_niches"]:
            assert ns["intent_component"] == pytest.approx(0.5, abs=1e-9), (
                f"Neutral prefs must give intent_component=0.5 for all niches, "
                f"got {ns['intent_component']} for '{ns['niche']}'"
            )

    def test_none_prefs_produce_equal_intent_scores(self, artifact):
        """user_preferences=None must behave identically to all-zero prefs (FR-014)."""
        result = compute_fit_score([0] * 40, artifact, user_preferences=None)
        for ns in result["ranked_niches"]:
            assert ns["intent_component"] == pytest.approx(0.5, abs=1e-9), (
                f"None prefs must give intent_component=0.5, got {ns['intent_component']} "
                f"for '{ns['niche']}'"
            )

    def test_office_preference_boosts_technical_management_intent(self, artifact):
        """
        work_environment=2 (strongly office) must produce higher intent_component
        for technical_management than for automotive (which is site-heavy).
        This tests that the affinity map shifts intent in the expected direction.
        """
        prefs = {"work_environment": 2}
        config = load_scoring_config()
        affinity = config["intent_affinity_map"]

        tm_intent = _compute_intent_component("technical_management", prefs, affinity)
        auto_intent = _compute_intent_component("automotive", prefs, affinity)

        assert tm_intent > auto_intent, (
            f"Office preference should boost technical_management intent above automotive. "
            f"technical_management={tm_intent:.3f}, automotive={auto_intent:.3f}"
        )

    def test_missing_preference_key_defaults_to_neutral_delta(self, artifact):
        """
        A preferences dict with a missing key must behave as if that key is 0.
        This tests FR-014: no error, no penalty/bonus for omitted dimensions.
        """
        config = load_scoring_config()
        affinity = config["intent_affinity_map"]

        # Explicitly neutral for work_environment, omit the rest
        prefs_partial = {"work_environment": 0}
        prefs_full_neutral = {
            "work_environment": 0, "system_level": 0,
            "industry_interest": 0, "team_scale": 0, "travel_tolerance": 0,
        }

        for niche in CANONICAL_NICHES:
            intent_partial = _compute_intent_component(niche, prefs_partial, affinity)
            intent_full = _compute_intent_component(niche, prefs_full_neutral, affinity)
            assert intent_partial == pytest.approx(intent_full, abs=1e-9), (
                f"Partial prefs must equal full neutral prefs for '{niche}': "
                f"partial={intent_partial}, full_neutral={intent_full}"
            )

    def test_extreme_stacked_preferences_clamp_to_range(self, artifact):
        """
        Multiple extreme preferences that would push intent > 1.0 or < 0.0 must
        be clamped. We use extreme values across all 5 dimensions (all at +2 or -2)
        and verify intent_component stays in [0.0, 1.0] for all niches.
        """
        config = load_scoring_config()
        affinity = config["intent_affinity_map"]

        extreme_positive = {key: 2 for key in
                            ["work_environment", "system_level", "industry_interest",
                             "team_scale", "travel_tolerance"]}
        extreme_negative = {key: -2 for key in
                            ["work_environment", "system_level", "industry_interest",
                             "team_scale", "travel_tolerance"]}

        for niche in CANONICAL_NICHES:
            pos_intent = _compute_intent_component(niche, extreme_positive, affinity)
            neg_intent = _compute_intent_component(niche, extreme_negative, affinity)
            assert 0.0 <= pos_intent <= 1.0, (
                f"Extreme positive prefs: intent_component for '{niche}' out of range: {pos_intent}"
            )
            assert 0.0 <= neg_intent <= 1.0, (
                f"Extreme negative prefs: intent_component for '{niche}' out of range: {neg_intent}"
            )

    def test_strong_prefs_change_ranking_vs_neutral(self, artifact):
        """
        Strong preferences on at least one dimension must change the ranked order
        compared to neutral preferences for a uniform feature vector. This validates
        SC-003 (soft preferences affect rankings).
        """
        # Uniform vector: each niche gets the same CV component, so intent drives ranking
        uniform_vector = [1] * 40

        result_neutral = compute_fit_score(uniform_vector, artifact, {
            k: 0 for k in ["work_environment", "system_level", "industry_interest",
                            "team_scale", "travel_tolerance"]
        })

        # Strong office + software preference → should push embedded/technical up
        result_strong = compute_fit_score(uniform_vector, artifact, {
            "work_environment":  2,
            "system_level":      2,
            "industry_interest": 2,
            "team_scale":        2,
            "travel_tolerance":  2,
        })

        ranks_neutral = _rank_by_niche(result_neutral)
        ranks_strong = _rank_by_niche(result_strong)

        # At least one niche must have changed rank position
        any_rank_change = any(
            ranks_neutral[n] != ranks_strong[n] for n in CANONICAL_NICHES
        )
        assert any_rank_change, (
            "Strong preferences must change the rank order compared to neutral. "
            f"Neutral ranks: {ranks_neutral}, Strong ranks: {ranks_strong}"
        )


# ── TESTS: override/pivot logic (User Story 3) ───────────────────────────────

class TestOverridePivotLogic:
    """
    Tests for override penalty application (US3 — Override & Pivot Logic).
    """

    def test_neutral_prefs_produce_zero_penalties(self, artifact, neutral_prefs):
        """
        Neutral preferences (all zeros) must not trigger any penalty.
        All penalty values in ranked_niches must be 0.0.
        """
        result = compute_fit_score(_niche_vector("automotive"), artifact, neutral_prefs)
        for ns in result["ranked_niches"]:
            assert ns["penalty"] == pytest.approx(0.0, abs=1e-9), (
                f"Neutral prefs must produce penalty=0.0, got {ns['penalty']} "
                f"for '{ns['niche']}'"
            )

    def test_neutral_prefs_pivot_applied_false(self, artifact, neutral_prefs):
        """Neutral preferences must produce pivot_applied=False."""
        result = compute_fit_score(_niche_vector("automotive"), artifact, neutral_prefs)
        assert result["pivot_applied"] is False, (
            f"Neutral prefs must have pivot_applied=False, got {result['pivot_applied']}"
        )

    def test_strong_office_preference_applies_penalty_to_automotive(self, artifact):
        """
        work_environment=2 (strongly office) must apply a penalty to 'automotive'
        (which is a site-heavy niche). This tests R-003 directly.
        """
        prefs = {"work_environment": 2}
        result = compute_fit_score(_niche_vector("automotive"), artifact, prefs)

        auto_entry = next(ns for ns in result["ranked_niches"] if ns["niche"] == "automotive")
        assert auto_entry["penalty"] > 0.0, (
            f"work_environment=2 must apply penalty to automotive, got penalty={auto_entry['penalty']}"
        )
        assert result["pivot_applied"] is True, (
            "Any applied penalty must set pivot_applied=True"
        )

    def test_strong_office_preference_demotes_automotive_rank(self, artifact):
        """
        When automotive is the highest-scoring CV niche (neutral prefs), applying
        work_environment=2 penalty must knock it from rank #1 (SC-004, FR-006).

        We use _competitive_auto_vector() — a vector where automotive leads with neutral
        prefs but the gap to technical_management is narrow enough that the 0.25 penalty
        plus the -0.10 intent shift for automotive (work_environment=2) tips the balance.
        Using the maximal automotive-only signal (_niche_vector) would give automotive a
        cosine advantage so large that even a 0.25 penalty cannot overcome it — which would
        be unrealistic for a real user who has BOTH automotive CV signal AND office preferences.
        """
        # Competitive vector: automotive leads, but technical_management is a real competitor
        auto_vector = _competitive_auto_vector()
        result_neutral = compute_fit_score(auto_vector, artifact, {"work_environment": 0})
        neutral_ranks = _rank_by_niche(result_neutral)
        assert neutral_ranks["automotive"] == 1, (
            f"Competitive automotive vector must rank automotive #1 with neutral prefs, "
            f"got rank {neutral_ranks['automotive']}"
        )

        # Now apply strong office preference — combined intent shift + penalty must demote automotive
        result_office = compute_fit_score(auto_vector, artifact, {"work_environment": 2})
        office_ranks = _rank_by_niche(result_office)
        assert office_ranks["automotive"] > 1, (
            f"work_environment=2 must demote automotive from #1, "
            f"got rank {office_ranks['automotive']}"
        )

    def test_penalty_demotes_niche_by_at_least_one_rank(self, artifact):
        """
        The max penalty (0.25) must be sufficient to demote a conflicted niche
        by at least 1 rank position (FR-006, SC-004).

        We use _competitive_auto_vector() where automotive leads neutrally but the
        competitor (technical_management) is close enough to benefit from the demotion.
        """
        auto_vector = _competitive_auto_vector()
        result_neutral = compute_fit_score(auto_vector, artifact, {})
        ranks_neutral = _rank_by_niche(result_neutral)
        automotive_neutral_rank = ranks_neutral["automotive"]

        result_penalized = compute_fit_score(auto_vector, artifact, {"work_environment": 2})
        ranks_penalized = _rank_by_niche(result_penalized)
        automotive_penalized_rank = ranks_penalized["automotive"]

        assert automotive_penalized_rank > automotive_neutral_rank, (
            f"Penalty must demote automotive by ≥1 rank. "
            f"Neutral rank: {automotive_neutral_rank}, Penalized rank: {automotive_penalized_rank}"
        )

    def test_penalty_values_in_score_breakdown_match_deduction(self, artifact):
        """
        The 'penalty' field in each NicheScore must reflect the actual deduction
        applied to that niche's composite score (not a generic value).
        """
        prefs = {"work_environment": 2}
        config = load_scoring_config()

        # Find the expected penalty for automotive from the config
        expected_auto_penalty = sum(
            rule["penalty"]
            for rule in config["override_penalty_config"]
            if rule["preference"] == "work_environment"
            and rule["threshold"] == 2
            and rule["niche"] == "automotive"
        )

        result = compute_fit_score(_niche_vector("automotive"), artifact, prefs)
        auto_entry = next(ns for ns in result["ranked_niches"] if ns["niche"] == "automotive")

        assert auto_entry["penalty"] == pytest.approx(expected_auto_penalty, abs=1e-6), (
            f"Automotive penalty must be {expected_auto_penalty}, got {auto_entry['penalty']}"
        )

    def test_pivot_applied_reflects_any_penalty_applied(self, artifact):
        """
        pivot_applied must be True if and only if at least one niche received a
        penalty > 0 in this call. With zero-preference it's False, with a
        conflict-triggering preference it's True.
        """
        # Case 1: No conflict → pivot_applied=False
        result_no_conflict = compute_fit_score(_niche_vector("robotics"), artifact, {})
        assert result_no_conflict["pivot_applied"] is False

        # Case 2: Conflict → pivot_applied=True
        result_conflict = compute_fit_score(
            _niche_vector("automotive"), artifact, {"work_environment": 2}
        )
        assert result_conflict["pivot_applied"] is True

    def test_apply_override_penalties_helper_applies_matching_rules(self):
        """
        Direct unit test of _apply_override_penalties(): matching rules are applied,
        non-matching rules are skipped.
        """
        # Set up a simple penalty config with two rules
        penalty_config = [
            {"preference": "work_environment", "threshold": 2, "niche": "automotive",
             "penalty": 0.25},
            {"preference": "work_environment", "threshold": 2, "niche": "mechanical_design",
             "penalty": 0.20},
            {"preference": "system_level", "threshold": -2, "niche": "embedded_systems",
             "penalty": 0.15},
        ]
        composite_scores = {n: 0.8 for n in CANONICAL_NICHES}
        user_prefs = {"work_environment": 2, "system_level": 0}  # system_level rule won't fire
        tracker = {n: 0.0 for n in CANONICAL_NICHES}

        penalized, any_applied = _apply_override_penalties(
            composite_scores, user_prefs, penalty_config, tracker
        )

        # work_environment=2 rules must fire
        assert penalized["automotive"] == pytest.approx(0.8 - 0.25, abs=1e-9)
        assert penalized["mechanical_design"] == pytest.approx(0.8 - 0.20, abs=1e-9)
        # system_level=0 rule (threshold=-2) must NOT fire
        assert penalized["embedded_systems"] == pytest.approx(0.8, abs=1e-9)
        assert any_applied is True

    def test_apply_override_penalties_no_match_returns_false(self):
        """_apply_override_penalties() must return any_applied=False when no rules match."""
        penalty_config = [
            {"preference": "work_environment", "threshold": 2, "niche": "automotive",
             "penalty": 0.25},
        ]
        composite_scores = {n: 0.7 for n in CANONICAL_NICHES}
        user_prefs = {"work_environment": 0}  # 0 != 2, rule won't fire
        tracker = {n: 0.0 for n in CANONICAL_NICHES}

        penalized, any_applied = _apply_override_penalties(
            composite_scores, user_prefs, penalty_config, tracker
        )

        assert any_applied is False
        # All scores must be unchanged
        for niche in CANONICAL_NICHES:
            assert penalized[niche] == pytest.approx(0.7, abs=1e-9)

    def test_penalty_clamps_score_to_zero_not_negative(self):
        """
        If the penalty exceeds the composite score, the result must be 0.0 (not negative).
        Behavioral Guarantee #3: composite_score in [0.0, 1.0].
        """
        penalty_config = [
            {"preference": "work_environment", "threshold": 2, "niche": "automotive",
             "penalty": 0.99},  # Larger than any realistic composite score
        ]
        composite_scores = {n: 0.5 for n in CANONICAL_NICHES}
        user_prefs = {"work_environment": 2}
        tracker = {n: 0.0 for n in CANONICAL_NICHES}

        penalized, _ = _apply_override_penalties(
            composite_scores, user_prefs, penalty_config, tracker
        )

        assert penalized["automotive"] >= 0.0, (
            f"Penalty must not produce negative composite score, got {penalized['automotive']}"
        )


# ── TESTS: performance and robustness (SC-002, SC-005) ───────────────────────

class TestPerformanceAndRobustness:
    """
    SC-002: <50ms per compute_fit_score() call on 100 randomized inputs.
    SC-005: All 6 niches always present with no exceptions across 100 random inputs.
    """

    def test_sc002_compute_fit_score_under_50ms(self, artifact, neutral_prefs):
        """
        SC-002: compute_fit_score() must complete in under 50ms per call.
        We test 100 randomized 40D inputs and assert each finishes within budget.
        The scoring is pure in-memory numpy ops — no I/O after config is cached.
        """
        rng = np.random.RandomState(999)
        times_ms = []

        for _ in range(100):
            # Random integer vector in [0, 10] — realistic feature vector range
            fv = [int(x) for x in rng.randint(0, 11, 40)]
            start = time.perf_counter()
            result = compute_fit_score(fv, artifact, neutral_prefs)
            elapsed_ms = (time.perf_counter() - start) * 1000
            times_ms.append(elapsed_ms)

            # Each individual call must be under 50ms
            assert elapsed_ms < 50.0, (
                f"compute_fit_score() took {elapsed_ms:.1f}ms — exceeds 50ms budget (SC-002)"
            )

        # Log the max time for diagnostic purposes (visible in pytest -v output)
        max_ms = max(times_ms)
        avg_ms = sum(times_ms) / len(times_ms)
        assert max_ms < 50.0, (
            f"Max latency {max_ms:.2f}ms exceeds 50ms budget across 100 calls. "
            f"Average: {avg_ms:.2f}ms"
        )

    def test_sc005_all_6_niches_always_present_no_exceptions(self, artifact):
        """
        SC-005: All 6 canonical niches must appear in every compute_fit_score() result
        across 100 randomized inputs, and no exceptions must be raised.
        """
        rng = np.random.RandomState(777)

        for i in range(100):
            fv = [int(x) for x in rng.randint(0, 11, 40)]
            # Random preferences: values from {-2, -1, 0, 1, 2}
            prefs = {
                "work_environment":  int(rng.choice([-2, -1, 0, 1, 2])),
                "system_level":      int(rng.choice([-2, -1, 0, 1, 2])),
                "industry_interest": int(rng.choice([-2, -1, 0, 1, 2])),
                "team_scale":        int(rng.choice([-2, -1, 0, 1, 2])),
                "travel_tolerance":  int(rng.choice([-2, -1, 0, 1, 2])),
            }

            try:
                result = compute_fit_score(fv, artifact, prefs)
            except Exception as exc:
                pytest.fail(
                    f"compute_fit_score() raised {type(exc).__name__} on iteration {i}: {exc}"
                )

            assert "ranked_niches" in result, (
                f"Iteration {i}: result missing 'ranked_niches'. Got: {result}"
            )
            returned_niches = {ns["niche"] for ns in result["ranked_niches"]}
            missing = set(CANONICAL_NICHES) - returned_niches
            assert not missing, (
                f"Iteration {i}: ranked_niches missing niches {missing} for fv={fv}"
            )

    def test_missing_centroids_returns_error_not_exception(self):
        """
        An artifact without niche_centroids must return a MISSING_CENTROIDS error
        dict — not raise an exception (FR-009).
        """
        from src.model_io import current_schema_version
        from sklearn.ensemble import RandomForestClassifier

        X = np.array([[float(i)] * 40 for i in range(60)])
        y = np.array([CANONICAL_NICHES[i % 6] for i in range(60)])
        clf = RandomForestClassifier(n_estimators=5, random_state=0)
        clf.fit(X, y)

        legacy_artifact = {
            "model":           clf,
            "niche_labels":    [str(c) for c in clf.classes_],
            "training_record": {},
            "schema_version":  current_schema_version(),
            "model_version":   "legacy",
        }

        try:
            result = compute_fit_score([1] * 40, legacy_artifact)
        except Exception as exc:
            pytest.fail(
                f"compute_fit_score() must not raise for missing centroids, "
                f"raised {type(exc).__name__}: {exc}"
            )

        assert result.get("error") == "MISSING_CENTROIDS", (
            f"Missing centroids must produce MISSING_CENTROIDS error, got: {result}"
        )
