"""
test_scorer_contract.py — Contract Tests for src/scorer.py (Phase 4)

Contract tests verify the locked public interface of compute_fit_score():
  - Return dict structure (ranked_niches, pivot_applied, fallback_used always present)
  - Each NicheScore has all required keys (niche, composite_score, cv_component,
    intent_component, penalty, rank)
  - All 6 canonical niches always appear in ranked_niches
  - composite_score values are in [0.0, 1.0]
  - Error dict structure when invalid input is passed
  - Function NEVER raises (Behavioral Guarantee #1)

These tests do NOT test the accuracy of the scoring formula — that is the job
of the unit tests. Here we only check "does the output have the right shape and
types?" so that Phase 5 (gap analysis) and beyond can depend on a stable contract.

Spec reference: specs/004-hybrid-scoring-logic/contracts/scorer_contract.md
"""

import numpy as np
import pytest

from src.model_io import CANONICAL_NICHES
from src.scorer import compute_fit_score


# ── FIXTURES ──────────────────────────────────────────────────────────────────

def _make_artifact_with_centroids(n_per_niche: int = 20, seed: int = 42) -> dict:
    """
    Builds a minimal model artifact that includes niche_centroids.
    We create a small trained RandomForestClassifier inline so the artifact
    passes model_io validation without needing a full training run on disk.
    The centroid vectors are synthetic but correctly shaped (6 niches, 40D each).
    """
    from sklearn.ensemble import RandomForestClassifier
    from src.model_io import current_schema_version

    rng = np.random.RandomState(seed)

    # Build a synthetic separable dataset — each niche has a unique dominant feature.
    X, y = [], []
    centroids = {}
    for i, niche in enumerate(CANONICAL_NICHES):
        niche_vecs = []
        for _ in range(n_per_niche):
            v = [0.0] * 40
            v[i * 4] = float(10 + rng.randint(0, 3))   # Unique dominant dim per niche
            X.append(v)
            y.append(niche)
            niche_vecs.append(v)
        # Centroid = mean of this niche's training vectors
        centroids[niche] = list(np.array(niche_vecs).mean(axis=0))

    clf = RandomForestClassifier(n_estimators=10, random_state=seed)
    clf.fit(np.array(X), np.array(y))

    return {
        "model":            clf,
        "niche_labels":     [str(c) for c in clf.classes_],
        "training_record":  {},
        "schema_version":   current_schema_version(),
        "model_version":    "test_contract_artifact",
        "niche_centroids":  centroids,
    }


@pytest.fixture(scope="module")
def artifact():
    """Shared Phase 4-compatible model artifact for all contract tests."""
    return _make_artifact_with_centroids()


@pytest.fixture
def neutral_prefs():
    """Neutral user preferences — all zero (no bias toward any niche)."""
    return {
        "work_environment":  0,
        "system_level":      0,
        "industry_interest": 0,
        "team_scale":        0,
        "travel_tolerance":  0,
    }


def _robotics_vector() -> list[int]:
    """
    A feature vector strongly resembling the robotics niche profile.
    Dimension 4 (index 4) is the dominant dimension for robotics in our test
    artifact (niche index 1 → dimension 1 * 4 = 4).
    """
    v = [0] * 40
    v[4] = 10   # Strong robotics signal at dimension 4
    return v


# ── TESTS: top-level result structure ────────────────────────────────────────

class TestFitScoreResultStructure:
    """
    Verifies that compute_fit_score() always returns a dict with the three
    required top-level keys: ranked_niches, pivot_applied, fallback_used.
    """

    def test_result_has_ranked_niches_key(self, artifact, neutral_prefs):
        """ranked_niches must always be present in the result dict."""
        result = compute_fit_score(_robotics_vector(), artifact, neutral_prefs)
        assert "ranked_niches" in result, (
            "compute_fit_score() result must contain 'ranked_niches' key. "
            f"Got keys: {list(result.keys())}"
        )

    def test_result_has_pivot_applied_key(self, artifact, neutral_prefs):
        """pivot_applied must always be present — even when no penalty was applied."""
        result = compute_fit_score(_robotics_vector(), artifact, neutral_prefs)
        assert "pivot_applied" in result, (
            "compute_fit_score() result must contain 'pivot_applied' key. "
            f"Got keys: {list(result.keys())}"
        )

    def test_result_has_fallback_used_key(self, artifact, neutral_prefs):
        """fallback_used must always be present — even when the vector is non-zero."""
        result = compute_fit_score(_robotics_vector(), artifact, neutral_prefs)
        assert "fallback_used" in result, (
            "compute_fit_score() result must contain 'fallback_used' key. "
            f"Got keys: {list(result.keys())}"
        )

    def test_pivot_applied_is_bool(self, artifact, neutral_prefs):
        """pivot_applied must be a Python bool, not just a truthy/falsy value."""
        result = compute_fit_score(_robotics_vector(), artifact, neutral_prefs)
        assert isinstance(result["pivot_applied"], bool), (
            f"pivot_applied must be bool, got {type(result['pivot_applied'])}"
        )

    def test_fallback_used_is_bool(self, artifact, neutral_prefs):
        """fallback_used must be a Python bool."""
        result = compute_fit_score(_robotics_vector(), artifact, neutral_prefs)
        assert isinstance(result["fallback_used"], bool), (
            f"fallback_used must be bool, got {type(result['fallback_used'])}"
        )

    def test_no_error_key_on_success(self, artifact, neutral_prefs):
        """A successful call must not have an 'error' key in the result."""
        result = compute_fit_score(_robotics_vector(), artifact, neutral_prefs)
        assert "error" not in result, (
            f"Successful result must not contain 'error'. Got: {result}"
        )


# ── TESTS: ranked_niches list structure ───────────────────────────────────────

class TestRankedNichesListContract:
    """
    Verifies that ranked_niches is always a list of exactly 6 NicheScore dicts,
    one per canonical niche, sorted descending by composite_score.
    """

    def test_ranked_niches_is_a_list(self, artifact, neutral_prefs):
        """ranked_niches must be a list (not a dict, tuple, or generator)."""
        result = compute_fit_score(_robotics_vector(), artifact, neutral_prefs)
        assert isinstance(result["ranked_niches"], list), (
            f"ranked_niches must be a list, got {type(result['ranked_niches'])}"
        )

    def test_ranked_niches_has_exactly_6_entries(self, artifact, neutral_prefs):
        """ranked_niches must always contain exactly 6 entries — one per canonical niche."""
        result = compute_fit_score(_robotics_vector(), artifact, neutral_prefs)
        assert len(result["ranked_niches"]) == 6, (
            f"ranked_niches must have exactly 6 entries, got {len(result['ranked_niches'])}"
        )

    def test_all_canonical_niches_present(self, artifact, neutral_prefs):
        """Every canonical niche must appear in ranked_niches — no niche may be omitted."""
        result = compute_fit_score(_robotics_vector(), artifact, neutral_prefs)
        returned_niches = {ns["niche"] for ns in result["ranked_niches"]}
        canonical_set = set(CANONICAL_NICHES)
        missing = canonical_set - returned_niches
        assert not missing, (
            f"ranked_niches is missing canonical niches: {missing}. "
            f"Got niches: {returned_niches}"
        )

    def test_ranked_niches_sorted_descending_by_composite_score(self, artifact, neutral_prefs):
        """ranked_niches must be ordered by composite_score descending (rank 1 = best)."""
        result = compute_fit_score(_robotics_vector(), artifact, neutral_prefs)
        scores = [ns["composite_score"] for ns in result["ranked_niches"]]
        assert scores == sorted(scores, reverse=True), (
            f"ranked_niches is not sorted descending by composite_score. Got: {scores}"
        )

    def test_rank_values_are_1_through_6(self, artifact, neutral_prefs):
        """rank values must be integers 1 through 6 (1 = best fit)."""
        result = compute_fit_score(_robotics_vector(), artifact, neutral_prefs)
        ranks = sorted(ns["rank"] for ns in result["ranked_niches"])
        assert ranks == [1, 2, 3, 4, 5, 6], (
            f"rank values must be 1–6, got sorted ranks: {ranks}"
        )


# ── TESTS: NicheScore dict structure ─────────────────────────────────────────

class TestNicheScoreDictContract:
    """
    Verifies that each entry in ranked_niches has all required NicheScore keys
    with the correct types and value ranges.
    """

    REQUIRED_KEYS = {
        "niche", "composite_score", "cv_component",
        "intent_component", "penalty", "rank",
    }

    def test_each_niche_score_has_required_keys(self, artifact, neutral_prefs):
        """Every NicheScore dict must have all 6 required keys."""
        result = compute_fit_score(_robotics_vector(), artifact, neutral_prefs)
        for ns in result["ranked_niches"]:
            missing = self.REQUIRED_KEYS - set(ns.keys())
            assert not missing, (
                f"NicheScore for '{ns.get('niche', '?')}' is missing keys: {missing}"
            )

    def test_composite_score_is_float_in_range(self, artifact, neutral_prefs):
        """composite_score must be a float in [0.0, 1.0] for every niche."""
        result = compute_fit_score(_robotics_vector(), artifact, neutral_prefs)
        for ns in result["ranked_niches"]:
            score = ns["composite_score"]
            assert isinstance(score, float), (
                f"composite_score for '{ns['niche']}' must be float, got {type(score)}"
            )
            assert 0.0 <= score <= 1.0, (
                f"composite_score for '{ns['niche']}' must be in [0.0, 1.0], got {score}"
            )

    def test_cv_component_is_float_in_range(self, artifact, neutral_prefs):
        """cv_component must be a float in [0.0, 1.0] for every niche."""
        result = compute_fit_score(_robotics_vector(), artifact, neutral_prefs)
        for ns in result["ranked_niches"]:
            score = ns["cv_component"]
            assert isinstance(score, float), (
                f"cv_component for '{ns['niche']}' must be float, got {type(score)}"
            )
            assert 0.0 <= score <= 1.0, (
                f"cv_component for '{ns['niche']}' must be in [0.0, 1.0], got {score}"
            )

    def test_intent_component_is_float_in_range(self, artifact, neutral_prefs):
        """intent_component must be a float in [0.0, 1.0] for every niche."""
        result = compute_fit_score(_robotics_vector(), artifact, neutral_prefs)
        for ns in result["ranked_niches"]:
            score = ns["intent_component"]
            assert isinstance(score, float), (
                f"intent_component for '{ns['niche']}' must be float, got {type(score)}"
            )
            assert 0.0 <= score <= 1.0, (
                f"intent_component for '{ns['niche']}' must be in [0.0, 1.0], got {score}"
            )

    def test_penalty_is_non_negative_float(self, artifact, neutral_prefs):
        """penalty must be a non-negative float (0.0 when no penalty was applied)."""
        result = compute_fit_score(_robotics_vector(), artifact, neutral_prefs)
        for ns in result["ranked_niches"]:
            penalty = ns["penalty"]
            assert isinstance(penalty, float), (
                f"penalty for '{ns['niche']}' must be float, got {type(penalty)}"
            )
            assert penalty >= 0.0, (
                f"penalty for '{ns['niche']}' must be >= 0.0, got {penalty}"
            )

    def test_niche_field_is_canonical_niche_name(self, artifact, neutral_prefs):
        """niche must be one of the 6 canonical niche names."""
        result = compute_fit_score(_robotics_vector(), artifact, neutral_prefs)
        for ns in result["ranked_niches"]:
            assert ns["niche"] in CANONICAL_NICHES, (
                f"niche '{ns['niche']}' is not a canonical niche name. "
                f"Canonical niches: {CANONICAL_NICHES}"
            )

    def test_rank_is_integer(self, artifact, neutral_prefs):
        """rank must be an integer."""
        result = compute_fit_score(_robotics_vector(), artifact, neutral_prefs)
        for ns in result["ranked_niches"]:
            assert isinstance(ns["rank"], int), (
                f"rank for '{ns['niche']}' must be int, got {type(ns['rank'])}"
            )


# ── TESTS: error dict structure ───────────────────────────────────────────────

class TestErrorDictContract:
    """
    Verifies that invalid inputs produce a properly shaped error dict (not an exception).
    """

    def test_invalid_vector_length_returns_error_dict_not_exception(self, artifact):
        """A 39-element vector must return an error dict — never raise."""
        result = compute_fit_score([0] * 39, artifact)
        assert "error" in result, (
            f"Invalid vector should return error dict, got: {result}"
        )
        assert result["error"] == "INVALID_VECTOR", (
            f"Error code for wrong-length vector must be INVALID_VECTOR, got: {result['error']}"
        )
        assert "message" in result, "Error dict must have a 'message' key"

    def test_missing_centroids_returns_error_dict(self):
        """An artifact without niche_centroids must return MISSING_CENTROIDS error."""
        from src.model_io import current_schema_version
        from sklearn.ensemble import RandomForestClassifier

        # Build a valid artifact but without niche_centroids (pre-Phase-4 style)
        X = np.array([[float(i)] * 40 for i in range(60)])
        y = np.array([CANONICAL_NICHES[i % 6] for i in range(60)])
        clf = RandomForestClassifier(n_estimators=5, random_state=0)
        clf.fit(X, y)

        legacy_artifact = {
            "model":           clf,
            "niche_labels":    [str(c) for c in clf.classes_],
            "training_record": {},
            "schema_version":  current_schema_version(),
            "model_version":   "legacy_no_centroids",
            # Deliberately omitting "niche_centroids"
        }

        result = compute_fit_score([0] * 40, legacy_artifact)
        assert "error" in result, (
            f"Missing centroids should return error dict, got: {result}"
        )
        assert result["error"] == "MISSING_CENTROIDS", (
            f"Error code must be MISSING_CENTROIDS, got: {result['error']}"
        )

    def test_error_dict_never_raises(self, artifact):
        """compute_fit_score() must NEVER raise an exception for any input."""
        # Try a variety of bad inputs — all must return a dict, never raise
        bad_inputs = [
            ([0] * 39, artifact),          # Wrong length
            (None, artifact),              # None vector
            ("not_a_list", artifact),      # Wrong type
            ([0] * 40, {}),               # Empty artifact
            ([0] * 40, None),             # None artifact
        ]
        for fv, art in bad_inputs:
            try:
                result = compute_fit_score(fv, art)
                # We allow any dict return, but it must not be an exception
                assert isinstance(result, dict), (
                    f"compute_fit_score must return a dict, got {type(result)} "
                    f"for inputs fv={fv!r}, artifact={art!r}"
                )
            except Exception as exc:
                pytest.fail(
                    f"compute_fit_score raised {type(exc).__name__} for inputs "
                    f"fv={fv!r}, artifact={art!r}: {exc}"
                )

    def test_error_dict_has_message_key(self, artifact):
        """Error dicts must always contain a 'message' key for human-readable context."""
        result = compute_fit_score([0] * 5, artifact)  # Too short
        assert "message" in result, (
            f"Error dict must have 'message' key, got: {result}"
        )
        assert isinstance(result["message"], str), (
            f"'message' must be a str, got {type(result['message'])}"
        )


# ── TESTS: behavioural guarantees ────────────────────────────────────────────

class TestBehaviouralGuarantees:
    """
    Verifies specific behavioural guarantees from scorer_contract.md that
    don't fit neatly into structure tests.
    """

    def test_top_niche_composite_score_is_1_0(self, artifact, neutral_prefs):
        """
        After max-normalization, the highest-ranked niche must always have
        composite_score == 1.0 (as long as max_score > 0). This is the
        contract guarantee that the best niche always reads as perfect fit.
        """
        result = compute_fit_score(_robotics_vector(), artifact, neutral_prefs)
        top_score = result["ranked_niches"][0]["composite_score"]
        assert top_score == pytest.approx(1.0, abs=1e-6), (
            f"Top niche composite_score must be 1.0 after normalization, got {top_score}"
        )

    def test_none_preferences_equivalent_to_neutral_dict(self, artifact):
        """
        user_preferences=None must produce the same result as all-zero preferences.
        Both represent "no preference expressed — fully neutral".
        """
        fv = _robotics_vector()
        result_none = compute_fit_score(fv, artifact, user_preferences=None)
        result_neutral = compute_fit_score(fv, artifact, user_preferences={
            "work_environment": 0, "system_level": 0,
            "industry_interest": 0, "team_scale": 0, "travel_tolerance": 0,
        })

        # Extract composite scores from both results and compare
        scores_none = {ns["niche"]: ns["composite_score"] for ns in result_none["ranked_niches"]}
        scores_neutral = {ns["niche"]: ns["composite_score"] for ns in result_neutral["ranked_niches"]}

        for niche in CANONICAL_NICHES:
            assert scores_none[niche] == pytest.approx(scores_neutral[niche], abs=1e-6), (
                f"None prefs and neutral dict prefs gave different scores for '{niche}': "
                f"None={scores_none[niche]}, neutral={scores_neutral[niche]}"
            )

    def test_pivot_applied_false_with_neutral_prefs(self, artifact, neutral_prefs):
        """
        Neutral preferences (all zeros) trigger no override penalties, so
        pivot_applied must be False.
        """
        result = compute_fit_score(_robotics_vector(), artifact, neutral_prefs)
        assert result["pivot_applied"] is False, (
            f"Neutral preferences must not trigger any override penalty. "
            f"pivot_applied was {result['pivot_applied']}"
        )

    def test_fallback_used_false_for_nonzero_vector(self, artifact, neutral_prefs):
        """
        A non-zero feature vector must have fallback_used=False.
        fallback_used=True is only for all-zero vectors.
        """
        result = compute_fit_score(_robotics_vector(), artifact, neutral_prefs)
        assert result["fallback_used"] is False, (
            f"Non-zero feature vector must have fallback_used=False. "
            f"Got fallback_used={result['fallback_used']}"
        )

    def test_fallback_used_true_for_zero_vector(self, artifact, neutral_prefs):
        """
        An all-zeros feature vector must have fallback_used=True (FR-015).
        """
        zero_vector = [0] * 40
        result = compute_fit_score(zero_vector, artifact, neutral_prefs)
        assert result["fallback_used"] is True, (
            f"Zero feature vector must have fallback_used=True. "
            f"Got fallback_used={result['fallback_used']}"
        )

    def test_zero_vector_returns_6_niches(self, artifact, neutral_prefs):
        """
        Even a zero vector must return all 6 niches — no niche is omitted (FR-007).
        """
        zero_vector = [0] * 40
        result = compute_fit_score(zero_vector, artifact, neutral_prefs)
        assert "ranked_niches" in result, "Zero vector must still return ranked_niches"
        assert len(result["ranked_niches"]) == 6, (
            f"Zero vector must return 6 niches, got {len(result['ranked_niches'])}"
        )

    def test_deterministic_identical_inputs_identical_outputs(self, artifact, neutral_prefs):
        """
        compute_fit_score() must be deterministic: two calls with the same inputs
        must produce identical outputs (Behavioral Guarantee #9).
        """
        fv = _robotics_vector()
        result1 = compute_fit_score(fv, artifact, neutral_prefs)
        result2 = compute_fit_score(fv, artifact, neutral_prefs)

        # Compare ranked niches order and scores
        for ns1, ns2 in zip(result1["ranked_niches"], result2["ranked_niches"]):
            assert ns1["niche"] == ns2["niche"], "Same input produced different niche order"
            assert ns1["composite_score"] == ns2["composite_score"], (
                f"Same input produced different composite_score for '{ns1['niche']}'"
            )
