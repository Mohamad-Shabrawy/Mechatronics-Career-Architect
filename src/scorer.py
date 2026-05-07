"""
scorer.py — Hybrid Scoring Logic & Weighted Averaging (Phase 4)

This module is the composite fit-score engine. Given a CV's 40D feature vector,
a loaded Phase 3 model artifact (which includes per-niche centroid vectors), and
optional soft intent preferences from the user, it produces a ranked list of all
6 Mechatronics niches with their fit scores.

Scoring formula (authoritative — from spec.md and plan.md):

    For each niche n in CANONICAL_NICHES:
        cosine_n    = dot(fv, centroid_n) / (norm(fv) * norm(centroid_n))
                      → 0.0 if fv is a zero vector (zero-vector guard)
        intent_n    = clamp(0.5 + Σ(affinity_deltas for user prefs), 0.0, 1.0)
        composite_n = 0.60 * cosine_n + 0.40 * intent_n
        penalized_n = composite_n - penalty_n         # penalty_n >= 0.0

    Normalize:
        max_score   = max(penalized_n for all niches)
        normalized_n = penalized_n / max_score  if max_score > 0  else 0.0

Design decisions (research.md):
  - R-001: Cosine similarity via numpy dot/norm — transparent, microsecond-fast for 40D
  - R-002: Intent affinity uses -2 to +2 integer scale; YAML config; baseline 0.5
  - R-003: Override penalties are additive post-composite subtractions; max 0.25 default
  - R-004: Post-penalty max-normalization so top niche always reads as 1.0
  - R-005: Centroids are loaded from ModelArtifact, never re-computed at inference

Public API:
  compute_fit_score(feature_vector, model_artifact, user_preferences=None) -> dict

Constitution compliance:
  - Principle IV (Separation of Concerns): pure scoring logic; no UI, no I/O after config load
  - Principle V (Responsible AI): score_breakdown exposes all 3 components per niche;
    pivot_applied flag surfaces override transparency; fallback_used explains zero-vector behaviour

Contract reference: specs/004-hybrid-scoring-logic/contracts/scorer_contract.md
Data model:         specs/004-hybrid-scoring-logic/data-model.md
"""

import pathlib

import numpy as np
import yaml

from src.model_io import CANONICAL_NICHES


# ── MODULE-LEVEL CONFIG CACHE ─────────────────────────────────────────────────
# The scoring config (intent affinity map + override penalty rules) is loaded
# once on the first call and cached here. Subsequent calls reuse the cached value
# with no file I/O — satisfying the SC-002 <50ms performance requirement and
# Behavioral Guarantee #10 from the contract.
#
# None means "not yet loaded"; after first successful load this holds the config dict.
_SCORING_CONFIG: dict | None = None

# The config file lives in the project's config/ directory.
# We resolve the path relative to this source file so the scorer works
# regardless of the working directory from which it is invoked.
_CONFIG_FILE_PATH = pathlib.Path(__file__).parent.parent / "config" / "scoring_config.yaml"


# ── PUBLIC API ────────────────────────────────────────────────────────────────

def compute_fit_score(
    feature_vector: list[int],
    model_artifact: dict,
    user_preferences: dict | None = None,
) -> dict:
    """
    Computes a ranked composite fit score for all 6 Mechatronics niches.

    This is the sole public entry point for Phase 4. It combines:
      - 60% CV component: cosine similarity between the user's feature vector and
        each niche's centroid (mean feature vector from Phase 3 training data).
      - 40% intent component: user preference dimensions mapped through the
        IntentAffinityMap config to produce a per-niche affinity score in [0, 1].
      - Post-composite override penalties for strong preference/niche conflicts.
      - Post-penalty max-normalization so the best niche always scores 1.0.

    Parameters
    ----------
    feature_vector : list[int]
        40-dimensional Phase 2 feature vector. Must have exactly 40 elements.
        All-zeros is valid — returns 0.0 cosine for all niches and sets fallback_used=True.

    model_artifact : dict
        Loaded Phase 3 artifact from model_io.load_model(). Must contain
        "niche_centroids" (added by Phase 4 trainer.py backfill).

    user_preferences : dict | None
        Optional soft intent preferences. Recognised keys:
            work_environment   : int, -2 to +2 (site vs office)
            system_level       : int, -2 to +2 (hardware vs software)
            industry_interest  : int, -2 to +2 (heavy industry vs consumer electronics)
            team_scale         : int, -2 to +2 (large org vs startup)
            travel_tolerance   : int, -2 to +2 (high travel OK vs no travel)
        Missing keys default to 0 (neutral). None is treated as fully neutral.
        Values outside [-2, +2] are clamped silently.

    Returns
    -------
    dict
        On success — a FitScoreResult dict:
            {
                "ranked_niches": [
                    {
                        "niche":            str,    # canonical niche name
                        "composite_score":  float,  # normalised score in [0.0, 1.0]
                        "cv_component":     float,  # cosine similarity in [0.0, 1.0]
                        "intent_component": float,  # intent affinity in [0.0, 1.0]
                        "penalty":          float,  # override penalty subtracted
                        "rank":             int,    # 1 = best fit
                    },
                    ... (5 more, one per remaining niche)
                ],
                "pivot_applied":  bool,  # True if any override penalty was applied
                "fallback_used":  bool,  # True if zero-vector or missing ml_prediction
            }

        On error — an error dict (never raises):
            {"error": str, "message": str}
            # "error" values: "INVALID_VECTOR" | "MISSING_CENTROIDS" | "CONFIG_ERROR" | "SCORING_ERROR"
    """
    # Wrap the entire body in try/except so no unexpected exception ever escapes
    # to the caller. Behavioral Guarantee #1: this function NEVER raises.
    try:
        # ── Input validation: feature vector length ───────────────────────────
        # The Phase 2 contract locks the vector at exactly 40 dimensions.
        # A wrong-length vector means the cosine similarity is meaningless — we
        # return an error dict immediately rather than producing a bogus result.
        if not isinstance(feature_vector, list) or len(feature_vector) != 40:
            actual_len = len(feature_vector) if isinstance(feature_vector, list) else "not a list"
            return {
                "error":   "INVALID_VECTOR",
                "message": (
                    f"feature_vector must be a list of exactly 40 elements, "
                    f"got {actual_len}. "
                    "Ensure the input comes from Phase 2 enrich_cv()."
                ),
            }

        # ── Input validation: centroid availability ───────────────────────────
        # Phase 4 requires "niche_centroids" in the artifact. Artifacts trained
        # before Phase 4 won't have this key. We surface the absence as a clear
        # error so users know they need to retrain, not debug silently wrong scores.
        if "niche_centroids" not in model_artifact:
            return {
                "error":   "MISSING_CENTROIDS",
                "message": (
                    "model_artifact does not contain 'niche_centroids'. "
                    "This artifact was trained before Phase 4 support. "
                    "Retrain the model using Phase 4-compatible trainer.py."
                ),
            }

        # ── Load scoring config (once, cached) ───────────────────────────────
        # load_scoring_config() handles caching internally; subsequent calls
        # return the cached value instantly without touching the filesystem.
        try:
            config = load_scoring_config()
        except RuntimeError as config_err:
            # Config file missing or malformed YAML — surface as CONFIG_ERROR dict.
            return {
                "error":   "CONFIG_ERROR",
                "message": str(config_err),
            }

        # ── Normalise user_preferences ────────────────────────────────────────
        # None means fully neutral — treat as an empty dict (all keys default to 0).
        # Values outside [-2, +2] are clamped here before any downstream lookup.
        prefs = _normalise_preferences(user_preferences or {})

        # ── Detect zero-vector input ──────────────────────────────────────────
        # A zero feature vector produces 0.0 cosine similarity for all niches.
        # We set fallback_used=True so the caller knows the ranking is based
        # entirely on the 40% intent component, not on CV content.
        fv_array = np.array(feature_vector, dtype=float)
        is_zero_vector = float(np.linalg.norm(fv_array)) == 0.0

        niche_centroids = model_artifact["niche_centroids"]

        # ── Compute per-niche CV and intent components ────────────────────────
        # We iterate in CANONICAL_NICHES order so the output dict has a predictable
        # structure. Each intermediate value is stored before compositing so the
        # score_breakdown in the output is fully transparent (Constitution V).
        cv_components: dict[str, float] = {}
        intent_components: dict[str, float] = {}

        for niche in CANONICAL_NICHES:
            # CV component: cosine similarity to this niche's centroid.
            # Returns 0.0 if the feature vector is a zero vector (guard is inside
            # _cosine_similarity so we don't need to duplicate the check here).
            centroid = niche_centroids[niche]
            cv_components[niche] = _cosine_similarity(fv_array, centroid)

            # Intent component: preference-driven affinity score in [0.0, 1.0].
            # Starts at 0.5 baseline (neutral → all niches equal), shifted by deltas.
            intent_components[niche] = _compute_intent_component(
                niche, prefs, config["intent_affinity_map"]
            )

        # ── Compute raw composite scores (60% CV + 40% intent) ───────────────
        # This is the central formula from plan.md and spec.md FR-002.
        # We compute all 6 composites before applying penalties so each niche's
        # raw composite is recorded in score_breakdown for transparency.
        composite_scores: dict[str, float] = {
            niche: 0.60 * cv_components[niche] + 0.40 * intent_components[niche]
            for niche in CANONICAL_NICHES
        }

        # ── Apply override/pivot penalties ────────────────────────────────────
        # Penalties are subtracted from composite scores when a user preference
        # strongly conflicts with a niche's primary work context (R-003).
        # _apply_override_penalties returns the updated scores dict and a flag
        # indicating whether any penalty was actually applied (for pivot_applied).
        penalties_applied: dict[str, float] = {niche: 0.0 for niche in CANONICAL_NICHES}
        penalized_scores, any_penalty_applied = _apply_override_penalties(
            composite_scores,
            prefs,
            config["override_penalty_config"],
            penalties_applied,
        )

        # ── Post-penalty max-normalization ────────────────────────────────────
        # After penalties, divide all scores by the maximum so the best niche
        # reads as 1.0. This gives users an intuitive relative fit score (R-004).
        # If all penalized scores are 0.0 (degenerate case: zero vector + full penalty),
        # leave all scores at 0.0 rather than dividing by zero.
        max_score = max(penalized_scores.values())

        normalized_scores: dict[str, float] = {}
        if max_score > 0.0:
            for niche in CANONICAL_NICHES:
                normalized_scores[niche] = penalized_scores[niche] / max_score
        else:
            # All scores are zero — keep them at 0.0 to avoid division by zero.
            normalized_scores = {niche: 0.0 for niche in CANONICAL_NICHES}

        # ── Build the ranked_niches list ──────────────────────────────────────
        # Sort niches by normalized score descending, then assign rank 1–6.
        # The score_breakdown per niche exposes all 3 components for transparency
        # (Behavioral Guarantee #6, Constitution V, FR-010).
        sorted_niches = sorted(
            CANONICAL_NICHES,
            key=lambda n: normalized_scores[n],
            reverse=True,
        )

        ranked_niches = []
        for rank_position, niche in enumerate(sorted_niches, start=1):
            ranked_niches.append({
                "niche":            niche,
                "composite_score":  round(normalized_scores[niche], 6),
                "cv_component":     round(cv_components[niche], 6),
                "intent_component": round(intent_components[niche], 6),
                "penalty":          round(penalties_applied[niche], 6),
                "rank":             rank_position,
            })

        # ── Build and return the FitScoreResult ──────────────────────────────
        return {
            "ranked_niches": ranked_niches,
            "pivot_applied": any_penalty_applied,
            "fallback_used": is_zero_vector,
        }

    except Exception as exc:
        # Catch-all: any unexpected error inside the scoring logic is returned
        # as a SCORING_ERROR dict. This ensures Behavioral Guarantee #1 is never
        # violated — the function truly never raises to its caller.
        return {
            "error":   "SCORING_ERROR",
            "message": f"Unexpected error during scoring: {exc}",
        }


# ── CONFIG LOADING ────────────────────────────────────────────────────────────

def load_scoring_config() -> dict:
    """
    Loads and returns the scoring configuration from config/scoring_config.yaml.

    The config is cached in module-level _SCORING_CONFIG after the first successful
    load. All subsequent calls return the cached value instantly — no file I/O is
    performed after the first call (Behavioral Guarantee #10, SC-002 performance).

    The config contains two keys:
        "intent_affinity_map"     — nested dict for computing intent components
        "override_penalty_config" — list of penalty rule dicts

    Returns
    -------
    dict
        The full scoring config dict from the YAML file.

    Raises
    ------
    RuntimeError
        If the config file is missing or contains invalid YAML. The caller
        (compute_fit_score) catches this and returns a CONFIG_ERROR dict.
    """
    global _SCORING_CONFIG

    # Return the cached config immediately on all calls after the first.
    # This is the hot path once the module is warmed up.
    if _SCORING_CONFIG is not None:
        return _SCORING_CONFIG

    # First call — load from disk. We use yaml.safe_load() to avoid executing
    # arbitrary Python code that an unsafe YAML loader would permit.
    config_path = _CONFIG_FILE_PATH
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        raise RuntimeError(
            f"Scoring config not found at '{config_path}'. "
            "Ensure config/scoring_config.yaml exists in the project root."
        )
    except yaml.YAMLError as exc:
        raise RuntimeError(
            f"Failed to parse scoring config '{config_path}': {exc}. "
            "Check the YAML syntax in config/scoring_config.yaml."
        )

    # Validate top-level structure before caching — a completely malformed YAML
    # that parsed successfully (e.g., a bare string) would cause confusing errors
    # later during scoring. We check for the two required top-level keys here.
    if not isinstance(config, dict):
        raise RuntimeError(
            f"scoring_config.yaml must be a YAML mapping, got {type(config).__name__}."
        )
    if "intent_affinity_map" not in config:
        raise RuntimeError(
            "scoring_config.yaml is missing 'intent_affinity_map'. "
            "Check the config file structure."
        )
    if "override_penalty_config" not in config:
        raise RuntimeError(
            "scoring_config.yaml is missing 'override_penalty_config'. "
            "Check the config file structure."
        )

    # Cache and return. Thread safety is not a concern here because this is a
    # single-call Python library (not a multi-threaded server), and worst case
    # two threads both load the YAML once — which is harmless (idempotent result).
    _SCORING_CONFIG = config
    return _SCORING_CONFIG


# ── PRIVATE HELPERS ───────────────────────────────────────────────────────────

def _normalise_preferences(user_preferences: dict) -> dict:
    """
    Sanitises user preferences: clamps all present values to [-2, +2] and returns
    the cleaned dict. Missing keys stay absent (they'll default to 0.0 delta in
    _compute_intent_component via .get() with a default).

    Why clamp here rather than in _compute_intent_component?
      Doing it once at the entry point is cleaner than repeating the clamp in every
      downstream lookup. The data-model.md spec says "Values outside [-2, +2] are
      clamped silently to the nearest boundary" — this is that boundary enforcement.
    """
    known_keys = {
        "work_environment",
        "system_level",
        "industry_interest",
        "team_scale",
        "travel_tolerance",
    }
    cleaned = {}
    for key, value in user_preferences.items():
        if key in known_keys:
            # Silently clamp out-of-range values to the nearest valid boundary.
            try:
                val_int = int(value)
                cleaned[key] = max(-2, min(2, val_int))
            except (TypeError, ValueError):
                # Non-integer value — treat as neutral (0) rather than crashing.
                cleaned[key] = 0
    return cleaned


def _cosine_similarity(fv: np.ndarray, centroid: list[float]) -> float:
    """
    Computes cosine similarity between the user's feature vector and a niche centroid.

    Formula: dot(fv, centroid) / (norm(fv) * norm(centroid))

    Why cosine similarity rather than Euclidean distance?
      Cosine similarity measures the angle between two vectors, not their magnitude.
      Two CVs with the same skill mix but different total keyword counts (one verbose,
      one terse) will have high cosine similarity — which is the right behaviour for
      "does this CV match this niche's profile?".

    Zero-vector guard (FR-015):
      If the feature vector is all zeros (e.g., the CV had no extractable keywords),
      numpy's norm is 0 and division would produce NaN or inf. We return 0.0 instead,
      which is semantically correct (a blank CV has zero similarity to any niche profile).
      The centroid can also theoretically be zero (if a niche had zero training samples),
      in which case we also return 0.0.

    Parameters
    ----------
    fv : np.ndarray
        The user's 40D feature vector as a numpy float array.
    centroid : list[float]
        The niche's 40D centroid vector (loaded from the model artifact).

    Returns
    -------
    float
        Cosine similarity in [0.0, 1.0]. Returns 0.0 if either vector is zero.
    """
    centroid_array = np.array(centroid, dtype=float)

    norm_fv = np.linalg.norm(fv)
    norm_centroid = np.linalg.norm(centroid_array)

    # Guard: division by zero when either vector is all-zeros.
    if norm_fv == 0.0 or norm_centroid == 0.0:
        return 0.0

    # Standard cosine similarity formula — fast on 40D vectors.
    similarity = float(np.dot(fv, centroid_array) / (norm_fv * norm_centroid))

    # Clamp to [0.0, 1.0] to handle floating-point rounding that occasionally
    # produces values like 1.0000000000000002 or -0.000000000000001.
    return max(0.0, min(1.0, similarity))


def _compute_intent_component(
    niche: str,
    user_preferences: dict,
    affinity_map: dict,
) -> float:
    """
    Computes the intent component for a single niche given the user's preferences.

    Formula (data-model.md, R-002):
        base = 0.5
        for each preference key in user_preferences:
            delta = affinity_map[pref_key][str(value)].get(niche, 0.0)
            base += delta
        intent_component = clamp(base, 0.0, 1.0)

    The 0.5 baseline ensures that fully neutral preferences (all zeros, or None)
    produce an intent_component of exactly 0.5 for every niche — so the 40% intent
    contribution is equal across all niches when no preferences are expressed. This
    means neutral preferences let the CV component (60%) fully determine the ranking.

    Missing config keys (pref_key not in affinity_map, or value not mapped, or niche
    not in the delta dict) all default to 0.0 — neutral, no effect (FR-014).

    Parameters
    ----------
    niche : str
        The canonical niche name to compute the intent score for.
    user_preferences : dict
        The sanitised preference dict (already clamped by _normalise_preferences).
    affinity_map : dict
        The full intent_affinity_map from scoring_config.yaml.

    Returns
    -------
    float
        Intent component in [0.0, 1.0].
    """
    base = 0.5  # Neutral baseline: equal intent score for all niches

    for pref_key, pref_value in user_preferences.items():
        # Look up the per-value mapping for this preference dimension.
        # YAML integer keys are loaded as strings when quoted (our config uses "2", "-2",
        # etc.), so we convert pref_value to str for the lookup.
        value_map = affinity_map.get(pref_key, {})
        niche_deltas = value_map.get(str(pref_value), {})

        # Get this niche's affinity delta for this preference value.
        # Default 0.0 if the niche isn't mentioned (no effect for unlisted niches).
        delta = niche_deltas.get(niche, 0.0)
        base += delta

    # Clamp to [0.0, 1.0] to prevent stacked extreme preferences from producing
    # scores outside the valid range (data-model.md validation rule).
    return max(0.0, min(1.0, base))


def _apply_override_penalties(
    composite_scores: dict[str, float],
    user_preferences: dict,
    penalty_config: list[dict],
    penalties_tracker: dict[str, float],
) -> tuple[dict[str, float], bool]:
    """
    Applies post-composite additive penalties when user preferences conflict with niches.

    For each rule in penalty_config:
        if user_preferences.get(rule["preference"]) == rule["threshold"]:
            composite_scores[rule["niche"]] -= rule["penalty"]  (clamped to >= 0.0)
            penalties_tracker[rule["niche"]] += rule["penalty"]

    Why additive post-composite rather than a multiplier?
      Additive subtraction is transparent: the score_breakdown shows the exact amount
      deducted. "automotive: composite 0.82, penalty 0.25, final 0.57" immediately
      explains why automotive dropped from #1. A multiplier would be harder to reason
      about (research.md R-003).

    Why post-composite rather than pre-composite?
      Applying the penalty after the 60/40 weighting gives it maximum punch — it
      directly reduces the final score by the full penalty amount regardless of whether
      the CV or intent component was dominant. This guarantees FR-006 and SC-004.

    Parameters
    ----------
    composite_scores : dict[str, float]
        Pre-penalty composite scores for all 6 niches (modified in place via copy).
    user_preferences : dict
        The sanitised preference dict.
    penalty_config : list[dict]
        The override_penalty_config list from scoring_config.yaml. Each entry has:
            preference : str    — UserPreferences key (e.g., "work_environment")
            threshold  : int    — Value at which the penalty triggers
            niche      : str    — Canonical niche to penalise
            penalty    : float  — Amount to subtract from composite score
    penalties_tracker : dict[str, float]
        Mutable dict tracking the total penalty applied per niche (for score_breakdown).
        Modified in place by this function.

    Returns
    -------
    tuple[dict[str, float], bool]
        (penalized_scores, any_penalty_applied)
        penalized_scores: updated composite scores (all >= 0.0)
        any_penalty_applied: True if at least one penalty > 0 was applied
    """
    # Work on a copy so we don't mutate the caller's composite_scores dict.
    penalized = dict(composite_scores)
    any_penalty_applied = False

    for rule in penalty_config:
        preference = rule.get("preference", "")
        threshold = rule.get("threshold")
        niche = rule.get("niche", "")
        penalty = float(rule.get("penalty", 0.0))

        # Only apply the penalty if:
        #   a) The niche is one we know about (safe guard against config typos)
        #   b) The user expressed a preference for this dimension
        #   c) That preference value exactly equals the conflict threshold
        if niche not in penalized:
            continue

        user_value = user_preferences.get(preference)
        if user_value is None:
            # User didn't express a preference for this dimension — skip.
            continue

        if user_value == threshold and penalty > 0.0:
            # Conflict detected — apply the penalty.
            penalized[niche] = max(0.0, penalized[niche] - penalty)
            penalties_tracker[niche] = round(penalties_tracker[niche] + penalty, 6)
            any_penalty_applied = True

    return penalized, any_penalty_applied
