"""
test_model_io.py — Unit Tests for src/model_io.py

These tests verify the model serialization/loading round-trip, schema version checking,
and feature importance extraction.

Key behaviors tested:
  - save_model() creates a .joblib file at the expected path
  - load_model() on a saved artifact returns a dict with all required keys
  - Serialize → load round-trip: predictions match for identical inputs (SC-006)
  - load_model() raises SchemaVersionError when schema version mismatches
  - load_model() raises ModelLoadError on a corrupted file
  - get_feature_importance() returns exactly 40 entries summing to ~1.0
  - current_schema_version() returns an 8-char hex string

Spec reference: specs/003-ml-niche-classifier/tasks.md (T031)
"""

import json
import os
import pathlib

import joblib
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.model_io import (
    CANONICAL_NICHES,
    SchemaVersionError,
    ModelLoadError,
    ModelIntegrityError,
    current_schema_version,
    save_model,
    load_model,
    get_feature_importance,
)
from src.feature_vector import VECTOR_INDEX_ORDER


# ── FIXTURES ──────────────────────────────────────────────────────────────────

def _make_trained_classifier() -> RandomForestClassifier:
    """
    Creates a properly trained RandomForestClassifier on the full set of 6 niches.
    Uses clearly separable synthetic data (each niche has a unique dominant feature)
    to guarantee the model fits correctly and has meaningful feature importances.
    """
    n_niches = len(CANONICAL_NICHES)
    n_per_niche = 10  # Small dataset for fast tests
    X = []
    y = []
    for i, niche in enumerate(CANONICAL_NICHES):
        for j in range(n_per_niche):
            v = [0.0] * 40
            v[i * 4] = float(10 + j)  # Unique strong signal per niche
            X.append(v)
            y.append(niche)
    clf = RandomForestClassifier(n_estimators=20, random_state=42)
    clf.fit(np.array(X), np.array(y))
    return clf


@pytest.fixture
def trained_classifier():
    """Returns a fitted RandomForestClassifier for use in tests."""
    return _make_trained_classifier()


def _make_synthetic_centroids() -> dict[str, list[float]]:
    """
    Builds synthetic per-niche centroid vectors for testing.
    Each centroid is a 40D vector with a unique dominant dimension (niche_idx * 4),
    matching the synthetic training data pattern used in _make_trained_classifier().
    These are used to satisfy the Phase 4 niche_centroids requirement in load_model().
    """
    centroids = {}
    for i, niche in enumerate(CANONICAL_NICHES):
        centroid = [0.0] * 40
        centroid[i * 4] = 14.5  # Mean of (10 + 0) ... (10 + 9) = 14.5
        centroids[niche] = centroid
    return centroids


def _make_synthetic_niche_top_skills(centroids: dict) -> dict:
    """
    Builds synthetic per-niche top-10 skill dimension lists for testing.
    Each entry is a list of 10 (dim_index, label) tuples ranked by centroid
    value descending — the same shape produced by trainer.py Step 10b.
    These are required by load_model() Stage 5 validation (Phase 5 backfill).
    """
    niche_top_skills = {}
    for niche in CANONICAL_NICHES:
        centroid = centroids[niche]
        ranked = sorted(enumerate(centroid), key=lambda x: x[1], reverse=True)
        niche_top_skills[niche] = [
            (idx, f"{VECTOR_INDEX_ORDER[idx][0]}_{VECTOR_INDEX_ORDER[idx][1]}")
            for idx, _ in ranked[:10]
        ]
    return niche_top_skills


@pytest.fixture
def saved_artifact_path(trained_classifier, tmp_path):
    """
    Saves a Phase 5-compatible model artifact to tmp_path and returns the path string.
    Includes all required fields, niche_centroids (Phase 4), and niche_top_skills
    (Phase 5). load_model() validates both, so both must be present for load tests.
    """
    niche_labels = [str(c) for c in trained_classifier.classes_]
    training_record = {
        "run_id": "test_run_model_io",
        "schema_version": current_schema_version(),
        "evaluation_metrics": {
            "overall_accuracy": 0.92,
            "per_niche": {n: {"precision": 0.9, "recall": 0.9, "f1": 0.9, "support": 2}
                          for n in CANONICAL_NICHES},
        },
    }
    centroids = _make_synthetic_centroids()
    # Phase 4: niche_centroids; Phase 5: niche_top_skills — both required by load_model().
    path = save_model(
        model=trained_classifier,
        niche_labels=niche_labels,
        training_record=training_record,
        run_id="test_run_model_io",
        output_dir=str(tmp_path),
        niche_centroids=centroids,
        niche_top_skills=_make_synthetic_niche_top_skills(centroids),
    )
    return path


@pytest.fixture
def loaded_artifact(saved_artifact_path):
    """Returns a fully loaded and validated model artifact dict."""
    return load_model(saved_artifact_path)


# ── TESTS: current_schema_version ─────────────────────────────────────────────

class TestCurrentSchemaVersion:
    """Tests for the schema version hash function."""

    def test_returns_8_char_string(self):
        """current_schema_version() must return exactly an 8-character string."""
        version = current_schema_version()
        assert isinstance(version, str), f"Schema version must be str, got {type(version)}"
        assert len(version) == 8, f"Schema version must be 8 chars, got {len(version)}"

    def test_returns_hex_string(self):
        """The schema version must be a valid hex string (subset of SHA-256)."""
        version = current_schema_version()
        assert all(c in "0123456789abcdef" for c in version), (
            f"Schema version must be lowercase hex, got: {version!r}"
        )

    def test_deterministic_across_calls(self):
        """Two calls in the same process must return the same version."""
        v1 = current_schema_version()
        v2 = current_schema_version()
        assert v1 == v2, f"Schema version must be deterministic, got {v1} vs {v2}"


# ── TESTS: CANONICAL_NICHES ───────────────────────────────────────────────────

class TestCanonicalNiches:
    """Tests for the CANONICAL_NICHES constant."""

    def test_has_exactly_6_niches(self):
        """Phase 3 only supports exactly 6 niche labels (spec assumption)."""
        assert len(CANONICAL_NICHES) == 6, (
            f"Must have exactly 6 canonical niches, got {len(CANONICAL_NICHES)}"
        )

    def test_all_niches_are_lowercase(self):
        """All canonical niche names must be lowercase (downstream contract)."""
        for niche in CANONICAL_NICHES:
            assert niche == niche.lower(), f"Niche '{niche}' must be lowercase"

    def test_all_expected_niches_present(self):
        """All six domain names must be present in the canonical list."""
        expected = {
            "industrial_automation", "robotics", "embedded_systems",
            "automotive", "mechanical_design", "technical_management",
        }
        assert set(CANONICAL_NICHES) == expected, (
            f"Canonical niches mismatch: {set(CANONICAL_NICHES)} vs {expected}"
        )


# ── TESTS: save_model ─────────────────────────────────────────────────────────

class TestSaveModel:
    """Tests for the save_model() function."""

    def test_creates_joblib_file_at_expected_path(self, trained_classifier, tmp_path):
        """
        save_model() must create a .joblib file named <run_id>.joblib
        in the specified output directory.
        """
        run_id = "save_test_001"
        path = save_model(
            model=trained_classifier,
            niche_labels=[str(c) for c in trained_classifier.classes_],
            training_record={"run_id": run_id},
            run_id=run_id,
            output_dir=str(tmp_path),
        )

        assert os.path.exists(path), f"Model file must exist at {path}"
        assert path.endswith(".joblib"), f"File must be .joblib, got: {path}"
        expected_filename = f"{run_id}.joblib"
        assert os.path.basename(path) == expected_filename, (
            f"Expected filename {expected_filename}, got {os.path.basename(path)}"
        )

    def test_creates_output_directory_if_missing(self, trained_classifier, tmp_path):
        """
        save_model() must create the output directory automatically if it doesn't exist.
        """
        new_dir = str(tmp_path / "nonexistent" / "subdir")
        assert not os.path.exists(new_dir)

        save_model(
            model=trained_classifier,
            niche_labels=[str(c) for c in trained_classifier.classes_],
            training_record={},
            run_id="auto_create_dir_test",
            output_dir=new_dir,
        )

        assert os.path.exists(new_dir), "Output directory must be created automatically"

    def test_returns_absolute_path(self, trained_classifier, tmp_path):
        """save_model() must return an absolute path to the created file."""
        path = save_model(
            model=trained_classifier,
            niche_labels=[str(c) for c in trained_classifier.classes_],
            training_record={},
            run_id="abs_path_test",
            output_dir=str(tmp_path),
        )
        assert os.path.isabs(path), f"Returned path must be absolute, got: {path}"

    def test_artifact_contains_schema_version(self, trained_classifier, tmp_path):
        """
        The saved artifact must contain the current schema version.
        This is verified by loading the raw joblib file (bypassing validation).
        """
        path = save_model(
            model=trained_classifier,
            niche_labels=[str(c) for c in trained_classifier.classes_],
            training_record={},
            run_id="schema_embed_test",
            output_dir=str(tmp_path),
        )
        # Load raw without validation to inspect the embedded schema version
        raw = joblib.load(path)
        assert raw.get("schema_version") == current_schema_version(), (
            "Artifact must embed the current schema version"
        )

    def test_artifact_contains_all_required_keys(self, trained_classifier, tmp_path):
        """
        The saved artifact dict must have all 5 required keys.
        """
        path = save_model(
            model=trained_classifier,
            niche_labels=[str(c) for c in trained_classifier.classes_],
            training_record={"run_id": "key_check"},
            run_id="key_check",
            output_dir=str(tmp_path),
        )
        raw = joblib.load(path)
        required = {"model", "niche_labels", "training_record", "schema_version", "model_version"}
        missing = required - set(raw.keys())
        assert not missing, f"Artifact missing required keys: {missing}"


# ── TESTS: load_model ─────────────────────────────────────────────────────────

class TestLoadModel:
    """Tests for the load_model() function — happy path and error paths."""

    def test_loads_valid_artifact_without_error(self, saved_artifact_path):
        """
        load_model() on a valid, properly-versioned artifact must succeed
        and return a dict (not raise any exception).
        """
        artifact = load_model(saved_artifact_path)
        assert isinstance(artifact, dict), f"load_model must return dict, got {type(artifact)}"

    def test_loaded_artifact_has_all_required_keys(self, loaded_artifact):
        """
        The loaded artifact dict must have all 5 required keys ready for inference.
        """
        required = {"model", "niche_labels", "training_record", "schema_version", "model_version"}
        missing = required - set(loaded_artifact.keys())
        assert not missing, f"Loaded artifact missing keys: {missing}"

    def test_loaded_artifact_model_is_fitted_classifier(self, loaded_artifact):
        """
        The 'model' key in the loaded artifact must be a fitted sklearn classifier
        that can call predict_proba().
        """
        model = loaded_artifact["model"]
        assert isinstance(model, RandomForestClassifier), (
            f"model must be RandomForestClassifier, got {type(model)}"
        )
        # Must be fitted (has classes_ attribute after fit)
        assert hasattr(model, "classes_"), "Loaded model must be fitted (has classes_)"

    def test_raises_model_load_error_on_missing_file(self, tmp_path):
        """
        Trying to load a file that doesn't exist must raise ModelLoadError.
        """
        with pytest.raises(ModelLoadError):
            load_model(str(tmp_path / "nonexistent.joblib"))

    def test_raises_model_load_error_on_corrupted_file(self, tmp_path):
        """
        Loading a file that exists but is not a valid joblib artifact
        must raise ModelLoadError (not a raw pickle/joblib exception).
        """
        bad_file = tmp_path / "bad.joblib"
        bad_file.write_bytes(b"this is garbage data, not joblib")
        with pytest.raises(ModelLoadError):
            load_model(str(bad_file))

    def test_raises_schema_version_error_on_version_mismatch(self, trained_classifier, tmp_path):
        """
        An artifact with a wrong schema_version must raise SchemaVersionError.
        This is the core schema compatibility check (FR-009).
        """
        # Manually write an artifact with a wrong schema version
        artifact = {
            "model":           trained_classifier,
            "niche_labels":    [str(c) for c in trained_classifier.classes_],
            "training_record": {},
            "schema_version":  "00000000",  # Wrong — not the current schema version
            "model_version":   "bad_schema_run",
        }
        bad_path = str(tmp_path / "bad_schema.joblib")
        joblib.dump(artifact, bad_path)

        with pytest.raises(SchemaVersionError) as exc_info:
            load_model(bad_path)

        # The error message must mention both versions
        error_msg = str(exc_info.value)
        assert "00000000" in error_msg, "Error must show the artifact's wrong schema version"
        assert current_schema_version() in error_msg, "Error must show the current schema version"

    def test_raises_model_integrity_error_on_missing_keys(self, tmp_path):
        """
        An artifact missing required keys must raise ModelIntegrityError.
        """
        incomplete_artifact = {
            "model": _make_trained_classifier(),
            # Missing: niche_labels, training_record, schema_version, model_version
        }
        bad_path = str(tmp_path / "incomplete.joblib")
        joblib.dump(incomplete_artifact, bad_path)

        # Should raise either ModelIntegrityError or SchemaVersionError
        with pytest.raises((ModelIntegrityError, SchemaVersionError)):
            load_model(bad_path)


# ── TESTS: serialize → load round-trip ────────────────────────────────────────

class TestSerializeLoadRoundtrip:
    """
    Tests that the serialize→load round-trip produces identical predictions.
    SC-006: a serialized model loaded in a new context must produce identical
    predictions to the original in-memory model.
    """

    def test_loaded_model_predictions_match_in_memory_model(
        self, trained_classifier, saved_artifact_path
    ):
        """
        The loaded model must produce identical predictions to the in-memory model
        for 10 synthetic test inputs. This is the SC-006 reproducibility guarantee.
        """
        loaded_artifact = load_model(saved_artifact_path)
        loaded_model = loaded_artifact["model"]

        # Generate 10 test vectors with distinctive patterns
        test_vectors = []
        for i in range(10):
            v = [0.0] * 40
            v[i % 40] = float(i + 1)
            test_vectors.append(v)

        X_test = np.array(test_vectors)

        # Compare predictions from the original and loaded models
        in_memory_preds = trained_classifier.predict(X_test)
        loaded_preds = loaded_model.predict(X_test)

        for i, (orig, loaded) in enumerate(zip(in_memory_preds, loaded_preds)):
            assert orig == loaded, (
                f"Prediction {i} mismatch: in-memory='{orig}', loaded='{loaded}'"
            )

    def test_loaded_model_probabilities_match_in_memory_model(
        self, trained_classifier, saved_artifact_path
    ):
        """
        predict_proba() output must be numerically identical between the original
        and loaded models. This verifies that weights are preserved exactly.
        """
        loaded_artifact = load_model(saved_artifact_path)
        loaded_model = loaded_artifact["model"]

        # One distinctive test vector
        X_test = np.array([[10.0, 0.0] + [0.0] * 38])

        orig_proba = trained_classifier.predict_proba(X_test)
        loaded_proba = loaded_model.predict_proba(X_test)

        np.testing.assert_array_almost_equal(
            orig_proba, loaded_proba, decimal=10,
            err_msg="predict_proba() must produce identical results after save→load"
        )


# ── TESTS: get_feature_importance ────────────────────────────────────────────

class TestGetFeatureImportance:
    """
    Tests for the get_feature_importance() helper function.
    T031: "get_feature_importance(artifact) returns a dict of 40 entries
    with float values summing to ~1.0"
    """

    def test_returns_dict_with_40_entries(self, loaded_artifact):
        """
        get_feature_importance() must return a dict with exactly 40 entries —
        one for each (type, section) dimension in the feature vector.
        """
        fi = get_feature_importance(loaded_artifact)
        assert isinstance(fi, dict), f"Feature importance must be dict, got {type(fi)}"
        assert len(fi) == 40, (
            f"Feature importance must have 40 entries (10 types × 4 sections), "
            f"got {len(fi)}"
        )

    def test_importance_values_are_floats(self, loaded_artifact):
        """All feature importance values must be floats."""
        fi = get_feature_importance(loaded_artifact)
        for key, val in fi.items():
            assert isinstance(val, float), (
                f"Importance for '{key}' must be float, got {type(val)}"
            )

    def test_importance_values_sum_to_approximately_one(self, loaded_artifact):
        """
        Random Forest feature importances must sum to 1.0 (by definition of
        Gini importance normalization). Floating-point rounding may cause slight
        deviation, so we allow a tolerance of 1e-6.
        """
        fi = get_feature_importance(loaded_artifact)
        total = sum(fi.values())
        assert abs(total - 1.0) < 1e-6, (
            f"Feature importances must sum to ~1.0, got {total}"
        )

    def test_importance_keys_use_type_section_format(self, loaded_artifact):
        """
        Keys in the feature importance dict must have the format '<type>_<section>'.
        Examples: 'plc_hardware_skills', 'robotic_framework_projects'.
        """
        fi = get_feature_importance(loaded_artifact)
        # Verify each key corresponds to a valid (type, section) pair from VECTOR_INDEX_ORDER
        valid_labels = {f"{t}_{s}" for t, s in VECTOR_INDEX_ORDER}
        for key in fi:
            assert key in valid_labels, (
                f"Feature importance key '{key}' not in valid labels. "
                f"Expected format: '<entity_type>_<section_name>'"
            )

    def test_importance_sorted_descending(self, loaded_artifact):
        """
        Feature importance values must be sorted in descending order
        (highest importance first) for easy interpretation.
        """
        fi = get_feature_importance(loaded_artifact)
        values = list(fi.values())
        assert values == sorted(values, reverse=True), (
            "Feature importance must be sorted descending (highest first)"
        )

    def test_all_non_negative_values(self, loaded_artifact):
        """
        All importance values must be ≥ 0.0. Gini importance can never be negative.
        """
        fi = get_feature_importance(loaded_artifact)
        for key, val in fi.items():
            assert val >= 0.0, f"Importance for '{key}' must be ≥ 0, got {val}"


# ── TESTS: Phase 4 centroid validation in load_model (T006) ──────────────────

class TestPhase4CentroidValidation:
    """
    Phase 4 backfill: load_model() must validate that 'niche_centroids' is present
    and correctly shaped. These tests verify tasks.md T006 and T008.

    load_model() raises ModelIntegrityError when:
      - 'niche_centroids' key is absent
      - 'niche_centroids' is not a dict
      - Any canonical niche is missing from the dict
      - Any centroid has length != 40
    """

    def test_load_model_raises_when_niche_centroids_absent(self, trained_classifier, tmp_path):
        """
        An artifact without 'niche_centroids' must raise ModelIntegrityError.
        This catches pre-Phase-4 artifacts that cannot support scoring.
        """
        # Build a valid artifact but deliberately omit niche_centroids
        artifact_no_centroids = {
            "model":           trained_classifier,
            "niche_labels":    [str(c) for c in trained_classifier.classes_],
            "training_record": {},
            "schema_version":  current_schema_version(),
            "model_version":   "no_centroids_test",
            # "niche_centroids" intentionally missing
        }
        path = str(tmp_path / "no_centroids.joblib")
        joblib.dump(artifact_no_centroids, path)

        with pytest.raises(ModelIntegrityError) as exc_info:
            load_model(path)

        # Error message must mention niche_centroids so the user knows what to do
        assert "niche_centroids" in str(exc_info.value).lower() or \
               "centroid" in str(exc_info.value).lower(), (
            f"ModelIntegrityError for missing centroids must mention 'centroid', "
            f"got: {exc_info.value}"
        )

    def test_load_model_raises_when_centroid_has_wrong_length(self, trained_classifier, tmp_path):
        """
        An artifact where any centroid has length != 40 must raise ModelIntegrityError.
        A 39-element centroid would produce incorrect cosine similarity in scorer.py.
        """
        bad_centroids = _make_synthetic_centroids()
        # Corrupt one centroid by giving it only 39 dimensions
        bad_centroids["robotics"] = [0.0] * 39

        artifact_bad_centroid = {
            "model":           trained_classifier,
            "niche_labels":    [str(c) for c in trained_classifier.classes_],
            "training_record": {},
            "schema_version":  current_schema_version(),
            "model_version":   "bad_centroid_length",
            "niche_centroids": bad_centroids,
        }
        path = str(tmp_path / "bad_centroid.joblib")
        joblib.dump(artifact_bad_centroid, path)

        with pytest.raises(ModelIntegrityError) as exc_info:
            load_model(path)

        # The error must mention the offending niche and/or the length
        error_msg = str(exc_info.value)
        assert "robotics" in error_msg or "40" in error_msg, (
            f"Error for wrong centroid length must mention 'robotics' or '40'. "
            f"Got: {error_msg}"
        )

    def test_load_model_raises_when_niche_missing_from_centroids(self, trained_classifier, tmp_path):
        """
        An artifact where 'niche_centroids' is missing one canonical niche must
        raise ModelIntegrityError. Phase 4 scorer.py requires all 6 niches.
        """
        incomplete_centroids = _make_synthetic_centroids()
        # Remove one niche from the centroids dict
        del incomplete_centroids["automotive"]

        artifact_incomplete = {
            "model":           trained_classifier,
            "niche_labels":    [str(c) for c in trained_classifier.classes_],
            "training_record": {},
            "schema_version":  current_schema_version(),
            "model_version":   "incomplete_centroids",
            "niche_centroids": incomplete_centroids,
        }
        path = str(tmp_path / "incomplete_centroids.joblib")
        joblib.dump(artifact_incomplete, path)

        with pytest.raises(ModelIntegrityError) as exc_info:
            load_model(path)

        # The error must mention the missing niche
        assert "automotive" in str(exc_info.value), (
            f"Error for missing centroid niche must mention 'automotive'. "
            f"Got: {exc_info.value}"
        )

    def test_load_model_succeeds_with_valid_centroids(self, saved_artifact_path):
        """
        An artifact with correctly formed niche_centroids must load without error.
        The loaded artifact must include 'niche_centroids' in the returned dict.
        """
        # saved_artifact_path fixture now includes niche_centroids — must succeed
        artifact = load_model(saved_artifact_path)
        assert "niche_centroids" in artifact, (
            "Loaded artifact must contain 'niche_centroids' when saved with centroids"
        )

    def test_loaded_centroids_have_correct_shape(self, loaded_artifact):
        """
        After loading, niche_centroids must be a dict with 6 niches, each a 40D list.
        """
        centroids = loaded_artifact.get("niche_centroids", {})
        assert isinstance(centroids, dict), "niche_centroids must be a dict"
        assert len(centroids) == 6, f"niche_centroids must have 6 entries, got {len(centroids)}"

        for niche in CANONICAL_NICHES:
            assert niche in centroids, f"niche_centroids missing '{niche}'"
            centroid = centroids[niche]
            assert isinstance(centroid, list), f"Centroid for '{niche}' must be a list"
            assert len(centroid) == 40, (
                f"Centroid for '{niche}' must have 40 elements, got {len(centroid)}"
            )

    def test_save_model_with_centroids_embeds_them_in_artifact(self, trained_classifier, tmp_path):
        """
        save_model() called with niche_centroids must embed them in the artifact.
        Without centroids (None), the key should be absent.
        """
        centroids = _make_synthetic_centroids()

        # Save with centroids
        path_with = save_model(
            model=trained_classifier,
            niche_labels=[str(c) for c in trained_classifier.classes_],
            training_record={},
            run_id="with_centroids",
            output_dir=str(tmp_path),
            niche_centroids=centroids,
        )
        raw_with = joblib.load(path_with)
        assert "niche_centroids" in raw_with, (
            "Artifact saved with niche_centroids must include the key"
        )
        assert raw_with["niche_centroids"] == centroids

        # Save without centroids (legacy behaviour)
        path_without = save_model(
            model=trained_classifier,
            niche_labels=[str(c) for c in trained_classifier.classes_],
            training_record={},
            run_id="without_centroids",
            output_dir=str(tmp_path),
            niche_centroids=None,   # Explicitly None → key should be absent
        )
        raw_without = joblib.load(path_without)
        assert "niche_centroids" not in raw_without, (
            "Artifact saved with niche_centroids=None must NOT include the key"
        )


# ── TESTS: Phase 5 niche_top_skills validation in load_model (T007) ──────────

def _make_valid_artifact_with_top_skills(
    trained_classifier,
    niches=None,
):
    """
    Builds a minimal in-memory artifact dict that includes BOTH niche_centroids
    (Phase 4) AND niche_top_skills (Phase 5).

    This helper is used to create test fixtures without going through the full
    train() pipeline — faster and more deterministic for unit tests.
    The niche_top_skills format is: {niche: [(dim_idx, label), ...] × 10}
    """
    from src.model_io import current_schema_version

    if niches is None:
        niches = CANONICAL_NICHES

    centroids = {}
    for i, niche in enumerate(niches):
        c = [0.0] * 40
        c[i * 4] = 14.5
        centroids[niche] = c

    # Build niche_top_skills: pick the top-10 dims by centroid value descending
    # (for these synthetic centroids, only one dim is nonzero per niche)
    niche_top_skills = {}
    for niche in niches:
        centroid = centroids[niche]
        ranked = sorted(enumerate(centroid), key=lambda x: x[1], reverse=True)
        top10 = [(idx, f"type_{idx % 10}_section_{idx % 4}") for idx, _ in ranked[:10]]
        niche_top_skills[niche] = top10

    return {
        "model":            trained_classifier,
        "niche_labels":     [str(c) for c in trained_classifier.classes_],
        "training_record":  {},
        "schema_version":   current_schema_version(),
        "model_version":    "phase5_test",
        "niche_centroids":  centroids,
        "niche_top_skills": niche_top_skills,
    }


class TestPhase5NicheTopSkillsValidation:
    """
    Phase 5 backfill: load_model() must validate that 'niche_top_skills' is present
    and correctly shaped in the artifact.

    These tests are written FIRST per TDD (T007). They FAIL before T009 is implemented
    (load_model() does not yet check niche_top_skills) and PASS after T009.

    Why validate at load time?
      Silent bad data is worse than a loud error. If an artifact is missing
      niche_top_skills, analyze_skills_gap() will fail later with a confusing
      "MISSING_BENCHMARK" error. Catching this at load time means the user gets
      a clear "retrain your model" message immediately.
    """

    def test_load_model_raises_when_niche_top_skills_absent(
        self, trained_classifier, tmp_path
    ):
        """
        An artifact without 'niche_top_skills' must raise ModelIntegrityError
        with a message telling the user to retrain.
        This catches Phase 4 artifacts that predate Phase 5 support.
        """
        # Build a Phase 4 artifact (has centroids, but no niche_top_skills)
        artifact_phase4_only = {
            "model":           trained_classifier,
            "niche_labels":    [str(c) for c in trained_classifier.classes_],
            "training_record": {},
            "schema_version":  current_schema_version(),
            "model_version":   "phase4_only_test",
            "niche_centroids": _make_synthetic_centroids(),
            # "niche_top_skills" intentionally absent
        }
        path = str(tmp_path / "phase4_only.joblib")
        joblib.dump(artifact_phase4_only, path)

        with pytest.raises(ModelIntegrityError) as exc_info:
            load_model(path)

        # The error message must mention Phase 5 or niche_top_skills so the user
        # knows exactly what's missing and what action to take (retrain).
        error_msg = str(exc_info.value).lower()
        assert "niche_top_skills" in error_msg or "phase 5" in error_msg, (
            f"ModelIntegrityError for missing niche_top_skills must mention "
            f"'niche_top_skills' or 'Phase 5', got: {exc_info.value}"
        )

    def test_load_model_raises_when_niche_top_skills_missing_niches(
        self, trained_classifier, tmp_path
    ):
        """
        An artifact where 'niche_top_skills' has fewer than 6 niches must raise
        ModelIntegrityError. All 6 canonical niches must be present.
        """
        # Build artifact with only 5 niches in niche_top_skills
        partial_niches = CANONICAL_NICHES[:5]  # Missing "technical_management"
        centroids = _make_synthetic_centroids()

        partial_top_skills = {}
        for i, niche in enumerate(partial_niches):
            c = centroids[niche]
            ranked = sorted(enumerate(c), key=lambda x: x[1], reverse=True)
            partial_top_skills[niche] = [(idx, f"label_{idx}") for idx, _ in ranked[:10]]

        artifact_partial = {
            "model":            trained_classifier,
            "niche_labels":     [str(c) for c in trained_classifier.classes_],
            "training_record":  {},
            "schema_version":   current_schema_version(),
            "model_version":    "partial_top_skills",
            "niche_centroids":  centroids,
            "niche_top_skills": partial_top_skills,
        }
        path = str(tmp_path / "partial_top_skills.joblib")
        joblib.dump(artifact_partial, path)

        with pytest.raises(ModelIntegrityError) as exc_info:
            load_model(path)

        # Must mention the missing niche or reference the incomplete structure
        error_msg = str(exc_info.value).lower()
        assert (
            "niche_top_skills" in error_msg
            or "technical_management" in error_msg
            or "6" in error_msg
        ), (
            f"ModelIntegrityError for incomplete niche_top_skills must mention the issue. "
            f"Got: {exc_info.value}"
        )

    def test_load_model_raises_when_niche_entry_has_wrong_count(
        self, trained_classifier, tmp_path
    ):
        """
        An artifact where any niche's niche_top_skills entry has length != 10
        must raise ModelIntegrityError. The benchmark size is fixed at 10.
        """
        centroids = _make_synthetic_centroids()

        # Build correct top_skills for all niches
        niche_top_skills = {}
        for i, niche in enumerate(CANONICAL_NICHES):
            c = centroids[niche]
            ranked = sorted(enumerate(c), key=lambda x: x[1], reverse=True)
            niche_top_skills[niche] = [(idx, f"label_{idx}") for idx, _ in ranked[:10]]

        # Corrupt one niche: give it only 9 entries instead of 10
        niche_top_skills["robotics"] = niche_top_skills["robotics"][:9]

        artifact_wrong_count = {
            "model":            trained_classifier,
            "niche_labels":     [str(c) for c in trained_classifier.classes_],
            "training_record":  {},
            "schema_version":   current_schema_version(),
            "model_version":    "wrong_count_test",
            "niche_centroids":  centroids,
            "niche_top_skills": niche_top_skills,
        }
        path = str(tmp_path / "wrong_count.joblib")
        joblib.dump(artifact_wrong_count, path)

        with pytest.raises(ModelIntegrityError) as exc_info:
            load_model(path)

        # Error must mention the problem niche or the expected count
        error_msg = str(exc_info.value).lower()
        assert "robotics" in error_msg or "10" in error_msg, (
            f"ModelIntegrityError for wrong niche_top_skills count must mention "
            f"'robotics' or '10'. Got: {exc_info.value}"
        )

    def test_load_model_succeeds_with_valid_niche_top_skills(
        self, trained_classifier, tmp_path
    ):
        """
        An artifact with correctly formed niche_top_skills (all 6 niches, each 10 entries)
        must load without error. The loaded artifact must contain 'niche_top_skills'.
        """
        full_artifact = _make_valid_artifact_with_top_skills(trained_classifier)
        path = str(tmp_path / "valid_phase5.joblib")
        joblib.dump(full_artifact, path)

        artifact = load_model(path)
        assert "niche_top_skills" in artifact, (
            "Loaded artifact must contain 'niche_top_skills' when saved with it"
        )
        assert len(artifact["niche_top_skills"]) == 6, (
            "Loaded niche_top_skills must have 6 niches"
        )

    def test_save_model_with_niche_top_skills_embeds_them(
        self, trained_classifier, tmp_path
    ):
        """
        save_model() called with niche_top_skills parameter must embed them in
        the artifact. This test verifies the save_model() signature change (T009).
        """
        centroids = _make_synthetic_centroids()

        niche_top_skills = {}
        for i, niche in enumerate(CANONICAL_NICHES):
            c = centroids[niche]
            ranked = sorted(enumerate(c), key=lambda x: x[1], reverse=True)
            niche_top_skills[niche] = [(idx, f"label_{idx}") for idx, _ in ranked[:10]]

        path = save_model(
            model=trained_classifier,
            niche_labels=[str(c) for c in trained_classifier.classes_],
            training_record={},
            run_id="phase5_save_test",
            output_dir=str(tmp_path),
            niche_centroids=centroids,
            niche_top_skills=niche_top_skills,
        )
        raw = joblib.load(path)
        assert "niche_top_skills" in raw, (
            "Artifact saved with niche_top_skills must include the key"
        )
        assert raw["niche_top_skills"] == niche_top_skills, (
            "niche_top_skills in artifact must match what was passed to save_model()"
        )
