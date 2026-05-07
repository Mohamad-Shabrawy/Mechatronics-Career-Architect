"""
test_classifier_contract.py — Phase 3 Classifier Contract Tests

These tests verify the BEHAVIORAL GUARANTEES of predict_niche() and load_model()
as defined in the classifier contract (classifier_contract.md).

Contract tests focus on the public interface and behavioral promises:
  - predict_niche() never raises (even on garbage input)
  - predict_niche() always writes a PredictionLogEntry to logs/predictions.jsonl
  - predicted_niche is always one of the 6 canonical niche names (on success)
  - confidence is always in [0.0, 1.0]
  - All-zero feature vector returns a valid prediction (not an error)
  - Invalid vector length returns an error dict (not an exception)
  - load_model() raises SchemaVersionError on mismatched schema version

These tests use a minimal synthetic model artifact (a pre-fitted RandomForest
on tiny data) to avoid requiring a full training run as a test dependency.

Contract reference: specs/003-ml-niche-classifier/contracts/classifier_contract.md
"""

import json
import os
import pathlib
import tempfile

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.classifier import predict_niche
from src.model_io import (
    CANONICAL_NICHES,
    SchemaVersionError,
    ModelLoadError,
    ModelIntegrityError,
    current_schema_version,
    save_model,
    load_model,
)


# ── FIXTURES ──────────────────────────────────────────────────────────────────

def _make_minimal_model() -> tuple:
    """
    Creates and fits a RandomForest on tiny synthetic data — just enough to
    produce valid predict_proba() output. We use 12 samples (2 per niche) with
    distinct feature patterns so the model can fit without warning about single-sample classes.
    This is NOT a production-quality model — it's a test fixture.

    Returns both the fitted classifier and the per-niche centroids computed from
    the same training data — required by Phase 4 model_io validation (load_model
    now enforces that niche_centroids is present in every saved artifact).
    """
    # Create 24 samples (4 per niche) with clearly separated feature patterns.
    # Each niche gets a unique "activated" dimension so the tiny model can still learn.
    n_niches = len(CANONICAL_NICHES)
    X = np.zeros((n_niches * 4, 40), dtype=float)
    y = []
    for i, niche in enumerate(CANONICAL_NICHES):
        # Set a distinctive dimension for each niche so the model can fit
        for j in range(4):
            X[i * 4 + j, i * 4] = float(j + 1)  # unique feature pattern per niche
        y.extend([niche] * 4)

    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, np.array(y))

    # Compute per-niche centroids — same formula used by trainer.py (Phase 4 backfill).
    # Each centroid is the mean feature vector for all training samples of that niche.
    y_arr = np.array(y)
    niche_centroids = {}
    for i, niche in enumerate(CANONICAL_NICHES):
        mask = y_arr == niche
        if mask.any():
            niche_centroids[niche] = X[mask].mean(axis=0).tolist()
        else:
            niche_centroids[niche] = [0.0] * 40

    return clf, niche_centroids


@pytest.fixture
def minimal_artifact(tmp_path):
    """
    Returns a complete, valid model artifact dict built from the minimal test model.
    The artifact is saved to and loaded from disk (via save_model/load_model) so
    it's identical to what production inference would use.

    Phase 4 note: load_model() now validates that niche_centroids is present and
    well-formed. We pass the centroids computed from our training data so this
    fixture continues to work after the Phase 4 backfill.
    """
    from src.feature_vector import VECTOR_INDEX_ORDER

    clf, niche_centroids = _make_minimal_model()
    niche_labels = [str(c) for c in clf.classes_]
    training_record = {
        "run_id": "test_run",
        "evaluation_metrics": {"overall_accuracy": 0.85, "per_niche": {}},
    }
    run_id = "test_run"

    # Phase 5: build niche_top_skills so load_model() Stage 5 validation passes.
    # Each niche gets its top-10 dims ranked by centroid value descending.
    niche_top_skills = {}
    for niche in CANONICAL_NICHES:
        centroid = niche_centroids[niche]
        ranked = sorted(enumerate(centroid), key=lambda x: x[1], reverse=True)
        niche_top_skills[niche] = [
            (idx, f"{VECTOR_INDEX_ORDER[idx][0]}_{VECTOR_INDEX_ORDER[idx][1]}")
            for idx, _ in ranked[:10]
        ]

    artifact_path = save_model(
        model=clf,
        niche_labels=niche_labels,
        training_record=training_record,
        run_id=run_id,
        output_dir=str(tmp_path),
        niche_centroids=niche_centroids,
        niche_top_skills=niche_top_skills,
    )
    return load_model(artifact_path)


@pytest.fixture
def valid_40d_vector():
    """A valid 40-dimensional feature vector with realistic values."""
    # This vector has some non-zero values to make it "interesting" to the model.
    # The pattern here loosely resembles an Embedded Systems profile.
    v = [0] * 40
    v[8] = 3   # microcontroller_skills
    v[9] = 1   # microcontroller_projects
    v[12] = 2  # communication_protocol_skills
    return v


@pytest.fixture
def zero_40d_vector():
    """A valid all-zero 40-dimensional feature vector (represents a CV with no recognized entities)."""
    return [0] * 40


# ── CONTRACT TESTS: predict_niche() return schema ─────────────────────────────

class TestPredictNicheReturnSchema:
    """
    Tests that verify the shape and types of predict_niche() return values.
    These tests correspond directly to the success-path contract in classifier_contract.md.
    """

    def test_success_returns_two_required_keys(self, minimal_artifact, valid_40d_vector):
        """
        The success return value must have exactly the keys 'predicted_niche' and 'confidence'.
        Phase 4 and 5 will access both by name — extra or missing keys break them.
        """
        result = predict_niche(valid_40d_vector, minimal_artifact)
        assert "error" not in result, f"Expected success, got error: {result}"
        assert set(result.keys()) == {"predicted_niche", "confidence"}, (
            f"Expected exactly {{'predicted_niche', 'confidence'}}, got {set(result.keys())}"
        )

    def test_predicted_niche_is_always_in_canonical_list(self, minimal_artifact, valid_40d_vector):
        """
        The predicted niche must ALWAYS be one of the 6 canonical names.
        An out-of-vocabulary label would break every downstream component.
        This is behavioral guarantee #3 in the contract.
        """
        result = predict_niche(valid_40d_vector, minimal_artifact)
        assert "error" not in result
        assert result["predicted_niche"] in CANONICAL_NICHES, (
            f"predicted_niche '{result['predicted_niche']}' is not in CANONICAL_NICHES: {CANONICAL_NICHES}"
        )

    def test_confidence_is_float_in_valid_range(self, minimal_artifact, valid_40d_vector):
        """
        Confidence must be a float in [0.0, 1.0]. This is behavioral guarantee #4.
        A confidence > 1.0 would be nonsensical; < 0.0 would break downstream scoring.
        """
        result = predict_niche(valid_40d_vector, minimal_artifact)
        assert "error" not in result
        conf = result["confidence"]
        assert isinstance(conf, float), f"confidence must be float, got {type(conf)}"
        assert 0.0 <= conf <= 1.0, f"confidence {conf} is outside [0.0, 1.0]"

    def test_predicted_niche_is_lowercase(self, minimal_artifact, valid_40d_vector):
        """
        The predicted niche must be lowercase to match the canonical form used
        throughout the pipeline. Mixed case would cause failures in downstream
        string comparison and reporting.
        """
        result = predict_niche(valid_40d_vector, minimal_artifact)
        assert "error" not in result
        assert result["predicted_niche"] == result["predicted_niche"].lower(), (
            f"predicted_niche '{result['predicted_niche']}' must be lowercase"
        )

    def test_json_serializable_success_result(self, minimal_artifact, valid_40d_vector):
        """
        The success result must survive json.dumps() — Phase 4 and 5 will serialize it.
        """
        result = predict_niche(valid_40d_vector, minimal_artifact)
        assert "error" not in result
        # Must not raise
        serialized = json.dumps(result)
        roundtrip = json.loads(serialized)
        assert roundtrip["predicted_niche"] == result["predicted_niche"]
        assert abs(roundtrip["confidence"] - result["confidence"]) < 1e-10


# ── CONTRACT TESTS: all-zero vector behavior ──────────────────────────────────

class TestAllZeroVectorBehavior:
    """
    A CV with no recognized entities produces an all-zero feature vector.
    The spec requires this to return a valid prediction (not an error).
    This is User Story 3, Acceptance Scenario 3.
    """

    def test_zero_vector_returns_valid_prediction(self, minimal_artifact, zero_40d_vector):
        """
        An all-zero feature vector must return a prediction dict with a niche
        and confidence — not an error dict. The model must make some prediction
        even when it has no information to go on.
        """
        result = predict_niche(zero_40d_vector, minimal_artifact)
        # This must NOT have an "error" key — zero vector is a valid (if weak) input.
        assert "error" not in result, (
            f"All-zero vector should return a valid prediction, got: {result}"
        )
        assert "predicted_niche" in result
        assert "confidence" in result
        assert result["predicted_niche"] in CANONICAL_NICHES

    def test_zero_vector_confidence_is_valid_float(self, minimal_artifact, zero_40d_vector):
        """
        Even for a zero vector (maximum uncertainty), confidence must be a valid float
        in [0.0, 1.0]. The model will return an equal distribution across classes,
        so confidence will be ~0.16 (1/6) for a balanced model — still valid.
        """
        result = predict_niche(zero_40d_vector, minimal_artifact)
        assert "error" not in result
        assert 0.0 <= result["confidence"] <= 1.0


# ── CONTRACT TESTS: invalid input handling ────────────────────────────────────

class TestInvalidInputHandling:
    """
    Tests for the "never raise" guarantee and error dict format.
    Behavioral guarantee #1: predict_niche() NEVER raises an exception.
    """

    def test_wrong_length_vector_returns_error_dict(self, minimal_artifact):
        """
        A vector with the wrong number of elements must return an error dict,
        not raise an exception. Phase 4 may receive malformed inputs from corrupt
        Phase 2 runs and must not crash.
        """
        bad_vector = [0] * 10  # Wrong length: 10 instead of 40
        result = predict_niche(bad_vector, minimal_artifact)
        assert "error" in result, f"Expected error dict, got: {result}"
        assert result["error"] == "INVALID_VECTOR"
        assert "message" in result

    def test_empty_vector_returns_error_dict(self, minimal_artifact):
        """An empty list [] is clearly wrong (length 0, not 40)."""
        result = predict_niche([], minimal_artifact)
        assert "error" in result
        assert result["error"] == "INVALID_VECTOR"

    def test_too_long_vector_returns_error_dict(self, minimal_artifact):
        """A 41-element vector is also invalid — strict 40D contract."""
        result = predict_niche([0] * 41, minimal_artifact)
        assert "error" in result
        assert result["error"] == "INVALID_VECTOR"

    def test_none_vector_returns_error_dict_not_exception(self, minimal_artifact):
        """
        None is definitely not a valid feature vector.
        Must return an error dict without raising, even though len(None) would raise TypeError.
        """
        try:
            result = predict_niche(None, minimal_artifact)
            # If it returns, it must be an error dict
            assert "error" in result, f"Expected error dict for None input, got: {result}"
        except Exception as exc:
            pytest.fail(
                f"predict_niche(None, ...) raised {type(exc).__name__}: {exc} — "
                "it must never raise (behavioral guarantee #1)"
            )

    def test_broken_model_returns_error_dict_not_exception(self):
        """
        Even a completely malformed model artifact must not cause an exception to propagate.
        The error dict protects the caller from internal model failures.
        """
        broken_artifact = {"model": None, "niche_labels": CANONICAL_NICHES, "model_version": "test"}
        valid_vector = [0] * 40
        try:
            result = predict_niche(valid_vector, broken_artifact)
            assert "error" in result, f"Expected error dict for broken model, got: {result}"
        except Exception as exc:
            pytest.fail(
                f"predict_niche with broken model raised {type(exc).__name__}: {exc} — "
                "it must never raise (behavioral guarantee #1)"
            )


# ── CONTRACT TESTS: prediction logging ────────────────────────────────────────

class TestPredictionLogging:
    """
    Tests that verify the prediction log (logs/predictions.jsonl) is always written.
    Behavioral guarantee #2: predict_niche() always writes a PredictionLogEntry.
    SC-008: 100% of invocations produce a log entry, including error cases.
    """

    def test_successful_prediction_writes_log_entry(self, minimal_artifact, valid_40d_vector, tmp_path, monkeypatch):
        """
        After a successful prediction, at least one new line must appear in
        logs/predictions.jsonl. We redirect the log path to a temp directory
        to avoid polluting the real logs/ directory during tests.
        """
        # Redirect the prediction log to the tmp_path so tests don't write to real logs/
        import src.classifier as clf_module
        log_path = tmp_path / "predictions.jsonl"
        monkeypatch.setattr(clf_module, "_PREDICTION_LOG_PATH", log_path)

        # Count lines before the call
        initial_lines = 0
        if log_path.exists():
            initial_lines = sum(1 for _ in log_path.open())

        predict_niche(valid_40d_vector, minimal_artifact)

        # There must be exactly one new line in the log
        assert log_path.exists(), "Log file was not created"
        lines = [line for line in log_path.open() if line.strip()]
        assert len(lines) == initial_lines + 1, (
            f"Expected {initial_lines + 1} log lines, got {len(lines)}"
        )

    def test_error_prediction_also_writes_log_entry(self, minimal_artifact, tmp_path, monkeypatch):
        """
        Even when predict_niche() returns an error (wrong vector length), it must
        still write a log entry — with predicted_niche=null (as per the contract).
        This ensures the audit trail is complete even for bad inputs.
        """
        import src.classifier as clf_module
        log_path = tmp_path / "predictions.jsonl"
        monkeypatch.setattr(clf_module, "_PREDICTION_LOG_PATH", log_path)

        bad_vector = [0] * 5  # Wrong length — will trigger INVALID_VECTOR error
        result = predict_niche(bad_vector, minimal_artifact)

        assert "error" in result, "Expected an error result for 5-element vector"
        assert log_path.exists(), "Log file must exist even for error predictions"

        # Read the log entry and verify it has the null predicted_niche
        log_lines = [json.loads(line) for line in log_path.open() if line.strip()]
        assert len(log_lines) >= 1, "At least one log entry must be written"
        last_entry = log_lines[-1]
        assert last_entry["predicted_niche"] is None, (
            f"Error prediction log entry must have predicted_niche=null, got: {last_entry}"
        )

    def test_log_entry_has_correct_schema(self, minimal_artifact, valid_40d_vector, tmp_path, monkeypatch):
        """
        The log entry must have all 5 required fields from the PredictionLogEntry schema
        in data-model.md: timestamp, model_version, input_hash, predicted_niche, confidence.
        """
        import src.classifier as clf_module
        log_path = tmp_path / "predictions.jsonl"
        monkeypatch.setattr(clf_module, "_PREDICTION_LOG_PATH", log_path)

        predict_niche(valid_40d_vector, minimal_artifact)

        log_entry = json.loads(log_path.read_text().strip().split("\n")[-1])
        required_fields = {"timestamp", "model_version", "input_hash", "predicted_niche", "confidence"}
        assert set(log_entry.keys()) == required_fields, (
            f"Log entry must have exactly {required_fields}, got {set(log_entry.keys())}"
        )

    def test_log_entry_input_hash_is_not_raw_vector(self, minimal_artifact, valid_40d_vector, tmp_path, monkeypatch):
        """
        The input_hash in the log must NOT be the raw feature vector itself.
        It must be an 8-character hash string (PII protection per FR-014).
        """
        import src.classifier as clf_module
        log_path = tmp_path / "predictions.jsonl"
        monkeypatch.setattr(clf_module, "_PREDICTION_LOG_PATH", log_path)

        predict_niche(valid_40d_vector, minimal_artifact)

        log_entry = json.loads(log_path.read_text().strip().split("\n")[-1])
        input_hash = log_entry["input_hash"]

        # Must be exactly 8 characters (our truncated SHA-256)
        assert isinstance(input_hash, str), f"input_hash must be str, got {type(input_hash)}"
        assert len(input_hash) == 8, f"input_hash must be 8 chars, got {len(input_hash)}"

        # Must NOT contain the raw vector values (e.g., not the vector as a list string)
        vector_str = str(valid_40d_vector)
        assert input_hash not in vector_str, "input_hash should not be a substring of the raw vector"
        assert vector_str not in log_entry.get("input_hash", ""), (
            "Raw feature vector must never appear in the log entry"
        )


# ── CONTRACT TESTS: load_model() schema version validation ────────────────────

class TestLoadModelSchemaValidation:
    """
    Tests for load_model()'s schema version checking behavior (FR-009).
    A model trained on a different VECTOR_INDEX_ORDER must be rejected.
    """

    def test_load_model_raises_schema_version_error_on_mismatch(self, tmp_path):
        """
        If we deliberately embed a wrong schema version in the artifact file,
        load_model() must raise SchemaVersionError (not return silently with wrong data).
        This is behavioral guarantee #5 in the contract.
        """
        import joblib

        # _make_minimal_model() returns (clf, centroids) — unpack both even though we only
        # need clf here. The artifact intentionally has a wrong schema_version so load_model()
        # raises SchemaVersionError before it even gets to centroid validation.
        clf, _centroids = _make_minimal_model()
        niche_labels = [str(c) for c in clf.classes_]

        # Build an artifact with a deliberately wrong schema version
        artifact = {
            "model":           clf,
            "niche_labels":    niche_labels,
            "training_record": {},
            "schema_version":  "deadbeef",  # Definitely wrong — 8-char placeholder
            "model_version":   "test_run_bad_schema",
        }
        artifact_path = tmp_path / "bad_schema.joblib"
        joblib.dump(artifact, str(artifact_path))

        # load_model() must raise SchemaVersionError, not silently return
        with pytest.raises(SchemaVersionError) as exc_info:
            load_model(str(artifact_path))

        # The error message must show both versions for debugging
        error_msg = str(exc_info.value)
        assert "deadbeef" in error_msg, "Error must show the artifact's schema version"
        assert current_schema_version() in error_msg, "Error must show the current schema version"

    def test_load_model_raises_model_load_error_on_missing_file(self, tmp_path):
        """
        Loading a model from a path that doesn't exist must raise ModelLoadError.
        """
        with pytest.raises(ModelLoadError):
            load_model(str(tmp_path / "does_not_exist.joblib"))

    def test_load_model_raises_model_load_error_on_corrupted_file(self, tmp_path):
        """
        Loading a corrupted (non-joblib) file must raise ModelLoadError,
        not a raw joblib exception.
        """
        corrupted = tmp_path / "corrupted.joblib"
        corrupted.write_bytes(b"this is not a valid joblib file at all")

        with pytest.raises(ModelLoadError):
            load_model(str(corrupted))

    def test_valid_artifact_loads_without_error(self, minimal_artifact):
        """
        A properly constructed artifact (from our fixture) must load cleanly.
        This verifies the happy path works after all the error path tests.
        """
        # If the fixture loaded successfully, we already know this works.
        # This test makes it explicit and documents the expected behavior.
        assert "model" in minimal_artifact
        assert "niche_labels" in minimal_artifact
        assert "schema_version" in minimal_artifact
        assert minimal_artifact["schema_version"] == current_schema_version()
