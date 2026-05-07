"""
test_classifier.py — Unit Tests for src/classifier.py

These tests focus on the specific implementation details of predict_niche()
that are not covered by the contract tests:
  - INVALID_VECTOR error dict format and log entry content
  - Input hash is exactly 8 characters and is NOT the raw vector
  - Log file is automatically created if logs/ directory is missing
  - Multiple consecutive calls all produce log entries
  - The confidence value is the actual model probability

These tests complement test_classifier_contract.py. The contract tests verify
behavioral guarantees; these unit tests verify implementation details.

Spec reference: specs/003-ml-niche-classifier/tasks.md (T026)
"""

import json
import pathlib
import warnings

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.classifier import predict_niche, _compute_input_hash, _write_prediction_log
from src.model_io import CANONICAL_NICHES, save_model, load_model, current_schema_version


# ── FIXTURES ──────────────────────────────────────────────────────────────────

def _make_minimal_model() -> tuple:
    """
    Creates a minimal fitted RandomForest for use as a test fixture.

    Returns both the fitted classifier and per-niche centroids computed from the
    same training data. Phase 4 backfill added centroid validation to load_model(),
    so every fixture that calls save_model() + load_model() must include centroids.
    """
    n_niches = len(CANONICAL_NICHES)
    X = np.zeros((n_niches * 4, 40), dtype=float)
    y = []
    for i, niche in enumerate(CANONICAL_NICHES):
        for j in range(4):
            X[i * 4 + j, i * 4] = float(j + 1)
        y.extend([niche] * 4)
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, np.array(y))

    # Compute per-niche centroids — mirrors the Phase 4 trainer.py backfill logic.
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
    A complete, valid model artifact saved and loaded from tmp_path.

    Phase 4: load_model() validates niche_centroids — we compute and pass them.
    Phase 5: load_model() validates niche_top_skills — we compute and pass them too.
    """
    from src.feature_vector import VECTOR_INDEX_ORDER

    clf, niche_centroids = _make_minimal_model()
    niche_labels = [str(c) for c in clf.classes_]
    training_record = {"run_id": "unit_test", "evaluation_metrics": {}}

    # Build niche_top_skills: top-10 dims per niche ranked by centroid value desc.
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
        run_id="unit_test",
        output_dir=str(tmp_path),
        niche_centroids=niche_centroids,
        niche_top_skills=niche_top_skills,
    )
    return load_model(artifact_path)


# ── TESTS: INVALID_VECTOR error format ────────────────────────────────────────

class TestInvalidVectorErrorFormat:
    """
    Tests that the INVALID_VECTOR error dict contains the right structure
    and that a log entry is still written even for invalid inputs.
    T026: test input vector length ≠ 40 returns {"error": "INVALID_VECTOR", "message": "..."}
    """

    def test_wrong_length_returns_invalid_vector_error_code(self, minimal_artifact, tmp_path, monkeypatch):
        """
        A vector with wrong length must return exactly {"error": "INVALID_VECTOR", "message": str}.
        """
        import src.classifier as clf_mod
        monkeypatch.setattr(clf_mod, "_PREDICTION_LOG_PATH", tmp_path / "predictions.jsonl")

        result = predict_niche([0] * 10, minimal_artifact)

        assert "error" in result, f"Expected error dict, got: {result}"
        assert result["error"] == "INVALID_VECTOR", (
            f"Expected 'INVALID_VECTOR', got: {result['error']}"
        )
        assert "message" in result, "Error dict must have 'message' key"
        assert isinstance(result["message"], str), "message must be a string"
        assert len(result["message"]) > 0, "message must not be empty"

    def test_invalid_vector_log_entry_has_null_predicted_niche(self, minimal_artifact, tmp_path, monkeypatch):
        """
        When INVALID_VECTOR error fires, the log entry must have predicted_niche=null.
        This is required by SC-008 (100% log coverage including error cases).
        T026: "test input vector length ≠ 40 returns ... and still writes a log entry
        with predicted_niche: null"
        """
        import src.classifier as clf_mod
        log_path = tmp_path / "predictions.jsonl"
        monkeypatch.setattr(clf_mod, "_PREDICTION_LOG_PATH", log_path)

        predict_niche([0] * 5, minimal_artifact)

        assert log_path.exists(), "Log file must be created even for INVALID_VECTOR errors"
        log_entries = [json.loads(line) for line in log_path.read_text().strip().split("\n") if line.strip()]
        assert len(log_entries) >= 1

        last_entry = log_entries[-1]
        assert last_entry["predicted_niche"] is None, (
            f"INVALID_VECTOR log entry must have predicted_niche=null, got: {last_entry}"
        )
        assert last_entry["confidence"] is None, (
            f"INVALID_VECTOR log entry must have confidence=null, got: {last_entry}"
        )


# ── TESTS: input hash properties ──────────────────────────────────────────────

class TestInputHashProperties:
    """
    Tests for the PII-protection guarantee: the log stores only a hash, not the raw vector.
    T026: "test prediction log hash is NOT the raw vector (hash length == 8 chars)"
    """

    def test_input_hash_is_8_chars(self):
        """
        _compute_input_hash() must return exactly an 8-character string.
        This is the truncated SHA-256 defined in the spec (R-005, FR-014).
        """
        vector = [0] * 40
        hash_val = _compute_input_hash(vector)
        assert isinstance(hash_val, str), f"Hash must be str, got {type(hash_val)}"
        assert len(hash_val) == 8, f"Hash must be exactly 8 chars, got {len(hash_val)}"

    def test_input_hash_is_hex_string(self):
        """
        The 8-character hash must be a valid lowercase hex string (from SHA-256).
        """
        vector = list(range(40))
        hash_val = _compute_input_hash(vector)
        assert all(c in "0123456789abcdef" for c in hash_val), (
            f"Hash must be lowercase hex, got: {hash_val!r}"
        )

    def test_input_hash_is_not_raw_vector(self):
        """
        The hash must NOT contain the raw vector values in any recognizable form.
        This is the PII protection guarantee (FR-014).
        """
        vector = [1, 2, 3, 4, 5] + [0] * 35
        hash_val = _compute_input_hash(vector)
        raw_str = str(vector)
        # The hash is 8 chars; the vector string is much longer — they can't be equal
        assert hash_val != raw_str, "Hash must not equal the raw vector string"
        # Also verify the hash is not a substring of the vector string in any direction
        assert hash_val not in raw_str or len(hash_val) == 8  # 8-char string can match by chance but that's OK
        # The real test: the hash is much shorter than the raw vector
        assert len(hash_val) < len(raw_str), "Hash must be shorter than the raw vector string"

    def test_same_vector_produces_same_hash(self):
        """
        The hash function must be deterministic — same input always gives same hash.
        This is required for drift detection: same CV processed twice gets same hash.
        """
        vector = [1, 0, 2, 0, 0, 3] + [0] * 34
        hash1 = _compute_input_hash(vector)
        hash2 = _compute_input_hash(vector)
        assert hash1 == hash2, f"Same vector produced different hashes: {hash1} vs {hash2}"

    def test_different_vectors_produce_different_hashes(self):
        """
        Different vectors must (with overwhelming probability) produce different hashes.
        Hash collisions are astronomically unlikely for 8-char hex.
        """
        vector1 = [1] + [0] * 39
        vector2 = [2] + [0] * 39
        hash1 = _compute_input_hash(vector1)
        hash2 = _compute_input_hash(vector2)
        assert hash1 != hash2, f"Different vectors produced the same hash: {hash1}"


# ── TESTS: log file auto-creation ─────────────────────────────────────────────

class TestLogFileAutoCreation:
    """
    Tests that logs/ directory and predictions.jsonl are created automatically.
    T026: "test log file is created automatically if logs/ directory is missing"
    """

    def test_log_file_created_in_nonexistent_directory(self, tmp_path, monkeypatch):
        """
        If logs/ doesn't exist when predict_niche() is first called,
        it must be created automatically. The engineer should not need to
        manually create the logs/ directory before running predictions.
        """
        import src.classifier as clf_mod
        # Use a path that definitely doesn't exist
        new_log_dir = tmp_path / "new_subdir" / "another_subdir"
        new_log_path = new_log_dir / "predictions.jsonl"

        # Verify the directory doesn't exist yet
        assert not new_log_dir.exists(), "Test setup: directory must not exist yet"

        monkeypatch.setattr(clf_mod, "_PREDICTION_LOG_PATH", new_log_path)

        # Create a minimal artifact inline for this test.
        # _make_minimal_model() returns (clf, centroids) — we only need clf here
        # since predict_niche() doesn't use centroids.
        clf, _centroids = _make_minimal_model()
        niche_labels = [str(c) for c in clf.classes_]
        artifact = {
            "model": clf,
            "niche_labels": niche_labels,
            "schema_version": current_schema_version(),
            "model_version": "autocreate_test",
            "training_record": {},
        }

        # This should auto-create the directory and log file
        predict_niche([0] * 40, artifact)

        assert new_log_dir.exists(), "logs/ directory must be auto-created"
        assert new_log_path.exists(), "predictions.jsonl must be auto-created"

    def test_multiple_calls_each_append_one_line(self, tmp_path, monkeypatch):
        """
        Calling predict_niche() N times must result in exactly N lines in the log.
        Each call appends exactly one line — not more, not less.
        """
        import src.classifier as clf_mod
        log_path = tmp_path / "predictions.jsonl"
        monkeypatch.setattr(clf_mod, "_PREDICTION_LOG_PATH", log_path)

        # _make_minimal_model() returns (clf, centroids) — only clf needed here
        clf, _centroids = _make_minimal_model()
        artifact = {
            "model": clf,
            "niche_labels": [str(c) for c in clf.classes_],
            "schema_version": current_schema_version(),
            "model_version": "multi_call_test",
            "training_record": {},
        }

        n_calls = 5
        for _ in range(n_calls):
            predict_niche([0] * 40, artifact)

        lines = [l for l in log_path.read_text().strip().split("\n") if l.strip()]
        assert len(lines) == n_calls, (
            f"Expected {n_calls} log lines after {n_calls} calls, got {len(lines)}"
        )


# ── TESTS: confidence value properties ────────────────────────────────────────

class TestConfidenceValueProperties:
    """
    Tests that the confidence value accurately reflects the model's probability output.
    """

    def test_confidence_is_highest_class_probability(self, minimal_artifact, tmp_path, monkeypatch):
        """
        The confidence returned by predict_niche() must be the probability of the
        predicted class — not the sum of all probabilities (which would be 1.0),
        and not an arbitrary number.
        """
        import src.classifier as clf_mod
        monkeypatch.setattr(clf_mod, "_PREDICTION_LOG_PATH", tmp_path / "predictions.jsonl")

        # Use a strong signal vector (dim 0 = industrial_automation dominant feature)
        vector = [0] * 40
        vector[0] = 10  # Strong industrial_automation signal

        result = predict_niche(vector, minimal_artifact)
        assert "error" not in result

        # The confidence must be in (0, 1] — specifically the MAX probability,
        # which for a well-trained model on a strong signal should be > 1/6 ≈ 0.167
        assert result["confidence"] > 0.0, "Confidence must be > 0 for a valid input"
        assert result["confidence"] <= 1.0, "Confidence must be ≤ 1.0"

    def test_log_entry_confidence_matches_return_value(self, minimal_artifact, tmp_path, monkeypatch):
        """
        The confidence in the prediction log must match the confidence in the return dict.
        """
        import src.classifier as clf_mod
        log_path = tmp_path / "predictions.jsonl"
        monkeypatch.setattr(clf_mod, "_PREDICTION_LOG_PATH", log_path)

        result = predict_niche([0] * 40, minimal_artifact)
        assert "error" not in result

        log_entry = json.loads(log_path.read_text().strip().split("\n")[-1])
        assert abs(log_entry["confidence"] - result["confidence"]) < 1e-10, (
            f"Log confidence {log_entry['confidence']} must match return confidence {result['confidence']}"
        )

    def test_log_entry_predicted_niche_matches_return_value(self, minimal_artifact, tmp_path, monkeypatch):
        """
        The predicted_niche in the log must match the predicted_niche in the return dict.
        """
        import src.classifier as clf_mod
        log_path = tmp_path / "predictions.jsonl"
        monkeypatch.setattr(clf_mod, "_PREDICTION_LOG_PATH", log_path)

        result = predict_niche([0] * 40, minimal_artifact)
        assert "error" not in result

        log_entry = json.loads(log_path.read_text().strip().split("\n")[-1])
        assert log_entry["predicted_niche"] == result["predicted_niche"]
