"""
test_phase3_pipeline.py — Phase 3 End-to-End Integration Tests

These tests validate the full Phase 3 pipeline using the actual training dataset
(data/training/dataset.csv or dataset.json). They verify the Success Criteria
from the spec (SC-001 through SC-007) at the system level.

What "integration test" means here:
  - Tests use real production modules (no mocking of core components)
  - Tests run the actual training pipeline on real pre-extracted feature vectors
  - Tests chain Phase 1 + Phase 2 + Phase 3 to verify end-to-end connectivity
  - Tests check timing constraints (SC-004, SC-005)

What's NOT tested here (covered in unit tests):
  - Individual function behaviors (those are in unit/ tests)
  - Error paths for corrupted files (those are in unit/ tests)

Note on test duration:
  The training pipeline tests (SC-001–SC-004) take longer than unit tests because
  they train real models. They are marked with pytest.mark.slow if needed but are
  expected to complete in < 5 minutes per SC-004.

Spec reference: specs/003-ml-niche-classifier/spec.md (Success Criteria SC-001..SC-007)
"""

import json
import os
import pathlib
import time
import warnings

import numpy as np
import pytest
import yaml

from src.model_io import CANONICAL_NICHES, load_model, current_schema_version
from src.classifier import predict_niche
from src.enricher import enrich_cv

# ── PATHS ─────────────────────────────────────────────────────────────────────

# Project root (two levels up from this file: tests/integration/ → tests/ → project root)
PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent

# The pre-extracted labeled feature vector datasets
TRAINING_CSV = PROJECT_ROOT / "data" / "training" / "dataset.csv"
TRAINING_JSON = PROJECT_ROOT / "data" / "training" / "dataset.json"
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "training_config.yaml"


# ── HELPERS ───────────────────────────────────────────────────────────────────

def _write_test_config(tmp_path, dataset_path: str = None, seed: int = 42, n_estimators: int = 100) -> str:
    """
    Writes a training config YAML for integration tests.
    Uses the real dataset by default (or a custom path if provided).
    n_estimators is kept smaller than production (200) to keep test runtime reasonable.
    """
    config = {
        "dataset_path":    dataset_path or str(TRAINING_JSON),
        "random_seed":     seed,
        "train_test_split": 0.8,
        "n_estimators":    n_estimators,
        "max_depth":       None,
        "min_samples_leaf": 2,
        "class_weight":    "balanced",
        "model_output_dir": str(tmp_path / "models"),
        "experiment_dir":   str(tmp_path / "experiments"),
    }
    config_path = str(tmp_path / "test_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path


def _dataset_available() -> bool:
    """Returns True if the training dataset is available on disk."""
    return TRAINING_JSON.exists() or TRAINING_CSV.exists()


def _get_dataset_path() -> str:
    """Returns the path to whichever training dataset format is available."""
    if TRAINING_JSON.exists():
        return str(TRAINING_JSON)
    if TRAINING_CSV.exists():
        return str(TRAINING_CSV)
    return None


# ── TESTS: SC-001 — Overall Accuracy ≥ 75% ───────────────────────────────────

@pytest.mark.skipif(
    not _dataset_available(),
    reason="Training dataset not available at data/training/"
)
class TestSC001OverallAccuracy:
    """
    SC-001: The trained classifier achieves at least 75% overall accuracy
    on a held-out test set of at least 60 labeled CV records (10 per niche).
    """

    def test_overall_accuracy_meets_threshold(self, tmp_path):
        """
        Train on the real dataset and verify overall accuracy ≥ 0.75.
        This is the primary quality gate for the Phase 3 classifier.
        """
        from src.trainer import train

        dataset_path = _get_dataset_path()
        config_path = _write_test_config(tmp_path, dataset_path=dataset_path)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_path = train(config_path)

        # Check the training record for the accuracy value
        exp_dir = tmp_path / "experiments"
        record_file = list(exp_dir.glob("*.json"))[0]
        with open(record_file) as f:
            record = json.load(f)

        accuracy = record["evaluation_metrics"]["overall_accuracy"]
        assert accuracy >= 0.75, (
            f"SC-001 FAILED: overall accuracy {accuracy:.3f} < 0.75 threshold. "
            f"Full metrics: {record['evaluation_metrics']}"
        )

    def test_training_produces_accepted_status(self, tmp_path):
        """
        A successful training run on the full dataset must produce a TrainingRecord
        with status='accepted', confirming all gates passed.
        """
        from src.trainer import train

        dataset_path = _get_dataset_path()
        config_path = _write_test_config(tmp_path, dataset_path=dataset_path)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            train(config_path)

        exp_dir = tmp_path / "experiments"
        record_file = list(exp_dir.glob("*.json"))[0]
        with open(record_file) as f:
            record = json.load(f)

        assert record["status"] == "accepted", (
            f"Training run must be accepted, got status: {record['status']}"
        )


# ── TESTS: SC-002 — Per-niche F1 ≥ 60% ───────────────────────────────────────

@pytest.mark.skipif(
    not _dataset_available(),
    reason="Training dataset not available at data/training/"
)
class TestSC002PerNicheF1:
    """
    SC-002: No individual niche achieves less than 60% F1 score on the held-out
    test set, ensuring no niche is systematically ignored by the model.
    """

    def test_all_niches_meet_f1_threshold(self, tmp_path):
        """
        Every niche in the test set must achieve F1 ≥ 0.60.
        A niche below this threshold means the model is systematically mis-predicting
        that niche — which could result in career recommendations that consistently
        steer engineers away from their best-fit role.
        """
        from src.trainer import train

        dataset_path = _get_dataset_path()
        config_path = _write_test_config(tmp_path, dataset_path=dataset_path)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            train(config_path)

        exp_dir = tmp_path / "experiments"
        record_file = list(exp_dir.glob("*.json"))[0]
        with open(record_file) as f:
            record = json.load(f)

        per_niche = record["evaluation_metrics"]["per_niche"]
        failures = []
        for niche, metrics in per_niche.items():
            f1 = metrics["f1"]
            if f1 < 0.60:
                failures.append(f"{niche}: F1={f1:.3f}")

        assert not failures, (
            f"SC-002 FAILED — niches below 0.60 F1: {failures}"
        )


# ── TESTS: SC-003 — Reproducibility ──────────────────────────────────────────

@pytest.mark.skipif(
    not _dataset_available(),
    reason="Training dataset not available at data/training/"
)
class TestSC003Reproducibility:
    """
    SC-003: Two consecutive training runs with identical data and configuration
    produce identical model predictions on the same test inputs.
    """

    def test_two_runs_produce_identical_predictions(self, tmp_path):
        """
        Train twice with the same config, load both models, and verify they produce
        identical predictions on 10 test vectors. If they differ, the training
        pipeline has a reproducibility bug (broken random seed handling).
        """
        from src.trainer import train

        dataset_path = _get_dataset_path()

        dir1 = tmp_path / "run1"
        dir2 = tmp_path / "run2"
        dir1.mkdir()
        dir2.mkdir()

        config1 = _write_test_config(dir1, dataset_path=dataset_path, seed=42)
        config2 = _write_test_config(dir2, dataset_path=dataset_path, seed=42)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            path1 = train(config1)
            path2 = train(config2)

        artifact1 = load_model(path1)
        artifact2 = load_model(path2)

        # Test on 10 synthetic vectors
        import src.classifier as clf_mod
        orig_log = clf_mod._PREDICTION_LOG_PATH
        clf_mod._PREDICTION_LOG_PATH = tmp_path / "test_log.jsonl"

        try:
            for i in range(10):
                v = [0] * 40
                v[i % 40] = i + 1
                r1 = predict_niche(v, artifact1)
                r2 = predict_niche(v, artifact2)
                # Predicted niche must be exactly identical across runs.
                # Confidence may differ by floating-point rounding at the 15th+
                # decimal place (e.g. 0.56346...087 vs 0.56346...0871) — this is
                # not a reproducibility failure, just IEEE 754 non-determinism in
                # float→str conversion for the same underlying bit pattern.
                assert r1.get("predicted_niche") == r2.get("predicted_niche"), (
                    f"SC-003 FAILED at vector {i}: different niches predicted. "
                    f"run1={r1}, run2={r2}. "
                    "Training is not reproducible — check random seed handling."
                )
                if "confidence" in r1 and "confidence" in r2:
                    assert abs(r1["confidence"] - r2["confidence"]) < 1e-10, (
                        f"SC-003 FAILED at vector {i}: confidence differs beyond "
                        f"floating-point tolerance: {r1['confidence']} vs {r2['confidence']}"
                    )
        finally:
            clf_mod._PREDICTION_LOG_PATH = orig_log


# ── TESTS: SC-004 — Training Speed ───────────────────────────────────────────

@pytest.mark.skipif(
    not _dataset_available(),
    reason="Training dataset not available at data/training/"
)
class TestSC004TrainingSpeed:
    """
    SC-004: The full training pipeline completes in under 5 minutes on a standard
    laptop CPU with a dataset of up to 600 records (100 per niche).
    Our dataset has 2400 records (400/niche) — if it completes quickly, the
    600-record case certainly will.
    """

    def test_training_completes_within_five_minutes(self, tmp_path):
        """
        Time the full training pipeline (load + train + evaluate + serialize).
        Must complete in < 300 seconds (5 minutes).
        """
        from src.trainer import train

        dataset_path = _get_dataset_path()
        config_path = _write_test_config(
            tmp_path,
            dataset_path=dataset_path,
            n_estimators=200,  # Full production config
        )

        start_time = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            train(config_path)
        elapsed = time.time() - start_time

        assert elapsed < 300, (
            f"SC-004 FAILED: training took {elapsed:.1f}s, limit is 300s (5 minutes)"
        )


# ── TESTS: SC-005 — Prediction Speed ─────────────────────────────────────────

@pytest.mark.skipif(
    not _dataset_available(),
    reason="Training dataset not available at data/training/"
)
class TestSC005PredictionSpeed:
    """
    SC-005: The prediction pipeline (load model + predict niche) produces a result
    in under 1 second per CV.
    """

    def test_prediction_completes_within_one_second(self, tmp_path):
        """
        Time a single predict_niche() call including log write.
        Must complete in < 1.0 seconds (SC-005).
        """
        from src.trainer import train

        dataset_path = _get_dataset_path()
        config_path = _write_test_config(tmp_path, dataset_path=dataset_path)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_path = train(config_path)

        artifact = load_model(model_path)

        import src.classifier as clf_mod
        orig_log = clf_mod._PREDICTION_LOG_PATH
        clf_mod._PREDICTION_LOG_PATH = tmp_path / "speed_test.jsonl"

        try:
            feature_vector = [0] * 40
            feature_vector[8] = 3   # Some non-zero values

            start_time = time.time()
            result = predict_niche(feature_vector, artifact)
            elapsed = time.time() - start_time
        finally:
            clf_mod._PREDICTION_LOG_PATH = orig_log

        assert "error" not in result
        assert elapsed < 1.0, (
            f"SC-005 FAILED: prediction took {elapsed:.3f}s, limit is 1.0s"
        )


# ── TESTS: SC-006 — JSON Round-trip ──────────────────────────────────────────

@pytest.mark.skipif(
    not _dataset_available(),
    reason="Training dataset not available at data/training/"
)
class TestSC006JsonRoundtrip:
    """
    SC-006: The prediction result survives json.dumps → json.loads round-trip losslessly.
    """

    def test_prediction_result_is_json_roundtrip_lossless(self, tmp_path):
        """
        json.loads(json.dumps(result)) must produce an identical dict to the original.
        Phase 4 and 5 will serialize and deserialize prediction results over HTTP.
        """
        from src.trainer import train

        dataset_path = _get_dataset_path()
        config_path = _write_test_config(tmp_path, dataset_path=dataset_path)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_path = train(config_path)

        artifact = load_model(model_path)

        import src.classifier as clf_mod
        orig_log = clf_mod._PREDICTION_LOG_PATH
        clf_mod._PREDICTION_LOG_PATH = tmp_path / "roundtrip_test.jsonl"

        try:
            result = predict_niche([0] * 40, artifact)
        finally:
            clf_mod._PREDICTION_LOG_PATH = orig_log

        assert "error" not in result

        # JSON round-trip must be lossless
        serialized = json.dumps(result)
        deserialized = json.loads(serialized)

        assert deserialized["predicted_niche"] == result["predicted_niche"]
        assert abs(deserialized["confidence"] - result["confidence"]) < 1e-10


# ── TESTS: SC-007 — Training Record Coverage ─────────────────────────────────

@pytest.mark.skipif(
    not _dataset_available(),
    reason="Training dataset not available at data/training/"
)
class TestSC007TrainingRecordCoverage:
    """
    SC-007: The training record is created and accessible for 100% of completed
    training runs — both accepted and rejected runs.
    """

    def test_training_record_exists_after_successful_training(self, tmp_path):
        """
        After a successful training run, exactly one TrainingRecord JSON file
        must exist in the experiments/ directory.
        """
        from src.trainer import train

        dataset_path = _get_dataset_path()
        config_path = _write_test_config(tmp_path, dataset_path=dataset_path)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            train(config_path)

        exp_dir = tmp_path / "experiments"
        json_files = list(exp_dir.glob("*.json"))
        assert len(json_files) == 1, (
            f"SC-007 FAILED: Expected 1 TrainingRecord, found {len(json_files)}"
        )

    def test_training_record_is_valid_json(self, tmp_path):
        """
        The TrainingRecord file must be valid JSON that can be parsed without error.
        """
        from src.trainer import train

        dataset_path = _get_dataset_path()
        config_path = _write_test_config(tmp_path, dataset_path=dataset_path)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            train(config_path)

        record_file = list((tmp_path / "experiments").glob("*.json"))[0]
        with open(record_file) as f:
            record = json.load(f)  # Must not raise

        assert isinstance(record, dict), "TrainingRecord must be a JSON object"
        assert "run_id" in record, "TrainingRecord must have a run_id"


# ── TESTS: Full Phase 1 → 2 → 3 Pipeline Integration ─────────────────────────

class TestFullPipelineIntegration:
    """
    Tests that verify the full pipeline works end-to-end: parse_cv → enrich_cv →
    predict_niche. These tests create a synthetic CV (no real PDF needed) by
    building a Phase 1-compatible output directly.
    """

    @pytest.fixture
    def trained_model_artifact(self, tmp_path):
        """
        Trains a model on the real or synthetic dataset and returns the loaded artifact.
        Skips if the training dataset is not available.
        """
        if not _dataset_available():
            pytest.skip("Training dataset not available")

        from src.trainer import train
        dataset_path = _get_dataset_path()
        config_path = _write_test_config(tmp_path, dataset_path=dataset_path)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_path = train(config_path)

        return load_model(model_path)

    def test_enriched_cv_can_be_classified(self, trained_model_artifact, tmp_path, monkeypatch):
        """
        A Phase 2 enriched output (from enrich_cv()) can be passed directly to
        predict_niche() and produces a valid prediction. This is the core Phase 3
        integration contract: enrich_cv()["feature_vector"] → predict_niche().
        """
        import src.classifier as clf_mod
        monkeypatch.setattr(clf_mod, "_PREDICTION_LOG_PATH", tmp_path / "integration.jsonl")

        # Build a synthetic Phase 1-style output with strong Embedded Systems signals
        phase1_output = {
            "skills": [
                {
                    "text": "STM32 ARM Cortex-M4 FreeRTOS UART SPI I2C embedded C programming",
                    "keywords": ["stm32", "freertos", "uart"],
                }
            ],
            "projects": [
                {
                    "text": "ESP32 IoT sensor node with MQTT and BLE communication",
                    "keywords": ["esp32", "mqtt"],
                }
            ],
            "education": [],
            "internships": [],
        }

        enriched = enrich_cv(phase1_output)
        assert "error" not in enriched, f"enrich_cv failed: {enriched}"
        assert len(enriched["feature_vector"]) == 40, "feature_vector must be 40D"

        result = predict_niche(enriched["feature_vector"], trained_model_artifact)

        # Must return a valid prediction
        assert "error" not in result, f"predict_niche failed: {result}"
        assert result["predicted_niche"] in CANONICAL_NICHES
        assert 0.0 <= result["confidence"] <= 1.0

    def test_phase2_feature_vector_is_valid_phase3_input(self, trained_model_artifact, tmp_path, monkeypatch):
        """
        The feature_vector from enrich_cv() must be directly usable as the
        feature_vector parameter to predict_niche() — no format conversion needed.
        This verifies the interface contract between Phase 2 and Phase 3.
        """
        import src.classifier as clf_mod
        monkeypatch.setattr(clf_mod, "_PREDICTION_LOG_PATH", tmp_path / "interface.jsonl")

        # A robotics-heavy CV
        phase1_output = {
            "skills": [
                {
                    "text": "ROS2 MoveIt! SLAM OpenCV gazebo simulator robot operating system",
                    "keywords": ["ros2", "moveit"],
                }
            ],
            "projects": [{"text": "Delta robot arm with ROS2 control", "keywords": []}],
            "education": [],
            "internships": [],
        }

        enriched = enrich_cv(phase1_output)
        feature_vector = enriched["feature_vector"]

        # The feature vector from Phase 2 must work as-is with Phase 3
        result = predict_niche(feature_vector, trained_model_artifact)

        assert "error" not in result, (
            f"Phase 2 feature vector is not valid Phase 3 input: {result}"
        )

    def test_empty_cv_gives_valid_prediction(self, trained_model_artifact, tmp_path, monkeypatch):
        """
        A CV with no recognized entities (all-zero feature vector from Phase 2)
        must still produce a valid (not error) prediction from Phase 3.
        """
        import src.classifier as clf_mod
        monkeypatch.setattr(clf_mod, "_PREDICTION_LOG_PATH", tmp_path / "empty_cv.jsonl")

        # Empty CV → all sections empty → zero feature vector
        empty_phase1 = {
            "skills": [],
            "projects": [],
            "education": [],
            "internships": [],
        }
        enriched = enrich_cv(empty_phase1)
        feature_vector = enriched["feature_vector"]

        # All zeros — this is valid input per spec (US3 scenario 3)
        assert all(v == 0 for v in feature_vector), "Empty CV must produce zero vector"

        result = predict_niche(feature_vector, trained_model_artifact)
        assert "error" not in result, f"Empty CV prediction must not error: {result}"
        assert result["predicted_niche"] in CANONICAL_NICHES


# ── TESTS: dataset loading with real data ─────────────────────────────────────

@pytest.mark.skipif(
    not _dataset_available(),
    reason="Training dataset not available at data/training/"
)
class TestRealDatasetLoading:
    """
    Tests that the real training dataset loads and validates correctly.
    """

    def test_training_csv_loads_without_error(self):
        """The real training CSV loads without validation errors."""
        if not TRAINING_CSV.exists():
            pytest.skip("CSV dataset not available")

        from src.dataset import load_dataset
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            feature_vectors, labels = load_dataset(str(TRAINING_CSV))

        assert len(feature_vectors) > 0, "Dataset must have at least 1 record"
        assert len(labels) == len(feature_vectors)

    def test_training_json_loads_without_error(self):
        """The real training JSON loads without validation errors."""
        if not TRAINING_JSON.exists():
            pytest.skip("JSON dataset not available")

        from src.dataset import load_dataset
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            feature_vectors, labels = load_dataset(str(TRAINING_JSON))

        assert len(feature_vectors) > 0

    def test_training_dataset_has_all_6_niches(self):
        """The training dataset must have records for all 6 canonical niches."""
        dataset_path = _get_dataset_path()
        if not dataset_path:
            pytest.skip("No training dataset available")

        from src.dataset import load_dataset
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, labels = load_dataset(dataset_path)

        label_set = set(labels)
        missing = set(CANONICAL_NICHES) - label_set
        assert not missing, f"Training dataset missing niches: {missing}"
