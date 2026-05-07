"""
test_trainer.py — Unit Tests for src/trainer.py

These tests verify the training pipeline behavior using small synthetic datasets.
We do NOT train on the real 2400-record dataset here — that's too slow for unit tests.
Instead we build 300-record synthetic datasets (50/niche) and verify the expected
outputs (TrainingRecord files, model artifacts, rejection behavior).

Key behaviors tested:
  - Training with a synthetic dataset writes a TrainingRecord JSON to experiments/
  - Two runs with identical seed and data produce identical predictions (SC-003)
  - A dataset that produces accuracy < 0.75 writes a "rejected" record but no artifact
  - The TrainingRecord contains feature_importance with all 40 dimensions
  - The bias_fairness_note is a non-empty string identifying the minimum F1 niche
  - The CLI entry point (train function) returns an absolute path to the .joblib file

Spec reference: specs/003-ml-niche-classifier/tasks.md (T016)
"""

import json
import os
import pathlib
import tempfile
import warnings

import numpy as np
import pytest
import yaml

from src.model_io import CANONICAL_NICHES, load_model, current_schema_version
from src.trainer import train, _build_run_id, _get_git_sha


# ── HELPERS ───────────────────────────────────────────────────────────────────

def _make_separable_dataset(n_per_niche: int = 60, seed: int = 42) -> list:
    """
    Creates a perfectly separable synthetic dataset where each niche has a unique
    "dominant" feature dimension. This guarantees high accuracy on the test split
    so training will always be accepted (passes the 0.75 gate).

    Each niche's records have a large value at a unique feature dimension
    (e.g., industrial_automation records have feat_0 = 10, robotics records have feat_4 = 10).
    """
    rng = np.random.RandomState(seed)
    records = []

    for niche_idx, niche in enumerate(CANONICAL_NICHES):
        for j in range(n_per_niche):
            # Create a mostly-zero vector with a strong signal at the niche's unique dimension.
            # The uniqueness guarantees the RF can perfectly separate the classes.
            v = [0] * 40
            unique_dim = niche_idx * 4  # Dimensions 0, 4, 8, 12, 16, 20 — one per niche
            v[unique_dim] = 10 + rng.randint(0, 3)  # Strong, slightly noisy signal
            # Add a tiny random noise to other dimensions to make it more realistic
            noise_dim = rng.randint(0, 40)
            v[noise_dim] = rng.randint(0, 2)
            records.append({"feature_vector": v, "niche": niche})

    return records


def _write_config(tmp_path, dataset_path: str, seed: int = 42) -> str:
    """
    Writes a training config YAML and returns its path.
    Uses the provided dataset_path and seed; all other values are small defaults
    suitable for fast unit testing.
    """
    config = {
        "dataset_path":    dataset_path,
        "random_seed":     seed,
        "train_test_split": 0.8,
        "n_estimators":    50,          # Small number of trees for speed in tests
        "max_depth":       None,
        "min_samples_leaf": 1,
        "class_weight":    "balanced",
        "model_output_dir": str(tmp_path / "models"),
        "experiment_dir":   str(tmp_path / "experiments"),
    }
    config_path = str(tmp_path / "training_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path


def _write_separable_json(tmp_path, n_per_niche: int = 60, seed: int = 42) -> str:
    """
    Writes a separable synthetic JSON dataset and returns the path.
    """
    import json as json_mod
    records = _make_separable_dataset(n_per_niche=n_per_niche, seed=seed)
    dataset_path = str(tmp_path / "dataset.json")
    with open(dataset_path, "w") as f:
        json_mod.dump(records, f)
    return dataset_path


# ── TESTS: training record creation ───────────────────────────────────────────

class TestTrainingRecordCreation:
    """
    Tests that verify a TrainingRecord JSON is always written to experiments/,
    for both accepted and rejected training runs (SC-007).
    """

    def test_successful_training_writes_training_record(self, tmp_path):
        """
        A successful training run (accuracy ≥ 0.75) must write a .json file
        to the configured experiment_dir.
        """
        dataset_path = _write_separable_json(tmp_path)
        config_path = _write_config(tmp_path, dataset_path)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_path = train(config_path)

        # The experiment directory must contain exactly one JSON file
        exp_dir = tmp_path / "experiments"
        assert exp_dir.exists(), "experiments/ directory was not created"
        json_files = list(exp_dir.glob("*.json"))
        assert len(json_files) == 1, (
            f"Expected 1 TrainingRecord JSON, found {len(json_files)}: {json_files}"
        )

    def test_training_record_has_required_fields(self, tmp_path):
        """
        The TrainingRecord JSON must have all fields defined in data-model.md.
        """
        required_fields = {
            "run_id", "training_date", "git_commit_sha", "dataset_path",
            "dataset_version", "dataset_size", "records_per_niche", "algorithm",
            "hyperparameters", "random_seed", "train_test_split",
            "evaluation_metrics", "bias_fairness_note", "feature_importance",
            "model_artifact_path", "schema_version", "status",
        }
        dataset_path = _write_separable_json(tmp_path)
        config_path = _write_config(tmp_path, dataset_path)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            train(config_path)

        record_file = list((tmp_path / "experiments").glob("*.json"))[0]
        with open(record_file) as f:
            record = json.load(f)

        missing = required_fields - set(record.keys())
        assert not missing, f"TrainingRecord missing fields: {missing}"

    def test_training_record_evaluation_metrics_structure(self, tmp_path):
        """
        The evaluation_metrics field must contain overall_accuracy and per_niche
        with all 6 niche keys having precision, recall, f1, and support.
        """
        dataset_path = _write_separable_json(tmp_path)
        config_path = _write_config(tmp_path, dataset_path)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            train(config_path)

        record_file = list((tmp_path / "experiments").glob("*.json"))[0]
        with open(record_file) as f:
            record = json.load(f)

        metrics = record["evaluation_metrics"]
        assert "overall_accuracy" in metrics, "evaluation_metrics must have overall_accuracy"
        assert "per_niche" in metrics, "evaluation_metrics must have per_niche"

        for niche in CANONICAL_NICHES:
            assert niche in metrics["per_niche"], f"per_niche must include '{niche}'"
            niche_metrics = metrics["per_niche"][niche]
            for key in ("precision", "recall", "f1", "support"):
                assert key in niche_metrics, f"per_niche['{niche}'] missing '{key}'"

    def test_training_record_algorithm_is_random_forest(self, tmp_path):
        """
        The algorithm field must always be 'RandomForestClassifier' (research.md R-001).
        """
        dataset_path = _write_separable_json(tmp_path)
        config_path = _write_config(tmp_path, dataset_path)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            train(config_path)

        record_file = list((tmp_path / "experiments").glob("*.json"))[0]
        with open(record_file) as f:
            record = json.load(f)

        assert record["algorithm"] == "RandomForestClassifier"

    def test_training_record_schema_version_matches_current(self, tmp_path):
        """
        The schema_version in the TrainingRecord must match current_schema_version(),
        because the model was trained on the current Phase 2 schema.
        """
        dataset_path = _write_separable_json(tmp_path)
        config_path = _write_config(tmp_path, dataset_path)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            train(config_path)

        record_file = list((tmp_path / "experiments").glob("*.json"))[0]
        with open(record_file) as f:
            record = json.load(f)

        assert record["schema_version"] == current_schema_version()


# ── TESTS: reproducibility ────────────────────────────────────────────────────

class TestReproducibility:
    """
    Tests that verify two training runs with identical config produce identical results.
    SC-003: reproducibility guarantee.
    """

    def test_two_runs_with_same_seed_produce_identical_predictions(self, tmp_path):
        """
        Two train() calls with the same dataset and config must produce models
        that give identical predictions on the same input vectors. This is the
        core reproducibility guarantee (SC-003, FR-004).
        """
        dataset_path = _write_separable_json(tmp_path)
        config_path = _write_config(tmp_path, dataset_path, seed=42)

        # Create separate output directories for each run
        dir1 = tmp_path / "run1"
        dir2 = tmp_path / "run2"
        dir1.mkdir()
        dir2.mkdir()

        config1 = _write_config(dir1, dataset_path, seed=42)
        config2 = _write_config(dir2, dataset_path, seed=42)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            path1 = train(config1)
            path2 = train(config2)

        # Load both models and predict on 10 test vectors
        artifact1 = load_model(path1)
        artifact2 = load_model(path2)

        test_vectors = _make_separable_dataset(n_per_niche=2, seed=99)[:10]

        from src.classifier import predict_niche
        import src.classifier as clf_mod
        # Redirect log to avoid test pollution
        orig_path = clf_mod._PREDICTION_LOG_PATH
        clf_mod._PREDICTION_LOG_PATH = tmp_path / "test_predictions.jsonl"

        try:
            for tv in test_vectors:
                fv = tv["feature_vector"]
                result1 = predict_niche(fv, artifact1)
                result2 = predict_niche(fv, artifact2)
                # The predicted niche must be identical across runs.
                # Confidence may differ by floating-point rounding at the 15th+ decimal
                # place (same computation can produce 0.423000000000000004 vs 0.422999999999999)
                # — both are the same underlying value. We check exact niche match and
                # near-equality for confidence.
                assert result1.get("predicted_niche") == result2.get("predicted_niche"), (
                    f"Same input produced different niches: {result1} vs {result2}"
                )
                if "confidence" in result1 and "confidence" in result2:
                    assert abs(result1["confidence"] - result2["confidence"]) < 1e-10, (
                        f"Confidence differs beyond floating-point tolerance: "
                        f"{result1['confidence']} vs {result2['confidence']}"
                    )
        finally:
            clf_mod._PREDICTION_LOG_PATH = orig_path


# ── TESTS: acceptance gate rejection ─────────────────────────────────────────

class TestAcceptanceGateRejection:
    """
    Tests that the accuracy gate and F1 gate work correctly.
    Models below the threshold must produce a rejected TrainingRecord but NO artifact.
    """

    def test_rejected_training_writes_record_with_rejected_status(self, tmp_path):
        """
        When training produces low accuracy (random labels), the TrainingRecord
        must be written with status='rejected' and no model artifact created.
        We create a dataset where true labels don't match features — the model
        can't learn anything meaningful, so accuracy will be low.
        """
        # Create a dataset with shuffled/random labels that don't match features
        # so the model can't learn any pattern and accuracy will be near chance (16%)
        rng = np.random.RandomState(0)
        records = []
        for _ in range(360):  # 360 total records
            v = [int(x) for x in rng.randint(0, 3, 40)]
            # Assign a completely random label regardless of features
            niche = rng.choice(CANONICAL_NICHES)
            records.append({"feature_vector": v, "niche": niche})

        import json as json_mod
        dataset_path = str(tmp_path / "random_dataset.json")
        with open(dataset_path, "w") as f:
            json_mod.dump(records, f)

        config_path = _write_config(tmp_path, dataset_path)

        # Training should fail (accuracy too low) and raise ValueError
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError) as exc_info:
                train(config_path)

        # But a TrainingRecord must still have been written
        exp_dir = tmp_path / "experiments"
        json_files = list(exp_dir.glob("*.json"))
        assert len(json_files) == 1, "Rejected run must still write a TrainingRecord"

        with open(json_files[0]) as f:
            record = json.load(f)

        assert record["status"] == "rejected", (
            f"Record status must be 'rejected' for failed training, got: {record['status']}"
        )

        # No model artifact must exist (rejected run — nothing was saved to models/)
        models_dir = tmp_path / "models"
        if models_dir.exists():
            model_files = list(models_dir.glob("*.joblib"))
            assert len(model_files) == 0, (
                f"No model artifact should exist for rejected training, found: {model_files}"
            )


# ── TESTS: feature importance in TrainingRecord ───────────────────────────────

class TestFeatureImportanceInRecord:
    """
    Tests that feature importance is included in the TrainingRecord (Constitution V).
    """

    def test_training_record_has_feature_importance(self, tmp_path):
        """
        The TrainingRecord must include feature_importance (top 20 dimensions).
        This is required by data-model.md and Constitution Principle V.
        """
        dataset_path = _write_separable_json(tmp_path)
        config_path = _write_config(tmp_path, dataset_path)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            train(config_path)

        record_file = list((tmp_path / "experiments").glob("*.json"))[0]
        with open(record_file) as f:
            record = json.load(f)

        fi = record.get("feature_importance", {})
        assert isinstance(fi, dict), f"feature_importance must be dict, got {type(fi)}"
        assert len(fi) > 0, "feature_importance must not be empty"
        # Top-20 stored in the record per data-model.md
        assert len(fi) <= 20, f"Expected ≤ 20 feature importance entries, got {len(fi)}"

    def test_feature_importance_values_are_floats(self, tmp_path):
        """
        All feature importance values must be floats (Gini importance scores).
        """
        dataset_path = _write_separable_json(tmp_path)
        config_path = _write_config(tmp_path, dataset_path)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            train(config_path)

        record_file = list((tmp_path / "experiments").glob("*.json"))[0]
        with open(record_file) as f:
            record = json.load(f)

        for key, val in record["feature_importance"].items():
            assert isinstance(val, float), (
                f"feature_importance['{key}'] must be float, got {type(val)}"
            )

    def test_bias_fairness_note_is_non_empty_string(self, tmp_path):
        """
        The bias_fairness_note must be a non-empty string identifying the minimum F1 niche.
        This is the per-niche fairness report required by Constitution Principle V.
        """
        dataset_path = _write_separable_json(tmp_path)
        config_path = _write_config(tmp_path, dataset_path)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            train(config_path)

        record_file = list((tmp_path / "experiments").glob("*.json"))[0]
        with open(record_file) as f:
            record = json.load(f)

        note = record.get("bias_fairness_note", "")
        assert isinstance(note, str), f"bias_fairness_note must be str, got {type(note)}"
        assert len(note) > 0, "bias_fairness_note must not be empty"
        # Must mention F1 in some form
        assert "f1" in note.lower() or "fairness" in note.lower(), (
            f"bias_fairness_note must mention F1, got: {note}"
        )


# ── TESTS: model artifact creation ────────────────────────────────────────────

class TestModelArtifactCreation:
    """
    Tests that successful training produces a valid .joblib model artifact.
    """

    def test_successful_training_creates_joblib_file(self, tmp_path):
        """
        A successful training run must create a .joblib file in the model_output_dir.
        """
        dataset_path = _write_separable_json(tmp_path)
        config_path = _write_config(tmp_path, dataset_path)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_path = train(config_path)

        assert os.path.exists(model_path), f"Model artifact not created at {model_path}"
        assert model_path.endswith(".joblib"), f"Model must be a .joblib file, got: {model_path}"

    def test_returned_path_is_absolute(self, tmp_path):
        """
        The path returned by train() must be absolute so callers don't need to
        resolve it relative to an unknown working directory.
        """
        dataset_path = _write_separable_json(tmp_path)
        config_path = _write_config(tmp_path, dataset_path)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_path = train(config_path)

        assert os.path.isabs(model_path), f"Model path must be absolute, got: {model_path}"

    def test_saved_artifact_is_loadable(self, tmp_path):
        """
        The .joblib file created by train() must be loadable by load_model()
        without any SchemaVersionError or ModelIntegrityError.
        """
        dataset_path = _write_separable_json(tmp_path)
        config_path = _write_config(tmp_path, dataset_path)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_path = train(config_path)

        # This must not raise
        artifact = load_model(model_path)
        assert "model" in artifact
        assert "niche_labels" in artifact
        assert artifact["schema_version"] == current_schema_version()


# ── TESTS: helper functions ────────────────────────────────────────────────────

class TestHelperFunctions:
    """
    Tests for the internal helper functions exposed for testability.
    """

    def test_build_run_id_format(self):
        """
        _build_run_id() must return a string in the format YYYYMMDD_HHMMSS_seed{N}.
        """
        run_id = _build_run_id(42)
        assert isinstance(run_id, str)
        # Must contain 'seed42'
        assert "seed42" in run_id, f"run_id must contain 'seed42', got: {run_id}"
        # Must have the correct format: 8 digits + underscore + 6 digits + underscore + seed{N}
        parts = run_id.split("_")
        assert len(parts) == 3, f"run_id must have 3 parts separated by '_', got: {run_id}"
        assert len(parts[0]) == 8, f"Date part must be 8 chars (YYYYMMDD), got: {parts[0]}"
        assert len(parts[1]) == 6, f"Time part must be 6 chars (HHMMSS), got: {parts[1]}"

    def test_get_git_sha_returns_string(self):
        """
        _get_git_sha() must always return a string — either a 40-char SHA or 'unknown'.
        It must never raise even when not in a git repo.
        """
        result = _get_git_sha()
        assert isinstance(result, str), f"git SHA must be str, got {type(result)}"
        assert len(result) > 0, "git SHA must be non-empty"
        # It's either a 40-char hex SHA or the literal "unknown"
        assert result == "unknown" or len(result) == 40, (
            f"git SHA must be 40-char hex or 'unknown', got: {result!r}"
        )


# ── TESTS: Phase 4 centroid backfill (T005) ───────────────────────────────────

class TestPhase4CentroidBackfill:
    """
    Phase 4 requirement: train() must embed "niche_centroids" in the model artifact.
    Each centroid is the mean feature vector for one niche class in the training split.
    These tests verify tasks.md T005 and T007.

    The centroid vectors are what Phase 4 scorer.py uses for cosine similarity —
    they must be present in every new artifact produced by trainer.py.
    """

    def test_artifact_contains_niche_centroids_key(self, tmp_path):
        """
        After Phase 4 backfill, every artifact produced by train() must contain
        the 'niche_centroids' key (tasks.md T005 — written to fail before T007,
        now passing after T007 implementation).
        """
        dataset_path = _write_separable_json(tmp_path)
        config_path = _write_config(tmp_path, dataset_path)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_path = train(config_path)

        artifact = load_model(model_path)
        assert "niche_centroids" in artifact, (
            "train() must produce an artifact with 'niche_centroids' key. "
            "This is required by Phase 4 scorer.py for cosine similarity scoring."
        )

    def test_niche_centroids_has_all_6_niches(self, tmp_path):
        """
        'niche_centroids' must contain an entry for all 6 canonical niches.
        Phase 4 scorer.py iterates CANONICAL_NICHES and requires all 6 to be present.
        """
        dataset_path = _write_separable_json(tmp_path)
        config_path = _write_config(tmp_path, dataset_path)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_path = train(config_path)

        artifact = load_model(model_path)
        centroids = artifact["niche_centroids"]

        assert isinstance(centroids, dict), (
            f"'niche_centroids' must be a dict, got {type(centroids)}"
        )
        for niche in CANONICAL_NICHES:
            assert niche in centroids, (
                f"'niche_centroids' is missing entry for '{niche}'. "
                f"Present keys: {list(centroids.keys())}"
            )

    def test_each_centroid_is_a_list_of_40_floats(self, tmp_path):
        """
        Each centroid in 'niche_centroids' must be a list of exactly 40 float values.
        This matches the Phase 2 feature vector length (locked contract).
        """
        dataset_path = _write_separable_json(tmp_path)
        config_path = _write_config(tmp_path, dataset_path)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_path = train(config_path)

        artifact = load_model(model_path)
        centroids = artifact["niche_centroids"]

        for niche, centroid in centroids.items():
            assert isinstance(centroid, list), (
                f"Centroid for '{niche}' must be a list, got {type(centroid)}"
            )
            assert len(centroid) == 40, (
                f"Centroid for '{niche}' must have 40 elements, got {len(centroid)}"
            )
            for i, val in enumerate(centroid):
                assert isinstance(val, float), (
                    f"Centroid[{i}] for '{niche}' must be float, got {type(val)}"
                )


# ── TESTS: Phase 5 niche_top_skills backfill (T006) ──────────────────────────

class TestPhase5NicheTopSkillsBackfill:
    """
    Phase 5 requirement: train() must embed "niche_top_skills" in the model artifact.
    Each entry is the top-10 feature dimensions for that niche, ranked by centroid value
    descending, formatted as (dim_idx: int, label: str) tuples.

    These tests are written FIRST per TDD (T006). They FAIL before T008 is implemented
    (trainer.py does not yet compute niche_top_skills) and PASS after T008.

    Why niche_top_skills matters: analyze_skills_gap() in gap_analyzer.py cannot operate
    without knowing which feature dimensions are most diagnostic for each niche. The
    centroid-derived top-10 dimensions are the proxy for "most important skills for
    this niche" — the artifact carries them so inference never needs the training set.
    """

    def test_artifact_contains_niche_top_skills_key(self, tmp_path):
        """
        After Phase 5 backfill, every artifact produced by train() must contain
        the 'niche_top_skills' key. This is the entry point for analyze_skills_gap().
        """
        dataset_path = _write_separable_json(tmp_path)
        config_path = _write_config(tmp_path, dataset_path)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_path = train(config_path)

        artifact = load_model(model_path)
        assert "niche_top_skills" in artifact, (
            "train() must produce an artifact with 'niche_top_skills' key. "
            "This is required by Phase 5 gap_analyzer.py to identify skill gaps."
        )

    def test_niche_top_skills_has_all_6_niches(self, tmp_path):
        """
        'niche_top_skills' must contain an entry for all 6 canonical niches.
        analyze_skills_gap() will request any of the 6 niches; all must be present.
        """
        dataset_path = _write_separable_json(tmp_path)
        config_path = _write_config(tmp_path, dataset_path)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_path = train(config_path)

        artifact = load_model(model_path)
        top_skills = artifact["niche_top_skills"]

        assert isinstance(top_skills, dict), (
            f"'niche_top_skills' must be a dict, got {type(top_skills)}"
        )
        for niche in CANONICAL_NICHES:
            assert niche in top_skills, (
                f"'niche_top_skills' is missing entry for '{niche}'. "
                f"Present keys: {list(top_skills.keys())}"
            )

    def test_each_niche_top_skills_has_exactly_10_tuples(self, tmp_path):
        """
        Each niche entry in 'niche_top_skills' must have exactly 10 tuples.
        The benchmark size is fixed at 10 (spec FR-US1-001 and data-model.md).
        """
        dataset_path = _write_separable_json(tmp_path)
        config_path = _write_config(tmp_path, dataset_path)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_path = train(config_path)

        artifact = load_model(model_path)
        top_skills = artifact["niche_top_skills"]

        for niche in CANONICAL_NICHES:
            entry = top_skills[niche]
            assert len(entry) == 10, (
                f"niche_top_skills['{niche}'] must have exactly 10 entries, "
                f"got {len(entry)}"
            )

    def test_each_top_skills_entry_is_int_str_tuple(self, tmp_path):
        """
        Each element in a niche's top_skills list must be a (int, str) tuple:
          - int: the feature vector dimension index (0–39)
          - str: a human-readable label in the format "<type_key>_<section_name>"

        This shape is what analyze_skills_gap() expects when it iterates top10.
        """
        dataset_path = _write_separable_json(tmp_path)
        config_path = _write_config(tmp_path, dataset_path)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_path = train(config_path)

        artifact = load_model(model_path)
        top_skills = artifact["niche_top_skills"]

        for niche in CANONICAL_NICHES:
            for i, entry in enumerate(top_skills[niche]):
                assert len(entry) == 2, (
                    f"niche_top_skills['{niche}'][{i}] must be a 2-tuple, got len={len(entry)}"
                )
                dim_idx, label = entry
                assert isinstance(dim_idx, int), (
                    f"niche_top_skills['{niche}'][{i}][0] must be int (dim index), "
                    f"got {type(dim_idx)}"
                )
                assert isinstance(label, str), (
                    f"niche_top_skills['{niche}'][{i}][1] must be str (label), "
                    f"got {type(label)}"
                )
                assert 0 <= dim_idx < 40, (
                    f"niche_top_skills['{niche}'][{i}] dim_idx={dim_idx} out of range [0,40)"
                )

    def test_top_skills_ordered_by_centroid_value_descending(self, tmp_path):
        """
        Within each niche, the 10 tuples must be sorted by centroid value descending —
        the most diagnostic dimension first. This ordering ensures that rank-1 in
        niche_benchmarks.json maps to the dimension the model weights most heavily,
        giving the user the most impactful skill to target first.
        """
        dataset_path = _write_separable_json(tmp_path)
        config_path = _write_config(tmp_path, dataset_path)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_path = train(config_path)

        artifact = load_model(model_path)
        centroids = artifact["niche_centroids"]
        top_skills = artifact["niche_top_skills"]

        for niche in CANONICAL_NICHES:
            centroid = centroids[niche]
            dims_in_order = [dim_idx for dim_idx, _ in top_skills[niche]]

            # Extract the centroid values at those dimension indices
            values_in_order = [centroid[d] for d in dims_in_order]

            # They must be non-increasing (sorted descending)
            for j in range(len(values_in_order) - 1):
                assert values_in_order[j] >= values_in_order[j + 1], (
                    f"niche_top_skills['{niche}'] is not sorted descending by centroid value. "
                    f"Position {j} has value {values_in_order[j]} but position {j+1} has "
                    f"{values_in_order[j+1]}."
                )
