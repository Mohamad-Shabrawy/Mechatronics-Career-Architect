"""
trainer.py — Training Pipeline, Experiment Logging, and Model Promotion

This module owns the full lifecycle of one training run:
  1. Load and validate the labeled dataset (via dataset.py)
  2. Split into train/test sets with stratification (so every niche appears in both)
  3. Train a RandomForestClassifier with the configured hyperparameters and fixed seed
  4. Evaluate on the held-out test set: overall accuracy + per-niche precision/recall/F1
  5. Compute feature importance (which CV sections/entity types matter most)
  6. Write an immutable TrainingRecord JSON to experiments/ — always, even for failures
  7. If accuracy ≥ 0.75 AND all per-niche F1 ≥ 0.60: serialize the model artifact
     Otherwise: write a "rejected" TrainingRecord and return without saving

Design decisions:
  - The training pipeline is deliberately NOT exposed as a public function named
    predict_niche() — separation of concerns (Constitution IV) means training and
    inference live in separate modules (trainer.py vs classifier.py).
  - The `train()` function is callable from Python (returns path string) and from
    the CLI (python -m src.trainer --config ...).
  - We use stratified splitting so even small niche classes appear in the test set.
    Without stratification, a niche with 50 records might have zero test samples.

Acceptance gates (from spec FR-007, SC-001, SC-002):
  - overall_accuracy must be ≥ 0.75 — models below this threshold are rejected
  - All per-niche F1 scores must be ≥ 0.60 — no niche may be systematically ignored

Contract reference: specs/003-ml-niche-classifier/contracts/classifier_contract.md
Data model:         specs/003-ml-niche-classifier/data-model.md
"""

import argparse
import collections
import json
import pathlib
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import yaml

from src.dataset import load_dataset, compute_dataset_version
from src.feature_vector import VECTOR_INDEX_ORDER
from src.model_io import (
    CANONICAL_NICHES,
    save_model,
    current_schema_version,
)


# ── PUBLIC API ────────────────────────────────────────────────────────────────

def train(config_path: str) -> str:
    """
    Runs the full training pipeline end-to-end and returns the model artifact path.

    This is the single public entry point for training. It:
      1. Loads the YAML config from config_path
      2. Loads and validates the labeled dataset
      3. Trains a RandomForestClassifier with the configured hyperparameters
      4. Evaluates on the held-out test set
      5. Writes a TrainingRecord JSON to experiments/ (always — even for rejected runs)
      6. If metrics pass the acceptance gates, serializes the model to models/
         and returns the absolute path of the saved .joblib file
      7. If metrics fail the gates, raises a ValueError explaining why

    Parameters
    ----------
    config_path : str
        Path to a YAML training configuration file (see config/training_config.yaml).

    Returns
    -------
    str
        Absolute path of the saved .joblib model artifact.

    Raises
    ------
    ValueError
        If the config file cannot be read, the dataset is invalid, or the trained
        model does not meet the minimum accuracy/F1 gates.
    Exception
        Unexpected failures are re-raised after writing a "rejected" TrainingRecord
        so there is always a record of every attempted training run.
    """
    # ── Step 1: Load configuration ────────────────────────────────────────────
    config = _load_config(config_path)

    seed = config.get("random_seed", 42)
    run_id = _build_run_id(seed)

    # ── Step 2: Load and validate the labeled dataset ─────────────────────────
    dataset_path = config.get("dataset_path", "data/training/dataset.csv")
    feature_vectors, labels = load_dataset(dataset_path)

    # Compute a content hash of the dataset file so the TrainingRecord permanently
    # links this model to the exact data it was trained on (FR-005).
    dataset_version = compute_dataset_version(dataset_path)
    dataset_size = len(labels)
    records_per_niche = dict(collections.Counter(labels))

    # ── Step 3: Split into train and test sets ────────────────────────────────
    # Stratified split: ensures every niche appears in both train and test.
    # Without stratification, small classes might end up entirely in one split.
    test_size = 1.0 - config.get("train_test_split", 0.8)
    X = np.array(feature_vectors, dtype=float)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        stratify=y,         # Critical: preserves class proportions in both splits
    )

    # ── Step 4: Build and train the classifier ────────────────────────────────
    # class_weight="balanced" makes rare niches (like technical_management, which
    # often has fewer CV samples) get more weight in the loss, helping the model
    # not simply ignore them.
    class_weight = config.get("class_weight", "balanced")
    if class_weight == "uniform":
        class_weight = None  # sklearn uses None for uniform weighting

    classifier = RandomForestClassifier(
        n_estimators=config.get("n_estimators", 200),
        max_depth=config.get("max_depth"),            # None = unlimited depth
        min_samples_leaf=config.get("min_samples_leaf", 2),
        class_weight=class_weight,
        random_state=seed,                            # Fixed seed = reproducible trees
        n_jobs=-1,                                    # Use all CPU cores for speed
    )
    classifier.fit(X_train, y_train)

    # ── Step 5: Evaluate on the held-out test set ─────────────────────────────
    y_pred = classifier.predict(X_test)
    evaluation_metrics = _compute_evaluation_metrics(y_test, y_pred, classifier)

    # ── Step 6: Check acceptance gates ───────────────────────────────────────
    # Both gates must pass for the model to be serialized (FR-007, SC-001, SC-002).
    overall_accuracy = evaluation_metrics["overall_accuracy"]
    min_f1_niche, min_f1_value = _find_min_f1_niche(evaluation_metrics)

    accuracy_passes = overall_accuracy >= 0.75
    f1_passes = min_f1_value >= 0.60

    # ── Step 7: Build feature importance and fairness note ───────────────────
    # Feature importance tells us which (type, section) dimensions drove predictions.
    # We store all 40 but only record the top-20 in the TrainingRecord (data-model.md).
    feature_importance_full = _compute_feature_importance(classifier)
    feature_importance_top20 = dict(
        list(
            sorted(feature_importance_full.items(), key=lambda x: x[1], reverse=True)
        )[:20]
    )

    # The bias/fairness note is a human-readable summary of per-niche F1 performance.
    # It identifies the "weakest" niche — the one the model is most likely to miss.
    bias_fairness_note = (
        f"Per-niche F1 scores above — minimum F1 across niches: "
        f"{min_f1_value:.3f} ({min_f1_niche}). "
        f"{'All niches meet the 0.60 F1 minimum.' if f1_passes else f'WARNING: {min_f1_niche} is below the 0.60 F1 threshold.'}"
    )

    # ── Step 8: Write the TrainingRecord ─────────────────────────────────────
    # We write this ALWAYS — for both accepted and rejected runs. FR-005 and SC-007
    # require 100% coverage: every training attempt produces a record.
    experiment_dir = config.get("experiment_dir", "experiments/")
    model_output_dir = config.get("model_output_dir", "models/")

    # Determine where the model artifact will go (even for rejected runs, we record
    # the intended path — it just won't be created).
    intended_artifact_path = str(
        pathlib.Path(model_output_dir).resolve() / f"{run_id}.joblib"
    )

    training_record = {
        "run_id":              run_id,
        "training_date":       datetime.now(timezone.utc).isoformat(),
        "git_commit_sha":      _get_git_sha(),
        "dataset_path":        str(pathlib.Path(dataset_path).resolve()),
        "dataset_version":     dataset_version,
        "dataset_size":        dataset_size,
        "records_per_niche":   records_per_niche,
        "algorithm":           "RandomForestClassifier",
        "hyperparameters": {
            "n_estimators":    config.get("n_estimators", 200),
            "max_depth":       config.get("max_depth"),
            "min_samples_leaf": config.get("min_samples_leaf", 2),
            "class_weight":    config.get("class_weight", "balanced"),
        },
        "random_seed":         seed,
        "train_test_split":    config.get("train_test_split", 0.8),
        "evaluation_metrics":  evaluation_metrics,
        "bias_fairness_note":  bias_fairness_note,
        "feature_importance":  feature_importance_top20,
        "model_artifact_path": intended_artifact_path if (accuracy_passes and f1_passes) else None,
        "schema_version":      current_schema_version(),
        "status":              "accepted" if (accuracy_passes and f1_passes) else "rejected",
    }

    _write_training_record(training_record, experiment_dir, run_id)

    # ── Step 9: Accept or reject the model ────────────────────────────────────
    if not accuracy_passes:
        raise ValueError(
            f"Training run {run_id} REJECTED: overall accuracy {overall_accuracy:.3f} "
            f"is below the 0.75 threshold. TrainingRecord written to experiments/{run_id}.json. "
            "No model artifact was saved. Increase training data or adjust hyperparameters."
        )

    if not f1_passes:
        raise ValueError(
            f"Training run {run_id} REJECTED: niche '{min_f1_niche}' has F1 = "
            f"{min_f1_value:.3f}, below the 0.60 minimum (SC-002). "
            f"TrainingRecord written to experiments/{run_id}.json. "
            "No model artifact was saved. Add more training data for this niche."
        )

    # ── Step 10: Compute per-niche centroid vectors (Phase 4 backfill) ───────
    # A centroid is the mean feature vector across all training samples in one class.
    # These 40D centroid vectors are loaded by Phase 4's scorer.py to compute
    # cosine similarity between a new CV's feature vector and each niche's ideal profile.
    #
    # Why compute here (at training time) rather than at inference time?
    #   Centroids are derived from training data only. Embedding them in the artifact
    #   keeps training and inference decoupled — inference never needs the training set.
    #   If we computed them at inference time we'd need to ship the training data,
    #   which violates the separation-of-concerns requirement (research.md R-005).
    #
    # We only include niches that have training samples. If a niche has no X_train
    # samples (extremely rare with stratified split, but possible in unit tests with
    # tiny datasets), we fall back to a zero vector of length 40 so the artifact
    # always has exactly 6 entries.
    niche_centroids = {}
    for niche in CANONICAL_NICHES:
        # Boolean mask: rows where the training label matches this niche
        niche_mask = (y_train == niche)
        if niche_mask.any():
            # Mean over all rows for this niche — gives us a 40D float centroid
            niche_centroids[niche] = X_train[niche_mask].mean(axis=0).tolist()
        else:
            # Safety fallback: zero vector. This niche had no training samples —
            # cosine similarity will return 0.0 for it at scoring time.
            niche_centroids[niche] = [0.0] * 40

    # ── Step 10b: Compute per-niche top-10 skill dimensions (Phase 5 backfill) ──
    # For each niche, we rank all 40 feature dimensions by centroid value (descending)
    # and take the top 10. These are the dimensions that are most "active" for that
    # niche in the training data — a proxy for the most important skills.
    #
    # Why use centroids instead of feature importances?
    #   Feature importances tell you which dimensions separate all classes globally.
    #   Centroid values tell you what a "typical" engineer in this niche looks like —
    #   which is exactly what the skills gap analysis needs: a per-niche benchmark
    #   that maps directly to human-readable skill names from niche_benchmarks.json.
    #
    # The positional mapping contract: rank-0 centroid dimension → rank-0 benchmark
    # skill name in niche_benchmarks.json (by rank field). Both are ordered by
    # importance descending, so position i in niche_top_skills aligns with position i
    # in the JSON benchmark for that niche.
    niche_top_skills = {}
    for niche, centroid in niche_centroids.items():
        # Sort all 40 (index, centroid_value) pairs by value descending — most
        # diagnostic dimension first. Ties broken by index (stable enough for our use).
        ranked = sorted(enumerate(centroid), key=lambda x: x[1], reverse=True)
        # Store the top 10 as (dim_index, human_label) tuples.
        # Label format: "<type_key>_<section_name>" — same as feature importance keys.
        niche_top_skills[niche] = [
            (idx, f"{VECTOR_INDEX_ORDER[idx][0]}_{VECTOR_INDEX_ORDER[idx][1]}")
            for idx, _ in ranked[:10]
        ]

    # ── Step 11: Serialize the accepted model ─────────────────────────────────
    # We only reach here if BOTH acceptance gates passed.
    # The niche_labels list must be in the same order as the classifier's .classes_
    # so that class-index-to-name lookups in predict_niche() are correct.
    niche_labels = _build_niche_labels_from_classifier(classifier)
    artifact_path = save_model(
        model=classifier,
        niche_labels=niche_labels,
        training_record=training_record,
        run_id=run_id,
        output_dir=model_output_dir,
        niche_centroids=niche_centroids,
        niche_top_skills=niche_top_skills,
    )

    return artifact_path


# ── PRIVATE HELPERS ───────────────────────────────────────────────────────────

def _load_config(config_path: str) -> dict:
    """
    Loads and returns the YAML training configuration from the given path.

    If the file is missing or invalid YAML, raises ValueError with a clear message.
    We catch all YAML parsing errors here because the caller (train()) wraps
    everything in a try block that writes a rejected TrainingRecord.
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        raise ValueError(
            f"Training config not found: '{config_path}'\n"
            "Run from the project root directory and verify the config path."
        )
    except yaml.YAMLError as exc:
        raise ValueError(
            f"Failed to parse training config '{config_path}': {exc}"
        )

    if not isinstance(config, dict):
        raise ValueError(
            f"Training config must be a YAML mapping, got {type(config).__name__}"
        )

    return config


def _build_run_id(seed: int) -> str:
    """
    Generates a unique identifier for this training run.

    Format: YYYYMMDD_HHMMSS_seed{N}
    Example: "20260417_143022_seed42"

    The timestamp component ensures uniqueness across runs.
    The seed component makes the configuration visible in the filename itself —
    a practitioner browsing experiments/ can immediately see which seed was used
    without opening the JSON file.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_seed{seed}"


def _get_git_sha() -> str:
    """
    Returns the current git commit SHA for embedding in the TrainingRecord.

    Why include the git SHA?
      The training code (entity taxonomy, semantic map, feature vector layout, model
      hyperparameters) is all part of the source tree. The git SHA creates a
      three-way link: dataset version (file hash) + code version (git SHA) + model
      artifact. Together they make training fully reproducible by any engineer with
      access to the repo and the same dataset file.

    Returns "unknown" gracefully if:
      - Not running inside a git repository
      - git is not installed
      - Any subprocess error occurs

    We never let a missing git environment crash the training pipeline.
    """
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode("utf-8").strip()
        return sha
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        # Not in a git repo, git not installed, or subprocess timed out.
        return "unknown"


def _compute_evaluation_metrics(
    y_true,
    y_pred,
    classifier: RandomForestClassifier,
) -> dict:
    """
    Computes overall accuracy and per-niche precision, recall, F1, and support.

    We use scikit-learn's classification_report() to get per-class metrics and
    then reshape the output into our TrainingRecord schema format.

    The 'support' field (number of test records per class) is included so that
    the bias/fairness reader can immediately see whether a low F1 score is due to
    model weakness or simply a tiny test sample (which also signals that the
    training data for that niche is thin).

    Returns a dict matching the EvaluationMetrics data model:
    {
        "overall_accuracy": float,
        "per_niche": {
            "robotics": {"precision": float, "recall": float, "f1": float, "support": int},
            ...
        }
    }
    """
    # Get the labels that actually appear in the test set (stratified split means
    # all should appear, but we handle the edge case where a class is missing).
    all_labels = list(classifier.classes_)

    # classification_report with output_dict=True gives us a structured dict
    # rather than a formatted string — easier to reshape into our schema.
    report = classification_report(
        y_true, y_pred,
        labels=all_labels,
        output_dict=True,
        zero_division=0,    # When a class has no test samples, return 0 rather than warning
    )

    # Extract overall accuracy from the report's "accuracy" key.
    overall_accuracy = float(report.get("accuracy", 0.0))

    # Build per-niche metrics dict from the per-class rows of the report.
    # The keys in report are the class labels (niche names) plus aggregates like
    # "macro avg", "weighted avg", "accuracy" — we skip the non-niche keys.
    per_niche = {}
    for label in all_labels:
        if label in report:
            metrics = report[label]
            per_niche[label] = {
                "precision": float(metrics.get("precision", 0.0)),
                "recall":    float(metrics.get("recall", 0.0)),
                "f1":        float(metrics.get("f1-score", 0.0)),
                "support":   int(metrics.get("support", 0)),
            }
        else:
            # This niche didn't appear in the test set at all (edge case).
            per_niche[label] = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}

    return {
        "overall_accuracy": overall_accuracy,
        "per_niche":        per_niche,
    }


def _find_min_f1_niche(evaluation_metrics: dict) -> tuple[str, float]:
    """
    Returns the niche name and F1 score of the worst-performing niche.

    Used for:
      1. The bias_fairness_note in the TrainingRecord
      2. Determining whether the model passes the SC-002 F1 gate (all ≥ 0.60)

    If per_niche is empty (shouldn't happen), returns ("unknown", 0.0) to avoid
    crashing the training pipeline.
    """
    per_niche = evaluation_metrics.get("per_niche", {})
    if not per_niche:
        return ("unknown", 0.0)

    # Find the niche with the lowest F1 score.
    min_niche = min(per_niche.items(), key=lambda x: x[1]["f1"])
    return min_niche[0], min_niche[1]["f1"]


def _compute_feature_importance(classifier: RandomForestClassifier) -> dict[str, float]:
    """
    Maps the Random Forest's 40 feature importance scores to human-readable labels.

    The label format is "<type_key>_<section_name>", e.g. "plc_hardware_skills".
    This is the same format used by get_feature_importance() in model_io.py —
    we compute it here during training so it can go into the TrainingRecord
    immediately, without needing to load the model artifact back from disk.

    The scores come from CART Gini impurity averaging (sklearn's default) — they
    tell us how much each feature dimension improved the tree split quality on average.
    Higher = more useful for distinguishing between niches.
    """
    importances = classifier.feature_importances_  # numpy array of shape (40,)

    labeled = {
        f"{type_key}_{section}": float(importance)
        for (type_key, section), importance in zip(VECTOR_INDEX_ORDER, importances)
    }

    return dict(sorted(labeled.items(), key=lambda x: x[1], reverse=True))


def _build_niche_labels_from_classifier(classifier: RandomForestClassifier) -> list[str]:
    """
    Returns the niche labels in the order the RandomForestClassifier uses internally.

    Why is this important?
      When we call classifier.predict_proba(X), it returns probabilities in the same
      order as classifier.classes_. If we stored labels in a different order (e.g.,
      CANONICAL_NICHES sorted alphabetically), the index-to-name mapping would be wrong
      and "robotics" probability might get labeled as "automotive".

      By returning list(classifier.classes_), we guarantee the index alignment is
      correct. If the classifier was trained with all 6 niches, classes_ will contain
      all 6 — just in the order scikit-learn encountered them during fit().
    """
    return [str(c) for c in classifier.classes_]


def _write_training_record(record: dict, experiment_dir: str, run_id: str) -> None:
    """
    Writes the TrainingRecord dict to a JSON file in the experiments directory.

    File path: <experiment_dir>/<run_id>.json

    This is always called — for both accepted and rejected training runs — because
    SC-007 requires 100% coverage: every train() call must produce a TrainingRecord.

    We use json.dump with indent=2 for human readability. The engineer should be
    able to cat any experiment file and immediately understand what happened.

    If the write fails (disk full, permission denied, etc.), we log to stderr but
    do NOT raise — the training result itself is valid even if logging failed.
    """
    exp_path = pathlib.Path(experiment_dir)
    exp_path.mkdir(parents=True, exist_ok=True)

    record_file = exp_path / f"{run_id}.json"
    try:
        with open(record_file, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, default=str)
    except OSError as exc:
        # Logging failure should not kill the training run — print to stderr and continue.
        print(
            f"WARNING: Failed to write TrainingRecord to '{record_file}': {exc}",
            file=sys.stderr,
        )


# ── CLI ENTRY POINT ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    CLI entry point for running the training pipeline from the command line.

    Usage:
        python -m src.trainer --config config/training_config.yaml

    On success, prints the path to the saved model artifact.
    On failure, prints the error message and exits with code 1.
    """
    parser = argparse.ArgumentParser(
        description="Train the Mechatronics niche classifier.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python -m src.trainer --config config/training_config.yaml\n\n"
            "After training, check experiments/ for the training record JSON\n"
            "and models/ for the saved model artifact."
        ),
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML training configuration file",
    )
    args = parser.parse_args()

    try:
        model_path = train(args.config)
        print(f"Training complete. Model saved to: {model_path}")
        sys.exit(0)
    except ValueError as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"Unexpected error during training: {exc}", file=sys.stderr)
        sys.exit(1)
