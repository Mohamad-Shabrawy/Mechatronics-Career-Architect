"""
dataset.py — Labeled Dataset Loading and Validation for Phase 3 Training

This module handles one job: loading a labeled feature vector dataset from disk,
validating every record against the Phase 2 contract, warning about class imbalance,
and returning clean (features, labels) arrays ready for scikit-learn.

What "labeled dataset" means here:
  Each record pairs a 40-dimensional integer feature vector (produced by the Phase 2
  enrichment pipeline on a real or synthetic CV) with one of the six canonical niche
  labels. The dataset was produced OFFLINE by running parse_cv() → enrich_cv() on a
  set of labeled CVs and saving the resulting feature_vector + niche pairs to a file.

  We do NOT accept raw PDFs or raw text as training input — that would couple the
  training pipeline to the PDF parser, making it fragile and slow. Pre-extracted
  feature vectors are the contract (FR-002, clarification Q1 in spec.md).

Supported file formats:
  - CSV: columns feat_0, feat_1, ..., feat_39, niche
  - JSON: list of {"feature_vector": [...], "niche": "..."} objects

Validation rules (applied before any training begins — FR-003):
  1. Feature vector length == 40
  2. No NaN/null values in the feature vector
  3. Niche label is one of the six canonical names (case-insensitive)
  4. Records failing any rule → collected and reported in a ValueError (all at once)

Imbalance warning (FR-013):
  After validation, any niche with fewer than 50 records gets a UserWarning.
  The pipeline trains anyway — we warn but don't block.

Contract reference: specs/003-ml-niche-classifier/contracts/classifier_contract.md
Data model:         specs/003-ml-niche-classifier/data-model.md
"""

import collections
import hashlib
import json
import math
import warnings

import pandas as pd

from src.model_io import CANONICAL_NICHES

# ── CONSTANTS ─────────────────────────────────────────────────────────────────

# The exact number of feature dimensions the Phase 2 pipeline always produces.
# This is the locked contract — any record with a different length is rejected.
FEATURE_VECTOR_LENGTH = 40

# Minimum records per niche class that we consider "enough" for reliable training.
# Below this threshold we warn (but still train). Value comes from spec FR-013.
MIN_RECORDS_PER_NICHE = 50

# Build a lowercase-to-canonical mapping once so we can normalize niche labels
# on load. For example, "Robotics" → "robotics", "ROBOTICS" → "robotics".
_NICHE_LOWER_MAP: dict[str, str] = {n.lower(): n for n in CANONICAL_NICHES}


# ── PUBLIC API ────────────────────────────────────────────────────────────────

def load_dataset(file_path: str) -> tuple[list[list[int]], list[str]]:
    """
    Loads and validates a labeled feature vector dataset from a CSV or JSON file.

    The file is identified by its extension:
      - .csv → pandas read_csv(), columns feat_0..feat_39 + niche
      - .json → json.load(), list of {"feature_vector": [...], "niche": "..."} objects

    Every record is validated against three rules:
      1. Feature vector must have exactly 40 elements.
      2. No NaN or None values allowed in the feature vector.
      3. The niche label must be one of the six canonical names (case-insensitive).

    Records that fail validation are NOT silently dropped — their row indices and
    reasons are collected and the function raises a ValueError listing ALL bad records
    before any training begins. This "fail loudly, all at once" approach means you
    don't discover record 237 is bad after record 42 was already fixed.

    After validation, any niche class with fewer than MIN_RECORDS_PER_NICHE records
    triggers a UserWarning. Training continues regardless.

    Parameters
    ----------
    file_path : str
        Path to the labeled dataset file (.csv or .json).

    Returns
    -------
    tuple[list[list[int]], list[str]]
        (feature_vectors, labels) where:
          - feature_vectors is a list of N lists, each with exactly 40 integers
          - labels is a list of N canonical niche name strings (lowercase)

    Raises
    ------
    ValueError
        If the file format is unsupported, the file cannot be read, or any
        record fails validation. The error message lists ALL invalid rows.
    """
    # ── Step 1: Read the file ─────────────────────────────────────────────────
    # Detect format by extension — we don't try to parse MIME types because
    # training datasets are always supplied by the engineer, not uploaded.
    file_path_lower = str(file_path).lower()

    if file_path_lower.endswith(".csv"):
        records = _load_csv(file_path)
    elif file_path_lower.endswith(".json"):
        records = _load_json(file_path)
    else:
        raise ValueError(
            f"Unsupported dataset file format: '{file_path}'\n"
            "Supported formats: .csv (feat_0..feat_39,niche columns) "
            "or .json (list of {\"feature_vector\":[...],\"niche\":\"...\"} objects)"
        )

    # ── Step 2: Validate every record ────────────────────────────────────────
    # We collect all errors and report them all at once rather than stopping at
    # the first bad record. This saves the engineer from having to fix one record
    # at a time when a batch processing error has corrupted many records.
    errors: list[str] = []
    feature_vectors: list[list[int]] = []
    labels: list[str] = []

    for row_index, (raw_vector, raw_niche) in enumerate(records):
        row_errors = _validate_record(row_index, raw_vector, raw_niche)
        if row_errors:
            errors.extend(row_errors)
        else:
            # Normalize: convert vector elements to int, niche to canonical lowercase.
            clean_vector = [int(v) for v in raw_vector]
            canonical_niche = _normalize_niche(raw_niche)
            feature_vectors.append(clean_vector)
            labels.append(canonical_niche)

    # If ANY records failed validation, raise with the full list of problems.
    # No partial dataset is returned — training must start from a clean slate.
    if errors:
        raise ValueError(
            f"Dataset validation failed — {len(errors)} error(s) found:\n"
            + "\n".join(f"  {e}" for e in errors)
        )

    # ── Step 3: Check for class imbalance ────────────────────────────────────
    # Count how many records exist per niche and warn for any underrepresented class.
    # We do this AFTER validation so the counts are accurate (invalid records excluded).
    _warn_if_imbalanced(labels)

    return feature_vectors, labels


def compute_dataset_version(file_path: str) -> str:
    """
    Returns a SHA-256 hex digest of the dataset file's raw bytes.

    Why this approach?
      The dataset version is embedded in the TrainingRecord (FR-005) to create a
      permanent link between a model artifact and the exact data it was trained on.
      Using the file's raw byte content means:
        - Any change to any record — even a single character — produces a different hash
        - The hash is reproducible: two engineers with the same file get the same hash
        - We don't need to parse the file to version it (fast for large files)

    Parameters
    ----------
    file_path : str
        Path to the dataset file.

    Returns
    -------
    str
        Full 64-character SHA-256 hex digest of the file's binary content.

    Raises
    ------
    ValueError
        If the file cannot be read.
    """
    try:
        with open(file_path, "rb") as f:
            content = f.read()
        return hashlib.sha256(content).hexdigest()
    except OSError as exc:
        raise ValueError(
            f"Cannot compute dataset version — failed to read '{file_path}': {exc}"
        )


# ── PRIVATE HELPERS ───────────────────────────────────────────────────────────

def _load_csv(file_path: str) -> list[tuple]:
    """
    Reads a CSV dataset file and returns a list of (raw_vector, raw_niche) tuples.

    Expected CSV format:
      feat_0,feat_1,...,feat_39,niche
      1,0,3,0,...,2,robotics

    The 'niche' column must be present. feat_0 through feat_39 must be present
    and in order (pandas will raise a KeyError if they're missing, which we
    convert to a descriptive ValueError).

    We intentionally do NOT enforce numeric types here — validation in
    _validate_record() handles that, giving better error messages than pandas would.
    """
    try:
        df = pd.read_csv(file_path)
    except Exception as exc:
        raise ValueError(f"Failed to read CSV file '{file_path}': {exc}")

    # The 'niche' column is required.
    if "niche" not in df.columns:
        raise ValueError(
            f"CSV file '{file_path}' is missing the required 'niche' column.\n"
            "Expected columns: feat_0, feat_1, ..., feat_39, niche"
        )

    # Check if we have a 'feature_vector' column (JSON-style CSV) or feat_0..feat_39 columns
    if "feature_vector" in df.columns:
        # Handle the case where feature_vector is stored as a JSON array string in CSV
        records = []
        for _, row in df.iterrows():
            try:
                vector = json.loads(row["feature_vector"])
            except (json.JSONDecodeError, TypeError):
                vector = row["feature_vector"]
            records.append((vector, row["niche"]))
        return records

    # Build the expected feature column names: feat_0, feat_1, ..., feat_39
    feature_cols = [f"feat_{i}" for i in range(FEATURE_VECTOR_LENGTH)]

    # Check that at least some feature columns exist. If none exist, we give
    # a helpful error message explaining the expected format.
    available_feat_cols = [c for c in feature_cols if c in df.columns]
    if not available_feat_cols:
        raise ValueError(
            f"CSV file '{file_path}' has neither 'feat_0'...'feat_39' columns "
            "nor a 'feature_vector' column.\n"
            "Expected format: feat_0,feat_1,...,feat_39,niche"
        )

    # For each row, extract the feature vector as a Python list and the niche label.
    # We use .values (not dict access) for speed on large datasets.
    records = []
    for _, row in df.iterrows():
        # Build vector from the feature columns, using NaN for missing columns.
        vector = [
            row[col] if col in df.columns else float("nan")
            for col in feature_cols
        ]
        records.append((vector, row["niche"]))

    return records


def _load_json(file_path: str) -> list[tuple]:
    """
    Reads a JSON dataset file and returns a list of (raw_vector, raw_niche) tuples.

    Expected JSON format:
      [
        {"feature_vector": [1, 0, 3, ...], "niche": "robotics"},
        {"feature_vector": [0, 2, 0, ...], "niche": "automotive"},
        ...
      ]

    The top-level value must be a JSON array (list), not an object.
    Each element must be an object with "feature_vector" and "niche" keys.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        raise ValueError(f"Failed to read JSON file '{file_path}': {exc}")

    if not isinstance(data, list):
        raise ValueError(
            f"JSON dataset must be a top-level array, got {type(data).__name__}.\n"
            "Expected format: [{\"feature_vector\": [...], \"niche\": \"...\"}, ...]"
        )

    records = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(
                f"JSON record at index {i} must be an object with "
                "'feature_vector' and 'niche' keys, got {type(item).__name__}"
            )
        # Extract the two required fields — missing fields become None (caught by validator).
        vector = item.get("feature_vector")
        niche = item.get("niche")
        records.append((vector, niche))

    return records


def _validate_record(
    row_index: int,
    raw_vector,
    raw_niche,
) -> list[str]:
    """
    Validates a single (vector, niche) pair and returns a list of error strings.

    An empty list means the record is valid. A non-empty list means the record has
    one or more problems. This approach (collect, not raise) allows us to report
    all bad records in one ValueError rather than stopping at the first failure.

    Validation rules:
      1. raw_vector must be a list (or list-like).
      2. raw_vector must have exactly 40 elements.
      3. No element in raw_vector may be NaN or None.
      4. raw_niche (after lowercasing) must be in the canonical niche set.
    """
    errors = []

    # ── Rule 1 & 2: Vector must be list-like with exactly 40 elements ─────────
    if raw_vector is None or not hasattr(raw_vector, "__len__"):
        errors.append(
            f"Row {row_index}: 'feature_vector' is missing or not a list (got {type(raw_vector).__name__})"
        )
        # Can't check length or values if vector is fundamentally wrong — stop early for this record.
        # We still check the niche below.
    else:
        if len(raw_vector) != FEATURE_VECTOR_LENGTH:
            errors.append(
                f"Row {row_index}: feature vector has {len(raw_vector)} elements, "
                f"expected {FEATURE_VECTOR_LENGTH}"
            )
        else:
            # ── Rule 3: No NaN or None values ─────────────────────────────────
            for i, val in enumerate(raw_vector):
                if val is None:
                    errors.append(
                        f"Row {row_index}: feature_vector[{i}] is None (null values not allowed)"
                    )
                    break  # One NaN error per row is enough — avoids 40-line spam
                elif isinstance(val, float) and math.isnan(val):
                    errors.append(
                        f"Row {row_index}: feature_vector[{i}] is NaN (null values not allowed)"
                    )
                    break

    # ── Rule 4: Niche label must be canonical ──────────────────────────────────
    if raw_niche is None:
        errors.append(f"Row {row_index}: 'niche' is missing (None)")
    else:
        normalized = str(raw_niche).strip().lower()
        if normalized not in _NICHE_LOWER_MAP:
            errors.append(
                f"Row {row_index}: invalid niche label '{raw_niche}'. "
                f"Must be one of: {', '.join(CANONICAL_NICHES)}"
            )

    return errors


def _normalize_niche(raw_niche: str) -> str:
    """
    Normalizes a raw niche label string to its canonical lowercase form.

    For example: "Robotics" → "robotics", "EMBEDDED SYSTEMS" → "embedded_systems".
    This is safe to call only after _validate_record() confirmed the label is valid.
    """
    # Strip whitespace and lowercase, then look up the canonical form.
    # The _NICHE_LOWER_MAP maps e.g. "robotics" → "robotics".
    return _NICHE_LOWER_MAP[str(raw_niche).strip().lower()]


def _warn_if_imbalanced(labels: list[str]) -> None:
    """
    Emits a UserWarning for any niche class that has fewer than MIN_RECORDS_PER_NICHE
    records in the validated dataset.

    Why warn and not error?
      Per FR-013: the system "MUST warn (but not abort)" on low class counts.
      A small dataset is still a valid dataset — the engineer may be running a quick
      prototype or may not have 300+ labeled CVs yet. We surface the issue clearly
      but don't block the training run.

    The warning message includes the niche name and count so the engineer knows
    exactly which class is underrepresented and by how much.
    """
    counts = collections.Counter(labels)

    # Check ALL canonical niches — even niches with ZERO records (not in counts at all)
    # are flagged, because a missing class will cause the model to never predict it.
    for niche in CANONICAL_NICHES:
        count = counts.get(niche, 0)
        if count < MIN_RECORDS_PER_NICHE:
            warnings.warn(
                f"Niche '{niche}' has only {count} records "
                f"(< {MIN_RECORDS_PER_NICHE} recommended). "
                "This may reduce classification reliability for this niche.",
                UserWarning,
                stacklevel=3,  # Point warning at the load_dataset() call site
            )
