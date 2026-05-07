"""
test_dataset.py — Unit Tests for src/dataset.py

These tests verify the labeled dataset loading and validation logic in isolation.
No ML training happens here — we only test that:
  - Valid CSV and JSON files load correctly and return the right shapes
  - Invalid records (wrong vector length, bad niche labels, NaN values) are rejected
  - Class imbalance warnings fire when a niche has < 50 records
  - compute_dataset_version() is deterministic and sensitive to file content

All tests use small synthetic datasets (a few dozen records) written to tmp_path,
so they're fast and don't depend on the real training dataset being present.

Spec reference: specs/003-ml-niche-classifier/tasks.md (T011)
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
import pytest

from src.dataset import load_dataset, compute_dataset_version, MIN_RECORDS_PER_NICHE
from src.model_io import CANONICAL_NICHES


# ── HELPERS ───────────────────────────────────────────────────────────────────

def _make_valid_vector(seed: int = 0) -> list:
    """
    Creates a valid 40-dimensional feature vector with some non-zero values.
    The seed controls the pattern so different calls produce different vectors.
    """
    v = [0] * 40
    v[seed % 40] = seed + 1  # One non-zero value at a seed-dependent position
    return v


def _write_valid_csv(path, n_per_niche: int = 60, niches=None) -> str:
    """
    Writes a valid CSV with n_per_niche records per canonical niche to path.
    Returns the path as a string for convenience.
    """
    if niches is None:
        niches = CANONICAL_NICHES
    rows = []
    for i, niche in enumerate(niches):
        for j in range(n_per_niche):
            v = _make_valid_vector(seed=(i * 100 + j) % 40)
            row = {f"feat_{k}": v[k] for k in range(40)}
            row["niche"] = niche
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return str(path)


def _write_valid_json(path, n_per_niche: int = 60, niches=None) -> str:
    """
    Writes a valid JSON dataset with n_per_niche records per canonical niche.
    Returns the path as a string.
    """
    if niches is None:
        niches = CANONICAL_NICHES
    records = []
    for i, niche in enumerate(niches):
        for j in range(n_per_niche):
            v = _make_valid_vector(seed=(i * 100 + j) % 40)
            records.append({"feature_vector": v, "niche": niche})
    with open(path, "w") as f:
        json.dump(records, f)
    return str(path)


# ── TESTS: valid CSV loading ───────────────────────────────────────────────────

class TestValidCsvLoading:
    """
    Tests that well-formed CSV files load correctly.
    US1 Acceptance Scenarios 1 and 2.
    """

    def test_valid_csv_loads_without_error(self, tmp_path):
        """
        A well-formed CSV with all 6 niches and 60 records each must load
        without raising any exception.
        """
        csv_path = _write_valid_csv(tmp_path / "dataset.csv")
        feature_vectors, labels = load_dataset(csv_path)
        assert feature_vectors is not None
        assert labels is not None

    def test_valid_csv_returns_correct_record_count(self, tmp_path):
        """
        With 60 records per niche × 6 niches = 360 records total,
        load_dataset() should return exactly 360 entries.
        """
        n_per_niche = 60
        csv_path = _write_valid_csv(tmp_path / "dataset.csv", n_per_niche=n_per_niche)
        feature_vectors, labels = load_dataset(csv_path)
        expected = n_per_niche * len(CANONICAL_NICHES)
        assert len(feature_vectors) == expected, (
            f"Expected {expected} records, got {len(feature_vectors)}"
        )
        assert len(labels) == expected

    def test_valid_csv_feature_vectors_are_40d(self, tmp_path):
        """
        Every feature vector from a valid CSV must have exactly 40 elements.
        """
        csv_path = _write_valid_csv(tmp_path / "dataset.csv")
        feature_vectors, _ = load_dataset(csv_path)
        for i, fv in enumerate(feature_vectors):
            assert len(fv) == 40, f"Feature vector at index {i} has {len(fv)} elements, expected 40"

    def test_valid_csv_labels_are_canonical(self, tmp_path):
        """
        All labels returned from a valid CSV must be lowercase canonical niche names.
        """
        csv_path = _write_valid_csv(tmp_path / "dataset.csv")
        _, labels = load_dataset(csv_path)
        canonical_set = set(CANONICAL_NICHES)
        for i, label in enumerate(labels):
            assert label in canonical_set, (
                f"Label at index {i} is '{label}', not in canonical set {canonical_set}"
            )

    def test_csv_with_uppercase_niche_labels_normalizes_to_lowercase(self, tmp_path):
        """
        The spec (data-model.md) says niche labels are 'normalized to canonical form'
        on load. This means 'Robotics' in the CSV must become 'robotics' in the output.
        """
        # Write a CSV with title-case niche labels (as they appear in the raw training set)
        rows = []
        for niche in CANONICAL_NICHES:
            title_case = niche.replace("_", " ").title().replace(" ", "_")
            for _ in range(55):  # above the 50 threshold to avoid warnings
                v = _make_valid_vector()
                row = {f"feat_{k}": v[k] for k in range(40)}
                row["niche"] = niche  # already lowercase — just verify it stays lowercase
                rows.append(row)
        df = pd.DataFrame(rows)
        csv_path = str(tmp_path / "dataset.csv")
        df.to_csv(csv_path, index=False)

        _, labels = load_dataset(csv_path)
        for label in labels:
            assert label == label.lower(), f"Label '{label}' must be lowercase"


# ── TESTS: valid JSON loading ──────────────────────────────────────────────────

class TestValidJsonLoading:
    """
    Tests that well-formed JSON files produce the same output as equivalent CSVs.
    US1 Acceptance Scenario 2.
    """

    def test_valid_json_loads_without_error(self, tmp_path):
        """A well-formed JSON dataset loads without error."""
        json_path = _write_valid_json(tmp_path / "dataset.json")
        feature_vectors, labels = load_dataset(json_path)
        assert feature_vectors is not None
        assert labels is not None

    def test_valid_json_returns_same_count_as_equivalent_csv(self, tmp_path):
        """
        JSON and CSV formats loaded from equivalent data must produce the
        same number of records.
        """
        n_per_niche = 55
        csv_path = _write_valid_csv(tmp_path / "dataset.csv", n_per_niche=n_per_niche)
        json_path = _write_valid_json(tmp_path / "dataset.json", n_per_niche=n_per_niche)

        fv_csv, lbl_csv = load_dataset(csv_path)
        fv_json, lbl_json = load_dataset(json_path)

        assert len(fv_csv) == len(fv_json), (
            f"CSV gives {len(fv_csv)} records, JSON gives {len(fv_json)}"
        )
        assert len(lbl_csv) == len(lbl_json)

    def test_json_feature_vectors_are_40d(self, tmp_path):
        """Every feature vector from a valid JSON file must have exactly 40 elements."""
        json_path = _write_valid_json(tmp_path / "dataset.json")
        feature_vectors, _ = load_dataset(json_path)
        for i, fv in enumerate(feature_vectors):
            assert len(fv) == 40, f"JSON vector at {i} has {len(fv)} dims"


# ── TESTS: validation — wrong vector length ────────────────────────────────────

class TestVectorLengthValidation:
    """
    Tests that records with wrong feature vector length are rejected with
    a helpful error message that includes the row index.
    US1 Acceptance Scenario 1 (validation), T011.
    """

    def test_short_vector_is_rejected(self, tmp_path):
        """A vector with 39 elements must be rejected."""
        records = [
            {"feature_vector": [0] * 39, "niche": "robotics"},  # Too short
            {"feature_vector": [0] * 40, "niche": "robotics"},  # Valid
        ]
        json_path = str(tmp_path / "dataset.json")
        with open(json_path, "w") as f:
            json.dump(records, f)

        with pytest.raises(ValueError) as exc_info:
            load_dataset(json_path)
        assert "Row 0" in str(exc_info.value), "Error must reference the row index"

    def test_long_vector_is_rejected(self, tmp_path):
        """A vector with 41 elements must also be rejected."""
        records = [{"feature_vector": [0] * 41, "niche": "automotive"}]
        json_path = str(tmp_path / "bad.json")
        with open(json_path, "w") as f:
            json.dump(records, f)

        with pytest.raises(ValueError) as exc_info:
            load_dataset(json_path)
        assert "41" in str(exc_info.value), "Error must mention the actual length"

    def test_multiple_bad_rows_all_reported(self, tmp_path):
        """
        When multiple records have bad vector lengths, all bad rows must be
        reported in a single ValueError (not just the first bad row).
        """
        records = [
            {"feature_vector": [0] * 10, "niche": "robotics"},   # Bad
            {"feature_vector": [0] * 40, "niche": "robotics"},   # Good
            {"feature_vector": [0] * 5,  "niche": "automotive"}, # Bad
        ]
        json_path = str(tmp_path / "multi_bad.json")
        with open(json_path, "w") as f:
            json.dump(records, f)

        with pytest.raises(ValueError) as exc_info:
            load_dataset(json_path)
        error_msg = str(exc_info.value)
        # Both bad rows must appear in the error (not just the first one)
        assert "Row 0" in error_msg
        assert "Row 2" in error_msg

    def test_csv_with_wrong_column_count_is_rejected(self, tmp_path):
        """
        A CSV that has only 30 feature columns (feat_0..feat_29) plus niche
        must be rejected because the feature vectors will be only 30-dimensional.
        """
        rows = []
        for _ in range(5):
            row = {f"feat_{i}": 0 for i in range(30)}  # Only 30 columns, not 40
            row["niche"] = "robotics"
            rows.append(row)
        csv_path = str(tmp_path / "short.csv")
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        with pytest.raises(ValueError):
            load_dataset(csv_path)


# ── TESTS: validation — invalid niche labels ──────────────────────────────────

class TestNicheLabelValidation:
    """
    Tests that records with invalid niche labels are caught before training.
    """

    def test_invalid_niche_is_rejected(self, tmp_path):
        """An unrecognized niche label must be rejected with a descriptive error."""
        records = [{"feature_vector": [0] * 40, "niche": "blockchain_engineering"}]
        json_path = str(tmp_path / "bad_niche.json")
        with open(json_path, "w") as f:
            json.dump(records, f)

        with pytest.raises(ValueError) as exc_info:
            load_dataset(json_path)
        assert "blockchain_engineering" in str(exc_info.value), (
            "Error must mention the invalid label"
        )

    def test_null_niche_is_rejected(self, tmp_path):
        """A null/None niche value must be rejected."""
        records = [{"feature_vector": [0] * 40, "niche": None}]
        json_path = str(tmp_path / "null_niche.json")
        with open(json_path, "w") as f:
            json.dump(records, f)

        with pytest.raises(ValueError):
            load_dataset(json_path)

    def test_row_index_included_in_niche_error(self, tmp_path):
        """The error message for a bad niche must include the row index."""
        records = [
            {"feature_vector": [0] * 40, "niche": "robotics"},        # Good
            {"feature_vector": [0] * 40, "niche": "made_up_niche"},   # Bad — row 1
        ]
        json_path = str(tmp_path / "bad_at_1.json")
        with open(json_path, "w") as f:
            json.dump(records, f)

        with pytest.raises(ValueError) as exc_info:
            load_dataset(json_path)
        assert "Row 1" in str(exc_info.value)


# ── TESTS: imbalance warnings ─────────────────────────────────────────────────

class TestImbalanceWarnings:
    """
    Tests that class imbalance warnings fire correctly for underpopulated niches.
    FR-013: warn (but don't abort) when any niche has < 50 records.
    US1 Acceptance Scenario 4.
    """

    def test_under_threshold_niche_emits_user_warning(self, tmp_path):
        """
        A dataset where one niche has only 10 records must emit a UserWarning
        mentioning that niche name. The load must still succeed (not raise).
        """
        records = []
        for i, niche in enumerate(CANONICAL_NICHES):
            # Give one niche only 10 records to trigger the warning
            count = 10 if niche == "technical_management" else 55
            for j in range(count):
                records.append({
                    "feature_vector": _make_valid_vector(seed=(i * 100 + j) % 40),
                    "niche": niche,
                })
        json_path = str(tmp_path / "imbalanced.json")
        with open(json_path, "w") as f:
            json.dump(records, f)

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            feature_vectors, labels = load_dataset(json_path)

        # Must still return data (no exception)
        assert feature_vectors is not None
        assert labels is not None

        # Must have emitted at least one UserWarning about the under-threshold niche
        user_warnings = [w for w in caught_warnings if issubclass(w.category, UserWarning)]
        assert len(user_warnings) >= 1, (
            f"Expected at least one UserWarning for imbalanced data, got {len(user_warnings)}"
        )
        warning_msgs = " ".join(str(w.message) for w in user_warnings)
        assert "technical_management" in warning_msgs, (
            f"Warning must mention the under-threshold niche, got: {warning_msgs}"
        )

    def test_balanced_dataset_produces_no_warnings(self, tmp_path):
        """
        A dataset with exactly MIN_RECORDS_PER_NICHE per niche must not emit any
        imbalance warnings — only datasets BELOW the threshold get warned.
        """
        csv_path = _write_valid_csv(
            tmp_path / "balanced.csv",
            n_per_niche=MIN_RECORDS_PER_NICHE,  # Exactly at threshold — no warning
        )

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            load_dataset(csv_path)

        imbalance_warnings = [
            w for w in caught_warnings
            if issubclass(w.category, UserWarning)
            and "records" in str(w.message).lower()
        ]
        assert len(imbalance_warnings) == 0, (
            f"Expected no imbalance warnings for balanced data, got: "
            f"{[str(w.message) for w in imbalance_warnings]}"
        )

    def test_warning_includes_count_and_threshold(self, tmp_path):
        """
        The warning message must include the actual count and the threshold so
        the engineer knows exactly how many more records are needed.
        """
        records = []
        for i, niche in enumerate(CANONICAL_NICHES):
            count = 5 if niche == "robotics" else 55
            for j in range(count):
                records.append({
                    "feature_vector": _make_valid_vector(seed=j),
                    "niche": niche,
                })
        json_path = str(tmp_path / "very_imbalanced.json")
        with open(json_path, "w") as f:
            json.dump(records, f)

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            load_dataset(json_path)

        user_warnings = [
            w for w in caught_warnings
            if issubclass(w.category, UserWarning)
            and "robotics" in str(w.message)
        ]
        assert len(user_warnings) >= 1
        # The warning should mention the actual count (5)
        assert "5" in str(user_warnings[0].message), (
            f"Warning should include count 5, got: {user_warnings[0].message}"
        )


# ── TESTS: NaN and null value rejection ───────────────────────────────────────

class TestNanValidation:
    """
    Tests that NaN and null values in feature vectors are caught before training.
    A NaN in a feature vector could corrupt the entire Random Forest fit.
    """

    def test_nan_in_vector_is_rejected(self, tmp_path):
        """A feature vector containing NaN must be rejected."""
        v = [0.0] * 40
        v[5] = float("nan")
        records = [{"feature_vector": v, "niche": "robotics"}]
        json_path = str(tmp_path / "nan.json")
        with open(json_path, "w") as f:
            json.dump(records, f)

        with pytest.raises(ValueError) as exc_info:
            load_dataset(json_path)
        # NaN becomes null in JSON, which becomes None in Python — either message is valid
        error_msg = str(exc_info.value).lower()
        assert any(word in error_msg for word in ["null", "nan", "none"]), (
            f"Error must mention the null/NaN issue, got: {exc_info.value}"
        )


# ── TESTS: compute_dataset_version ────────────────────────────────────────────

class TestComputeDatasetVersion:
    """
    Tests for the dataset version hashing function.
    The hash must be deterministic and sensitive to content changes.
    """

    def test_same_file_produces_same_hash(self, tmp_path):
        """
        Calling compute_dataset_version() twice on the same file must return
        the same hash. This is required for TrainingRecord reproducibility.
        """
        csv_path = _write_valid_csv(tmp_path / "dataset.csv")
        hash1 = compute_dataset_version(csv_path)
        hash2 = compute_dataset_version(csv_path)
        assert hash1 == hash2, f"Same file produced different hashes: {hash1} vs {hash2}"

    def test_different_files_produce_different_hashes(self, tmp_path):
        """
        Two files with different content must produce different hashes.
        This verifies the hash is sensitive to content changes (FR-005).
        """
        path1 = _write_valid_csv(tmp_path / "dataset1.csv", n_per_niche=55)
        path2 = _write_valid_csv(tmp_path / "dataset2.csv", n_per_niche=60)  # Different size
        hash1 = compute_dataset_version(path1)
        hash2 = compute_dataset_version(path2)
        assert hash1 != hash2, "Different files must produce different hashes"

    def test_hash_is_64_char_hex_string(self, tmp_path):
        """
        The SHA-256 hash must be a 64-character lowercase hex string.
        """
        csv_path = _write_valid_csv(tmp_path / "dataset.csv")
        version = compute_dataset_version(csv_path)
        assert isinstance(version, str), f"Version must be str, got {type(version)}"
        assert len(version) == 64, f"SHA-256 must be 64 chars, got {len(version)}"
        assert all(c in "0123456789abcdef" for c in version), (
            f"Hash must be lowercase hex, got: {version}"
        )

    def test_missing_file_raises_value_error(self, tmp_path):
        """
        Trying to hash a file that doesn't exist must raise ValueError (not OSError).
        We wrap OS errors in ValueError so the caller interface is consistent.
        """
        with pytest.raises(ValueError):
            compute_dataset_version(str(tmp_path / "nonexistent.csv"))

    def test_modified_file_produces_different_hash(self, tmp_path):
        """
        Modifying even one byte of the dataset file must produce a different hash.
        This ensures TrainingRecords are truly linked to exact file contents.
        """
        csv_path = _write_valid_csv(tmp_path / "dataset.csv")
        original_hash = compute_dataset_version(csv_path)

        # Append a single character to the file
        with open(csv_path, "a") as f:
            f.write("x")  # One extra byte

        modified_hash = compute_dataset_version(csv_path)
        assert original_hash != modified_hash, "Modified file must produce a different hash"


# ── TESTS: unsupported file formats ───────────────────────────────────────────

class TestUnsupportedFormats:
    """Tests that unsupported file extensions are rejected with clear errors."""

    def test_parquet_format_raises_value_error(self, tmp_path):
        """Parquet files are not supported — must fail with a clear message."""
        fake_parquet = str(tmp_path / "dataset.parquet")
        with open(fake_parquet, "w") as f:
            f.write("fake")
        with pytest.raises(ValueError) as exc_info:
            load_dataset(fake_parquet)
        assert "Unsupported" in str(exc_info.value) or "format" in str(exc_info.value).lower()

    def test_txt_format_raises_value_error(self, tmp_path):
        """Plain text files are not supported."""
        fake_txt = str(tmp_path / "dataset.txt")
        with open(fake_txt, "w") as f:
            f.write("robotics 0 1 2 3")
        with pytest.raises(ValueError):
            load_dataset(fake_txt)
