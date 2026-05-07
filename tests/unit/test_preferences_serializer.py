"""
test_preferences_serializer.py — Unit tests for serialise_preferences()

These tests verify that the preference form serialiser:
  - returns a dict with exactly the 5 expected keys
  - casts string form values to integers
  - defaults absent fields to 0
  - accepts boundary values -2 and +2
  - raises ValueError for out-of-range values (e.g. 3, -3)

Written BEFORE implementation (TDD — must FAIL until routes.py exists).
"""

import pytest

# Import the function we're about to write — this import will fail until
# routes.py is created, which is exactly what we want for the red phase.
from src.api.v1.routes import serialise_preferences

EXPECTED_KEYS = {
    "work_environment",
    "system_level",
    "industry_interest",
    "team_scale",
    "travel_tolerance",
}


class TestSerialisePrefencesKeys:
    """All 5 canonical preference keys must appear in the output."""

    def test_all_five_keys_present(self):
        # Pass a form-like dict with all five fields set to neutral
        form = {
            "work_environment":  "0",
            "system_level":      "0",
            "industry_interest": "0",
            "team_scale":        "0",
            "travel_tolerance":  "0",
        }
        result = serialise_preferences(form)
        assert set(result.keys()) == EXPECTED_KEYS

    def test_exactly_five_keys(self):
        form = {k: "0" for k in EXPECTED_KEYS}
        result = serialise_preferences(form)
        assert len(result) == 5

    def test_all_values_are_integers(self):
        form = {k: "1" for k in EXPECTED_KEYS}
        result = serialise_preferences(form)
        for k, v in result.items():
            assert isinstance(v, int), f"Value for {k!r} should be int, got {type(v)}"


class TestSerialisePrefencesDefaults:
    """Absent fields must default to 0, not raise."""

    def test_absent_field_defaults_to_zero(self):
        # Pass only one field; the other four should default to 0
        result = serialise_preferences({"work_environment": "1"})
        assert result["system_level"] == 0
        assert result["industry_interest"] == 0
        assert result["team_scale"] == 0
        assert result["travel_tolerance"] == 0

    def test_empty_form_all_zeros(self):
        result = serialise_preferences({})
        for k in EXPECTED_KEYS:
            assert result[k] == 0


class TestSerialisePrefencesBoundaryValues:
    """Values at the edges of the valid range (-2 and +2) must be accepted."""

    def test_positive_two_accepted(self):
        form = {k: "2" for k in EXPECTED_KEYS}
        result = serialise_preferences(form)
        for v in result.values():
            assert v == 2

    def test_negative_two_accepted(self):
        form = {k: "-2" for k in EXPECTED_KEYS}
        result = serialise_preferences(form)
        for v in result.values():
            assert v == -2


class TestSerialisePrefencesOutOfRange:
    """Values outside [-2, 2] must raise ValueError — not silently pass."""

    def test_value_three_raises(self):
        form = {"work_environment": "3"}
        with pytest.raises(ValueError):
            serialise_preferences(form)

    def test_value_negative_three_raises(self):
        form = {"work_environment": "-3"}
        with pytest.raises(ValueError):
            serialise_preferences(form)
