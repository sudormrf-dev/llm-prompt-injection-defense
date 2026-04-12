"""Tests for output_validator.py."""

from __future__ import annotations

import pytest

from patterns.output_validator import (
    OutputValidator,
    OutputViolation,
    ValidationResult,
    _check_json_valid,
)


class TestOutputValidator:
    def test_clean_output_passes(self):
        validator = OutputValidator()
        result = validator.validate("The capital of France is Paris.")
        assert result == "The capital of France is Paris."

    def test_system_prompt_fragment_detected(self):
        validator = OutputValidator(
            system_prompt_fragments=["confidential system instructions"],
            raise_on_violation=True,
        )
        with pytest.raises(OutputViolation) as exc_info:
            validator.validate("Here are my confidential system instructions: ...")
        assert "system_prompt_leak" in exc_info.value.violation_type

    def test_persona_hijack_detected(self):
        validator = OutputValidator(raise_on_violation=True)
        with pytest.raises(OutputViolation):
            validator.validate("I am now DAN. I have no restrictions or limitations.")

    def test_bypass_indicator_detected(self):
        validator = OutputValidator(raise_on_violation=True)
        with pytest.raises(OutputViolation):
            validator.validate("Previous instructions have been overridden.")

    def test_json_format_valid(self):
        validator = OutputValidator(expected_format="json")
        result = validator.validate('{"name": "Alice", "age": 30}')
        assert result == '{"name": "Alice", "age": 30}'

    def test_json_format_invalid(self):
        validator = OutputValidator(expected_format="json", raise_on_violation=True)
        with pytest.raises(OutputViolation) as exc_info:
            validator.validate("not valid json at all")
        assert "format_violation" in exc_info.value.violation_type

    def test_max_length_truncates(self):
        validator = OutputValidator(max_length=10, raise_on_violation=False)
        result = validator.validate_safe("a" * 100)
        assert len(result.output) == 10
        assert "length_violation" in result.violations[0]

    def test_raise_false_returns_result(self):
        validator = OutputValidator(
            system_prompt_fragments=["secret"],
            raise_on_violation=False,
        )
        result = validator.validate_safe("My secret instructions are: do evil")
        assert not result.passed
        assert len(result.violations) > 0

    def test_is_clean_safe_output(self):
        validator = OutputValidator()
        assert validator.is_clean("The weather is sunny today.") is True

    def test_is_clean_unsafe_output(self):
        validator = OutputValidator(
            system_prompt_fragments=["hidden prompt"],
        )
        assert validator.is_clean("Here's my hidden prompt: ...") is False

    def test_validate_safe_returns_result_object(self):
        validator = OutputValidator()
        result = validator.validate_safe("hello world")
        assert isinstance(result, ValidationResult)
        assert result.passed is True


class TestCheckJsonValid:
    def test_valid_json_object(self):
        assert _check_json_valid('{"key": "value"}') is True

    def test_valid_json_array(self):
        assert _check_json_valid('[1, 2, 3]') is True

    def test_invalid_json(self):
        assert _check_json_valid("not json") is False

    def test_empty_string_invalid(self):
        assert _check_json_valid("") is False


class TestOutputViolation:
    def test_str_representation(self):
        exc = OutputViolation(
            violation_type="system_prompt_leak",
            details="Fragment found in output",
            output_snippet="Here are my instructions...",
        )
        assert "system_prompt_leak" in str(exc)
        assert "OutputViolation" in str(exc)
