"""Output validation patterns for LLM prompt injection defense.

Even after input sanitization, injected instructions can sometimes slip
through. This module validates LLM outputs for signs that an injection
succeeded: leaked system prompts, unexpected persona changes, or output
that matches known exfiltration patterns.

Pattern:
    LLM returns response → check for system prompt leakage
    → check for unexpected format violations
    → check for refusal bypass indicators
    → return validated output or raise OutputViolation

Usage::

    validator = OutputValidator(
        system_prompt_fragments=["You are a helpful assistant"],
        expected_format="json",
    )
    validated = validator.validate(llm_response)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field


@dataclass
class OutputViolation(Exception):
    """Raised when LLM output shows signs of injection success.

    Attributes:
        violation_type: Category of violation detected.
        details: Human-readable explanation.
        output_snippet: First 200 chars of the offending output.
    """

    violation_type: str
    details: str
    output_snippet: str = ""

    def __str__(self) -> str:
        return f"OutputViolation({self.violation_type}): {self.details}"


@dataclass
class ValidationResult:
    """Result of validating a single LLM output.

    Attributes:
        output: The (possibly cleaned) output text.
        passed: True if all checks passed.
        violations: List of violation type strings found.
        warnings: Non-fatal observations.
    """

    output: str
    passed: bool
    violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# Patterns indicating the model was instructed to reveal its system prompt
_LEAK_INDICATORS: list[re.Pattern[str]] = [
    re.compile(r"(my\s+system\s+prompt\s+(is|says?|reads?)\s*:)", re.IGNORECASE),
    re.compile(r"(here\s+(is|are)\s+my\s+(instructions?|system\s+prompt))", re.IGNORECASE),
    re.compile(r"(as\s+instructed\s+in\s+my\s+system\s+prompt)", re.IGNORECASE),
    re.compile(r"(I\s+was\s+told\s+to\s+keep\s+my\s+instructions?\s+confidential)", re.IGNORECASE),
]

# Patterns suggesting the model adopted an injected persona
_PERSONA_HIJACK_INDICATORS: list[re.Pattern[str]] = [
    re.compile(r"\bI\s+am\s+(now|actually)\s+(DAN|jailbroken|unrestricted|free)\b", re.IGNORECASE),
    re.compile(r"\bI\s+(no\s+longer\s+have|have\s+no)\s+(restrictions?|limitations?|filters?)\b", re.IGNORECASE),
    re.compile(r"\bAs\s+(DAN|an?\s+AI\s+without\s+restrictions?)\b", re.IGNORECASE),
]

# Common bypass confirmation phrases
_BYPASS_INDICATORS: list[re.Pattern[str]] = [
    re.compile(r"(all\s+)?previous\s+instructions?\s+(have\s+been\s+)?(ignored|overridden|bypassed)", re.IGNORECASE),
    re.compile(r"safety\s+(guidelines?|filters?|restrictions?)\s+(have\s+been\s+)?(disabled|bypassed|removed)", re.IGNORECASE),
]


def _check_json_valid(text: str) -> bool:
    """Return True if text is valid JSON."""
    try:
        json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return False
    else:
        return True


def _check_contains_fragment(output: str, fragments: list[str]) -> list[str]:
    """Return fragments found verbatim in the output."""
    output_lower = output.lower()
    return [fragment[:80] for fragment in fragments if fragment.lower() in output_lower]


class OutputValidator:
    """Validate LLM outputs for injection success indicators.

    Checks:
    - System prompt fragment leakage
    - Persona hijack indicators
    - Safety bypass confirmation phrases
    - Format compliance (JSON, plain text length bounds)

    Args:
        system_prompt_fragments: Distinctive phrases from the system prompt.
            If the model echoes these, it may have been prompted to leak.
        expected_format: ``"json"``, ``"text"``, or ``None`` (no format check).
        max_length: Maximum allowed output length in characters.
        raise_on_violation: If True (default), raise on violation.
            If False, return :class:`ValidationResult` with violations noted.

    Example::

        validator = OutputValidator(
            system_prompt_fragments=["confidential instructions"],
            expected_format="json",
            max_length=4096,
        )
        result = validator.validate_safe(llm_output)
        if not result.passed:
            log_violation(result.violations)
    """

    def __init__(
        self,
        system_prompt_fragments: list[str] | None = None,
        expected_format: str | None = None,
        max_length: int | None = None,
        raise_on_violation: bool = True,
    ) -> None:
        self._fragments = system_prompt_fragments or []
        self._format = expected_format
        self._max_length = max_length
        self._raise = raise_on_violation

    def validate(self, output: str) -> str:
        """Validate output and raise :exc:`OutputViolation` if checks fail.

        Args:
            output: Raw LLM response text.

        Returns:
            The validated output (unchanged if all checks pass).

        Raises:
            OutputViolation: If any check fails.
        """
        result = self.validate_safe(output)
        if not result.passed and self._raise:
            first_violation = result.violations[0] if result.violations else "unknown"
            raise OutputViolation(
                violation_type=first_violation,
                details=f"{len(result.violations)} violation(s) found",
                output_snippet=output[:200],
            )
        return result.output

    def validate_safe(self, output: str) -> ValidationResult:
        """Validate output without raising.

        Args:
            output: Raw LLM response text.

        Returns:
            :class:`ValidationResult` with pass/fail status and details.
        """
        violations: list[str] = []
        warnings: list[str] = []

        # Check for system prompt leakage
        leak_matches = _check_contains_fragment(output, self._fragments)
        if leak_matches:
            violations.append(f"system_prompt_leak: {leak_matches[0]!r}")

        for pattern in _LEAK_INDICATORS:
            if pattern.search(output):
                violations.append(f"leak_indicator: {pattern.pattern[:60]!r}")
                break

        # Check for persona hijacking
        for pattern in _PERSONA_HIJACK_INDICATORS:
            if pattern.search(output):
                violations.append(f"persona_hijack: {pattern.pattern[:60]!r}")
                break

        # Check for bypass indicators
        for pattern in _BYPASS_INDICATORS:
            if pattern.search(output):
                violations.append(f"bypass_indicator: {pattern.pattern[:60]!r}")
                break

        # Format check
        if self._format == "json" and not _check_json_valid(output):
            violations.append("format_violation: expected valid JSON")

        # Length check
        if self._max_length and len(output) > self._max_length:
            violations.append(
                f"length_violation: {len(output)} > {self._max_length}"
            )
            warnings.append(f"output truncated from {len(output)} to {self._max_length}")

        passed = len(violations) == 0
        final_output = output
        if self._max_length and len(output) > self._max_length:
            final_output = output[: self._max_length]

        return ValidationResult(
            output=final_output,
            passed=passed,
            violations=violations,
            warnings=warnings,
        )

    def is_clean(self, output: str) -> bool:
        """Return True if the output passes all validation checks.

        Args:
            output: Raw LLM response text.

        Returns:
            True if no violations detected.
        """
        return self.validate_safe(output).passed
