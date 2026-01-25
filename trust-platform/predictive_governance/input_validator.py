"""
Input Validator
===============

Validates inputs for predictive AI models to catch data quality issues early.

Why Validate Inputs?
    Predictive models don't know when their inputs are nonsensical.
    A sepsis prediction model will happily compute a risk score given
    heart_rate = -50 or temperature = 100°C. The prediction will be
    garbage, but the model won't tell you that.

    Input validation catches these issues BEFORE wasting compute on
    meaningless predictions.

Validation Types:
    1. Range checks: Is the value within physiologically plausible bounds?
    2. Type checks: Is the value the expected type?
    3. Completeness checks: Are required inputs present?
    4. Consistency checks: Do related inputs make sense together?

Example:
    >>> validator = InputValidator()
    >>> validator.add_rule("heart_rate", RangeRule(min_val=30, max_val=250))
    >>> validator.add_rule("temperature", RangeRule(min_val=30, max_val=45))
    >>>
    >>> result = validator.validate({"heart_rate": 85, "temperature": 38.5})
    >>> print(result.is_valid)  # True
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod


__all__ = [
    'InputValidator',
    'InputValidationResult',
    'ValidationRule',
    'RangeRule',
    'TypeRule',
    'RequiredRule',
    'CustomRule',
    'ValidationSeverity',
]


# =============================================================================
# ENUMS & DATA CLASSES
# =============================================================================

class ValidationSeverity(str, Enum):
    """Severity of validation failure."""
    WARNING = "warning"      # Unusual but may be valid
    ERROR = "error"          # Invalid, but can proceed with caution
    CRITICAL = "critical"    # Invalid, must reject


@dataclass
class ValidationFailure:
    """Details of a validation failure."""
    input_name: str
    rule_name: str
    message: str
    severity: ValidationSeverity
    actual_value: Any
    expected: str


@dataclass
class InputValidationResult:
    """Result of validating model inputs."""
    is_valid: bool
    failures: List[ValidationFailure]
    warnings: List[ValidationFailure]
    n_validated: int
    n_passed: int

    @property
    def has_critical_failures(self) -> bool:
        """True if any critical failures."""
        return any(f.severity == ValidationSeverity.CRITICAL for f in self.failures)

    @property
    def error_messages(self) -> List[str]:
        """Get list of error messages."""
        return [f.message for f in self.failures]

    @property
    def warning_messages(self) -> List[str]:
        """Get list of warning messages."""
        return [f.message for f in self.warnings]


# =============================================================================
# VALIDATION RULES
# =============================================================================

class ValidationRule(ABC):
    """Base class for validation rules."""

    def __init__(
        self,
        name: str = "",
        severity: ValidationSeverity = ValidationSeverity.ERROR,
    ):
        self.name = name or self.__class__.__name__
        self.severity = severity

    @abstractmethod
    def validate(self, value: Any) -> Tuple[bool, Optional[str]]:
        """
        Validate a value.

        Returns:
            Tuple of (is_valid, error_message)
        """
        pass


class RangeRule(ValidationRule):
    """
    Validates that a numeric value is within a range.

    Example:
        >>> rule = RangeRule(min_val=30, max_val=250, name="heart_rate_range")
        >>> rule.validate(85)  # (True, None)
        >>> rule.validate(300)  # (False, "Value 300 exceeds maximum 250")
    """

    def __init__(
        self,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        name: str = "range",
        severity: ValidationSeverity = ValidationSeverity.ERROR,
    ):
        super().__init__(name, severity)
        self.min_val = min_val
        self.max_val = max_val

    def validate(self, value: Any) -> Tuple[bool, Optional[str]]:
        if value is None:
            return False, "Value is None"

        try:
            num_value = float(value)
        except (TypeError, ValueError):
            return False, f"Cannot convert {value} to number"

        if self.min_val is not None and num_value < self.min_val:
            return False, f"Value {num_value} below minimum {self.min_val}"

        if self.max_val is not None and num_value > self.max_val:
            return False, f"Value {num_value} exceeds maximum {self.max_val}"

        return True, None


class TypeRule(ValidationRule):
    """
    Validates that a value is of expected type.

    Example:
        >>> rule = TypeRule(expected_type=float, name="numeric_type")
        >>> rule.validate(3.14)  # (True, None)
        >>> rule.validate("hello")  # (False, "Expected float, got str")
    """

    def __init__(
        self,
        expected_type: type,
        allow_none: bool = False,
        name: str = "type",
        severity: ValidationSeverity = ValidationSeverity.ERROR,
    ):
        super().__init__(name, severity)
        self.expected_type = expected_type
        self.allow_none = allow_none

    def validate(self, value: Any) -> Tuple[bool, Optional[str]]:
        if value is None:
            if self.allow_none:
                return True, None
            else:
                return False, "Value is None but None not allowed"

        if isinstance(value, self.expected_type):
            return True, None

        # Special case: allow int for float
        if self.expected_type == float and isinstance(value, int):
            return True, None

        return False, f"Expected {self.expected_type.__name__}, got {type(value).__name__}"


class RequiredRule(ValidationRule):
    """
    Validates that a required value is present (not None, not empty).

    Example:
        >>> rule = RequiredRule(name="required")
        >>> rule.validate(85)  # (True, None)
        >>> rule.validate(None)  # (False, "Required value is missing")
    """

    def __init__(
        self,
        allow_zero: bool = True,
        allow_empty_string: bool = False,
        name: str = "required",
        severity: ValidationSeverity = ValidationSeverity.CRITICAL,
    ):
        super().__init__(name, severity)
        self.allow_zero = allow_zero
        self.allow_empty_string = allow_empty_string

    def validate(self, value: Any) -> Tuple[bool, Optional[str]]:
        if value is None:
            return False, "Required value is missing"

        if isinstance(value, str) and not value.strip() and not self.allow_empty_string:
            return False, "Required value is empty string"

        if isinstance(value, (int, float)) and value == 0 and not self.allow_zero:
            return False, "Required value is zero"

        return True, None


class CustomRule(ValidationRule):
    """
    Custom validation rule with user-defined function.

    Example:
        >>> rule = CustomRule(
        ...     validator=lambda x: x % 2 == 0,
        ...     error_message="Value must be even",
        ...     name="even_check"
        ... )
        >>> rule.validate(4)  # (True, None)
        >>> rule.validate(3)  # (False, "Value must be even")
    """

    def __init__(
        self,
        validator: Callable[[Any], bool],
        error_message: str = "Validation failed",
        name: str = "custom",
        severity: ValidationSeverity = ValidationSeverity.ERROR,
    ):
        super().__init__(name, severity)
        self.validator = validator
        self.error_message = error_message

    def validate(self, value: Any) -> Tuple[bool, Optional[str]]:
        try:
            if self.validator(value):
                return True, None
            else:
                return False, self.error_message
        except Exception as e:
            return False, f"Validation error: {str(e)}"


# =============================================================================
# MAIN VALIDATOR CLASS
# =============================================================================

class InputValidator:
    """
    Validates inputs for predictive AI models.

    Supports multiple validation rules per input, with configurable
    severity levels and detailed failure reporting.

    Example:
        >>> validator = InputValidator()
        >>>
        >>> # Add rules for vital signs
        >>> validator.add_rule("heart_rate", RangeRule(30, 250))
        >>> validator.add_rule("temperature", RangeRule(30, 45))
        >>> validator.add_rule("sbp", RangeRule(50, 250))
        >>> validator.add_rule("dbp", RangeRule(30, 150))
        >>>
        >>> # Validate inputs
        >>> result = validator.validate({
        ...     "heart_rate": 85,
        ...     "temperature": 38.5,
        ...     "sbp": 120,
        ...     "dbp": 80
        ... })
        >>> print(result.is_valid)
    """

    def __init__(self):
        """Initialize input validator."""
        self._rules: Dict[str, List[ValidationRule]] = {}
        self._required_inputs: set = set()

    def add_rule(
        self,
        input_name: str,
        rule: ValidationRule,
    ) -> 'InputValidator':
        """
        Add a validation rule for an input.

        Args:
            input_name: Name of the input to validate
            rule: Validation rule to apply

        Returns:
            self for chaining
        """
        if input_name not in self._rules:
            self._rules[input_name] = []
        self._rules[input_name].append(rule)
        return self

    def mark_required(self, input_name: str) -> 'InputValidator':
        """
        Mark an input as required.

        Args:
            input_name: Name of required input

        Returns:
            self for chaining
        """
        self._required_inputs.add(input_name)
        # Add required rule if not already present
        has_required = any(
            isinstance(r, RequiredRule)
            for r in self._rules.get(input_name, [])
        )
        if not has_required:
            self.add_rule(input_name, RequiredRule())
        return self

    def validate(
        self,
        inputs: Dict[str, Any],
        strict: bool = False,
    ) -> InputValidationResult:
        """
        Validate all inputs.

        Args:
            inputs: Dictionary of input name -> value
            strict: If True, fail on any issue; if False, allow warnings

        Returns:
            InputValidationResult with details
        """
        failures: List[ValidationFailure] = []
        warnings: List[ValidationFailure] = []
        n_validated = 0
        n_passed = 0

        # Check required inputs
        for required_input in self._required_inputs:
            if required_input not in inputs:
                failures.append(ValidationFailure(
                    input_name=required_input,
                    rule_name="required",
                    message=f"Missing required input: {required_input}",
                    severity=ValidationSeverity.CRITICAL,
                    actual_value=None,
                    expected="Present and valid",
                ))

        # Validate each input
        for input_name, value in inputs.items():
            rules = self._rules.get(input_name, [])

            if not rules:
                continue  # No rules for this input

            n_validated += 1
            input_valid = True

            for rule in rules:
                is_valid, error_msg = rule.validate(value)

                if not is_valid:
                    input_valid = False
                    failure = ValidationFailure(
                        input_name=input_name,
                        rule_name=rule.name,
                        message=f"{input_name}: {error_msg}",
                        severity=rule.severity,
                        actual_value=value,
                        expected=self._get_expected_description(rule),
                    )

                    if rule.severity == ValidationSeverity.WARNING:
                        warnings.append(failure)
                    else:
                        failures.append(failure)

            if input_valid:
                n_passed += 1

        # Determine overall validity
        if strict:
            is_valid = len(failures) == 0 and len(warnings) == 0
        else:
            is_valid = not any(
                f.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL)
                for f in failures
            )

        return InputValidationResult(
            is_valid=is_valid,
            failures=failures,
            warnings=warnings,
            n_validated=n_validated,
            n_passed=n_passed,
        )

    def validate_single(
        self,
        input_name: str,
        value: Any,
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate a single input.

        Args:
            input_name: Name of the input
            value: Value to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        rules = self._rules.get(input_name, [])

        for rule in rules:
            is_valid, error_msg = rule.validate(value)
            if not is_valid:
                return False, error_msg

        return True, None

    def _get_expected_description(self, rule: ValidationRule) -> str:
        """Get human-readable description of what rule expects."""
        if isinstance(rule, RangeRule):
            parts = []
            if rule.min_val is not None:
                parts.append(f">= {rule.min_val}")
            if rule.max_val is not None:
                parts.append(f"<= {rule.max_val}")
            return " and ".join(parts) if parts else "Any number"
        elif isinstance(rule, TypeRule):
            return f"Type: {rule.expected_type.__name__}"
        elif isinstance(rule, RequiredRule):
            return "Required (not None or empty)"
        else:
            return "Custom validation"

    @property
    def required_inputs(self) -> List[str]:
        """Get list of required input names."""
        return list(self._required_inputs)

    @property
    def all_inputs(self) -> List[str]:
        """Get list of all input names with rules."""
        return list(self._rules.keys())
