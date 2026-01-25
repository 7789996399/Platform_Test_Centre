"""
Clinical Risk Adapter
=====================

Domain-specific adapter for clinical risk prediction AI governance.

This adapter provides:
- Input validation for vital signs and lab values
- Physiologically plausible ranges
- Risk score action thresholds
- Mock EHR integration for testing

Supported Risk Scores:
- Sepsis risk (qSOFA-like)
- Readmission risk
- Deterioration early warning (NEWS-like)
- Mortality risk (APACHE/SAPS-like)

Example:
    >>> adapter = ClinicalRiskAdapter(risk_type=RiskScoreType.SEPSIS)
    >>> is_valid, error = adapter.validate_input("heart_rate", 95)
    >>> thresholds = adapter.get_action_thresholds()
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from predictive_governance.input_validator import (
    InputValidator,
    RangeRule,
    RequiredRule,
    ValidationSeverity,
)


__all__ = [
    'ClinicalRiskAdapter',
    'RiskScoreType',
    'VitalSign',
    'LabValue',
    'VITAL_SIGN_RANGES',
    'LAB_VALUE_RANGES',
    'RISK_ACTION_THRESHOLDS',
]


# =============================================================================
# ENUMS
# =============================================================================

class RiskScoreType(str, Enum):
    """Type of clinical risk score."""
    SEPSIS = "sepsis"
    READMISSION = "readmission"
    DETERIORATION = "deterioration"
    MORTALITY = "mortality"
    CARDIAC_ARREST = "cardiac_arrest"


class VitalSign(str, Enum):
    """Standard vital sign parameters."""
    HEART_RATE = "heart_rate"
    RESPIRATORY_RATE = "respiratory_rate"
    SYSTOLIC_BP = "systolic_bp"
    DIASTOLIC_BP = "diastolic_bp"
    MEAN_ARTERIAL_PRESSURE = "mean_arterial_pressure"
    TEMPERATURE = "temperature"
    OXYGEN_SATURATION = "oxygen_saturation"
    LEVEL_OF_CONSCIOUSNESS = "level_of_consciousness"


class LabValue(str, Enum):
    """Standard laboratory values."""
    WBC = "wbc"  # White blood cell count
    LACTATE = "lactate"
    CREATININE = "creatinine"
    BILIRUBIN = "bilirubin"
    PLATELET_COUNT = "platelet_count"
    PH = "ph"
    PO2 = "po2"
    PCO2 = "pco2"
    GLUCOSE = "glucose"
    SODIUM = "sodium"
    POTASSIUM = "potassium"
    HEMOGLOBIN = "hemoglobin"
    BUN = "bun"


# =============================================================================
# PHYSIOLOGICAL RANGES
# =============================================================================

# Plausible vital sign ranges (adult)
VITAL_SIGN_RANGES: Dict[str, Tuple[float, float]] = {
    "heart_rate": (30, 250),          # bpm
    "respiratory_rate": (4, 60),       # breaths/min
    "systolic_bp": (50, 280),          # mmHg
    "diastolic_bp": (20, 180),         # mmHg
    "mean_arterial_pressure": (30, 200),  # mmHg
    "temperature": (30, 45),           # °C
    "oxygen_saturation": (50, 100),    # %
    "level_of_consciousness": (0, 15),  # GCS scale
}

# Plausible lab value ranges (adult)
LAB_VALUE_RANGES: Dict[str, Tuple[float, float]] = {
    "wbc": (0.5, 100),          # K/uL
    "lactate": (0.1, 30),       # mmol/L
    "creatinine": (0.1, 25),    # mg/dL
    "bilirubin": (0.1, 50),     # mg/dL
    "platelet_count": (5, 1000),  # K/uL
    "ph": (6.5, 8.0),           # pH units
    "po2": (20, 600),           # mmHg
    "pco2": (10, 150),          # mmHg
    "glucose": (20, 1000),      # mg/dL
    "sodium": (100, 180),       # mEq/L
    "potassium": (1.5, 10),     # mEq/L
    "hemoglobin": (2, 25),      # g/dL
    "bun": (1, 200),            # mg/dL
}

# Risk score action thresholds by score type
RISK_ACTION_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "sepsis": {
        "elevated": 0.20,   # 20% - increased monitoring
        "urgent": 0.40,     # 40% - prompt assessment
        "critical": 0.60,   # 60% - immediate intervention
    },
    "readmission": {
        "elevated": 0.30,
        "urgent": 0.50,
        "critical": 0.70,
    },
    "deterioration": {
        "elevated": 0.15,
        "urgent": 0.30,
        "critical": 0.50,
    },
    "mortality": {
        "elevated": 0.10,
        "urgent": 0.25,
        "critical": 0.50,
    },
    "cardiac_arrest": {
        "elevated": 0.05,
        "urgent": 0.15,
        "critical": 0.30,
    },
}


# =============================================================================
# CLINICAL RISK ADAPTER
# =============================================================================

class ClinicalRiskAdapter:
    """
    Clinical risk-specific adapter for predictive governance.

    Provides:
    - Input validation for vital signs and labs
    - Physiologically plausible ranges
    - Risk score action thresholds
    - Mock EHR data for testing

    Example:
        >>> adapter = ClinicalRiskAdapter(risk_type=RiskScoreType.SEPSIS)
        >>>
        >>> # Validate inputs
        >>> is_valid, error = adapter.validate_input("heart_rate", 95)
        >>>
        >>> # Get action thresholds
        >>> thresholds = adapter.get_action_thresholds()
    """

    def __init__(
        self,
        risk_type: RiskScoreType = RiskScoreType.SEPSIS,
        vital_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        lab_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        """
        Initialize clinical risk adapter.

        Args:
            risk_type: Type of risk score being computed
            vital_ranges: Custom vital sign ranges (uses defaults if not provided)
            lab_ranges: Custom lab value ranges (uses defaults if not provided)
        """
        self.risk_type = risk_type
        self.vital_ranges = vital_ranges or VITAL_SIGN_RANGES
        self.lab_ranges = lab_ranges or LAB_VALUE_RANGES

        # Build input validator
        self._validator = self._build_validator()

    def _build_validator(self) -> InputValidator:
        """Build input validator with all vital and lab ranges."""
        validator = InputValidator()

        # Add vital sign rules
        for vital_name, (min_val, max_val) in self.vital_ranges.items():
            validator.add_rule(
                vital_name,
                RangeRule(min_val=min_val, max_val=max_val, name=f"{vital_name}_range")
            )

        # Add lab value rules
        for lab_name, (min_val, max_val) in self.lab_ranges.items():
            validator.add_rule(
                lab_name,
                RangeRule(min_val=min_val, max_val=max_val, name=f"{lab_name}_range")
            )

        # Mark required inputs based on risk type
        for required in self.get_required_inputs():
            validator.mark_required(required)

        return validator

    def validate_input(
        self,
        input_name: str,
        value: Any,
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate a single input value.

        Args:
            input_name: Name of the input (e.g., "heart_rate")
            value: Value to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        return self._validator.validate_single(input_name, value)

    def validate_all(
        self,
        inputs: Dict[str, Any],
    ) -> Tuple[bool, List[str]]:
        """
        Validate all inputs.

        Args:
            inputs: Dictionary of input name -> value

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        result = self._validator.validate(inputs)
        return result.is_valid, result.error_messages

    def get_required_inputs(self) -> List[str]:
        """
        Get list of required input names for this risk type.

        Returns:
            List of required input names
        """
        required_by_type = {
            RiskScoreType.SEPSIS: [
                "heart_rate",
                "respiratory_rate",
                "systolic_bp",
                "temperature",
                "wbc",
            ],
            RiskScoreType.READMISSION: [
                "heart_rate",
                "respiratory_rate",
                "systolic_bp",
                "oxygen_saturation",
            ],
            RiskScoreType.DETERIORATION: [
                "heart_rate",
                "respiratory_rate",
                "systolic_bp",
                "oxygen_saturation",
                "temperature",
                "level_of_consciousness",
            ],
            RiskScoreType.MORTALITY: [
                "heart_rate",
                "respiratory_rate",
                "systolic_bp",
                "temperature",
                "wbc",
                "creatinine",
                "bilirubin",
            ],
            RiskScoreType.CARDIAC_ARREST: [
                "heart_rate",
                "respiratory_rate",
                "systolic_bp",
                "oxygen_saturation",
            ],
        }
        return required_by_type.get(self.risk_type, [])

    def get_action_thresholds(self) -> Dict[str, float]:
        """
        Get action thresholds for this risk type.

        Returns:
            Dictionary with elevated, urgent, critical thresholds
        """
        return RISK_ACTION_THRESHOLDS.get(
            self.risk_type.value,
            {"elevated": 0.3, "urgent": 0.5, "critical": 0.7}
        )

    def get_mock_ehr_data(
        self,
        patient_id: str,
        scenario: str = "normal",
    ) -> Dict[str, Any]:
        """
        Get mock EHR data for testing.

        Args:
            patient_id: Patient identifier
            scenario: Scenario type ("normal", "septic", "deteriorating")

        Returns:
            Mock EHR vital signs and labs
        """
        scenarios = {
            "normal": {
                "heart_rate": 75,
                "respiratory_rate": 16,
                "systolic_bp": 120,
                "diastolic_bp": 80,
                "temperature": 37.0,
                "oxygen_saturation": 98,
                "level_of_consciousness": 15,
                "wbc": 7.5,
                "lactate": 1.0,
                "creatinine": 1.0,
            },
            "septic": {
                "heart_rate": 110,
                "respiratory_rate": 24,
                "systolic_bp": 90,
                "diastolic_bp": 55,
                "temperature": 38.8,
                "oxygen_saturation": 92,
                "level_of_consciousness": 14,
                "wbc": 18.5,
                "lactate": 4.2,
                "creatinine": 2.1,
            },
            "deteriorating": {
                "heart_rate": 95,
                "respiratory_rate": 22,
                "systolic_bp": 105,
                "diastolic_bp": 65,
                "temperature": 37.8,
                "oxygen_saturation": 94,
                "level_of_consciousness": 15,
                "wbc": 12.0,
                "lactate": 2.5,
                "creatinine": 1.4,
            },
        }

        base_data = scenarios.get(scenario, scenarios["normal"])

        return {
            "patient_id": patient_id,
            "timestamp": "2024-01-15T14:30:00Z",
            "vitals": {k: v for k, v in base_data.items() if k in self.vital_ranges},
            "labs": {k: v for k, v in base_data.items() if k in self.lab_ranges},
            "clinical_notes": f"Mock EHR data for {scenario} scenario",
        }

    def compute_qsofa(
        self,
        respiratory_rate: float,
        systolic_bp: float,
        altered_mental_status: bool,
    ) -> int:
        """
        Compute qSOFA score (quick sepsis-related organ failure assessment).

        Args:
            respiratory_rate: Respiratory rate (breaths/min)
            systolic_bp: Systolic blood pressure (mmHg)
            altered_mental_status: True if GCS < 15

        Returns:
            qSOFA score (0-3)
        """
        score = 0

        if respiratory_rate >= 22:
            score += 1

        if systolic_bp <= 100:
            score += 1

        if altered_mental_status:
            score += 1

        return score

    def compute_news(
        self,
        vitals: Dict[str, float],
    ) -> int:
        """
        Compute NEWS score (National Early Warning Score).

        Simplified version for demonstration.

        Args:
            vitals: Dictionary of vital signs

        Returns:
            NEWS score (0-20)
        """
        score = 0

        # Respiratory rate scoring
        rr = vitals.get("respiratory_rate", 16)
        if rr <= 8:
            score += 3
        elif rr <= 11:
            score += 1
        elif rr <= 20:
            score += 0
        elif rr <= 24:
            score += 2
        else:
            score += 3

        # Oxygen saturation scoring
        spo2 = vitals.get("oxygen_saturation", 98)
        if spo2 <= 91:
            score += 3
        elif spo2 <= 93:
            score += 2
        elif spo2 <= 95:
            score += 1
        else:
            score += 0

        # Temperature scoring
        temp = vitals.get("temperature", 37.0)
        if temp <= 35.0:
            score += 3
        elif temp <= 36.0:
            score += 1
        elif temp <= 38.0:
            score += 0
        elif temp <= 39.0:
            score += 1
        else:
            score += 2

        # Systolic BP scoring
        sbp = vitals.get("systolic_bp", 120)
        if sbp <= 90:
            score += 3
        elif sbp <= 100:
            score += 2
        elif sbp <= 110:
            score += 1
        elif sbp <= 219:
            score += 0
        else:
            score += 3

        # Heart rate scoring
        hr = vitals.get("heart_rate", 75)
        if hr <= 40:
            score += 3
        elif hr <= 50:
            score += 1
        elif hr <= 90:
            score += 0
        elif hr <= 110:
            score += 1
        elif hr <= 130:
            score += 2
        else:
            score += 3

        # Consciousness scoring
        gcs = vitals.get("level_of_consciousness", 15)
        if gcs < 15:
            score += 3

        return score

    def get_score_description(self, score: int, score_type: str = "NEWS") -> str:
        """Get clinical description of risk score."""
        if score_type == "NEWS":
            if score <= 4:
                return "Low risk - routine monitoring"
            elif score <= 6:
                return "Medium risk - increased monitoring frequency"
            else:
                return "High risk - urgent clinical review required"
        elif score_type == "qSOFA":
            if score <= 1:
                return "Low risk of sepsis"
            else:
                return "High risk of sepsis - consider sepsis workup"
        else:
            return f"Score: {score}"
