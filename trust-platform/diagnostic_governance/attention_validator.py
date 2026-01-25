"""
Attention Validator
===================

Validates model attention maps to detect potential model failures.

Why Validate Attention?
    Deep learning models can achieve correct predictions for wrong reasons.
    For example, a chest X-ray model might:
    - Focus on patient positioning markers instead of lung fields
    - Use text labels burned into the image
    - Exploit spurious correlations (hospital-specific artifacts)

    By validating that attention is in expected regions, we can catch
    these failures before they cause harm.

Validation Strategies:
    1. Region-based: Check that attention overlaps expected anatomical regions
    2. Pattern-based: Check attention distribution matches expected patterns
    3. Anomaly-based: Flag unusual attention patterns not seen in training

Example:
    >>> validator = AttentionValidator()
    >>> result = validator.validate(
    ...     attention_map=model_attention,
    ...     predicted_class="pneumonia",
    ...     expected_regions=["right_lower_lobe", "left_lower_lobe"]
    ... )
    >>> if not result.is_valid:
    ...     print(f"Attention concerns: {result.concerns}")
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import math


__all__ = [
    'AttentionValidator',
    'AttentionValidationResult',
    'AttentionRegion',
    'AttentionConcern',
]


# =============================================================================
# ENUMS & DATA CLASSES
# =============================================================================

class AttentionConcern(str, Enum):
    """Types of attention validation concerns."""
    WRONG_REGION = "wrong_region"           # Attention in unexpected area
    OUTSIDE_ANATOMY = "outside_anatomy"      # Attention outside body/organ
    TOO_DIFFUSE = "too_diffuse"             # Attention spread too broadly
    TOO_FOCUSED = "too_focused"             # Attention on tiny spot
    EDGE_ARTIFACT = "edge_artifact"          # Attention on image edges
    TEXT_REGION = "text_region"             # Attention on text/labels
    LOW_CONFIDENCE = "low_confidence"        # Weak attention overall


@dataclass
class AttentionRegion:
    """
    A region of interest in the attention map.

    Coordinates are normalized to [0, 1] for both x and y.
    """
    name: str
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    expected_for: List[str]  # Classes where attention here is expected

    def contains(self, x: float, y: float) -> bool:
        """Check if point is within region."""
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    def overlap_fraction(
        self,
        other_x_min: float,
        other_y_min: float,
        other_x_max: float,
        other_y_max: float
    ) -> float:
        """Compute overlap fraction with another box."""
        # Intersection
        inter_x_min = max(self.x_min, other_x_min)
        inter_y_min = max(self.y_min, other_y_min)
        inter_x_max = min(self.x_max, other_x_max)
        inter_y_max = min(self.y_max, other_y_max)

        if inter_x_min >= inter_x_max or inter_y_min >= inter_y_max:
            return 0.0

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        self_area = (self.x_max - self.x_min) * (self.y_max - self.y_min)

        return inter_area / self_area if self_area > 0 else 0.0


@dataclass
class AttentionValidationResult:
    """Result of attention map validation."""
    is_valid: bool
    confidence: float  # Overall confidence in validation
    concerns: List[AttentionConcern]
    concern_details: List[str]
    top_attention_region: Optional[str]
    attention_coverage: float  # Fraction of expected region covered
    unexpected_attention_fraction: float  # Fraction in unexpected areas

    @property
    def has_critical_concerns(self) -> bool:
        """True if any critical concerns found."""
        critical = {
            AttentionConcern.TEXT_REGION,
            AttentionConcern.OUTSIDE_ANATOMY,
        }
        return any(c in critical for c in self.concerns)


# =============================================================================
# MAIN VALIDATOR CLASS
# =============================================================================

class AttentionValidator:
    """
    Validates attention maps for diagnostic AI systems.

    The validator checks that model attention is in expected regions
    given the predicted class. This helps catch cases where the model
    is "right for wrong reasons."

    Example:
        >>> # Define expected attention regions
        >>> regions = [
        ...     AttentionRegion("lungs", 0.1, 0.2, 0.9, 0.8, ["pneumonia", "atelectasis"]),
        ...     AttentionRegion("heart", 0.3, 0.3, 0.7, 0.7, ["cardiomegaly"]),
        ... ]
        >>> validator = AttentionValidator(expected_regions=regions)
        >>>
        >>> result = validator.validate(attention_map, "pneumonia")
    """

    def __init__(
        self,
        expected_regions: Optional[List[AttentionRegion]] = None,
        min_coverage_threshold: float = 0.3,
        max_unexpected_threshold: float = 0.5,
    ):
        """
        Initialize attention validator.

        Args:
            expected_regions: List of anatomical regions with expected classes
            min_coverage_threshold: Minimum attention in expected region
            max_unexpected_threshold: Maximum attention in unexpected areas
        """
        self.expected_regions = expected_regions or []
        self.min_coverage_threshold = min_coverage_threshold
        self.max_unexpected_threshold = max_unexpected_threshold

    def validate(
        self,
        attention_map: Any,
        predicted_class: str,
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> AttentionValidationResult:
        """
        Validate attention map for a predicted class.

        Args:
            attention_map: 2D attention/saliency map (or mock data for testing)
            predicted_class: The class predicted by the model
            image_shape: Optional (height, width) of original image

        Returns:
            AttentionValidationResult with validation outcome
        """
        concerns: List[AttentionConcern] = []
        concern_details: List[str] = []

        # MOCK: For testing, simulate attention validation
        # In production, this would analyze actual attention maps

        # Simulate attention statistics
        mock_attention_stats = self._get_mock_attention_stats(predicted_class)

        # Check 1: Is attention in expected region?
        expected_region = self._get_expected_region(predicted_class)
        attention_coverage = mock_attention_stats.get("expected_coverage", 0.5)

        if expected_region and attention_coverage < self.min_coverage_threshold:
            concerns.append(AttentionConcern.WRONG_REGION)
            concern_details.append(
                f"Only {attention_coverage:.0%} attention in expected region "
                f"({expected_region.name}), threshold is {self.min_coverage_threshold:.0%}"
            )

        # Check 2: Is attention too diffuse?
        entropy = mock_attention_stats.get("entropy", 0.5)
        if entropy > 0.9:
            concerns.append(AttentionConcern.TOO_DIFFUSE)
            concern_details.append("Attention is spread too broadly across image")

        # Check 3: Is attention too focused?
        if entropy < 0.1:
            concerns.append(AttentionConcern.TOO_FOCUSED)
            concern_details.append("Attention is concentrated on very small region")

        # Check 4: Is attention on edges (potential artifact)?
        edge_fraction = mock_attention_stats.get("edge_fraction", 0.1)
        if edge_fraction > 0.3:
            concerns.append(AttentionConcern.EDGE_ARTIFACT)
            concern_details.append(f"{edge_fraction:.0%} of attention on image edges")

        # Check 5: Unexpected attention fraction
        unexpected_fraction = mock_attention_stats.get("unexpected_fraction", 0.2)
        if unexpected_fraction > self.max_unexpected_threshold:
            concerns.append(AttentionConcern.WRONG_REGION)
            concern_details.append(
                f"{unexpected_fraction:.0%} attention in unexpected regions"
            )

        # Compute overall validity
        is_valid = len(concerns) == 0
        confidence = 1.0 - (len(concerns) * 0.2)  # Reduce confidence per concern
        confidence = max(0.0, confidence)

        return AttentionValidationResult(
            is_valid=is_valid,
            confidence=confidence,
            concerns=concerns,
            concern_details=concern_details,
            top_attention_region=expected_region.name if expected_region else None,
            attention_coverage=attention_coverage,
            unexpected_attention_fraction=unexpected_fraction,
        )

    def _get_expected_region(self, predicted_class: str) -> Optional[AttentionRegion]:
        """Get the expected attention region for a predicted class."""
        for region in self.expected_regions:
            if predicted_class in region.expected_for:
                return region
        return None

    def _get_mock_attention_stats(self, predicted_class: str) -> Dict[str, float]:
        """
        Generate mock attention statistics for testing.

        In production, these would be computed from actual attention maps.
        """
        # Simulate reasonable statistics based on class
        # Classes that are "easier" to detect get better attention stats

        easy_classes = {"normal", "cardiomegaly"}
        hard_classes = {"subtle_nodule", "early_pneumonia"}

        if predicted_class in easy_classes:
            return {
                "expected_coverage": 0.7,
                "entropy": 0.4,
                "edge_fraction": 0.05,
                "unexpected_fraction": 0.1,
            }
        elif predicted_class in hard_classes:
            return {
                "expected_coverage": 0.3,
                "entropy": 0.7,
                "edge_fraction": 0.15,
                "unexpected_fraction": 0.4,
            }
        else:
            # Default reasonable stats
            return {
                "expected_coverage": 0.5,
                "entropy": 0.5,
                "edge_fraction": 0.1,
                "unexpected_fraction": 0.2,
            }

    def add_region(self, region: AttentionRegion) -> None:
        """Add an expected attention region."""
        self.expected_regions.append(region)

    def set_regions_for_class(
        self,
        class_name: str,
        regions: List[AttentionRegion]
    ) -> None:
        """Set expected regions for a specific class."""
        for region in regions:
            if class_name not in region.expected_for:
                region.expected_for.append(class_name)
            if region not in self.expected_regions:
                self.expected_regions.append(region)
