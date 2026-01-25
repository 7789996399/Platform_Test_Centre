"""
Prior Context Integrator
========================

Integrates prior imaging studies and clinical context for diagnostic AI.

Why Prior Context Matters:
    Radiology is fundamentally comparative. A finding's significance depends on:
    - Whether it's new or stable compared to prior studies
    - The patient's clinical history and symptoms
    - The clinical question being asked

    A nodule that's been stable for 5 years is very different from a new nodule.
    AI systems that ignore this context will produce misleading results.

Integration Strategies:
    1. Comparison with most recent prior
    2. Trend analysis across multiple priors
    3. Clinical correlation (symptoms match finding?)
    4. Indication-aware analysis (what question are we answering?)

Example:
    >>> integrator = PriorContextIntegrator()
    >>> result = integrator.integrate(
    ...     current_finding="nodule",
    ...     prior_studies=[study_from_2022, study_from_2021],
    ...     clinical_context={"indication": "cough", "history": "smoker"}
    ... )
    >>> print(result.comparison_summary)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime


__all__ = [
    'PriorContextIntegrator',
    'PriorStudy',
    'ContextualFinding',
    'ComparisonResult',
    'ChangeType',
]


# =============================================================================
# ENUMS & DATA CLASSES
# =============================================================================

class ChangeType(str, Enum):
    """Type of change compared to prior."""
    NEW = "new"                    # Finding not present in prior
    STABLE = "stable"              # Finding unchanged from prior
    IMPROVED = "improved"          # Finding better than prior
    WORSENED = "worsened"          # Finding worse than prior
    RESOLVED = "resolved"          # Finding was present, now gone
    UNKNOWN = "unknown"            # Cannot determine (no prior or unclear)


class ClinicalCorrelation(str, Enum):
    """How well finding correlates with clinical context."""
    STRONG = "strong"              # Finding explains symptoms
    MODERATE = "moderate"          # Finding may explain symptoms
    WEAK = "weak"                  # Finding unlikely to explain symptoms
    INCIDENTAL = "incidental"      # Finding unrelated to indication


@dataclass
class PriorStudy:
    """
    A prior imaging study for comparison.

    Contains the findings from a previous study and metadata
    for temporal comparison.
    """
    study_id: str
    study_date: str  # ISO format date
    modality: str  # "XR", "CT", "MRI", etc.
    findings: List[str]  # List of findings from the study
    impression: str  # Overall impression
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def age_days(self) -> int:
        """Days since this study (mock implementation)."""
        # In production, compute from actual dates
        return 365  # Default to 1 year for testing

    def has_finding(self, finding: str) -> bool:
        """Check if this study had a particular finding."""
        finding_lower = finding.lower()
        return any(finding_lower in f.lower() for f in self.findings)


@dataclass
class ContextualFinding:
    """
    A finding with contextual information.

    Enriches a raw finding with comparison to priors, clinical
    correlation, and risk assessment.
    """
    finding: str
    change_type: ChangeType
    clinical_correlation: ClinicalCorrelation
    prior_reference: Optional[str]  # Reference to prior study showing finding
    risk_modifier: float  # Multiplier for risk (>1 = higher risk)
    context_notes: List[str]

    @property
    def is_new(self) -> bool:
        """True if this is a new finding."""
        return self.change_type == ChangeType.NEW

    @property
    def is_stable(self) -> bool:
        """True if finding is stable."""
        return self.change_type == ChangeType.STABLE


@dataclass
class ComparisonResult:
    """Result of comparing current study to priors."""
    has_priors: bool
    most_recent_prior: Optional[PriorStudy]
    comparison_summary: str
    new_findings: List[str]
    stable_findings: List[str]
    resolved_findings: List[str]
    worsened_findings: List[str]


# =============================================================================
# MAIN INTEGRATOR CLASS
# =============================================================================

class PriorContextIntegrator:
    """
    Integrates prior imaging and clinical context for diagnostic AI.

    This component ensures AI predictions are interpreted in proper
    clinical context, not as isolated findings.

    Example:
        >>> integrator = PriorContextIntegrator()
        >>>
        >>> # Add prior studies
        >>> prior = PriorStudy(
        ...     study_id="XR-2023-001",
        ...     study_date="2023-01-15",
        ...     modality="XR",
        ...     findings=["clear lungs", "normal heart size"],
        ...     impression="Normal chest radiograph"
        ... )
        >>>
        >>> # Integrate with current finding
        >>> result = integrator.compare_finding(
        ...     current_finding="pneumonia",
        ...     prior_studies=[prior]
        ... )
        >>> print(result.change_type)  # NEW
    """

    def __init__(
        self,
        stability_threshold_days: int = 730,  # 2 years
        significant_change_threshold: float = 0.2,
    ):
        """
        Initialize prior context integrator.

        Args:
            stability_threshold_days: Days for a finding to be considered "stable"
            significant_change_threshold: Threshold for detecting significant change
        """
        self.stability_threshold_days = stability_threshold_days
        self.significant_change_threshold = significant_change_threshold

    def integrate(
        self,
        current_findings: List[str],
        prior_studies: List[PriorStudy],
        clinical_context: Optional[Dict[str, Any]] = None,
    ) -> List[ContextualFinding]:
        """
        Integrate current findings with prior studies and clinical context.

        Args:
            current_findings: List of findings from current study
            prior_studies: List of prior studies for comparison
            clinical_context: Optional clinical context (indication, history)

        Returns:
            List of contextualized findings
        """
        clinical_context = clinical_context or {}
        contextualized: List[ContextualFinding] = []

        for finding in current_findings:
            contextual_finding = self._contextualize_finding(
                finding, prior_studies, clinical_context
            )
            contextualized.append(contextual_finding)

        return contextualized

    def compare_finding(
        self,
        current_finding: str,
        prior_studies: List[PriorStudy],
    ) -> ContextualFinding:
        """
        Compare a single finding to prior studies.

        Args:
            current_finding: The finding from current study
            prior_studies: List of prior studies

        Returns:
            ContextualFinding with comparison information
        """
        return self._contextualize_finding(current_finding, prior_studies, {})

    def get_comparison_summary(
        self,
        current_findings: List[str],
        prior_studies: List[PriorStudy],
    ) -> ComparisonResult:
        """
        Get summary comparing current study to priors.

        Args:
            current_findings: Findings from current study
            prior_studies: Prior studies for comparison

        Returns:
            ComparisonResult with summary information
        """
        if not prior_studies:
            return ComparisonResult(
                has_priors=False,
                most_recent_prior=None,
                comparison_summary="No prior studies available for comparison.",
                new_findings=current_findings,
                stable_findings=[],
                resolved_findings=[],
                worsened_findings=[],
            )

        # Sort by date (most recent first)
        sorted_priors = sorted(
            prior_studies,
            key=lambda p: p.study_date,
            reverse=True
        )
        most_recent = sorted_priors[0]

        # Categorize findings
        new_findings = []
        stable_findings = []
        resolved_findings = []

        for finding in current_findings:
            if most_recent.has_finding(finding):
                stable_findings.append(finding)
            else:
                new_findings.append(finding)

        # Check for resolved findings
        for prior_finding in most_recent.findings:
            if not any(prior_finding.lower() in cf.lower() for cf in current_findings):
                resolved_findings.append(prior_finding)

        # Build summary
        summary_parts = []
        if new_findings:
            summary_parts.append(f"New findings: {', '.join(new_findings)}")
        if stable_findings:
            summary_parts.append(f"Stable: {', '.join(stable_findings)}")
        if resolved_findings:
            summary_parts.append(f"Resolved: {', '.join(resolved_findings)}")

        if not summary_parts:
            comparison_summary = "No significant changes compared to prior."
        else:
            comparison_summary = ". ".join(summary_parts) + "."

        return ComparisonResult(
            has_priors=True,
            most_recent_prior=most_recent,
            comparison_summary=comparison_summary,
            new_findings=new_findings,
            stable_findings=stable_findings,
            resolved_findings=resolved_findings,
            worsened_findings=[],  # Would need more sophisticated analysis
        )

    def _contextualize_finding(
        self,
        finding: str,
        prior_studies: List[PriorStudy],
        clinical_context: Dict[str, Any],
    ) -> ContextualFinding:
        """Create a contextualized finding with comparison and correlation."""
        # Determine change type
        change_type = self._determine_change_type(finding, prior_studies)

        # Determine clinical correlation
        clinical_correlation = self._determine_clinical_correlation(
            finding, clinical_context
        )

        # Find prior reference
        prior_reference = None
        for prior in prior_studies:
            if prior.has_finding(finding):
                prior_reference = f"{prior.study_id} ({prior.study_date})"
                break

        # Compute risk modifier
        risk_modifier = self._compute_risk_modifier(
            change_type, clinical_correlation
        )

        # Build context notes
        context_notes = self._build_context_notes(
            finding, change_type, clinical_correlation, clinical_context
        )

        return ContextualFinding(
            finding=finding,
            change_type=change_type,
            clinical_correlation=clinical_correlation,
            prior_reference=prior_reference,
            risk_modifier=risk_modifier,
            context_notes=context_notes,
        )

    def _determine_change_type(
        self,
        finding: str,
        prior_studies: List[PriorStudy],
    ) -> ChangeType:
        """Determine how finding compares to priors."""
        if not prior_studies:
            return ChangeType.UNKNOWN

        # Check most recent prior
        sorted_priors = sorted(
            prior_studies,
            key=lambda p: p.study_date,
            reverse=True
        )

        most_recent = sorted_priors[0]

        if most_recent.has_finding(finding):
            # Finding was present before
            if most_recent.age_days < self.stability_threshold_days:
                return ChangeType.STABLE
            else:
                return ChangeType.STABLE  # Long-term stable
        else:
            # Finding not in most recent prior
            return ChangeType.NEW

    def _determine_clinical_correlation(
        self,
        finding: str,
        clinical_context: Dict[str, Any],
    ) -> ClinicalCorrelation:
        """Determine how well finding correlates with clinical context."""
        indication = clinical_context.get("indication", "").lower()
        symptoms = clinical_context.get("symptoms", [])
        history = clinical_context.get("history", "").lower()

        finding_lower = finding.lower()

        # Define correlations (simplified)
        correlations = {
            "pneumonia": ["cough", "fever", "shortness of breath", "respiratory"],
            "cardiomegaly": ["heart", "cardiac", "shortness of breath", "edema"],
            "nodule": ["cancer", "mass", "smoking", "weight loss"],
            "fracture": ["trauma", "pain", "fall", "injury"],
        }

        # Check for correlation
        for condition, related_terms in correlations.items():
            if condition in finding_lower:
                # Check if any related term is in indication or symptoms
                for term in related_terms:
                    if term in indication:
                        return ClinicalCorrelation.STRONG
                    if any(term in s.lower() for s in symptoms):
                        return ClinicalCorrelation.MODERATE

        # No strong correlation found
        if indication:
            return ClinicalCorrelation.INCIDENTAL
        else:
            return ClinicalCorrelation.WEAK

    def _compute_risk_modifier(
        self,
        change_type: ChangeType,
        clinical_correlation: ClinicalCorrelation,
    ) -> float:
        """Compute risk modifier based on context."""
        base = 1.0

        # New findings are higher risk
        if change_type == ChangeType.NEW:
            base *= 1.5
        elif change_type == ChangeType.WORSENED:
            base *= 2.0
        elif change_type == ChangeType.STABLE:
            base *= 0.8

        # Poor clinical correlation = more concerning
        if clinical_correlation == ClinicalCorrelation.INCIDENTAL:
            base *= 1.2
        elif clinical_correlation == ClinicalCorrelation.STRONG:
            base *= 0.9  # Expected finding

        return base

    def _build_context_notes(
        self,
        finding: str,
        change_type: ChangeType,
        clinical_correlation: ClinicalCorrelation,
        clinical_context: Dict[str, Any],
    ) -> List[str]:
        """Build list of context notes for the finding."""
        notes = []

        if change_type == ChangeType.NEW:
            notes.append("NEW FINDING - not present on prior studies")
        elif change_type == ChangeType.STABLE:
            notes.append("Stable compared to prior")
        elif change_type == ChangeType.WORSENED:
            notes.append("WORSENED compared to prior - clinical attention needed")

        if clinical_correlation == ClinicalCorrelation.INCIDENTAL:
            notes.append("Incidental finding - may not explain presenting symptoms")
        elif clinical_correlation == ClinicalCorrelation.STRONG:
            notes.append("Correlates with clinical indication")

        indication = clinical_context.get("indication", "")
        if indication:
            notes.append(f"Clinical indication: {indication}")

        return notes
