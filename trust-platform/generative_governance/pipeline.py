"""
Generative Governance Pipeline
==============================

Implements the EHR-First verification flow for generative AI systems.

Architecture:
    ┌──────────────────────────────────────────────────────────────────────┐
    │                    GenerativeGovernancePipeline                       │
    ├──────────────────────────────────────────────────────────────────────┤
    │  1. EXTRACT: Pull verifiable claims from AI output                   │
    │              ↓                                                        │
    │  2. VERIFY (Fast): Check claims against EHR/sources                  │
    │              ↓                                                        │
    │     ┌─────────────┴─────────────┐                                    │
    │     │                           │                                    │
    │  VERIFIED ✓                  UNVERIFIED ?                            │
    │  (Skip entropy)              (Run entropy)                           │
    │     │                           │                                    │
    │     └─────────────┬─────────────┘                                    │
    │              ↓                                                        │
    │  3. DETECT: Flag confident hallucinators (low entropy + wrong)       │
    │              ↓                                                        │
    │  4. OUTPUT: Risk assessment + review level recommendation            │
    └──────────────────────────────────────────────────────────────────────┘

The "Confident Hallucinator" Problem:
    Traditional NLI fails on medical text (~9.5% accuracy) because hallucinations
    often sound confident. The key insight is combining UNCERTAINTY (semantic
    entropy) with CORRECTNESS (source verification):

    ┌─────────────────┬──────────────────┬───────────────────┐
    │                 │ Source: VERIFIED │ Source: CONFLICTS │
    ├─────────────────┼──────────────────┼───────────────────┤
    │ High Entropy    │ Review needed    │ Likely error      │
    │ (uncertain)     │ (unsure but ok)  │ (unsure & wrong)  │
    ├─────────────────┼──────────────────┼───────────────────┤
    │ Low Entropy     │ Likely correct   │ ⚠️ CONFIDENT      │
    │ (confident)     │ (sure & right)   │ HALLUCINATOR ⚠️   │
    └─────────────────┴──────────────────┴───────────────────┘

Example:
    >>> from generative_governance import GenerativeGovernancePipeline
    >>> from adapters.healthcare import HealthcareAdapter
    >>>
    >>> pipeline = GenerativeGovernancePipeline(adapter=HealthcareAdapter())
    >>> result = await pipeline.analyze(
    ...     ai_output="Patient takes Metoprolol 50mg daily",
    ...     source_data={"medications": [{"name": "Metoprolol", "dose": "25mg"}]}
    ... )
    >>> if result.has_confident_hallucinator:
    ...     print("CRITICAL: Confident hallucination detected!")
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Tuple

from core_engine.faithfulness import (
    Claim,
    ClaimCategory,
    VerificationStatus,
    VerificationResult,
)
from core_engine.semantic_entropy import (
    EntropyResult,
    EntropyRiskLevel,
    SemanticEntropyCalculator,
)
from core_engine.ensemble_orchestrator import (
    OverallRiskLevel,
    ReviewLevel,
    HallucinationType,
    detect_confident_hallucinator,
)


__all__ = [
    'GenerativeGovernancePipeline',
    'GenerativeGovernanceResult',
    'ClaimVerificationSummary',
    'GovernanceRiskLevel',
]


# =============================================================================
# ENUMS & DATA CLASSES
# =============================================================================

class GovernanceRiskLevel(str, Enum):
    """Risk level for generative AI output."""
    SAFE = "safe"                  # All claims verified, low entropy
    LOW_RISK = "low_risk"          # Minor concerns, brief review
    MEDIUM_RISK = "medium_risk"    # Some unverified claims, standard review
    HIGH_RISK = "high_risk"        # Contradictions found, detailed review
    CRITICAL = "critical"          # Confident hallucinator detected


@dataclass
class ClaimVerificationSummary:
    """Summary of claim verification results."""
    total_claims: int
    verified_claims: int
    contradicted_claims: int
    unverified_claims: int
    confident_hallucinators: int

    @property
    def verification_rate(self) -> float:
        """Proportion of claims that were verified."""
        if self.total_claims == 0:
            return 1.0
        return self.verified_claims / self.total_claims

    @property
    def has_contradictions(self) -> bool:
        """True if any claims contradict sources."""
        return self.contradicted_claims > 0

    @property
    def has_confident_hallucinators(self) -> bool:
        """True if any confident hallucinators detected."""
        return self.confident_hallucinators > 0


@dataclass
class ClaimAnalysisDetail:
    """Detailed analysis of a single claim."""
    claim: Claim
    verification_status: VerificationStatus
    matched_source: Optional[str]
    entropy_checked: bool
    entropy_result: Optional[EntropyResult]
    is_confident_hallucinator: bool
    risk_contribution: GovernanceRiskLevel


@dataclass
class GenerativeGovernanceResult:
    """
    Complete result from generative governance pipeline.

    Contains:
    - Overall risk assessment
    - Claim-by-claim analysis
    - Confident hallucinator flags
    - Recommended review level
    - Compute savings from EHR-First flow
    """
    risk_level: GovernanceRiskLevel
    review_level: ReviewLevel
    summary: ClaimVerificationSummary
    claim_details: List[ClaimAnalysisDetail]
    overall_entropy: Optional[float]
    has_confident_hallucinator: bool
    recommended_action: str
    compute_savings: float  # Percentage of SE calls saved
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_safe(self) -> bool:
        """True if output is safe for use without review."""
        return self.risk_level == GovernanceRiskLevel.SAFE

    @property
    def needs_human_review(self) -> bool:
        """True if human review is recommended."""
        return self.review_level != ReviewLevel.NONE


# =============================================================================
# ADAPTER PROTOCOL
# =============================================================================

class GenerativeAdapter(Protocol):
    """Protocol for domain-specific adapters used with generative governance."""

    def extract_claims(self, response: str) -> List[Claim]:
        """Extract verifiable claims from AI output."""
        ...

    async def verify_claim(self, claim: Claim, source_data: Dict[str, Any]) -> VerificationResult:
        """Verify a claim against source data."""
        ...

    def get_entropy_thresholds(self) -> Any:
        """Get domain-specific entropy thresholds."""
        ...


# =============================================================================
# MAIN PIPELINE
# =============================================================================

class GenerativeGovernancePipeline:
    """
    EHR-First governance pipeline for generative AI systems.

    This pipeline implements the core TRUST verification strategy:
    verify against sources FIRST, then selectively apply semantic entropy
    only where needed. This provides both cost savings and better detection.

    Key Features:
    - 80% compute reduction via source-first verification
    - Confident hallucinator detection
    - Domain-agnostic core with adapter injection
    - Configurable review level thresholds
    """

    def __init__(
        self,
        adapter: GenerativeAdapter,
        entropy_calculator: Optional[SemanticEntropyCalculator] = None,
        run_entropy_on_verified: bool = False,
        entropy_threshold: float = 0.5,
    ):
        """
        Initialize generative governance pipeline.

        Args:
            adapter: Domain-specific adapter for claim extraction/verification
            entropy_calculator: Optional SE calculator (created if not provided)
            run_entropy_on_verified: If True, run SE even on verified claims
            entropy_threshold: Threshold for high entropy classification
        """
        self.adapter = adapter
        self.entropy_calculator = entropy_calculator or SemanticEntropyCalculator()
        self.run_entropy_on_verified = run_entropy_on_verified
        self.entropy_threshold = entropy_threshold

    async def analyze(
        self,
        ai_output: str,
        source_data: Dict[str, Any],
        query: Optional[str] = None,
    ) -> GenerativeGovernanceResult:
        """
        Analyze AI output using EHR-First flow.

        Args:
            ai_output: The generated text to analyze
            source_data: Authoritative source data (EHR, documents, etc.)
            query: Optional original query/prompt

        Returns:
            GenerativeGovernanceResult with complete analysis
        """
        # Step 1: Extract claims
        claims = self.adapter.extract_claims(ai_output)

        if not claims:
            return self._create_safe_result()

        # Step 2: Verify each claim against sources (FAST)
        claim_details: List[ClaimAnalysisDetail] = []
        verified_count = 0
        contradicted_count = 0
        unverified_count = 0
        entropy_checks_needed = 0
        entropy_checks_skipped = 0
        confident_hallucinators = 0

        for claim in claims:
            # Verify against source
            verification = await self.adapter.verify_claim(claim, source_data)

            # Determine if we need entropy check
            needs_entropy = self._needs_entropy_check(verification)

            if needs_entropy:
                entropy_checks_needed += 1
            else:
                entropy_checks_skipped += 1

            # Run entropy if needed (or if configured to always run)
            entropy_result = None
            if needs_entropy or self.run_entropy_on_verified:
                entropy_result = await self._run_entropy_check(claim, query or "")

            # Detect confident hallucinator
            is_confident_hallucinator = self._check_confident_hallucinator(
                verification, entropy_result
            )

            if is_confident_hallucinator:
                confident_hallucinators += 1

            # Categorize verification result
            if verification.status == VerificationStatus.VERIFIED:
                verified_count += 1
                risk_contribution = GovernanceRiskLevel.SAFE
            elif verification.status == VerificationStatus.CONTRADICTED:
                contradicted_count += 1
                risk_contribution = GovernanceRiskLevel.CRITICAL if is_confident_hallucinator else GovernanceRiskLevel.HIGH_RISK
            else:
                unverified_count += 1
                risk_contribution = GovernanceRiskLevel.MEDIUM_RISK

            claim_details.append(ClaimAnalysisDetail(
                claim=claim,
                verification_status=verification.status,
                matched_source=verification.matched_text if hasattr(verification, 'matched_text') else None,
                entropy_checked=entropy_result is not None,
                entropy_result=entropy_result,
                is_confident_hallucinator=is_confident_hallucinator,
                risk_contribution=risk_contribution,
            ))

        # Step 3: Compute summary and overall risk
        summary = ClaimVerificationSummary(
            total_claims=len(claims),
            verified_claims=verified_count,
            contradicted_claims=contradicted_count,
            unverified_claims=unverified_count,
            confident_hallucinators=confident_hallucinators,
        )

        overall_risk = self._compute_overall_risk(summary, claim_details)
        review_level = self._compute_review_level(overall_risk, summary)
        recommended_action = self._get_recommended_action(overall_risk, summary)

        # Compute savings
        total_possible_entropy = len(claims)
        if total_possible_entropy > 0:
            compute_savings = entropy_checks_skipped / total_possible_entropy
        else:
            compute_savings = 1.0

        # Compute overall entropy if we ran any checks
        entropy_values = [
            d.entropy_result.entropy for d in claim_details
            if d.entropy_result is not None
        ]
        overall_entropy = sum(entropy_values) / len(entropy_values) if entropy_values else None

        return GenerativeGovernanceResult(
            risk_level=overall_risk,
            review_level=review_level,
            summary=summary,
            claim_details=claim_details,
            overall_entropy=overall_entropy,
            has_confident_hallucinator=confident_hallucinators > 0,
            recommended_action=recommended_action,
            compute_savings=compute_savings,
            metadata={
                "entropy_checks_run": entropy_checks_needed,
                "entropy_checks_skipped": entropy_checks_skipped,
                "source_data_keys": list(source_data.keys()),
            }
        )

    def _needs_entropy_check(self, verification: VerificationResult) -> bool:
        """Determine if a claim needs semantic entropy check."""
        # Skip entropy for verified claims (source-first optimization)
        if verification.status == VerificationStatus.VERIFIED:
            return False

        # Always check contradicted claims (need to confirm it's not uncertain)
        if verification.status == VerificationStatus.CONTRADICTED:
            return True

        # Check unverified claims to assess uncertainty
        if verification.status in (VerificationStatus.NOT_FOUND, VerificationStatus.PARTIAL):
            return True

        return True

    async def _run_entropy_check(
        self,
        claim: Claim,
        query: str,
    ) -> EntropyResult:
        """
        Run semantic entropy check on a claim.

        Note: In production, this would generate multiple responses and
        compute actual semantic entropy. Here we return a mock result.
        """
        # MOCK: In production, this would actually compute SE
        # For testing, return a mock result based on claim category
        mock_entropy = 0.3 if claim.category == ClaimCategory.HIGH_RISK else 0.5

        return EntropyResult(
            entropy=mock_entropy,
            normalized_entropy=mock_entropy,
            risk_level=EntropyRiskLevel.LOW if mock_entropy < self.entropy_threshold else EntropyRiskLevel.MEDIUM,
            n_samples=5,
            n_clusters=2,
            responses=["mock_response_1", "mock_response_2"],
            cluster_sizes=[3, 2],
        )

    def _check_confident_hallucinator(
        self,
        verification: VerificationResult,
        entropy_result: Optional[EntropyResult],
    ) -> bool:
        """
        Check if this is a confident hallucinator.

        Confident hallucinator = contradicts source + low entropy (confident)
        """
        if verification.status != VerificationStatus.CONTRADICTED:
            return False

        if entropy_result is None:
            return False

        # Low entropy = confident
        return entropy_result.entropy < self.entropy_threshold

    def _compute_overall_risk(
        self,
        summary: ClaimVerificationSummary,
        claim_details: List[ClaimAnalysisDetail],
    ) -> GovernanceRiskLevel:
        """Compute overall risk level from individual claim analyses."""
        # Critical if any confident hallucinators
        if summary.confident_hallucinators > 0:
            return GovernanceRiskLevel.CRITICAL

        # High risk if any contradictions
        if summary.contradicted_claims > 0:
            return GovernanceRiskLevel.HIGH_RISK

        # Medium risk if significant unverified claims
        if summary.unverified_claims > summary.total_claims * 0.3:
            return GovernanceRiskLevel.MEDIUM_RISK

        # Low risk if some unverified
        if summary.unverified_claims > 0:
            return GovernanceRiskLevel.LOW_RISK

        return GovernanceRiskLevel.SAFE

    def _compute_review_level(
        self,
        risk_level: GovernanceRiskLevel,
        summary: ClaimVerificationSummary,
    ) -> ReviewLevel:
        """Compute recommended review level."""
        if risk_level == GovernanceRiskLevel.CRITICAL:
            return ReviewLevel.DETAILED
        elif risk_level == GovernanceRiskLevel.HIGH_RISK:
            return ReviewLevel.DETAILED
        elif risk_level == GovernanceRiskLevel.MEDIUM_RISK:
            return ReviewLevel.STANDARD
        elif risk_level == GovernanceRiskLevel.LOW_RISK:
            return ReviewLevel.BRIEF
        else:
            return ReviewLevel.NONE

    def _get_recommended_action(
        self,
        risk_level: GovernanceRiskLevel,
        summary: ClaimVerificationSummary,
    ) -> str:
        """Get human-readable recommended action."""
        if risk_level == GovernanceRiskLevel.CRITICAL:
            return f"CRITICAL: {summary.confident_hallucinators} confident hallucinator(s) detected. Detailed review required before use."
        elif risk_level == GovernanceRiskLevel.HIGH_RISK:
            return f"HIGH RISK: {summary.contradicted_claims} claim(s) contradict sources. Verify before use."
        elif risk_level == GovernanceRiskLevel.MEDIUM_RISK:
            return f"MEDIUM RISK: {summary.unverified_claims} claim(s) could not be verified. Standard review recommended."
        elif risk_level == GovernanceRiskLevel.LOW_RISK:
            return "LOW RISK: Minor unverified claims. Brief review recommended."
        else:
            return "SAFE: All claims verified against sources. Safe for use."

    def _create_safe_result(self) -> GenerativeGovernanceResult:
        """Create a safe result when no claims are extracted."""
        return GenerativeGovernanceResult(
            risk_level=GovernanceRiskLevel.SAFE,
            review_level=ReviewLevel.NONE,
            summary=ClaimVerificationSummary(
                total_claims=0,
                verified_claims=0,
                contradicted_claims=0,
                unverified_claims=0,
                confident_hallucinators=0,
            ),
            claim_details=[],
            overall_entropy=None,
            has_confident_hallucinator=False,
            recommended_action="SAFE: No verifiable claims detected.",
            compute_savings=1.0,
        )
