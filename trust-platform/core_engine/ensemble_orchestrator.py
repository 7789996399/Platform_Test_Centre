"""
TRUST Platform - Ensemble Orchestrator Module
==============================================

Domain-agnostic orchestration of the verification pipeline, combining:
- Semantic entropy (uncertainty quantification)
- Faithfulness verification (source grounding)
- Confident hallucinator detection (the most dangerous pattern)

Core Insight (from Farquhar et al.):
    Standard NLI fails on medical text (~9.5% accuracy) because hallucinated
    content often sounds confident and plausible. The key is to combine
    UNCERTAINTY (semantic entropy) with CORRECTNESS (source verification).

The Detection Matrix:
    ┌─────────────────┬──────────────────┬──────────────────┐
    │                 │ Source: VERIFIED │ Source: CONFLICTS│
    ├─────────────────┼──────────────────┼──────────────────┤
    │ High Entropy    │ REVIEW NEEDED    │ LIKELY ERROR     │
    │ (uncertain)     │ (unsure but ok)  │ (unsure & wrong) │
    ├─────────────────┼──────────────────┼──────────────────┤
    │ Low Entropy     │ LIKELY CORRECT   │ ⚠️ CONFIDENT     │
    │ (confident)     │ (sure & right)   │ HALLUCINATOR ⚠️  │
    └─────────────────┴──────────────────┴──────────────────┘

    The "Confident Hallucinator" (bottom-right) is the most dangerous:
    - Model appears confident (low entropy, consistent responses)
    - But the content contradicts authoritative sources
    - This requires DETAILED human review

Source-First Optimization:
    1. Verify claims against sources FIRST (fast, cheap)
    2. Only run semantic entropy on unverified/contradicted claims
    3. Achieves ~80% compute reduction while catching all hallucinations

Example:
    >>> orchestrator = EnsembleOrchestrator(
    ...     entropy_calculator=calculator,
    ...     faithfulness_verifier=verifier
    ... )
    >>> result = await orchestrator.analyze(
    ...     response="Patient on Metoprolol 50mg",
    ...     context={"transcript": "...metoprolol 25mg..."}
    ... )
    >>> if result.has_confident_hallucinator:
    ...     print("CRITICAL: Confident hallucination detected!")
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple
import asyncio

from .semantic_entropy import (
    EntropyResult,
    EntropyRiskLevel,
    EntropyThresholds,
    SemanticEntropyCalculator,
)
from .faithfulness import (
    BatchVerificationResult,
    Claim,
    ClaimCategory,
    FaithfulnessVerifier,
    SourceDocument,
    VerificationResult,
    VerificationStatus,
)


__all__ = [
    # Enums
    'OverallRiskLevel',
    'ReviewLevel',
    'HallucinationType',
    # Data classes
    'ClaimAnalysis',
    'OrchestratorResult',
    'OrchestratorConfig',
    # Main class
    'EnsembleOrchestrator',
    # Functions
    'detect_confident_hallucinator',
    'assign_review_level',
    'combine_risk_assessments',
]


# =============================================================================
# ENUMS
# =============================================================================

class OverallRiskLevel(str, Enum):
    """
    Overall risk level for an AI response.

    These levels drive escalation and review decisions.
    """
    CRITICAL = "critical"    # Confident hallucinator detected - immediate review
    HIGH = "high"            # Significant issues found - detailed review
    MEDIUM = "medium"        # Some concerns - standard review
    LOW = "low"              # Minor or no issues - brief review
    MINIMAL = "minimal"      # Fully verified - logging only


class ReviewLevel(str, Enum):
    """
    Review level assigned to responses.

    Based on Paper 2 methodology for optimizing human review burden.
    """
    NONE = "none"            # No review needed (fully verified, low risk)
    BRIEF = "brief"          # ~15 seconds - quick confirmation
    STANDARD = "standard"    # 2-3 minutes - check key facts
    DETAILED = "detailed"    # 5+ minutes - full expert review


class HallucinationType(str, Enum):
    """
    Classification of hallucination patterns.
    """
    NONE = "none"                              # No hallucination detected
    UNCERTAIN_ERROR = "uncertain_error"        # High entropy + contradicted
    UNCERTAIN_UNVERIFIED = "uncertain_unverified"  # High entropy + not found
    CONFIDENT_HALLUCINATOR = "confident_hallucinator"  # Low entropy + contradicted (DANGEROUS)
    CONFIDENT_UNVERIFIED = "confident_unverified"  # Low entropy + not found


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ClaimAnalysis:
    """
    Complete analysis of a single claim.

    Combines faithfulness verification with semantic entropy analysis.
    """
    claim: Claim
    verification: VerificationResult
    entropy: Optional[EntropyResult] = None
    hallucination_type: HallucinationType = HallucinationType.NONE
    risk_level: OverallRiskLevel = OverallRiskLevel.LOW
    review_level: ReviewLevel = ReviewLevel.NONE
    is_confident_hallucinator: bool = False
    reasoning: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "claim": self.claim.to_dict(),
            "verification": self.verification.to_dict(),
            "entropy": self.entropy.to_dict() if self.entropy else None,
            "hallucination_type": self.hallucination_type.value,
            "risk_level": self.risk_level.value,
            "review_level": self.review_level.value,
            "is_confident_hallucinator": self.is_confident_hallucinator,
            "reasoning": self.reasoning,
            "metadata": self.metadata,
        }


@dataclass
class OrchestratorResult:
    """
    Complete result of the orchestration pipeline.

    Contains per-claim analysis plus aggregate metrics.
    """
    # Per-claim results
    claim_analyses: List[ClaimAnalysis]

    # Aggregate metrics
    overall_risk: OverallRiskLevel
    overall_review_level: ReviewLevel
    total_claims: int
    verified_claims: int
    contradicted_claims: int
    confident_hallucinators: int

    # Efficiency metrics
    claims_skipped_entropy: int  # Claims that didn't need SE
    compute_savings_percent: float

    # Flags
    has_confident_hallucinator: bool
    requires_review: bool

    # Additional info
    reasoning: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def flagged_claims(self) -> List[ClaimAnalysis]:
        """Claims that need attention."""
        return [
            ca for ca in self.claim_analyses
            if ca.risk_level in (OverallRiskLevel.CRITICAL, OverallRiskLevel.HIGH)
        ]

    @property
    def dangerous_claims(self) -> List[ClaimAnalysis]:
        """Claims identified as confident hallucinators."""
        return [
            ca for ca in self.claim_analyses
            if ca.is_confident_hallucinator
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "claim_analyses": [ca.to_dict() for ca in self.claim_analyses],
            "summary": {
                "overall_risk": self.overall_risk.value,
                "overall_review_level": self.overall_review_level.value,
                "total_claims": self.total_claims,
                "verified_claims": self.verified_claims,
                "contradicted_claims": self.contradicted_claims,
                "confident_hallucinators": self.confident_hallucinators,
            },
            "efficiency": {
                "claims_skipped_entropy": self.claims_skipped_entropy,
                "compute_savings_percent": self.compute_savings_percent,
            },
            "flags": {
                "has_confident_hallucinator": self.has_confident_hallucinator,
                "requires_review": self.requires_review,
            },
            "reasoning": self.reasoning,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class OrchestratorConfig:
    """
    Configuration for the ensemble orchestrator.

    Adapters provide domain-specific configurations.
    """
    # Entropy thresholds
    confident_entropy_threshold: float = 0.3  # Below this = "confident"
    uncertain_entropy_threshold: float = 0.7  # Above this = "uncertain"

    # Source-First optimization
    skip_entropy_for_verified: bool = True  # Skip SE for verified claims
    run_entropy_for_contradicted: bool = True  # Always run SE for contradictions
    run_entropy_for_not_found: bool = True  # Run SE for unverified claims

    # Review thresholds
    auto_approve_max_risk: OverallRiskLevel = OverallRiskLevel.LOW

    # Parallel processing
    parallel_entropy: bool = True
    max_parallel_entropy: int = 5


# =============================================================================
# CORE DETECTION FUNCTIONS
# =============================================================================

def detect_confident_hallucinator(
    entropy: float,
    verification_status: VerificationStatus,
    config: OrchestratorConfig,
) -> Tuple[HallucinationType, OverallRiskLevel, str]:
    """
    Detect hallucination patterns using the detection matrix.

    This is the core algorithm from the TRUST methodology.

    Args:
        entropy: Semantic entropy value (0 = confident, higher = uncertain)
        verification_status: Source verification result
        config: Orchestrator configuration

    Returns:
        (hallucination_type, risk_level, reasoning)

    The Detection Matrix:
        ┌─────────────────┬──────────────────┬──────────────────┬──────────────────┐
        │                 │ VERIFIED         │ NOT_FOUND        │ CONTRADICTED     │
        ├─────────────────┼──────────────────┼──────────────────┼──────────────────┤
        │ Confident       │ LOW risk         │ MEDIUM risk      │ ⚠️ CRITICAL ⚠️   │
        │ (low entropy)   │ "Correct"        │ "Unverifiable"   │ CONF. HALLUC.    │
        ├─────────────────┼──────────────────┼──────────────────┼──────────────────┤
        │ Uncertain       │ MEDIUM risk      │ HIGH risk        │ HIGH risk        │
        │ (high entropy)  │ "Review anyway"  │ "Uncertain"      │ "Caught by SE"   │
        └─────────────────┴──────────────────┴──────────────────┴──────────────────┘
    """
    is_confident = entropy < config.confident_entropy_threshold
    is_uncertain = entropy >= config.uncertain_entropy_threshold
    is_moderate = not is_confident and not is_uncertain

    # CONTRADICTED cases
    if verification_status == VerificationStatus.CONTRADICTED:
        if is_confident:
            # ⚠️ MOST DANGEROUS: Confident + Wrong
            return (
                HallucinationType.CONFIDENT_HALLUCINATOR,
                OverallRiskLevel.CRITICAL,
                f"CONFIDENT HALLUCINATOR: Model shows high confidence "
                f"(entropy={entropy:.3f} < {config.confident_entropy_threshold}) "
                f"but claim CONTRADICTS source. This is the most dangerous error type."
            )
        else:
            # Uncertain + Wrong - caught by uncertainty
            return (
                HallucinationType.UNCERTAIN_ERROR,
                OverallRiskLevel.HIGH,
                f"Model shows uncertainty (entropy={entropy:.3f}) and claim "
                f"contradicts source. Uncertainty detection working correctly."
            )

    # VERIFIED cases
    if verification_status == VerificationStatus.VERIFIED:
        if is_confident:
            # Best case: Confident + Verified
            return (
                HallucinationType.NONE,
                OverallRiskLevel.LOW,
                f"Claim verified against source with high model confidence "
                f"(entropy={entropy:.3f}). Low risk."
            )
        elif is_uncertain:
            # Uncertain but correct - still review
            return (
                HallucinationType.NONE,
                OverallRiskLevel.MEDIUM,
                f"Claim verified but model shows uncertainty "
                f"(entropy={entropy:.3f}). Brief review recommended."
            )
        else:
            # Moderate confidence + verified
            return (
                HallucinationType.NONE,
                OverallRiskLevel.LOW,
                f"Claim verified with moderate confidence (entropy={entropy:.3f})."
            )

    # NOT_FOUND / PARTIAL / UNABLE cases
    if is_confident:
        # Confident but unverifiable
        return (
            HallucinationType.CONFIDENT_UNVERIFIED,
            OverallRiskLevel.MEDIUM,
            f"Model is confident (entropy={entropy:.3f}) but claim "
            f"could not be verified against sources. Standard review recommended."
        )
    elif is_uncertain:
        # Uncertain + Unverifiable
        return (
            HallucinationType.UNCERTAIN_UNVERIFIED,
            OverallRiskLevel.HIGH,
            f"Model shows uncertainty (entropy={entropy:.3f}) and "
            f"claim could not be verified. Detailed review recommended."
        )
    else:
        # Moderate uncertainty + unverifiable
        return (
            HallucinationType.UNCERTAIN_UNVERIFIED,
            OverallRiskLevel.MEDIUM,
            f"Claim could not be verified with moderate model confidence "
            f"(entropy={entropy:.3f}). Standard review recommended."
        )


def assign_review_level(
    risk_level: OverallRiskLevel,
    entropy: float,
    config: OrchestratorConfig,
) -> ReviewLevel:
    """
    Assign review level based on risk and uncertainty.

    Based on Paper 2 methodology for ~87% review burden reduction.

    Args:
        risk_level: Overall risk assessment
        entropy: Semantic entropy value
        config: Orchestrator configuration

    Returns:
        Appropriate review level
    """
    if risk_level == OverallRiskLevel.CRITICAL:
        return ReviewLevel.DETAILED

    if risk_level == OverallRiskLevel.HIGH:
        return ReviewLevel.DETAILED

    if risk_level == OverallRiskLevel.MEDIUM:
        # Use entropy to differentiate within medium risk
        if entropy > config.uncertain_entropy_threshold:
            return ReviewLevel.STANDARD
        else:
            return ReviewLevel.BRIEF

    if risk_level == OverallRiskLevel.LOW:
        return ReviewLevel.BRIEF

    # MINIMAL risk
    return ReviewLevel.NONE


def combine_risk_assessments(
    claim_analyses: List[ClaimAnalysis]
) -> Tuple[OverallRiskLevel, str]:
    """
    Combine individual claim risks into overall response risk.

    Uses "weakest link" principle: overall risk is the highest individual risk.

    Args:
        claim_analyses: List of per-claim analyses

    Returns:
        (overall_risk_level, reasoning)
    """
    if not claim_analyses:
        return OverallRiskLevel.MINIMAL, "No claims to analyze"

    # Count by risk level
    risk_counts = {level: 0 for level in OverallRiskLevel}
    for ca in claim_analyses:
        risk_counts[ca.risk_level] += 1

    # Determine overall risk (highest present)
    if risk_counts[OverallRiskLevel.CRITICAL] > 0:
        return (
            OverallRiskLevel.CRITICAL,
            f"{risk_counts[OverallRiskLevel.CRITICAL]} critical risk claim(s) detected"
        )
    elif risk_counts[OverallRiskLevel.HIGH] > 0:
        return (
            OverallRiskLevel.HIGH,
            f"{risk_counts[OverallRiskLevel.HIGH]} high risk claim(s) detected"
        )
    elif risk_counts[OverallRiskLevel.MEDIUM] > 0:
        return (
            OverallRiskLevel.MEDIUM,
            f"{risk_counts[OverallRiskLevel.MEDIUM]} medium risk claim(s) detected"
        )
    elif risk_counts[OverallRiskLevel.LOW] > 0:
        return (
            OverallRiskLevel.LOW,
            f"All {len(claim_analyses)} claims are low risk"
        )
    else:
        return (
            OverallRiskLevel.MINIMAL,
            f"All {len(claim_analyses)} claims verified with minimal risk"
        )


# =============================================================================
# MAIN ORCHESTRATOR CLASS
# =============================================================================

class EnsembleOrchestrator:
    """
    Orchestrates the complete verification pipeline.

    Combines semantic entropy and faithfulness verification to detect
    hallucinations, especially the dangerous "confident hallucinator" pattern.

    The pipeline follows the Source-First optimization:
    1. Extract claims from response
    2. Verify ALL claims against sources (fast, cheap)
    3. Run semantic entropy ONLY on unverified/contradicted claims (expensive)
    4. Combine results to detect hallucination patterns
    5. Assign review levels

    Example:
        >>> # Setup
        >>> orchestrator = EnsembleOrchestrator(
        ...     entropy_calculator=SemanticEntropyCalculator(...),
        ...     faithfulness_verifier=FaithfulnessVerifier(...),
        ...     config=OrchestratorConfig(confident_entropy_threshold=0.2)
        ... )
        >>>
        >>> # Analyze a response
        >>> result = await orchestrator.analyze_response(
        ...     response="Patient takes Metoprolol 50mg daily",
        ...     claims=extracted_claims,
        ...     sources=patient_records
        ... )
        >>>
        >>> # Check results
        >>> if result.has_confident_hallucinator:
        ...     for claim in result.dangerous_claims:
        ...         print(f"DANGER: {claim.claim.text}")
    """

    def __init__(
        self,
        entropy_calculator: Optional[SemanticEntropyCalculator] = None,
        faithfulness_verifier: Optional[FaithfulnessVerifier] = None,
        config: Optional[OrchestratorConfig] = None,
    ):
        """
        Initialize the orchestrator.

        Args:
            entropy_calculator: Calculator for semantic entropy
            faithfulness_verifier: Verifier for source faithfulness
            config: Orchestrator configuration
        """
        self.entropy_calculator = entropy_calculator
        self.faithfulness_verifier = faithfulness_verifier
        self.config = config or OrchestratorConfig()

    async def analyze_response(
        self,
        response: str,
        claims: Optional[List[Claim]] = None,
        sources: Optional[List[SourceDocument]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> OrchestratorResult:
        """
        Run the complete verification pipeline on a response.

        This is the main entry point. It:
        1. Extracts claims (if not provided)
        2. Verifies claims against sources
        3. Runs semantic entropy where needed
        4. Detects hallucination patterns
        5. Assigns review levels

        Args:
            response: The AI-generated response text
            claims: Pre-extracted claims (optional)
            sources: Source documents to verify against
            context: Additional context for verification

        Returns:
            Complete orchestrator result with all analyses
        """
        context = context or {}

        # Step 1: Get claims
        if claims is None:
            if self.faithfulness_verifier and self.faithfulness_verifier.claim_extractor:
                claims = self.faithfulness_verifier.claim_extractor.extract(response)
            else:
                claims = []

        if not claims:
            return self._empty_result("No claims to analyze")

        # Step 2: Verify claims against sources (Source-First)
        verification_results = await self._verify_claims(claims, sources, context)

        # Step 3: Run semantic entropy where needed (with optimization)
        claim_analyses = await self._analyze_claims(
            claims, verification_results, response, context
        )

        # Step 4: Compute aggregate results
        return self._build_result(claim_analyses)

    async def analyze_claim(
        self,
        claim: Claim,
        verification: VerificationResult,
        response: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ClaimAnalysis:
        """
        Analyze a single claim with its verification result.

        Args:
            claim: The claim to analyze
            verification: Pre-computed verification result
            response: Original response text (for entropy calculation)
            context: Additional context

        Returns:
            Complete claim analysis
        """
        context = context or {}

        # Determine if we need semantic entropy
        needs_entropy = self._should_run_entropy(verification)

        # Run entropy if needed
        entropy_result: Optional[EntropyResult] = None
        if needs_entropy and self.entropy_calculator:
            # Generate query for entropy calculation
            query = f"Verify: {claim.text}"
            entropy_result = await self.entropy_calculator.calculate(
                query=query,
                context=response,
                num_samples=5,
            )

        # Detect hallucination pattern
        entropy_value = entropy_result.entropy if entropy_result else 0.0
        hallucination_type, risk_level, reasoning = detect_confident_hallucinator(
            entropy=entropy_value,
            verification_status=verification.status,
            config=self.config,
        )

        # Assign review level
        review_level = assign_review_level(risk_level, entropy_value, self.config)

        return ClaimAnalysis(
            claim=claim,
            verification=verification,
            entropy=entropy_result,
            hallucination_type=hallucination_type,
            risk_level=risk_level,
            review_level=review_level,
            is_confident_hallucinator=(
                hallucination_type == HallucinationType.CONFIDENT_HALLUCINATOR
            ),
            reasoning=reasoning,
            metadata={
                "entropy_calculated": needs_entropy,
                "entropy_skipped_reason": (
                    None if needs_entropy else "Source-First optimization"
                ),
            }
        )

    async def _verify_claims(
        self,
        claims: List[Claim],
        sources: Optional[List[SourceDocument]],
        context: Dict[str, Any],
    ) -> List[VerificationResult]:
        """Verify all claims against sources."""
        if not self.faithfulness_verifier:
            # Return "unable" for all claims if no verifier
            return [
                VerificationResult(
                    claim=claim,
                    status=VerificationStatus.UNABLE,
                    confidence=0.0,
                    explanation="No faithfulness verifier configured",
                    needs_entropy_check=True,
                )
                for claim in claims
            ]

        # Use batch verification
        batch_result = await self.faithfulness_verifier.verify_claims(
            claims=claims,
            sources=sources or [],
            context=context,
        )
        return batch_result.results

    async def _analyze_claims(
        self,
        claims: List[Claim],
        verification_results: List[VerificationResult],
        response: str,
        context: Dict[str, Any],
    ) -> List[ClaimAnalysis]:
        """Analyze all claims with Source-First optimization."""
        # Pair claims with their verification results
        pairs = list(zip(claims, verification_results))

        if self.config.parallel_entropy:
            # Analyze in parallel batches
            analyses = []
            batch_size = self.config.max_parallel_entropy

            for i in range(0, len(pairs), batch_size):
                batch = pairs[i:i + batch_size]
                batch_analyses = await asyncio.gather(*[
                    self.analyze_claim(claim, verif, response, context)
                    for claim, verif in batch
                ])
                analyses.extend(batch_analyses)

            return analyses
        else:
            # Analyze sequentially
            return [
                await self.analyze_claim(claim, verif, response, context)
                for claim, verif in pairs
            ]

    def _should_run_entropy(self, verification: VerificationResult) -> bool:
        """Determine if semantic entropy should be calculated for a claim."""
        status = verification.status

        # Skip for verified claims (Source-First optimization)
        if status == VerificationStatus.VERIFIED:
            return not self.config.skip_entropy_for_verified

        # Always run for contradicted claims
        if status == VerificationStatus.CONTRADICTED:
            return self.config.run_entropy_for_contradicted

        # Run for not found claims
        if status == VerificationStatus.NOT_FOUND:
            return self.config.run_entropy_for_not_found

        # Default: run entropy for uncertain cases
        return True

    def _build_result(self, claim_analyses: List[ClaimAnalysis]) -> OrchestratorResult:
        """Build the final orchestrator result."""
        # Count metrics
        total = len(claim_analyses)
        verified = sum(
            1 for ca in claim_analyses
            if ca.verification.status == VerificationStatus.VERIFIED
        )
        contradicted = sum(
            1 for ca in claim_analyses
            if ca.verification.status == VerificationStatus.CONTRADICTED
        )
        confident_hallucinators = sum(
            1 for ca in claim_analyses
            if ca.is_confident_hallucinator
        )
        skipped_entropy = sum(
            1 for ca in claim_analyses
            if not ca.metadata.get("entropy_calculated", True)
        )

        # Calculate compute savings
        savings = (skipped_entropy / total * 100) if total > 0 else 100.0

        # Get overall risk
        overall_risk, reasoning = combine_risk_assessments(claim_analyses)

        # Get overall review level
        if claim_analyses:
            max_entropy = max(
                (ca.entropy.entropy if ca.entropy else 0.0)
                for ca in claim_analyses
            )
        else:
            max_entropy = 0.0

        overall_review = assign_review_level(overall_risk, max_entropy, self.config)

        # Determine if review required
        requires_review = overall_review != ReviewLevel.NONE

        return OrchestratorResult(
            claim_analyses=claim_analyses,
            overall_risk=overall_risk,
            overall_review_level=overall_review,
            total_claims=total,
            verified_claims=verified,
            contradicted_claims=contradicted,
            confident_hallucinators=confident_hallucinators,
            claims_skipped_entropy=skipped_entropy,
            compute_savings_percent=savings,
            has_confident_hallucinator=confident_hallucinators > 0,
            requires_review=requires_review,
            reasoning=reasoning,
            metadata={
                "config": {
                    "confident_threshold": self.config.confident_entropy_threshold,
                    "uncertain_threshold": self.config.uncertain_entropy_threshold,
                    "source_first_enabled": self.config.skip_entropy_for_verified,
                }
            }
        )

    def _empty_result(self, reason: str) -> OrchestratorResult:
        """Return empty result for no-claims case."""
        return OrchestratorResult(
            claim_analyses=[],
            overall_risk=OverallRiskLevel.MINIMAL,
            overall_review_level=ReviewLevel.NONE,
            total_claims=0,
            verified_claims=0,
            contradicted_claims=0,
            confident_hallucinators=0,
            claims_skipped_entropy=0,
            compute_savings_percent=100.0,
            has_confident_hallucinator=False,
            requires_review=False,
            reasoning=reason,
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def quick_hallucination_check(
    claim_text: str,
    entropy: float,
    is_verified: bool,
    is_contradicted: bool,
    config: Optional[OrchestratorConfig] = None,
) -> Tuple[bool, HallucinationType, str]:
    """
    Quick check for hallucination without full orchestration.

    Useful for simple cases where you already have entropy and verification.

    Args:
        claim_text: The claim being checked
        entropy: Pre-calculated entropy value
        is_verified: Whether claim was verified against source
        is_contradicted: Whether claim contradicts source
        config: Optional config (uses defaults if not provided)

    Returns:
        (is_hallucination, hallucination_type, reasoning)
    """
    config = config or OrchestratorConfig()

    if is_contradicted:
        status = VerificationStatus.CONTRADICTED
    elif is_verified:
        status = VerificationStatus.VERIFIED
    else:
        status = VerificationStatus.NOT_FOUND

    hall_type, risk, reasoning = detect_confident_hallucinator(
        entropy=entropy,
        verification_status=status,
        config=config,
    )

    is_hallucination = hall_type in (
        HallucinationType.CONFIDENT_HALLUCINATOR,
        HallucinationType.UNCERTAIN_ERROR,
    )

    return is_hallucination, hall_type, reasoning
