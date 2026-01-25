"""
TRUST Platform - Faithfulness Verification Module
==================================================

Domain-agnostic framework for verifying AI-generated claims against
authoritative sources.

Core Concept:
    "Faithfulness" means AI outputs are grounded in and consistent with
    source material. This module provides the infrastructure for:
    1. Extracting verifiable claims from AI responses
    2. Matching claims against authoritative sources
    3. Detecting contradictions and unsupported statements

Architecture:
    The module defines abstract interfaces that domain-specific adapters
    implement. The core engine handles orchestration while adapters
    provide domain knowledge (what to extract, how to verify, which
    sources are authoritative).

    ┌─────────────────────────────────────────────────────────────┐
    │                  FaithfulnessVerifier                        │
    │  (Orchestration - domain agnostic)                          │
    ├─────────────────────────────────────────────────────────────┤
    │  Dependencies (injected by adapters):                        │
    │  ├── ClaimExtractor      → Extract claims from text         │
    │  ├── SourceProvider      → Fetch authoritative data         │
    │  └── ClaimMatcher        → Compare claims to sources        │
    └─────────────────────────────────────────────────────────────┘

Optimization Strategy (EHR-First Pattern):
    Fast source verification BEFORE expensive semantic entropy.
    Verified claims skip SE entirely → ~80% compute reduction.

Example:
    >>> verifier = FaithfulnessVerifier(
    ...     claim_extractor=healthcare_extractor,
    ...     source_provider=ehr_provider,
    ...     claim_matcher=medical_matcher
    ... )
    >>> result = await verifier.verify_response(
    ...     response="Patient takes Metoprolol 50mg daily",
    ...     sources=patient_record
    ... )
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Generic, List, Optional, Protocol, Tuple, TypeVar
import asyncio


__all__ = [
    # Enums
    'VerificationStatus',
    'ClaimCategory',
    # Data classes
    'Claim',
    'SourceDocument',
    'VerificationResult',
    'BatchVerificationResult',
    'VerificationConfig',
    # Protocols/ABCs
    'ClaimExtractor',
    'SourceProvider',
    'ClaimMatcher',
    # Main class
    'FaithfulnessVerifier',
    # Utility functions
    'compute_verification_stats',
]


# =============================================================================
# ENUMS
# =============================================================================

class VerificationStatus(str, Enum):
    """
    Result status of claim verification against sources.

    These statuses drive downstream processing decisions:
    - VERIFIED: Claim matches source → skip semantic entropy
    - CONTRADICTED: Claim conflicts with source → flag for review
    - NOT_FOUND: No matching source data → run semantic entropy
    - PARTIAL: Partial match → may need additional verification
    - UNABLE: Verification not possible → treat as NOT_FOUND
    """
    VERIFIED = "verified"           # Claim matches authoritative source
    CONTRADICTED = "contradicted"   # Claim conflicts with source (dangerous!)
    NOT_FOUND = "not_found"         # No source data to verify against
    PARTIAL = "partial"             # Partial match, incomplete verification
    UNABLE = "unable"               # Verification could not be performed

    @property
    def needs_entropy_check(self) -> bool:
        """Whether this status requires semantic entropy analysis."""
        return self in (
            VerificationStatus.CONTRADICTED,
            VerificationStatus.NOT_FOUND,
            VerificationStatus.PARTIAL,
            VerificationStatus.UNABLE,
        )

    @property
    def is_safe(self) -> bool:
        """Whether this status indicates safe, verified content."""
        return self == VerificationStatus.VERIFIED

    @property
    def is_dangerous(self) -> bool:
        """Whether this status indicates potentially dangerous content."""
        return self == VerificationStatus.CONTRADICTED


class ClaimCategory(str, Enum):
    """
    Generic claim categories for risk-based processing.

    Adapters map domain-specific claim types to these categories.
    For example, healthcare adapter might map:
    - MEDICATION → HIGH_RISK
    - DEMOGRAPHIC → LOW_RISK
    """
    HIGH_RISK = "high_risk"       # Requires strict verification
    MEDIUM_RISK = "medium_risk"   # Standard verification
    LOW_RISK = "low_risk"         # Basic verification
    INFORMATIONAL = "informational"  # No verification needed


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Claim:
    """
    A single verifiable claim extracted from AI-generated text.

    Claims are the atomic units of verification. Each claim represents
    a factual assertion that can be checked against authoritative sources.

    Attributes:
        id: Unique identifier for this claim
        text: The claim text as extracted
        category: Risk category (HIGH_RISK, MEDIUM_RISK, etc.)
        claim_type: Domain-specific type (e.g., "medication", "date")
        source_span: Character positions in original text (start, end)
        section: Which section of the document this came from
        metadata: Additional domain-specific information
    """
    id: str
    text: str
    category: ClaimCategory
    claim_type: str
    source_span: Optional[Tuple[int, int]] = None
    section: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "text": self.text,
            "category": self.category.value,
            "claim_type": self.claim_type,
            "source_span": self.source_span,
            "section": self.section,
            "metadata": self.metadata,
        }


@dataclass
class SourceDocument:
    """
    An authoritative source document for claim verification.

    Sources can be documents, database records, API responses, etc.
    The adapter's SourceProvider fetches these.

    Attributes:
        id: Unique identifier for this source
        content: The source content (text, structured data, etc.)
        source_type: Type of source (e.g., "ehr", "transcript", "document")
        authority_level: How authoritative (1.0 = gold standard)
        metadata: Additional source information (timestamps, authors, etc.)
    """
    id: str
    content: Any  # Can be text, dict, or domain-specific structure
    source_type: str
    authority_level: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationResult:
    """
    Result of verifying a single claim against sources.

    Attributes:
        claim: The claim that was verified
        status: Verification outcome (VERIFIED, CONTRADICTED, etc.)
        confidence: Confidence in the verification (0.0 to 1.0)
        matched_source: The source document that matched (if any)
        matched_text: The specific text that matched in the source
        explanation: Human-readable explanation of the result
        needs_entropy_check: Whether semantic entropy should be run
        metadata: Additional verification details
    """
    claim: Claim
    status: VerificationStatus
    confidence: float
    matched_source: Optional[SourceDocument] = None
    matched_text: Optional[str] = None
    explanation: str = ""
    needs_entropy_check: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Auto-set needs_entropy_check based on status if not explicitly set
        if self.needs_entropy_check and self.status == VerificationStatus.VERIFIED:
            self.needs_entropy_check = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "claim": self.claim.to_dict(),
            "status": self.status.value,
            "confidence": self.confidence,
            "matched_source_id": self.matched_source.id if self.matched_source else None,
            "matched_text": self.matched_text,
            "explanation": self.explanation,
            "needs_entropy_check": self.needs_entropy_check,
            "metadata": self.metadata,
        }


@dataclass
class BatchVerificationResult:
    """
    Result of verifying multiple claims (batch operation).

    Includes statistics useful for optimization tracking and dashboards.

    Attributes:
        results: Individual verification results
        total_claims: Total number of claims processed
        verified_count: Claims successfully verified
        contradicted_count: Claims that contradict sources (dangerous!)
        not_found_count: Claims with no source data
        partial_count: Claims with partial matches
        needs_entropy_count: Claims requiring semantic entropy
        verification_rate: Percentage of claims verified (0.0 to 1.0)
        compute_savings_percent: Estimated compute savings from skipping SE
    """
    results: List[VerificationResult]
    total_claims: int
    verified_count: int
    contradicted_count: int
    not_found_count: int
    partial_count: int
    needs_entropy_count: int
    verification_rate: float
    compute_savings_percent: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def flagged_claims(self) -> List[VerificationResult]:
        """Claims that need attention (contradicted or not found)."""
        return [
            r for r in self.results
            if r.status in (VerificationStatus.CONTRADICTED, VerificationStatus.NOT_FOUND)
        ]

    @property
    def dangerous_claims(self) -> List[VerificationResult]:
        """Claims that contradict authoritative sources."""
        return [
            r for r in self.results
            if r.status == VerificationStatus.CONTRADICTED
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "results": [r.to_dict() for r in self.results],
            "summary": {
                "total_claims": self.total_claims,
                "verified": self.verified_count,
                "contradicted": self.contradicted_count,
                "not_found": self.not_found_count,
                "partial": self.partial_count,
                "needs_entropy": self.needs_entropy_count,
            },
            "metrics": {
                "verification_rate": self.verification_rate,
                "compute_savings_percent": self.compute_savings_percent,
            },
            "flagged_claim_ids": [r.claim.id for r in self.flagged_claims],
            "dangerous_claim_ids": [r.claim.id for r in self.dangerous_claims],
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class VerificationConfig:
    """
    Configuration for the verification process.

    Adapters provide domain-specific configurations.

    Attributes:
        min_confidence_threshold: Minimum confidence for VERIFIED status
        partial_match_threshold: Confidence below this becomes PARTIAL
        verify_low_risk_claims: Whether to verify LOW_RISK claims
        parallel_verification: Whether to verify claims in parallel
        max_parallel_claims: Maximum claims to verify in parallel
    """
    min_confidence_threshold: float = 0.8
    partial_match_threshold: float = 0.5
    verify_low_risk_claims: bool = True
    parallel_verification: bool = True
    max_parallel_claims: int = 10


# =============================================================================
# PROTOCOLS (Interfaces for dependency injection)
# =============================================================================

class ClaimExtractor(Protocol):
    """
    Protocol for extracting claims from AI-generated text.

    Implementations are domain-specific. For example:
    - Healthcare: Extract medications, diagnoses, allergies
    - Legal: Extract citations, dates, parties
    - Finance: Extract amounts, accounts, transactions
    """

    def extract(self, text: str, section: Optional[str] = None) -> List[Claim]:
        """
        Extract verifiable claims from text.

        Args:
            text: The AI-generated text to extract claims from
            section: Optional section identifier for context

        Returns:
            List of extracted claims
        """
        ...

    def extract_from_structured(self, data: Dict[str, Any]) -> List[Claim]:
        """
        Extract claims from structured data (e.g., parsed JSON response).

        Args:
            data: Structured data containing claims

        Returns:
            List of extracted claims
        """
        ...


class SourceProvider(Protocol):
    """
    Protocol for providing authoritative source documents.

    Implementations connect to domain-specific data sources:
    - Healthcare: EHR systems (FHIR), medical databases
    - Legal: Case law databases, document management
    - Finance: Transaction systems, regulatory databases
    """

    async def get_sources(
        self,
        claim: Claim,
        context: Optional[Dict[str, Any]] = None
    ) -> List[SourceDocument]:
        """
        Fetch authoritative sources relevant to a claim.

        Args:
            claim: The claim to find sources for
            context: Additional context (e.g., patient ID, case number)

        Returns:
            List of relevant source documents
        """
        ...

    async def get_all_sources(
        self,
        context: Optional[Dict[str, Any]] = None
    ) -> List[SourceDocument]:
        """
        Fetch all available source documents for a context.

        Useful when verifying multiple claims against the same sources.

        Args:
            context: Context identifying which sources to fetch

        Returns:
            List of all available source documents
        """
        ...


class ClaimMatcher(Protocol):
    """
    Protocol for matching claims against source documents.

    Implementations provide domain-specific matching logic:
    - Healthcare: Medical terminology matching, dosage comparison
    - Legal: Citation verification, date matching
    - Finance: Amount verification, account matching
    """

    async def match(
        self,
        claim: Claim,
        sources: List[SourceDocument]
    ) -> VerificationResult:
        """
        Match a claim against source documents.

        Args:
            claim: The claim to verify
            sources: Available source documents

        Returns:
            VerificationResult with status and confidence
        """
        ...

    def compute_similarity(
        self,
        claim_text: str,
        source_text: str
    ) -> float:
        """
        Compute similarity between claim and source text.

        Args:
            claim_text: Text from the claim
            source_text: Text from the source

        Returns:
            Similarity score (0.0 to 1.0)
        """
        ...


# =============================================================================
# MAIN VERIFIER CLASS
# =============================================================================

class FaithfulnessVerifier:
    """
    Domain-agnostic faithfulness verification orchestrator.

    Coordinates claim extraction, source fetching, and claim matching.
    Domain-specific logic is injected through protocols.

    The verifier implements the "Source-First" optimization pattern:
    1. Extract all claims from the AI response
    2. Verify claims against authoritative sources (fast, cheap)
    3. Only claims that can't be verified need semantic entropy (slow, expensive)

    This pattern typically achieves 70-90% compute savings.

    Example:
        >>> # Setup with domain-specific implementations
        >>> verifier = FaithfulnessVerifier(
        ...     claim_extractor=MedicalClaimExtractor(),
        ...     source_provider=EHRSourceProvider(fhir_client),
        ...     claim_matcher=MedicalClaimMatcher()
        ... )
        >>>
        >>> # Verify a response
        >>> result = await verifier.verify_response(
        ...     response="Patient on Metoprolol 50mg BID",
        ...     context={"patient_id": "12345"}
        ... )
        >>> print(f"Verified: {result.verified_count}/{result.total_claims}")
    """

    def __init__(
        self,
        claim_extractor: Optional[ClaimExtractor] = None,
        source_provider: Optional[SourceProvider] = None,
        claim_matcher: Optional[ClaimMatcher] = None,
        config: Optional[VerificationConfig] = None,
    ):
        """
        Initialize the verifier.

        All dependencies are optional at construction time, allowing
        for flexible configuration. However, verify operations will
        fail if required dependencies are not set.

        Args:
            claim_extractor: Extracts claims from text
            source_provider: Provides authoritative sources
            claim_matcher: Matches claims to sources
            config: Verification configuration
        """
        self.claim_extractor = claim_extractor
        self.source_provider = source_provider
        self.claim_matcher = claim_matcher
        self.config = config or VerificationConfig()

    def with_extractor(self, extractor: ClaimExtractor) -> 'FaithfulnessVerifier':
        """Set claim extractor (fluent interface)."""
        self.claim_extractor = extractor
        return self

    def with_provider(self, provider: SourceProvider) -> 'FaithfulnessVerifier':
        """Set source provider (fluent interface)."""
        self.source_provider = provider
        return self

    def with_matcher(self, matcher: ClaimMatcher) -> 'FaithfulnessVerifier':
        """Set claim matcher (fluent interface)."""
        self.claim_matcher = matcher
        return self

    async def verify_response(
        self,
        response: str,
        context: Optional[Dict[str, Any]] = None,
        sources: Optional[List[SourceDocument]] = None,
    ) -> BatchVerificationResult:
        """
        Verify all claims in an AI-generated response.

        This is the main entry point for verification. It:
        1. Extracts claims from the response
        2. Fetches relevant sources (if not provided)
        3. Verifies each claim against sources
        4. Returns aggregated results with statistics

        Args:
            response: The AI-generated response text
            context: Domain-specific context (e.g., patient ID)
            sources: Pre-fetched sources (optional, will fetch if not provided)

        Returns:
            BatchVerificationResult with all claim verifications

        Raises:
            ValueError: If required dependencies not configured
        """
        self._validate_dependencies()

        # Step 1: Extract claims
        claims = self.claim_extractor.extract(response)

        if not claims:
            return BatchVerificationResult(
                results=[],
                total_claims=0,
                verified_count=0,
                contradicted_count=0,
                not_found_count=0,
                partial_count=0,
                needs_entropy_count=0,
                verification_rate=1.0,
                compute_savings_percent=100.0,
                metadata={"note": "No claims extracted from response"}
            )

        # Step 2: Fetch sources if not provided
        if sources is None:
            sources = await self.source_provider.get_all_sources(context)

        # Step 3: Verify claims
        return await self.verify_claims(claims, sources, context)

    async def verify_claims(
        self,
        claims: List[Claim],
        sources: List[SourceDocument],
        context: Optional[Dict[str, Any]] = None,
    ) -> BatchVerificationResult:
        """
        Verify a list of claims against source documents.

        Use this when you already have extracted claims.

        Args:
            claims: List of claims to verify
            sources: Source documents to verify against
            context: Additional context

        Returns:
            BatchVerificationResult with all verifications
        """
        self._validate_matcher()

        if not claims:
            return self._empty_result()

        # Filter claims based on config
        claims_to_verify = [
            c for c in claims
            if self.config.verify_low_risk_claims or c.category != ClaimCategory.LOW_RISK
        ]

        # Verify claims (parallel or sequential)
        if self.config.parallel_verification and len(claims_to_verify) > 1:
            results = await self._verify_parallel(claims_to_verify, sources)
        else:
            results = await self._verify_sequential(claims_to_verify, sources)

        # Compute statistics
        return compute_verification_stats(results)

    async def verify_claim(
        self,
        claim: Claim,
        sources: List[SourceDocument],
    ) -> VerificationResult:
        """
        Verify a single claim against source documents.

        Args:
            claim: The claim to verify
            sources: Source documents to verify against

        Returns:
            VerificationResult for this claim
        """
        self._validate_matcher()
        return await self.claim_matcher.match(claim, sources)

    async def _verify_parallel(
        self,
        claims: List[Claim],
        sources: List[SourceDocument],
    ) -> List[VerificationResult]:
        """Verify claims in parallel batches."""
        results = []
        batch_size = self.config.max_parallel_claims

        for i in range(0, len(claims), batch_size):
            batch = claims[i:i + batch_size]
            batch_results = await asyncio.gather(*[
                self.claim_matcher.match(claim, sources)
                for claim in batch
            ])
            results.extend(batch_results)

        return results

    async def _verify_sequential(
        self,
        claims: List[Claim],
        sources: List[SourceDocument],
    ) -> List[VerificationResult]:
        """Verify claims sequentially."""
        return [
            await self.claim_matcher.match(claim, sources)
            for claim in claims
        ]

    def _validate_dependencies(self) -> None:
        """Validate all required dependencies are set."""
        if self.claim_extractor is None:
            raise ValueError("claim_extractor must be configured")
        if self.source_provider is None:
            raise ValueError("source_provider must be configured")
        if self.claim_matcher is None:
            raise ValueError("claim_matcher must be configured")

    def _validate_matcher(self) -> None:
        """Validate claim matcher is set."""
        if self.claim_matcher is None:
            raise ValueError("claim_matcher must be configured")

    def _empty_result(self) -> BatchVerificationResult:
        """Return empty result for no claims case."""
        return BatchVerificationResult(
            results=[],
            total_claims=0,
            verified_count=0,
            contradicted_count=0,
            not_found_count=0,
            partial_count=0,
            needs_entropy_count=0,
            verification_rate=1.0,
            compute_savings_percent=100.0,
        )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def compute_verification_stats(results: List[VerificationResult]) -> BatchVerificationResult:
    """
    Compute verification statistics from a list of results.

    Args:
        results: List of verification results

    Returns:
        BatchVerificationResult with computed statistics
    """
    if not results:
        return BatchVerificationResult(
            results=[],
            total_claims=0,
            verified_count=0,
            contradicted_count=0,
            not_found_count=0,
            partial_count=0,
            needs_entropy_count=0,
            verification_rate=1.0,
            compute_savings_percent=100.0,
        )

    total = len(results)
    verified = sum(1 for r in results if r.status == VerificationStatus.VERIFIED)
    contradicted = sum(1 for r in results if r.status == VerificationStatus.CONTRADICTED)
    not_found = sum(1 for r in results if r.status == VerificationStatus.NOT_FOUND)
    partial = sum(1 for r in results if r.status == VerificationStatus.PARTIAL)
    needs_entropy = sum(1 for r in results if r.needs_entropy_check)

    verification_rate = verified / total if total > 0 else 0.0
    # Compute savings: claims that don't need SE / total claims
    compute_savings = ((total - needs_entropy) / total * 100) if total > 0 else 100.0

    return BatchVerificationResult(
        results=results,
        total_claims=total,
        verified_count=verified,
        contradicted_count=contradicted,
        not_found_count=not_found,
        partial_count=partial,
        needs_entropy_count=needs_entropy,
        verification_rate=verification_rate,
        compute_savings_percent=compute_savings,
    )


# =============================================================================
# SIMPLE IMPLEMENTATIONS (for testing/examples)
# =============================================================================

class SimpleTextMatcher:
    """
    Simple text-based claim matcher for testing.

    Uses basic string matching. Real implementations should use
    domain-specific matching logic (NLP, ontologies, etc.).
    """

    def __init__(
        self,
        case_sensitive: bool = False,
        partial_threshold: float = 0.5,
    ):
        self.case_sensitive = case_sensitive
        self.partial_threshold = partial_threshold

    async def match(
        self,
        claim: Claim,
        sources: List[SourceDocument],
    ) -> VerificationResult:
        """Match claim using simple text search."""
        claim_text = claim.text if self.case_sensitive else claim.text.lower()

        best_match: Optional[Tuple[SourceDocument, str, float]] = None

        for source in sources:
            source_text = str(source.content)
            if not self.case_sensitive:
                source_text = source_text.lower()

            similarity = self.compute_similarity(claim_text, source_text)

            if best_match is None or similarity > best_match[2]:
                best_match = (source, source_text, similarity)

        if best_match is None or best_match[2] < self.partial_threshold:
            return VerificationResult(
                claim=claim,
                status=VerificationStatus.NOT_FOUND,
                confidence=0.0,
                explanation="No matching source found",
                needs_entropy_check=True,
            )

        source, source_text, confidence = best_match

        if confidence >= 0.8:
            return VerificationResult(
                claim=claim,
                status=VerificationStatus.VERIFIED,
                confidence=confidence,
                matched_source=source,
                matched_text=source_text[:200],  # Truncate for storage
                explanation=f"Claim verified with {confidence:.0%} confidence",
                needs_entropy_check=False,
            )
        else:
            return VerificationResult(
                claim=claim,
                status=VerificationStatus.PARTIAL,
                confidence=confidence,
                matched_source=source,
                matched_text=source_text[:200],
                explanation=f"Partial match with {confidence:.0%} confidence",
                needs_entropy_check=True,
            )

    def compute_similarity(self, claim_text: str, source_text: str) -> float:
        """Simple substring-based similarity."""
        if claim_text in source_text:
            return 1.0

        # Check word overlap
        claim_words = set(claim_text.split())
        source_words = set(source_text.split())

        if not claim_words:
            return 0.0

        overlap = len(claim_words & source_words)
        return overlap / len(claim_words)
