"""
Base Adapter Module

Defines the abstract interface that all industry adapters must implement.
This enables the TRUST core engine to work across different domains while
allowing domain-specific customization of verification logic, risk thresholds,
and compliance requirements.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class RiskLevel(Enum):
    """Standardized risk levels for AI response classification."""
    CRITICAL = "critical"    # Immediate human review required
    HIGH = "high"            # Elevated scrutiny, potential escalation
    MEDIUM = "medium"        # Standard verification workflow
    LOW = "low"              # Minimal verification needed
    MINIMAL = "minimal"      # Pass-through with logging only


@dataclass
class VerificationResult:
    """Result of a verification check performed by an adapter."""
    passed: bool
    confidence: float  # 0.0 to 1.0
    risk_level: RiskLevel
    details: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    source_citations: List[str] = field(default_factory=list)


@dataclass
class AdapterConfig:
    """Configuration for an industry adapter."""
    industry_name: str
    version: str
    entropy_threshold: float = 0.3  # Semantic entropy threshold
    faithfulness_threshold: float = 0.8  # Minimum faithfulness score
    ensemble_agreement_threshold: float = 0.7  # Model consensus threshold
    max_risk_level_auto_approve: RiskLevel = RiskLevel.LOW
    custom_settings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Claim:
    """A single claim extracted from an AI response."""
    text: str
    source_span: Tuple[int, int]  # Character offsets in original response
    claim_type: str  # Domain-specific claim type
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationContext:
    """Context provided to verification methods."""
    query: str
    response: str
    claims: List[Claim]
    source_documents: List[Dict[str, Any]]
    session_metadata: Dict[str, Any] = field(default_factory=dict)


class IndustryAdapter(ABC):
    """
    Abstract base class for industry-specific adapters.

    Each industry adapter customizes the TRUST verification pipeline for
    domain-specific requirements including:
    - Risk threshold configuration
    - Domain terminology and ontologies
    - Regulatory compliance rules
    - Custom verification workflows
    - Escalation policies

    Implementations must provide all abstract methods to integrate with
    the core TRUST engine.

    Example:
        class HealthcareAdapter(IndustryAdapter):
            def __init__(self):
                config = AdapterConfig(
                    industry_name="healthcare",
                    version="1.0.0",
                    entropy_threshold=0.2,  # Stricter for medical
                    faithfulness_threshold=0.9,
                )
                super().__init__(config)

            def classify_risk(self, context: VerificationContext) -> RiskLevel:
                # Healthcare-specific risk classification
                ...
    """

    def __init__(self, config: AdapterConfig):
        """
        Initialize the adapter with configuration.

        Args:
            config: Adapter configuration including thresholds and settings
        """
        self._config = config
        self._initialized = False

    @property
    def config(self) -> AdapterConfig:
        """Get the adapter configuration."""
        return self._config

    @property
    def industry_name(self) -> str:
        """Get the industry name this adapter serves."""
        return self._config.industry_name

    # =========================================================================
    # Lifecycle Methods
    # =========================================================================

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize adapter resources (e.g., load ontologies, connect to DBs).

        Called once before the adapter is used. Implementations should:
        - Load domain-specific knowledge bases
        - Initialize external service connections
        - Validate configuration

        Raises:
            AdapterInitializationError: If initialization fails
        """
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """
        Clean up adapter resources.

        Called when the adapter is being decommissioned. Implementations should:
        - Close database connections
        - Flush any pending logs
        - Release external resources
        """
        pass

    # =========================================================================
    # Risk Assessment Methods
    # =========================================================================

    @abstractmethod
    def classify_risk(self, context: VerificationContext) -> RiskLevel:
        """
        Classify the risk level of a response in domain context.

        This method applies domain-specific logic to determine how risky
        an AI response is. For example, healthcare might classify any
        response mentioning dosages as HIGH risk.

        Args:
            context: The verification context including query and response

        Returns:
            The assessed risk level for this response
        """
        pass

    @abstractmethod
    def get_risk_thresholds(self) -> Dict[str, float]:
        """
        Get domain-specific risk thresholds.

        Returns a mapping of threshold names to values that the core engine
        uses to make verification decisions.

        Returns:
            Dictionary mapping threshold names to float values
            Expected keys: 'entropy', 'faithfulness', 'ensemble_agreement'
        """
        pass

    # =========================================================================
    # Claim Processing Methods
    # =========================================================================

    @abstractmethod
    def extract_claims(self, response: str) -> List[Claim]:
        """
        Extract verifiable claims from an AI response.

        Domain-specific claim extraction identifies statements that should
        be verified. For healthcare, this might identify drug names, dosages,
        and medical assertions.

        Args:
            response: The AI-generated response text

        Returns:
            List of extracted claims with their positions and types
        """
        pass

    @abstractmethod
    def classify_claim(self, claim: Claim) -> str:
        """
        Classify a claim into a domain-specific category.

        Claim categories help determine which verification strategies to use.

        Args:
            claim: The claim to classify

        Returns:
            Domain-specific claim category string
        """
        pass

    # =========================================================================
    # Verification Methods
    # =========================================================================

    @abstractmethod
    async def verify_claim(
        self,
        claim: Claim,
        context: VerificationContext
    ) -> VerificationResult:
        """
        Verify a single claim against domain knowledge.

        Performs domain-specific verification including:
        - Checking against authoritative sources
        - Validating terminology usage
        - Ensuring regulatory compliance

        Args:
            claim: The claim to verify
            context: Full verification context

        Returns:
            Verification result with confidence and details
        """
        pass

    @abstractmethod
    async def verify_response(
        self,
        context: VerificationContext
    ) -> VerificationResult:
        """
        Verify an entire AI response.

        Performs holistic verification of the response including:
        - Individual claim verification
        - Cross-claim consistency checking
        - Overall coherence assessment

        Args:
            context: The verification context

        Returns:
            Aggregate verification result
        """
        pass

    # =========================================================================
    # Source Attribution Methods
    # =========================================================================

    @abstractmethod
    def get_authoritative_sources(self, claim: Claim) -> List[str]:
        """
        Get list of authoritative sources for verifying a claim.

        Returns domain-specific sources that should be consulted when
        verifying claims of this type.

        Args:
            claim: The claim needing verification

        Returns:
            List of source identifiers or URIs
        """
        pass

    @abstractmethod
    def validate_source(self, source_id: str) -> bool:
        """
        Validate that a source is authoritative for this domain.

        Args:
            source_id: Identifier of the source to validate

        Returns:
            True if the source is authoritative, False otherwise
        """
        pass

    # =========================================================================
    # Compliance Methods
    # =========================================================================

    @abstractmethod
    def get_compliance_requirements(self) -> List[str]:
        """
        Get regulatory compliance requirements for this domain.

        Returns:
            List of compliance requirement identifiers (e.g., 'HIPAA', 'SOX')
        """
        pass

    @abstractmethod
    async def check_compliance(
        self,
        context: VerificationContext
    ) -> VerificationResult:
        """
        Check response against domain compliance requirements.

        Verifies the response meets all regulatory and policy requirements
        for the domain.

        Args:
            context: The verification context

        Returns:
            Compliance verification result
        """
        pass

    # =========================================================================
    # Escalation Methods
    # =========================================================================

    @abstractmethod
    def should_escalate(
        self,
        verification_result: VerificationResult,
        context: VerificationContext
    ) -> bool:
        """
        Determine if a response should be escalated for human review.

        Domain-specific logic determines escalation based on risk level,
        verification confidence, and regulatory requirements.

        Args:
            verification_result: Result of verification checks
            context: The verification context

        Returns:
            True if human review is required
        """
        pass

    @abstractmethod
    def get_escalation_reason(
        self,
        verification_result: VerificationResult,
        context: VerificationContext
    ) -> str:
        """
        Get human-readable explanation for escalation.

        Args:
            verification_result: Result that triggered escalation
            context: The verification context

        Returns:
            Explanation string for reviewers
        """
        pass

    # =========================================================================
    # Utility Methods (with default implementations)
    # =========================================================================

    def get_domain_terminology(self) -> Dict[str, str]:
        """
        Get domain-specific terminology mappings.

        Returns:
            Dictionary mapping terms to their definitions
        """
        return {}

    def normalize_response(self, response: str) -> str:
        """
        Normalize response text for consistent processing.

        Default implementation returns response unchanged.
        Override for domain-specific normalization.

        Args:
            response: Raw response text

        Returns:
            Normalized response text
        """
        return response

    def get_adapter_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about this adapter for logging/monitoring.

        Returns:
            Dictionary containing adapter metadata
        """
        return {
            "industry": self._config.industry_name,
            "version": self._config.version,
            "entropy_threshold": self._config.entropy_threshold,
            "faithfulness_threshold": self._config.faithfulness_threshold,
            "ensemble_agreement_threshold": self._config.ensemble_agreement_threshold,
        }
