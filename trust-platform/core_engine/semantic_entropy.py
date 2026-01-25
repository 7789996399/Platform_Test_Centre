"""
TRUST Platform - Semantic Entropy Module
=========================================

Domain-agnostic implementation of semantic entropy for uncertainty quantification.

Based on Farquhar et al. (Nature, 2024): "Detecting hallucinations in large
language models using semantic entropy."

Core Insight:
    When an LLM is uncertain, it generates semantically different responses
    to the same prompt. High entropy over semantic clusters indicates
    uncertainty and potential hallucination risk.

Algorithm:
    1. Generate N responses to the same prompt
    2. Cluster responses by bidirectional entailment (semantic equivalence)
    3. Calculate Shannon entropy over cluster distribution
    4. High entropy = high uncertainty = elevated risk

This module is domain-agnostic. Risk thresholds and interpretation
are provided by industry-specific adapters.

Example:
    >>> calculator = SemanticEntropyCalculator()
    >>> result = await calculator.calculate(
    ...     query="What medication is the patient taking?",
    ...     context="Patient reports taking metoprolol.",
    ...     num_samples=5
    ... )
    >>> print(f"Entropy: {result.entropy}, Risk: {result.risk_level}")
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple
import math


__all__ = [
    'EntropyRiskLevel',
    'EntropyResult',
    'EntropyThresholds',
    'SemanticEntropyCalculator',
    'EntailmentChecker',
    'ResponseGenerator',
    'ClusteringStrategy',
    'ExactMatchClustering',
    'EntailmentClustering',
    'calculate_shannon_entropy',
    'normalize_entropy',
]


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class EntropyRiskLevel(str, Enum):
    """
    Risk level based on semantic entropy.

    These are generic levels - adapters map them to domain-specific actions.
    """
    LOW = "low"          # Consistent responses, low uncertainty
    MEDIUM = "medium"    # Some variation, moderate uncertainty
    HIGH = "high"        # Significant variation, high uncertainty

    @classmethod
    def from_entropy(
        cls,
        entropy: float,
        thresholds: 'EntropyThresholds'
    ) -> 'EntropyRiskLevel':
        """Classify risk level based on entropy value and thresholds."""
        if entropy < thresholds.low_medium_boundary:
            return cls.LOW
        elif entropy < thresholds.medium_high_boundary:
            return cls.MEDIUM
        else:
            return cls.HIGH


@dataclass(frozen=True)
class EntropyThresholds:
    """
    Configurable thresholds for entropy-based risk classification.

    Adapters provide domain-specific thresholds. For example:
    - Healthcare: stricter thresholds (0.2, 0.5)
    - General Q&A: relaxed thresholds (0.5, 1.0)

    Attributes:
        low_medium_boundary: Entropy below this is LOW risk
        medium_high_boundary: Entropy above this is HIGH risk
    """
    low_medium_boundary: float = 0.5
    medium_high_boundary: float = 1.0

    def __post_init__(self):
        if self.low_medium_boundary >= self.medium_high_boundary:
            raise ValueError(
                f"low_medium_boundary ({self.low_medium_boundary}) must be less than "
                f"medium_high_boundary ({self.medium_high_boundary})"
            )
        if self.low_medium_boundary < 0:
            raise ValueError("Thresholds must be non-negative")


@dataclass
class EntropyResult:
    """
    Result of semantic entropy calculation.

    Attributes:
        entropy: Raw Shannon entropy value (0 = all same, higher = more varied)
        normalized_entropy: Entropy normalized to [0, 1] range
        num_clusters: Number of semantically distinct response groups
        num_samples: Total number of responses generated
        cluster_sizes: Number of responses in each cluster
        cluster_representatives: One representative response per cluster
        risk_level: Classified risk based on entropy and thresholds
        samples: All generated response samples (optional, for debugging)
        metadata: Additional information from the calculation
    """
    entropy: float
    normalized_entropy: float
    num_clusters: int
    num_samples: int
    cluster_sizes: List[int]
    cluster_representatives: List[str]
    risk_level: EntropyRiskLevel
    samples: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_confident(self) -> bool:
        """True if model shows high confidence (low entropy)."""
        return self.risk_level == EntropyRiskLevel.LOW

    @property
    def is_uncertain(self) -> bool:
        """True if model shows significant uncertainty (high entropy)."""
        return self.risk_level == EntropyRiskLevel.HIGH

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "entropy": self.entropy,
            "normalized_entropy": self.normalized_entropy,
            "num_clusters": self.num_clusters,
            "num_samples": self.num_samples,
            "cluster_sizes": self.cluster_sizes,
            "cluster_representatives": self.cluster_representatives,
            "risk_level": self.risk_level.value,
            "is_confident": self.is_confident,
            "is_uncertain": self.is_uncertain,
            "metadata": self.metadata,
        }


# =============================================================================
# PROTOCOLS (Interfaces for dependency injection)
# =============================================================================

class ResponseGenerator(Protocol):
    """
    Protocol for generating multiple LLM responses.

    Implementations can use any LLM provider (OpenAI, Anthropic, local models).
    """

    async def generate(
        self,
        prompt: str,
        context: str,
        num_samples: int,
        temperature: float = 1.0,
        **kwargs
    ) -> List[str]:
        """
        Generate multiple responses to the same prompt.

        Args:
            prompt: The question or statement to respond to
            context: Additional context for the response
            num_samples: Number of responses to generate
            temperature: Sampling temperature (higher = more varied)
            **kwargs: Provider-specific parameters

        Returns:
            List of response strings
        """
        ...


class EntailmentChecker(Protocol):
    """
    Protocol for checking semantic entailment between texts.

    Implementations can use NLI models (DeBERTa, RoBERTa) or LLM-based checking.
    """

    async def check_entailment(
        self,
        premise: str,
        hypothesis: str
    ) -> bool:
        """
        Check if premise entails hypothesis.

        Args:
            premise: The premise text
            hypothesis: The hypothesis text

        Returns:
            True if premise entails hypothesis
        """
        ...

    async def check_bidirectional_entailment(
        self,
        text_a: str,
        text_b: str
    ) -> bool:
        """
        Check if two texts are semantically equivalent.

        Bidirectional entailment: A entails B AND B entails A.
        This is the key insight from Farquhar et al.

        Args:
            text_a: First text
            text_b: Second text

        Returns:
            True if texts are semantically equivalent
        """
        ...


# =============================================================================
# CLUSTERING STRATEGIES
# =============================================================================

class ClusteringStrategy(ABC):
    """Abstract base class for response clustering strategies."""

    @abstractmethod
    async def cluster(self, responses: List[str]) -> List[List[int]]:
        """
        Cluster responses by semantic similarity.

        Args:
            responses: List of response strings

        Returns:
            List of clusters, where each cluster is a list of response indices
        """
        pass


class ExactMatchClustering(ClusteringStrategy):
    """
    Simple clustering by normalized string matching.

    Fast but less accurate. Use for testing or when NLI models unavailable.
    """

    def __init__(self, case_sensitive: bool = False, strip_whitespace: bool = True):
        self.case_sensitive = case_sensitive
        self.strip_whitespace = strip_whitespace

    def _normalize(self, text: str) -> str:
        """Normalize text for comparison."""
        if self.strip_whitespace:
            text = ' '.join(text.split())
        if not self.case_sensitive:
            text = text.lower()
        return text

    async def cluster(self, responses: List[str]) -> List[List[int]]:
        """Cluster by exact normalized string match."""
        if not responses:
            return []

        clusters: List[List[int]] = []
        seen: Dict[str, int] = {}

        for i, response in enumerate(responses):
            normalized = self._normalize(response)
            if normalized in seen:
                clusters[seen[normalized]].append(i)
            else:
                seen[normalized] = len(clusters)
                clusters.append([i])

        return clusters


class EntailmentClustering(ClusteringStrategy):
    """
    Clustering by bidirectional entailment using NLI.

    This is the method from Farquhar et al. - more accurate but requires
    an NLI model or LLM-based entailment checking.

    Algorithm (greedy):
        1. First response starts first cluster
        2. For each new response:
           - Check bidirectional entailment with each cluster representative
           - If entails any, add to that cluster
           - Otherwise, create new cluster
    """

    def __init__(self, entailment_checker: EntailmentChecker):
        self.entailment_checker = entailment_checker

    async def cluster(self, responses: List[str]) -> List[List[int]]:
        """Cluster by bidirectional entailment."""
        if not responses:
            return []

        # First response starts first cluster
        clusters: List[List[int]] = [[0]]

        for i in range(1, len(responses)):
            found_cluster = False

            for cluster in clusters:
                # Compare with first response in cluster (representative)
                representative_idx = cluster[0]

                is_equivalent = await self.entailment_checker.check_bidirectional_entailment(
                    responses[representative_idx],
                    responses[i]
                )

                if is_equivalent:
                    cluster.append(i)
                    found_cluster = True
                    break

            if not found_cluster:
                clusters.append([i])

        return clusters


# =============================================================================
# CORE ENTROPY FUNCTIONS
# =============================================================================

def calculate_shannon_entropy(cluster_sizes: List[int], total: int) -> float:
    """
    Calculate Shannon entropy over cluster distribution.

    Formula: H = -Σ p(x) * log2(p(x))

    Args:
        cluster_sizes: Number of items in each cluster
        total: Total number of items

    Returns:
        Shannon entropy value (bits)

    Examples:
        - All in one cluster: entropy = 0 (no uncertainty)
        - Uniform distribution: entropy = log2(n) (maximum uncertainty)

    >>> calculate_shannon_entropy([5], 5)  # All same
    0.0
    >>> calculate_shannon_entropy([1, 1, 1, 1, 1], 5)  # All different
    2.321928094887362
    """
    if total == 0:
        return 0.0

    entropy = 0.0
    for size in cluster_sizes:
        if size > 0:
            p = size / total
            entropy -= p * math.log2(p)

    return entropy


def normalize_entropy(entropy: float, num_samples: int) -> float:
    """
    Normalize entropy to [0, 1] range.

    Normalized by maximum possible entropy for given sample size.
    Maximum entropy occurs when all samples are in separate clusters.

    Args:
        entropy: Raw Shannon entropy
        num_samples: Number of samples

    Returns:
        Normalized entropy in [0, 1]
    """
    if num_samples <= 1:
        return 0.0

    max_entropy = math.log2(num_samples)
    if max_entropy == 0:
        return 0.0

    return min(1.0, entropy / max_entropy)


# =============================================================================
# MAIN CALCULATOR CLASS
# =============================================================================

class SemanticEntropyCalculator:
    """
    Domain-agnostic semantic entropy calculator.

    Calculates uncertainty in LLM responses using the semantic entropy method.
    This class is stateless and can be configured with different:
    - Response generators (OpenAI, Anthropic, local models)
    - Clustering strategies (exact match, NLI-based)
    - Risk thresholds (via adapters)

    Example:
        >>> # Setup with dependencies
        >>> generator = OpenAIResponseGenerator(api_key="...")
        >>> checker = DeBERTaEntailmentChecker()
        >>> clustering = EntailmentClustering(checker)
        >>> calculator = SemanticEntropyCalculator(
        ...     response_generator=generator,
        ...     clustering_strategy=clustering
        ... )
        >>>
        >>> # Calculate entropy
        >>> result = await calculator.calculate(
        ...     query="What is the dosage?",
        ...     context="Doctor prescribed 50mg daily.",
        ...     num_samples=5,
        ...     thresholds=EntropyThresholds(0.3, 0.7)  # Strict thresholds
        ... )
    """

    def __init__(
        self,
        response_generator: Optional[ResponseGenerator] = None,
        clustering_strategy: Optional[ClusteringStrategy] = None,
        default_thresholds: Optional[EntropyThresholds] = None,
    ):
        """
        Initialize the calculator.

        Args:
            response_generator: Generator for LLM responses (required for calculate())
            clustering_strategy: Strategy for clustering responses (default: ExactMatch)
            default_thresholds: Default risk thresholds (can be overridden per-call)
        """
        self.response_generator = response_generator
        self.clustering_strategy = clustering_strategy or ExactMatchClustering()
        self.default_thresholds = default_thresholds or EntropyThresholds()

    async def calculate(
        self,
        query: str,
        context: str,
        num_samples: int = 5,
        temperature: float = 1.0,
        thresholds: Optional[EntropyThresholds] = None,
        include_samples: bool = False,
        **generator_kwargs
    ) -> EntropyResult:
        """
        Calculate semantic entropy for a query.

        Generates multiple responses, clusters them by semantic similarity,
        and calculates entropy over the cluster distribution.

        Args:
            query: The question/statement to evaluate
            context: Context for generating responses
            num_samples: Number of responses to generate (default: 5)
            temperature: Sampling temperature (default: 1.0)
            thresholds: Risk classification thresholds (optional)
            include_samples: Include raw samples in result (default: False)
            **generator_kwargs: Additional args for response generator

        Returns:
            EntropyResult with entropy value and risk classification

        Raises:
            ValueError: If response_generator not configured
        """
        if self.response_generator is None:
            raise ValueError(
                "response_generator must be configured to use calculate(). "
                "Use calculate_from_responses() if you already have responses."
            )

        # Generate responses
        responses = await self.response_generator.generate(
            prompt=query,
            context=context,
            num_samples=num_samples,
            temperature=temperature,
            **generator_kwargs
        )

        return await self.calculate_from_responses(
            responses=responses,
            thresholds=thresholds,
            include_samples=include_samples,
        )

    async def calculate_from_responses(
        self,
        responses: List[str],
        thresholds: Optional[EntropyThresholds] = None,
        include_samples: bool = False,
    ) -> EntropyResult:
        """
        Calculate semantic entropy from pre-generated responses.

        Use this when you already have responses and don't need to generate them.

        Args:
            responses: List of LLM responses to analyze
            thresholds: Risk classification thresholds (optional)
            include_samples: Include raw samples in result (default: False)

        Returns:
            EntropyResult with entropy value and risk classification
        """
        thresholds = thresholds or self.default_thresholds

        if not responses:
            return EntropyResult(
                entropy=0.0,
                normalized_entropy=0.0,
                num_clusters=0,
                num_samples=0,
                cluster_sizes=[],
                cluster_representatives=[],
                risk_level=EntropyRiskLevel.LOW,
                samples=[],
                metadata={"error": "No responses provided"}
            )

        # Cluster responses
        clusters = await self.clustering_strategy.cluster(responses)

        # Calculate entropy
        cluster_sizes = [len(c) for c in clusters]
        num_samples = len(responses)
        num_clusters = len(clusters)

        entropy = calculate_shannon_entropy(cluster_sizes, num_samples)
        normalized = normalize_entropy(entropy, num_samples)

        # Get cluster representatives
        representatives = [responses[cluster[0]] for cluster in clusters]

        # Classify risk
        risk_level = EntropyRiskLevel.from_entropy(entropy, thresholds)

        return EntropyResult(
            entropy=entropy,
            normalized_entropy=normalized,
            num_clusters=num_clusters,
            num_samples=num_samples,
            cluster_sizes=cluster_sizes,
            cluster_representatives=representatives,
            risk_level=risk_level,
            samples=responses if include_samples else [],
            metadata={
                "thresholds": {
                    "low_medium": thresholds.low_medium_boundary,
                    "medium_high": thresholds.medium_high_boundary,
                },
                "clustering_strategy": self.clustering_strategy.__class__.__name__,
            }
        )

    def calculate_from_clusters(
        self,
        clusters: List[List[int]],
        responses: List[str],
        thresholds: Optional[EntropyThresholds] = None,
        include_samples: bool = False,
    ) -> EntropyResult:
        """
        Calculate entropy from pre-computed clusters.

        Use this when you have already clustered responses externally.

        Args:
            clusters: Pre-computed clusters (list of index lists)
            responses: Original responses (for representatives)
            thresholds: Risk classification thresholds
            include_samples: Include raw samples in result

        Returns:
            EntropyResult with entropy value and risk classification
        """
        thresholds = thresholds or self.default_thresholds

        cluster_sizes = [len(c) for c in clusters]
        num_samples = len(responses)
        num_clusters = len(clusters)

        entropy = calculate_shannon_entropy(cluster_sizes, num_samples)
        normalized = normalize_entropy(entropy, num_samples)

        representatives = [
            responses[cluster[0]] if cluster else ""
            for cluster in clusters
        ]

        risk_level = EntropyRiskLevel.from_entropy(entropy, thresholds)

        return EntropyResult(
            entropy=entropy,
            normalized_entropy=normalized,
            num_clusters=num_clusters,
            num_samples=num_samples,
            cluster_sizes=cluster_sizes,
            cluster_representatives=representatives,
            risk_level=risk_level,
            samples=responses if include_samples else [],
            metadata={
                "thresholds": {
                    "low_medium": thresholds.low_medium_boundary,
                    "medium_high": thresholds.medium_high_boundary,
                },
                "clustering_strategy": "external",
            }
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def calculate_semantic_entropy(
    responses: List[str],
    clustering_strategy: Optional[ClusteringStrategy] = None,
    thresholds: Optional[EntropyThresholds] = None,
) -> EntropyResult:
    """
    Convenience function for one-off entropy calculation.

    Args:
        responses: List of LLM responses
        clustering_strategy: Clustering method (default: ExactMatch)
        thresholds: Risk thresholds (default: standard)

    Returns:
        EntropyResult

    Example:
        >>> responses = ["The dose is 50mg", "50mg daily", "The dose is 50mg"]
        >>> result = await calculate_semantic_entropy(responses)
        >>> print(f"Entropy: {result.entropy}")
    """
    calculator = SemanticEntropyCalculator(
        clustering_strategy=clustering_strategy,
        default_thresholds=thresholds,
    )
    return await calculator.calculate_from_responses(responses)
