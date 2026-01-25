"""
TRUST Core Engine - Externalized Metacognitive Core for AI Governance

This module provides the core verification and validation components:
- Semantic Entropy: Uncertainty quantification through response sampling
- Faithfulness: Source attribution and claim verification
- Ensemble Orchestrator: Multi-model coordination and consensus
- Expert Routing: Domain-specific model selection
"""

from core_engine.semantic_entropy import (
    EntropyRiskLevel,
    EntropyResult,
    EntropyThresholds,
    SemanticEntropyCalculator,
    EntailmentChecker,
    ResponseGenerator,
    ClusteringStrategy,
    ExactMatchClustering,
    EntailmentClustering,
    calculate_shannon_entropy,
    normalize_entropy,
    calculate_semantic_entropy,
)

from core_engine.faithfulness import (
    VerificationStatus,
    ClaimCategory,
    Claim,
    SourceDocument,
    VerificationResult,
    BatchVerificationResult,
    VerificationConfig,
    ClaimExtractor,
    SourceProvider,
    ClaimMatcher,
    FaithfulnessVerifier,
    compute_verification_stats,
    SimpleTextMatcher,
)

from core_engine.ensemble_orchestrator import (
    OverallRiskLevel,
    ReviewLevel,
    HallucinationType,
    ClaimAnalysis,
    OrchestratorResult,
    OrchestratorConfig,
    EnsembleOrchestrator,
    detect_confident_hallucinator,
    assign_review_level,
    combine_risk_assessments,
    quick_hallucination_check,
)

__all__ = [
    # Semantic Entropy
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
    'calculate_semantic_entropy',
    # Faithfulness
    'VerificationStatus',
    'ClaimCategory',
    'Claim',
    'SourceDocument',
    'VerificationResult',
    'BatchVerificationResult',
    'VerificationConfig',
    'ClaimExtractor',
    'SourceProvider',
    'ClaimMatcher',
    'FaithfulnessVerifier',
    'compute_verification_stats',
    'SimpleTextMatcher',
    # Ensemble Orchestrator
    'OverallRiskLevel',
    'ReviewLevel',
    'HallucinationType',
    'ClaimAnalysis',
    'OrchestratorResult',
    'OrchestratorConfig',
    'EnsembleOrchestrator',
    'detect_confident_hallucinator',
    'assign_review_level',
    'combine_risk_assessments',
    'quick_hallucination_check',
]
