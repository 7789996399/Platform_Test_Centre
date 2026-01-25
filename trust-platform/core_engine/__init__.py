"""
TRUST Core Engine - Externalized Metacognitive Core for AI Governance

This module provides the core verification and validation components:
- Semantic Entropy: Uncertainty quantification through response sampling
- Faithfulness: Source attribution and claim verification
- Ensemble Orchestrator: Multi-model coordination and consensus
- Conformal Calibrator: Distribution-free uncertainty quantification
- Drift Monitor: Distribution drift detection for calibration validity
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

from core_engine.conformal_calibrator import (
    ConformalCalibrator,
    AdaptivePredictionSets,
    ConformizedQuantileRegression,
    PredictionSet,
    PredictionInterval,
    CalibrationResult,
)

from core_engine.drift_monitor import (
    DriftMonitor,
    CoverageMonitor,
    FeatureDriftDetector,
    DriftResult,
    DriftType,
    DriftSeverity,
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
    # Conformal Calibrator
    'ConformalCalibrator',
    'AdaptivePredictionSets',
    'ConformizedQuantileRegression',
    'PredictionSet',
    'PredictionInterval',
    'CalibrationResult',
    # Drift Monitor
    'DriftMonitor',
    'CoverageMonitor',
    'FeatureDriftDetector',
    'DriftResult',
    'DriftType',
    'DriftSeverity',
]
