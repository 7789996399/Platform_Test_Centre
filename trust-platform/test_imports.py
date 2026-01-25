#!/usr/bin/env python3
"""
TRUST Platform - Import Test Script

Verifies that all modules can be imported correctly.
Run from the trust-platform directory:
    python test_imports.py
"""

import sys
from pathlib import Path

# Add trust-platform to path
trust_platform_dir = Path(__file__).parent
sys.path.insert(0, str(trust_platform_dir))


def test_core_engine_imports():
    """Test core engine module imports."""
    print("Testing core_engine imports...")

    # Test semantic_entropy
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
    print("  ✓ core_engine.semantic_entropy")

    # Test faithfulness
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
    print("  ✓ core_engine.faithfulness")

    # Test ensemble_orchestrator
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
    print("  ✓ core_engine.ensemble_orchestrator")

    # Test package-level imports
    from core_engine import (
        SemanticEntropyCalculator,
        FaithfulnessVerifier,
        EnsembleOrchestrator,
    )
    print("  ✓ core_engine (package)")

    print("✓ All core_engine imports successful!\n")


def test_adapter_base_imports():
    """Test base adapter imports."""
    print("Testing adapters.base_adapter imports...")

    from adapters.base_adapter import (
        IndustryAdapter,
        AdapterConfig,
        Claim,
        RiskLevel,
        VerificationContext,
        VerificationResult,
    )
    print("  ✓ adapters.base_adapter")
    print("✓ Base adapter imports successful!\n")


def test_healthcare_adapter_imports():
    """Test healthcare adapter imports."""
    print("Testing adapters.healthcare imports...")

    from adapters.healthcare.healthcare_adapter import (
        HealthcareAdapter,
        HealthcareClaimType,
        HealthcareVerificationResult,
        MedicationComponents,
        ReviewLevel,
        AUTHORITATIVE_DRUG_SOURCES,
        AUTHORITATIVE_CLINICAL_SOURCES,
    )
    print("  ✓ adapters.healthcare.healthcare_adapter")

    from adapters.healthcare import (
        HealthcareAdapter,
        HealthcareClaimType,
    )
    print("  ✓ adapters.healthcare (package)")

    print("✓ Healthcare adapter imports successful!\n")


def test_legal_adapter_imports():
    """Test legal adapter imports."""
    print("Testing adapters.legal imports...")

    from adapters.legal.legal_adapter import (
        LegalAdapter,
        LegalClaimType,
        LegalReviewLevel,
        LegalVerificationResult,
        CaseCitationComponents,
        StatuteCitationComponents,
        AUTHORITATIVE_CASE_SOURCES,
        AUTHORITATIVE_STATUTE_SOURCES,
    )
    print("  ✓ adapters.legal.legal_adapter")

    from adapters.legal import (
        LegalAdapter,
        LegalClaimType,
    )
    print("  ✓ adapters.legal (package)")

    print("✓ Legal adapter imports successful!\n")


def test_finance_adapter_imports():
    """Test finance adapter imports."""
    print("Testing adapters.finance imports...")

    from adapters.finance.finance_adapter import (
        FinanceAdapter,
        FinanceClaimType,
        FinanceReviewLevel,
        FinanceVerificationResult,
        PerformanceComponents,
        FeeComponents,
        AUTHORITATIVE_MARKET_SOURCES,
        AUTHORITATIVE_FUND_SOURCES,
    )
    print("  ✓ adapters.finance.finance_adapter")

    from adapters.finance import (
        FinanceAdapter,
        FinanceClaimType,
    )
    print("  ✓ adapters.finance (package)")

    print("✓ Finance adapter imports successful!\n")


def test_adapters_package_imports():
    """Test main adapters package imports."""
    print("Testing adapters package imports...")

    from adapters import (
        # Base
        IndustryAdapter,
        AdapterConfig,
        RiskLevel,
        VerificationContext,
        VerificationResult,
        # Healthcare
        HealthcareAdapter,
        HealthcareClaimType,
        # Legal
        LegalAdapter,
        LegalClaimType,
        # Finance
        FinanceAdapter,
        FinanceClaimType,
    )
    print("  ✓ adapters (package)")

    print("✓ Adapters package imports successful!\n")


def test_instantiation():
    """Test that classes can be instantiated."""
    print("Testing class instantiation...")

    from core_engine.semantic_entropy import EntropyThresholds, SemanticEntropyCalculator
    from adapters.healthcare import HealthcareAdapter
    from adapters.legal import LegalAdapter
    from adapters.finance import FinanceAdapter

    # Test instantiation
    thresholds = EntropyThresholds()
    print(f"  ✓ EntropyThresholds(low_medium={thresholds.low_medium_boundary})")

    calculator = SemanticEntropyCalculator()
    print(f"  ✓ SemanticEntropyCalculator()")

    healthcare = HealthcareAdapter()
    print(f"  ✓ HealthcareAdapter(industry={healthcare._config.industry_name})")

    legal = LegalAdapter()
    print(f"  ✓ LegalAdapter(industry={legal._config.industry_name})")

    finance = FinanceAdapter()
    print(f"  ✓ FinanceAdapter(industry={finance._config.industry_name})")

    print("✓ All instantiation tests successful!\n")


def main():
    """Run all import tests."""
    print("=" * 60)
    print("TRUST Platform Import Tests")
    print("=" * 60)
    print()

    try:
        test_core_engine_imports()
        test_adapter_base_imports()
        test_healthcare_adapter_imports()
        test_legal_adapter_imports()
        test_finance_adapter_imports()
        test_adapters_package_imports()
        test_instantiation()

        print("=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        return 0

    except ImportError as e:
        print(f"\n❌ IMPORT ERROR: {e}")
        print("\nMake sure you're running from the trust-platform directory:")
        print("  cd trust-platform && python test_imports.py")
        return 1

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
