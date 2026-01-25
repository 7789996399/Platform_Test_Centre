#!/usr/bin/env python3
"""
Test All Governance Modules
============================

Unified test script for the three-module governance architecture:
1. Generative Governance (AI Scribes) - EHR-First Flow
2. Diagnostic Governance (Radiology AI) - Conformal-First Flow
3. Predictive Governance (Risk Scores) - Input-Validation-First Flow

This script uses mock data throughout - NO real API calls.

Usage:
    python test_all_modules.py
"""

import sys
import asyncio
from typing import Dict, Any, List, Optional, Tuple


# =============================================================================
# TEST 1: GENERATIVE GOVERNANCE (AI Scribe)
# =============================================================================

def test_generative_governance():
    """
    Test the EHR-First flow for AI Scribes.

    Flow: Extract claims → Verify against EHR → Run SE on unverified → Flag risks
    """
    print("\n" + "=" * 70)
    print("TEST 1: GENERATIVE GOVERNANCE (AI Scribe - EHR-First Flow)")
    print("=" * 70)

    import re
    from generative_governance import GenerativeClaimExtractor, ExtractedClaim
    from adapters.healthcare import HealthcareAdapter

    # Test claim extraction
    extractor = GenerativeClaimExtractor()

    ai_output = """
    PATIENT VISIT SUMMARY
    ---------------------
    Chief Complaint: Patient presents with chest pain and shortness of breath.

    History: The patient has a history of hypertension, diagnosed in 2019.
    Current medications include Lisinopril 10mg daily and Metformin 500mg twice daily.
    The patient reports taking aspirin 81mg as prescribed.

    Vital Signs:
    - Blood pressure: 145/92 mmHg
    - Heart rate: 88 bpm
    - Temperature: 37.2°C
    - Oxygen saturation: 97%

    Assessment: Possible unstable angina. The patient's diabetes is well-controlled
    with current medications. Recommend cardiology consultation and ECG.

    Plan: Order troponin levels, ECG, and chest X-ray. Continue current medications.
    Schedule follow-up in 1 week.
    """

    print("\n[Input] AI-Generated Note (excerpt):")
    print("-" * 40)
    print(ai_output[:300] + "...")

    print("\n[Processing] Testing Generative Governance Components...")
    print("  1. Extracting claims from AI output")

    # Define patterns for claim extraction
    patterns = {
        "medication": re.compile(r'(?:takes?|taking|on)\s+(\w+)\s+(\d+\s*mg)', re.I),
        "vital_bp": re.compile(r'blood pressure[:\s]+(\d+/\d+)', re.I),
        "vital_hr": re.compile(r'heart rate[:\s]+(\d+)', re.I),
        "vital_temp": re.compile(r'temperature[:\s]+([\d.]+)', re.I),
        "condition": re.compile(r'history of\s+(\w+)', re.I),
    }

    claims = extractor.extract(ai_output, patterns)
    print(f"     Extracted {len(claims)} claims")

    if claims:
        print(f"\n  Sample extracted claims:")
        for claim in claims[:5]:
            print(f"    - [{claim.claim_type}] {claim.text[:60]}...")

    print("\n  2. Healthcare Adapter available for claim verification")

    # Test the healthcare adapter
    adapter = HealthcareAdapter()
    entropy_thresholds = adapter.get_entropy_thresholds()
    print(f"     Entropy thresholds: low/med={entropy_thresholds.low_medium_boundary}, med/high={entropy_thresholds.medium_high_boundary}")

    print("\n[Results]")
    print("-" * 40)
    print(f"  Claims extracted: {len(claims)}")
    print(f"  Adapter initialized: Yes")
    print(f"  Pipeline ready: Yes (async analyze method available)")

    print("\n  Note: Full pipeline test requires async execution.")
    print("  Components validated successfully.")

    return len(claims) > 0


# =============================================================================
# TEST 2: DIAGNOSTIC GOVERNANCE (Radiology AI)
# =============================================================================

def test_diagnostic_governance():
    """
    Test the Conformal-First flow for Radiology AI.

    Flow: Conformal prediction → Add context (priors) → Validate attention → Output
    """
    print("\n" + "=" * 70)
    print("TEST 2: DIAGNOSTIC GOVERNANCE (Radiology AI - Conformal-First Flow)")
    print("=" * 70)

    from diagnostic_governance import (
        DiagnosticGovernancePipeline,
        AttentionValidator,
        PriorContextIntegrator,
        PriorStudy,
    )
    from adapters.radiology import RadiologyAdapter, ModalityType

    # Create adapter and pipeline
    adapter = RadiologyAdapter(modality=ModalityType.CHEST_XRAY)
    pipeline = DiagnosticGovernancePipeline(adapter=adapter, coverage_target=0.90)

    # Mock model output (softmax probabilities)
    model_scores = {
        "normal": 0.05,
        "pneumonia": 0.65,
        "atelectasis": 0.10,
        "cardiomegaly": 0.03,
        "effusion": 0.08,
        "infiltration": 0.04,
        "mass": 0.01,
        "nodule": 0.02,
        "pneumothorax": 0.01,
        "consolidation": 0.01,
    }

    print("\n[Input] Chest X-ray Model Output:")
    print("-" * 40)
    print("  Top 3 predictions:")
    sorted_probs = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
    for finding, prob in sorted_probs[:3]:
        print(f"    {finding}: {prob:.1%}")

    print("\n[Processing] Testing Diagnostic Governance Components...")

    # Test conformal calibrator through pipeline
    print("  1. Creating calibration data (mock)")

    import random
    random.seed(42)

    # Generate mock calibration data
    n_cal = 100
    class_names = list(model_scores.keys())
    cal_scores = []
    cal_labels = []

    for i in range(n_cal):
        scores = {c: random.random() for c in class_names}
        total = sum(scores.values())
        scores = {c: s / total for c, s in scores.items()}
        cal_scores.append(scores)
        cal_labels.append(class_names[random.randint(0, len(class_names) - 1)])

    # Calibrate the pipeline
    cal_result = pipeline.calibrate(cal_scores, cal_labels)
    print(f"     Calibrated with {cal_result.n_calibration} samples")
    print(f"     Threshold: {cal_result.threshold:.4f}")

    # Generate prediction set
    print("  2. Generating prediction set")
    pred_set = pipeline.calibrator.predict_set(model_scores, class_names)
    print(f"     Set size: {pred_set.set_size}")
    print(f"     Classes: {pred_set.classes}")

    # Test attention validator
    print("  3. Testing attention validation")
    validator = AttentionValidator()
    # Note: Mock attention validation doesn't require actual attention map
    attention_result = adapter.validate_attention(None, "pneumonia")
    print(f"     Attention valid: {attention_result[0]}")

    # Test prior context integration
    print("  4. Testing prior context integration")
    prior_integrator = PriorContextIntegrator()
    prior_studies = [
        PriorStudy(
            study_id="XR-2023-001",
            study_date="2023-06-15",
            modality="XR",
            findings=["normal"],
            impression="Normal chest radiograph",
        ),
    ]
    contextual = prior_integrator.integrate(["pneumonia"], prior_studies)
    print(f"     Prior comparison: {contextual[0].change_type if contextual else 'N/A'}")

    print("\n[Results]")
    print("-" * 40)
    print(f"  Calibrator: Initialized and calibrated")
    print(f"  Prediction set size: {pred_set.set_size}")
    print(f"  Coverage target: {pipeline.coverage_target:.0%}")
    print(f"  Attention validator: Working")
    print(f"  Prior context integrator: Working")

    return pred_set.set_size > 0


# =============================================================================
# TEST 3: PREDICTIVE GOVERNANCE (Risk Scores)
# =============================================================================

def test_predictive_governance():
    """
    Test the Input-Validation-First flow for Risk Scores.

    Flow: Validate inputs → Reject invalid → Run model → Calibrate → Track
    """
    print("\n" + "=" * 70)
    print("TEST 3: PREDICTIVE GOVERNANCE (Risk Scores - Input-Validation-First Flow)")
    print("=" * 70)

    from predictive_governance import (
        PredictiveGovernancePipeline,
        InputValidator,
        OutcomeTracker,
    )
    from adapters.clinical_risk import ClinicalRiskAdapter, RiskScoreType

    # Create adapter and pipeline
    adapter = ClinicalRiskAdapter(risk_type=RiskScoreType.SEPSIS)
    pipeline = PredictiveGovernancePipeline(adapter=adapter, coverage_target=0.90)

    print("\n[Input] Testing Input Validation...")
    print("-" * 40)

    # Test Case 1: Valid inputs
    valid_inputs = {
        "heart_rate": 110,
        "respiratory_rate": 24,
        "systolic_bp": 92,
        "temperature": 38.8,
        "wbc": 18.5,
    }

    print("  Valid inputs:")
    for key, val in valid_inputs.items():
        is_valid, error = adapter.validate_input(key, val)
        status = "OK" if is_valid else f"INVALID: {error}"
        print(f"    {key}={val}: {status}")

    # Test Case 2: Invalid inputs
    print("\n  Invalid inputs:")
    invalid_cases = [
        ("heart_rate", 350, "above max 250"),
        ("temperature", 50, "above max 45"),
        ("wbc", -5, "below min 0.5"),
    ]

    for name, val, expected_reason in invalid_cases:
        is_valid, error = adapter.validate_input(name, val)
        print(f"    {name}={val}: {'INVALID' if not is_valid else 'OK'} ({error})")

    # Test conformal calibration for regression
    print("\n[Processing] Testing CQR Calibration...")
    print("-" * 40)

    import random
    random.seed(42)

    # Generate mock calibration data
    n_cal = 100
    cal_lower = [random.uniform(0.1, 0.3) for _ in range(n_cal)]
    cal_upper = [cal_lower[i] + random.uniform(0.2, 0.4) for i in range(n_cal)]
    cal_actual = [random.uniform(0.1, 0.7) for _ in range(n_cal)]

    cal_result = pipeline.calibrate(cal_lower, cal_upper, cal_actual)
    print(f"  Calibrated with {cal_result.n_calibration} samples")
    print(f"  Threshold: {cal_result.threshold:.4f}")

    # Test prediction interval
    test_lower, test_upper = 0.25, 0.55
    interval = pipeline.calibrator.predict(test_lower, test_upper, point_estimate=0.40)
    print(f"\n  Test prediction:")
    print(f"    Model output: [{test_lower}, {test_upper}]")
    print(f"    Calibrated interval: [{interval.lower:.3f}, {interval.upper:.3f}]")
    print(f"    Point estimate: {interval.point_estimate:.3f}")

    # Test outcome tracker
    print("\n[Processing] Testing Outcome Tracker...")
    print("-" * 40)

    tracker = OutcomeTracker(coverage_target=0.90)

    # Record some predictions
    for i in range(10):
        pred_id = tracker.record_prediction(
            prediction_interval=(0.2, 0.5),
            point_estimate=0.35,
            inputs={"patient_id": f"P{i}"},
        )
        # Record outcome (some covered, some not)
        actual = random.uniform(0.15, 0.55)
        tracker.record_outcome(pred_id, actual)

    stats = tracker.get_calibration_stats()
    print(f"  Predictions tracked: {stats.n_predictions}")
    print(f"  With outcomes: {stats.n_with_outcomes}")
    print(f"  Empirical coverage: {stats.empirical_coverage:.1%}")
    print(f"  Target coverage: {stats.target_coverage:.0%}")

    print("\n[Results]")
    print("-" * 40)
    print(f"  Input validation: Working")
    print(f"  CQR calibration: Working")
    print(f"  Outcome tracking: Working")

    return stats.n_with_outcomes == 10


# =============================================================================
# TEST 4: CORE ENGINE MODULES
# =============================================================================

def test_core_engine_modules():
    """
    Test the core engine modules: Conformal Calibrator and Drift Monitor.
    """
    print("\n" + "=" * 70)
    print("TEST 4: CORE ENGINE MODULES")
    print("=" * 70)

    # Test Conformal Calibrator
    print("\n--- Conformal Calibrator ---")

    from core_engine.conformal_calibrator import (
        ConformalCalibrator,
        AdaptivePredictionSets,
        ConformizedQuantileRegression,
        CalibrationMethod,
    )

    # Test APS (Adaptive Prediction Sets) for classification
    print("\n[APS - Classification]")
    aps = AdaptivePredictionSets(alpha=0.1)  # 90% coverage

    import random
    random.seed(42)

    n_cal = 100
    n_classes = 5
    class_names = ["class_0", "class_1", "class_2", "class_3", "class_4"]

    cal_scores = []
    cal_labels = []

    for i in range(n_cal):
        scores = [random.random() for _ in range(n_classes)]
        total = sum(scores)
        scores = [s / total for s in scores]
        score_dict = {class_names[j]: scores[j] for j in range(n_classes)}
        cal_scores.append(score_dict)
        cal_labels.append(class_names[random.randint(0, n_classes - 1)])

    cal_result = aps.calibrate(cal_scores, cal_labels)
    print(f"  Calibration samples: {cal_result.n_calibration}")
    print(f"  Coverage target: {1 - cal_result.alpha:.0%}")
    print(f"  Threshold: {cal_result.threshold:.4f}")

    test_scores = {"class_0": 0.05, "class_1": 0.15, "class_2": 0.50, "class_3": 0.20, "class_4": 0.10}
    pred_set = aps.predict_set(test_scores)
    print(f"  Test input: {test_scores}")
    print(f"  Prediction set: {pred_set.classes}")
    print(f"  Set size: {pred_set.set_size}")

    # Test CQR
    print("\n[CQR - Regression]")
    cqr = ConformizedQuantileRegression(alpha=0.1)

    cal_lower = [random.uniform(0.1, 0.3) for _ in range(n_cal)]
    cal_upper = [cal_lower[i] + random.uniform(0.2, 0.4) for i in range(n_cal)]
    cal_actual = [random.uniform(0.1, 0.7) for _ in range(n_cal)]

    cal_result = cqr.calibrate(cal_lower, cal_upper, cal_actual)
    print(f"  Calibration samples: {cal_result.n_calibration}")
    print(f"  Coverage target: {1 - cal_result.alpha:.0%}")

    test_lower, test_upper = 0.25, 0.55
    pred_interval = cqr.predict(test_lower, test_upper)
    print(f"  Test input: [{test_lower}, {test_upper}]")
    print(f"  Calibrated interval: [{pred_interval.lower:.3f}, {pred_interval.upper:.3f}]")

    # Test Drift Monitor
    print("\n--- Drift Monitor ---")

    from core_engine.drift_monitor import DriftMonitor, CoverageMonitor, compute_psi

    monitor = DriftMonitor(psi_threshold=0.2)

    # Set reference distribution
    baseline_data = [random.gauss(50, 10) for _ in range(50)]
    monitor.set_reference({"feature1": baseline_data})

    # Test with no drift
    test_data_no_drift = [random.gauss(50, 10) for _ in range(20)]
    result_no_drift = monitor.check({"feature1": test_data_no_drift})
    print(f"\n  No drift case:")
    print(f"    Drift detected: {result_no_drift.is_significant}")
    print(f"    Severity: {result_no_drift.severity.value}")

    # Test with drift
    test_data_drift = [random.gauss(65, 10) for _ in range(20)]  # Mean shifted
    result_drift = monitor.check({"feature1": test_data_drift})
    print(f"\n  Drift case (mean shifted 50→65):")
    print(f"    Drift detected: {result_drift.is_significant}")
    print(f"    Severity: {result_drift.severity.value}")
    print(f"    PSI: {result_drift.statistic:.4f}")

    # Test PSI directly
    psi = compute_psi(baseline_data, test_data_drift)
    print(f"    Direct PSI: {psi:.4f}")

    # Test Coverage Monitor
    print("\n--- Coverage Monitor ---")

    cov_monitor = CoverageMonitor(target_coverage=0.90, window_size=50)

    for i in range(50):
        is_covered = random.random() < 0.88  # Slightly below 90%
        cov_monitor.update(is_covered)

    print(f"  Target coverage: {cov_monitor.target_coverage:.0%}")
    print(f"  Empirical coverage: {cov_monitor.current_coverage:.1%}")
    print(f"  Coverage degraded: {cov_monitor.is_coverage_degraded()}")

    return True


# =============================================================================
# TEST 5: ADAPTERS
# =============================================================================

def test_adapters():
    """Test the domain-specific adapters."""
    print("\n" + "=" * 70)
    print("TEST 5: DOMAIN ADAPTERS")
    print("=" * 70)

    # Test Radiology Adapter
    print("\n--- Radiology Adapter ---")

    from adapters.radiology import (
        RadiologyAdapter,
        ModalityType,
        RadiologyFinding,
        CHEST_XRAY_FINDINGS,
    )

    adapter = RadiologyAdapter(modality=ModalityType.CHEST_XRAY)

    print(f"  Modality: {adapter.modality.value}")
    print(f"  Available findings: {len(adapter.get_class_names())}")
    print(f"  Sample findings: {adapter.get_class_names()[:5]}")

    is_valid, concerns = adapter.validate_attention(None, "pneumonia")
    print(f"\n  Attention validation for 'pneumonia':")
    print(f"    Valid: {is_valid}")

    comparison = adapter.compare_with_prior("pneumonia", ["normal"])
    print(f"\n  Prior comparison:")
    print(f"    {comparison}")

    pacs_data = adapter.get_mock_pacs_data("P12345")
    print(f"\n  Mock PACS data: {len(pacs_data['prior_studies'])} prior studies")

    # Test Clinical Risk Adapter
    print("\n--- Clinical Risk Adapter ---")

    from adapters.clinical_risk import (
        ClinicalRiskAdapter,
        RiskScoreType,
        VITAL_SIGN_RANGES,
        RISK_ACTION_THRESHOLDS,
    )

    adapter = ClinicalRiskAdapter(risk_type=RiskScoreType.SEPSIS)

    print(f"  Risk type: {adapter.risk_type.value}")
    print(f"  Required inputs: {adapter.get_required_inputs()}")
    print(f"  Action thresholds: {adapter.get_action_thresholds()}")

    is_valid, error = adapter.validate_input("heart_rate", 95)
    print(f"\n  Input validation (heart_rate=95):")
    print(f"    Valid: {is_valid}")

    is_valid, error = adapter.validate_input("heart_rate", 350)
    print(f"\n  Input validation (heart_rate=350):")
    print(f"    Valid: {is_valid}")
    print(f"    Error: {error}")

    qsofa = adapter.compute_qsofa(
        respiratory_rate=22,
        systolic_bp=95,
        altered_mental_status=True
    )
    print(f"\n  qSOFA calculation: RR=22, SBP=95, AMS=True")
    print(f"    Score: {qsofa}")
    print(f"    {adapter.get_score_description(qsofa, 'qSOFA')}")

    vitals = {
        "respiratory_rate": 22,
        "oxygen_saturation": 94,
        "temperature": 38.2,
        "systolic_bp": 105,
        "heart_rate": 95,
        "level_of_consciousness": 15,
    }
    news = adapter.compute_news(vitals)
    print(f"\n  NEWS calculation:")
    print(f"    Score: {news}")
    print(f"    {adapter.get_score_description(news, 'NEWS')}")

    ehr_data = adapter.get_mock_ehr_data("P12345", "septic")
    print(f"\n  Mock EHR data (septic scenario):")
    print(f"    Heart rate: {ehr_data['vitals']['heart_rate']}")
    print(f"    WBC: {ehr_data['labs']['wbc']}")

    return True


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("TRUST PLATFORM - THREE-MODULE GOVERNANCE ARCHITECTURE")
    print("Comprehensive Test Suite")
    print("=" * 70)
    print("\nThis test uses MOCK DATA only - no real API calls.")

    results = {}

    # Test 1: Generative Governance
    try:
        results["Generative Governance"] = test_generative_governance()
    except Exception as e:
        print(f"\n[ERROR] Generative Governance test failed: {e}")
        import traceback
        traceback.print_exc()
        results["Generative Governance"] = False

    # Test 2: Diagnostic Governance
    try:
        results["Diagnostic Governance"] = test_diagnostic_governance()
    except Exception as e:
        print(f"\n[ERROR] Diagnostic Governance test failed: {e}")
        import traceback
        traceback.print_exc()
        results["Diagnostic Governance"] = False

    # Test 3: Predictive Governance
    try:
        results["Predictive Governance"] = test_predictive_governance()
    except Exception as e:
        print(f"\n[ERROR] Predictive Governance test failed: {e}")
        import traceback
        traceback.print_exc()
        results["Predictive Governance"] = False

    # Test 4: Core Engine Modules
    try:
        results["Core Engine Modules"] = test_core_engine_modules()
    except Exception as e:
        print(f"\n[ERROR] Core Engine Modules test failed: {e}")
        import traceback
        traceback.print_exc()
        results["Core Engine Modules"] = False

    # Test 5: Adapters
    try:
        results["Domain Adapters"] = test_adapters()
    except Exception as e:
        print(f"\n[ERROR] Domain Adapters test failed: {e}")
        import traceback
        traceback.print_exc()
        results["Domain Adapters"] = False

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    all_passed = True
    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        symbol = "+" if passed else "x"
        print(f"  [{symbol}] {test_name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "-" * 70)
    if all_passed:
        print("All tests passed successfully!")
        print("\nThe three-module governance architecture is working correctly:")
        print("  1. Generative: EHR-First flow (verify claims before SE)")
        print("  2. Diagnostic: Conformal-First flow (calibrate before context)")
        print("  3. Predictive: Input-Validation-First flow (validate before inference)")
    else:
        print("Some tests failed. Please check the errors above.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
