#!/usr/bin/env python3
"""
Run Conformal Prediction Comparison Experiment
===============================================

Compares OLD platform (SE-only) vs NEW platform (conformal prediction).

Outputs COMPLETE CSV files with ALL cases and ALL metrics.
NO interpretation in code - raw measurements only.

Usage:
    python run_experiment.py
"""

import sys
import os
import csv
import json
from datetime import datetime
from pathlib import Path

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.conformal_comparison.test_cases import generate_test_cases, TestCase
from experiments.conformal_comparison.old_platform_runner import OldPlatformRunner
from experiments.conformal_comparison.new_platform_runner import NewPlatformRunner


# =============================================================================
# CONFIGURATION
# =============================================================================

SEED = 42
N_TOTAL_CASES = 100
N_CALIBRATION_CASES = 20
N_TEST_CASES = N_TOTAL_CASES - N_CALIBRATION_CASES
COVERAGE_TARGET = 0.90

RESULTS_DIR = Path(__file__).parent / "results"


# =============================================================================
# CSV WRITING
# =============================================================================

def write_case_level_results(
    test_cases: list,
    old_results: list,
    new_results: list,
    filepath: Path
) -> None:
    """
    Write case-level results to CSV.

    One row per case, ALL metrics from BOTH platforms.
    """
    headers = [
        # Case info
        "case_id",
        "ground_truth_correct",
        "claim_type",
        "ehr_data_available",
        "difficulty",
        "claim_text",

        # OLD platform metrics
        "OLD_ehr_status",
        "OLD_semantic_entropy",
        "OLD_se_level",
        "OLD_risk_level",
        "OLD_requires_review",
        "OLD_computation_ms",

        # NEW platform metrics
        "NEW_ehr_status",
        "NEW_semantic_entropy",
        "NEW_se_level",
        "NEW_conformal_set",
        "NEW_conformal_set_size",
        "NEW_ground_truth_in_set",
        "NEW_coverage_target",
        "NEW_risk_level",
        "NEW_requires_review",
        "NEW_computation_ms",
    ]

    rows = []
    for case, old_res, new_res in zip(test_cases, old_results, new_results):
        row = {
            # Case info
            "case_id": case.case_id,
            "ground_truth_correct": int(case.ground_truth_correct),
            "claim_type": case.claim_type.value,
            "ehr_data_available": int(case.ehr_data_available),
            "difficulty": case.difficulty,
            "claim_text": case.claim_text,

            # OLD platform
            "OLD_ehr_status": old_res.ehr_status,
            "OLD_semantic_entropy": old_res.semantic_entropy,
            "OLD_se_level": old_res.se_level,
            "OLD_risk_level": old_res.risk_level,
            "OLD_requires_review": int(old_res.requires_review),
            "OLD_computation_ms": old_res.computation_time_ms,

            # NEW platform
            "NEW_ehr_status": new_res.ehr_status,
            "NEW_semantic_entropy": new_res.semantic_entropy,
            "NEW_se_level": new_res.se_level,
            "NEW_conformal_set": "|".join(new_res.conformal_prediction_set),
            "NEW_conformal_set_size": new_res.conformal_set_size,
            "NEW_ground_truth_in_set": int(new_res.ground_truth_in_set),
            "NEW_coverage_target": new_res.conformal_coverage_target,
            "NEW_risk_level": new_res.risk_level,
            "NEW_requires_review": int(new_res.requires_review),
            "NEW_computation_ms": new_res.computation_time_ms,
        }
        rows.append(row)

    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def write_calibration_data(
    calibration_cases: list,
    filepath: Path
) -> None:
    """
    Write calibration data to CSV.

    These are the 20 cases used for conformal calibration.
    """
    headers = [
        "case_id",
        "ground_truth_correct",
        "claim_type",
        "ehr_data_available",
        "difficulty",
        "expected_ehr_status",
        "claim_text",
    ]

    rows = []
    for case in calibration_cases:
        row = {
            "case_id": case.case_id,
            "ground_truth_correct": int(case.ground_truth_correct),
            "claim_type": case.claim_type.value,
            "ehr_data_available": int(case.ehr_data_available),
            "difficulty": case.difficulty,
            "expected_ehr_status": case.expected_ehr_status,
            "claim_text": case.claim_text,
        }
        rows.append(row)

    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def write_metadata(
    n_total: int,
    n_calibration: int,
    n_test: int,
    coverage_target: float,
    calibration_threshold: float,
    filepath: Path
) -> None:
    """Write experiment metadata to JSON."""
    metadata = {
        "experiment_name": "conformal_prediction_comparison",
        "description": "Comparing OLD platform (SE-only) vs NEW platform (conformal prediction)",
        "timestamp_start": datetime.now().isoformat(),
        "parameters": {
            "seed": SEED,
            "n_total_cases": n_total,
            "n_calibration_cases": n_calibration,
            "n_test_cases": n_test,
            "coverage_target": coverage_target,
            "calibration_threshold": calibration_threshold,
        },
        "platforms": {
            "OLD": {
                "name": "OLD_PLATFORM",
                "version": "1.0",
                "description": "EHR-First with Semantic Entropy only",
                "has_conformal": False,
            },
            "NEW": {
                "name": "NEW_PLATFORM",
                "version": "2.0",
                "description": "EHR-First with Conformal Prediction",
                "has_conformal": True,
            },
        },
        "output_files": [
            "case_level_results.csv",
            "calibration_data.csv",
            "experiment_metadata.json",
        ],
    }

    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=2)


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_experiment():
    """
    Run the full experiment comparing OLD and NEW platforms.

    Steps:
    1. Generate 100 deterministic test cases
    2. Split: 20 for calibration, 80 for test
    3. Run OLD platform on all 80 test cases
    4. Calibrate NEW platform on 20 calibration cases
    5. Run NEW platform on all 80 test cases
    6. Write COMPLETE CSV output
    """
    print("=" * 70)
    print("CONFORMAL PREDICTION COMPARISON EXPERIMENT")
    print("=" * 70)
    print(f"\nSeed: {SEED}")
    print(f"Total cases: {N_TOTAL_CASES}")
    print(f"Calibration cases: {N_CALIBRATION_CASES}")
    print(f"Test cases: {N_TEST_CASES}")
    print(f"Coverage target: {COVERAGE_TARGET:.0%}")

    # Create results directory
    RESULTS_DIR.mkdir(exist_ok=True)

    # Step 1: Generate test cases
    print("\n" + "-" * 70)
    print("STEP 1: Generating test cases...")
    all_cases = generate_test_cases(n_cases=N_TOTAL_CASES, seed=SEED)
    print(f"  Generated {len(all_cases)} cases")

    # Step 2: Split into calibration and test
    print("\nSTEP 2: Splitting data...")
    calibration_cases = all_cases[:N_CALIBRATION_CASES]
    test_cases = all_cases[N_CALIBRATION_CASES:]
    print(f"  Calibration: {len(calibration_cases)} cases")
    print(f"  Test: {len(test_cases)} cases")

    # Step 3: Initialize runners
    print("\nSTEP 3: Initializing platforms...")
    old_runner = OldPlatformRunner()
    new_runner = NewPlatformRunner(coverage_target=COVERAGE_TARGET)
    print(f"  OLD Platform: {old_runner.name} v{old_runner.version}")
    print(f"  NEW Platform: {new_runner.name} v{new_runner.version}")

    # Step 4: Calibrate NEW platform
    print("\nSTEP 4: Calibrating NEW platform on calibration set...")
    cal_result = new_runner.calibrate(calibration_cases)
    print(f"  Calibration threshold: {cal_result.threshold:.4f}")
    print(f"  Calibration samples: {cal_result.n_calibration}")

    # Step 5: Run OLD platform on test set
    print("\nSTEP 5: Running OLD platform on test set...")
    old_results = old_runner.process_batch(test_cases)
    print(f"  Processed {len(old_results)} cases")

    # Step 6: Run NEW platform on test set
    print("\nSTEP 6: Running NEW platform on test set...")
    new_results = new_runner.process_batch(test_cases)
    print(f"  Processed {len(new_results)} cases")

    # Step 7: Write output files
    print("\n" + "-" * 70)
    print("STEP 7: Writing output files...")

    # Case-level results
    case_level_path = RESULTS_DIR / "case_level_results.csv"
    write_case_level_results(test_cases, old_results, new_results, case_level_path)
    print(f"  Wrote: {case_level_path}")

    # Calibration data
    calibration_path = RESULTS_DIR / "calibration_data.csv"
    write_calibration_data(calibration_cases, calibration_path)
    print(f"  Wrote: {calibration_path}")

    # Metadata
    metadata_path = RESULTS_DIR / "experiment_metadata.json"
    write_metadata(
        n_total=N_TOTAL_CASES,
        n_calibration=N_CALIBRATION_CASES,
        n_test=N_TEST_CASES,
        coverage_target=COVERAGE_TARGET,
        calibration_threshold=cal_result.threshold,
        filepath=metadata_path
    )
    print(f"  Wrote: {metadata_path}")

    # Step 8: Print summary stats (for verification only)
    print("\n" + "=" * 70)
    print("RAW SUMMARY STATISTICS")
    print("(For verification - actual analysis should be done from CSV)")
    print("=" * 70)

    # Count EHR statuses
    old_ehr_counts = {}
    for r in old_results:
        old_ehr_counts[r.ehr_status] = old_ehr_counts.get(r.ehr_status, 0) + 1

    print(f"\nEHR Status Distribution (OLD):")
    for status, count in sorted(old_ehr_counts.items()):
        print(f"  {status}: {count}")

    # Count risk levels
    old_risk_counts = {}
    for r in old_results:
        old_risk_counts[r.risk_level] = old_risk_counts.get(r.risk_level, 0) + 1

    print(f"\nRisk Level Distribution (OLD):")
    for level, count in sorted(old_risk_counts.items()):
        print(f"  {level}: {count}")

    # NEW platform conformal stats
    n_covered = sum(1 for r in new_results if r.ground_truth_in_set)
    empirical_coverage = n_covered / len(new_results)
    avg_set_size = sum(r.conformal_set_size for r in new_results) / len(new_results)

    print(f"\nNEW Platform Conformal Stats:")
    print(f"  Ground truth in set: {n_covered}/{len(new_results)}")
    print(f"  Empirical coverage: {empirical_coverage:.4f}")
    print(f"  Target coverage: {COVERAGE_TARGET:.4f}")
    print(f"  Average set size: {avg_set_size:.2f}")

    # Set size distribution
    set_size_counts = {}
    for r in new_results:
        set_size_counts[r.conformal_set_size] = set_size_counts.get(r.conformal_set_size, 0) + 1

    print(f"\nSet Size Distribution (NEW):")
    for size, count in sorted(set_size_counts.items()):
        print(f"  Size {size}: {count}")

    # Review needed comparison
    old_review = sum(1 for r in old_results if r.requires_review)
    new_review = sum(1 for r in new_results if r.requires_review)

    print(f"\nReview Required:")
    print(f"  OLD: {old_review}/{len(old_results)}")
    print(f"  NEW: {new_review}/{len(new_results)}")

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"\nOutput files in: {RESULTS_DIR}")
    print("  - case_level_results.csv: All 80 test cases with all metrics")
    print("  - calibration_data.csv: The 20 calibration cases")
    print("  - experiment_metadata.json: Experiment parameters")

    return {
        "case_level_path": str(case_level_path),
        "calibration_path": str(calibration_path),
        "metadata_path": str(metadata_path),
        "n_test_cases": len(test_cases),
        "empirical_coverage": empirical_coverage,
    }


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    result = run_experiment()

    # Show first 10 rows of results
    print("\n" + "=" * 70)
    print("FIRST 10 ROWS OF case_level_results.csv")
    print("=" * 70)

    with open(result["case_level_path"], 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Print header
    headers = list(rows[0].keys())
    print("\nHeaders:")
    for h in headers:
        print(f"  - {h}")

    print(f"\nFirst 10 rows (selected columns):")
    print("-" * 100)
    print(f"{'case_id':<10} {'OLD_risk':<8} {'OLD_SE':<8} {'NEW_risk':<8} {'NEW_set_size':<12} {'NEW_in_set':<10}")
    print("-" * 100)

    for row in rows[:10]:
        print(f"{row['case_id']:<10} {row['OLD_risk_level']:<8} {row['OLD_semantic_entropy']:<8} "
              f"{row['NEW_risk_level']:<8} {row['NEW_conformal_set_size']:<12} {row['NEW_ground_truth_in_set']:<10}")

    print("-" * 100)
    print(f"\nTotal rows: {len(rows)}")

    # File sizes
    case_level_path = Path(result["case_level_path"])
    calibration_path = Path(result["calibration_path"])
    metadata_path = Path(result["metadata_path"])

    print(f"\nFile sizes:")
    print(f"  case_level_results.csv: {case_level_path.stat().st_size:,} bytes")
    print(f"  calibration_data.csv: {calibration_path.stat().st_size:,} bytes")
    print(f"  experiment_metadata.json: {metadata_path.stat().st_size:,} bytes")
