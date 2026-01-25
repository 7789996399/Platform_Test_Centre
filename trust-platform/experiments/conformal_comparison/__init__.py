"""
Conformal Prediction Comparison Experiment
==========================================

Compares OLD platform (SE-only) vs NEW platform (conformal prediction).

Files:
- test_cases.py: Generate 100 deterministic test cases
- old_platform_runner.py: OLD platform wrapper
- new_platform_runner.py: NEW platform wrapper with conformal
- run_experiment.py: Main experiment runner

Output (in results/):
- case_level_results.csv: All test cases with all metrics
- calibration_data.csv: Calibration cases
- experiment_metadata.json: Parameters
"""

from experiments.conformal_comparison.test_cases import (
    generate_test_cases,
    TestCase,
    ClaimType,
)
from experiments.conformal_comparison.old_platform_runner import (
    OldPlatformRunner,
    OldPlatformResult,
)
from experiments.conformal_comparison.new_platform_runner import (
    NewPlatformRunner,
    NewPlatformResult,
)
