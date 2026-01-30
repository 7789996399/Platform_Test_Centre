"""
Typed configuration loader for Layer 3 calibration.

Loads calibration_config.yaml into dataclasses with sensible defaults
if the YAML file is missing.
"""

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).parent

REVIEW_LEVELS = ["BRIEF", "STANDARD", "DETAILED", "CRITICAL"]


@dataclass
class ConformalConfig:
    coverage_target: float = 0.90
    min_calibration_samples: int = 100
    review_levels: List[str] = field(default_factory=lambda: list(REVIEW_LEVELS))


@dataclass
class HistogramConfig:
    score_bins: int = 10
    min_bin_count: int = 5


@dataclass
class PersistenceConfig:
    save_dir: str = "layer3_calibration"
    filename: str = "calibrator_state.json"


@dataclass
class Layer3Config:
    conformal: ConformalConfig = field(default_factory=ConformalConfig)
    histogram: HistogramConfig = field(default_factory=HistogramConfig)
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)


def _build_sub_config(cls, raw: dict):
    """Instantiate a dataclass from a dict, ignoring unknown keys."""
    import dataclasses
    valid_keys = {f.name for f in dataclasses.fields(cls)}
    filtered = {k: v for k, v in raw.items() if k in valid_keys}
    return cls(**filtered)


def load_config(path: Optional[str] = None) -> Layer3Config:
    """
    Load Layer3Config from a YAML file.

    Falls back to defaults if the file is missing or pyyaml is not installed.
    """
    if path is None:
        path = str(CONFIG_DIR / "calibration_config.yaml")

    if not os.path.exists(path):
        logger.warning("Config file %s not found, using defaults", path)
        return Layer3Config()

    try:
        import yaml
    except ImportError:
        logger.warning("pyyaml not installed, using default config")
        return Layer3Config()

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    return Layer3Config(
        conformal=_build_sub_config(ConformalConfig, raw.get("conformal", {})),
        histogram=_build_sub_config(HistogramConfig, raw.get("histogram", {})),
        persistence=_build_sub_config(PersistenceConfig, raw.get("persistence", {})),
    )
