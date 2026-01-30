"""
Typed configuration loader for Layer 2 training.

Loads training_config.yaml into a hierarchy of dataclasses with
sensible defaults if the YAML file is missing.
"""

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).parent


@dataclass
class ModelConfig:
    name: str = "meta-llama/Meta-Llama-3-8B"
    test_name: str = "sshleifer/tiny-gpt2"
    max_length: int = 512


@dataclass
class LoRAConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    test_target_modules: List[str] = field(
        default_factory=lambda: ["c_attn"]
    )


@dataclass
class QuantizationConfig:
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "float16"


@dataclass
class TrainingConfig:
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    num_epochs: int = 3
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.001
    max_grad_norm: float = 0.3
    fp16: bool = True
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    test_batch_size: int = 2
    test_max_steps: int = 2
    test_num_epochs: int = 1


@dataclass
class DataConfig:
    train_file: str = "data/train.ndjson"
    val_split: float = 0.1
    max_length: int = 512
    mock_num_examples: int = 100


@dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "layer2-meta-classifier"
    entity: Optional[str] = None
    run_name: Optional[str] = None


@dataclass
class Layer2Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


def _build_sub_config(cls, raw: dict):
    """Instantiate a dataclass from a dict, ignoring unknown keys."""
    import dataclasses
    valid_keys = {f.name for f in dataclasses.fields(cls)}
    filtered = {k: v for k, v in raw.items() if k in valid_keys}
    return cls(**filtered)


def load_config(path: Optional[str] = None) -> Layer2Config:
    """
    Load Layer2Config from a YAML file.

    Falls back to defaults if the file is missing or ``pyyaml`` is not
    installed.
    """
    if path is None:
        path = str(CONFIG_DIR / "training_config.yaml")

    if not os.path.exists(path):
        logger.warning("Config file %s not found, using defaults", path)
        return Layer2Config()

    try:
        import yaml
    except ImportError:
        logger.warning("pyyaml not installed, using default config")
        return Layer2Config()

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    return Layer2Config(
        model=_build_sub_config(ModelConfig, raw.get("model", {})),
        lora=_build_sub_config(LoRAConfig, raw.get("lora", {})),
        quantization=_build_sub_config(QuantizationConfig, raw.get("quantization", {})),
        training=_build_sub_config(TrainingConfig, raw.get("training", {})),
        data=_build_sub_config(DataConfig, raw.get("data", {})),
        wandb=_build_sub_config(WandbConfig, raw.get("wandb", {})),
    )
