"""
Converts NDJSON training files into HuggingFace Datasets for LoRA fine-tuning.

Handles validation, formatting via prompt_template, and provides a mock
data path for CPU testing.

CLI usage:
    python -m layer2.data.prepare_training_data --input data/train.ndjson --output data/hf_dataset [--test-mode]
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

from .prompt_template import format_training_example
from .mock_data_generator import generate_mock_dataset, write_ndjson

logger = logging.getLogger(__name__)

VALID_EHR_STATUSES = {"verified", "contradiction", "not_found", "not_checkable"}
VALID_REVIEW_LEVELS = {"BRIEF", "STANDARD", "DETAILED", "CRITICAL"}


def load_ndjson(path: str) -> List[dict]:
    """Read an NDJSON file, skipping malformed lines."""
    examples = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSON on line %d", i)
    logger.info("Loaded %d examples from %s", len(examples), path)
    return examples


def validate_example(ex: dict) -> Optional[str]:
    """
    Validate a single training example.

    Returns an error message string if invalid, or None if valid.
    """
    required = ["claim_type", "claim_text", "source_sentence", "ehr_status",
                 "risk_score", "review_level"]
    for key in required:
        if key not in ex:
            return f"missing required field: {key}"

    if ex["ehr_status"] not in VALID_EHR_STATUSES:
        return f"invalid ehr_status: {ex['ehr_status']}"

    if ex["review_level"] not in VALID_REVIEW_LEVELS:
        return f"invalid review_level: {ex['review_level']}"

    score = ex["risk_score"]
    if not isinstance(score, (int, float)) or score < 0.0 or score > 1.0:
        return f"risk_score out of range: {score}"

    hhem = ex.get("hhem_score")
    if hhem is not None:
        if not isinstance(hhem, (int, float)) or hhem < 0.0 or hhem > 1.0:
            return f"hhem_score out of range: {hhem}"

    se = ex.get("semantic_entropy")
    if se is not None:
        if not isinstance(se, (int, float)) or se < 0.0 or se > 1.0:
            return f"semantic_entropy out of range: {se}"

    return None


def example_to_training_text(ex: dict) -> str:
    """Convert a validated example dict into a formatted training string."""
    return format_training_example(
        claim_type=ex["claim_type"],
        claim_text=ex["claim_text"],
        source_sentence=ex["source_sentence"],
        ehr_status=ex["ehr_status"],
        hhem_score=ex.get("hhem_score"),
        semantic_entropy=ex.get("semantic_entropy"),
        risk_score=ex["risk_score"],
        review_level=ex["review_level"],
    )


def prepare_dataset(input_path: str, max_length: int = 512) -> "datasets.Dataset":
    """Load NDJSON, validate, format, and return a HuggingFace Dataset."""
    from datasets import Dataset

    raw = load_ndjson(input_path)
    texts = []
    skipped = 0
    for ex in raw:
        err = validate_example(ex)
        if err:
            logger.warning("Skipping invalid example: %s", err)
            skipped += 1
            continue
        texts.append(example_to_training_text(ex))

    if skipped:
        logger.info("Skipped %d invalid examples out of %d total", skipped, len(raw))

    return Dataset.from_dict({"text": texts})


def prepare_mock_dataset(
    num_examples: int = 100,
    seed: int = 42,
) -> "datasets.Dataset":
    """Generate mock data and return as a HuggingFace Dataset (CPU test path)."""
    from datasets import Dataset

    examples = generate_mock_dataset(num_examples=num_examples, seed=seed)
    texts = []
    for ex in examples:
        texts.append(format_training_example(
            claim_type=ex.claim_type,
            claim_text=ex.claim_text,
            source_sentence=ex.source_sentence,
            ehr_status=ex.ehr_status,
            hhem_score=ex.hhem_score,
            semantic_entropy=ex.semantic_entropy,
            risk_score=ex.risk_score,
            review_level=ex.review_level,
        ))

    return Dataset.from_dict({"text": texts})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare training data for Layer 2 meta-classifier"
    )
    parser.add_argument("--input", type=str, help="Path to NDJSON training file")
    parser.add_argument("--output", type=str, default="data/hf_dataset",
                        help="Output directory for HuggingFace Dataset")
    parser.add_argument("--test-mode", action="store_true",
                        help="Use mock data instead of real NDJSON")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.test_mode:
        logger.info("Test mode: generating mock dataset")
        dataset = prepare_mock_dataset(num_examples=50, seed=42)
    else:
        if not args.input:
            logger.error("--input is required when not using --test-mode")
            sys.exit(1)
        dataset = prepare_dataset(args.input)

    out_path = Path(args.output)
    out_path.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(out_path))
    logger.info("Saved dataset with %d examples to %s", len(dataset), out_path)


if __name__ == "__main__":
    main()
