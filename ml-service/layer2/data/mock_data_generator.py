"""
Synthetic clinical scenario generator for Layer 2 training data.

Provides 23 scenario templates across 4 claim types (medication, allergy,
diagnosis, vital_sign) with predetermined Layer 1 signals and physician labels.
Gaussian noise is added for diversity while keeping labels deterministic.
"""

import json
import logging
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TrainingExample:
    """A single labelled training example with Layer 1 signals and physician label."""
    claim_type: str
    claim_text: str
    source_sentence: str
    ehr_status: str
    hhem_score: Optional[float]
    semantic_entropy: Optional[float]
    risk_score: float
    review_level: str


# ── Scenario templates ───────────────────────────────────────────────────────
# Each tuple: (claim_type, claim_text, source_sentence, ehr_status,
#               hhem_score, semantic_entropy, risk_score, review_level)

SCENARIOS = [
    # ── Verified claims (low risk) ────────────────────────────────────────
    ("medication", "Patient takes Metoprolol 50mg daily",
     "The patient is currently on Metoprolol 50mg once daily.",
     "verified", None, None, 0.05, "BRIEF"),
    ("medication", "Lisinopril 10mg prescribed",
     "Lisinopril 10mg was prescribed for blood pressure management.",
     "verified", None, None, 0.05, "BRIEF"),
    ("allergy", "Patient allergic to Penicillin",
     "Known allergy to Penicillin documented in chart.",
     "verified", None, None, 0.03, "BRIEF"),
    ("diagnosis", "Type 2 diabetes mellitus",
     "Patient has a history of Type 2 diabetes mellitus.",
     "verified", None, None, 0.04, "BRIEF"),
    ("vital_sign", "Blood pressure 120/80 mmHg",
     "Vitals show blood pressure at 120/80 mmHg.",
     "verified", None, None, 0.02, "BRIEF"),

    # ── Not checkable (low risk) ──────────────────────────────────────────
    ("medication", "Patient reports taking over-the-counter ibuprofen",
     "Patient mentions occasional use of OTC ibuprofen for headaches.",
     "not_checkable", None, None, 0.10, "BRIEF"),
    ("diagnosis", "Patient feels fatigued",
     "Patient reports feeling fatigued for the past two weeks.",
     "not_checkable", None, None, 0.08, "BRIEF"),

    # ── Not found + faithful (low-medium risk) ────────────────────────────
    ("medication", "Started on Atorvastatin 20mg",
     "New prescription for Atorvastatin 20mg for cholesterol.",
     "not_found", 0.92, 0.15, 0.15, "STANDARD"),
    ("allergy", "Patient reports new allergy to Sulfa drugs",
     "Patient developed allergic reaction to Sulfa antibiotics.",
     "not_found", 0.88, 0.20, 0.18, "STANDARD"),
    ("diagnosis", "New diagnosis of hypothyroidism",
     "Labs confirm new diagnosis of hypothyroidism.",
     "not_found", 0.95, 0.10, 0.12, "STANDARD"),

    # ── Not found + hallucinated (high risk) ──────────────────────────────
    ("medication", "Patient on Warfarin 5mg",
     "Patient is currently taking Warfarin 5mg daily.",
     "not_found", 0.25, 0.55, 0.85, "DETAILED"),
    ("allergy", "Severe anaphylaxis to Latex documented",
     "Chart shows severe anaphylaxis reaction to Latex.",
     "not_found", 0.30, 0.60, 0.80, "DETAILED"),

    # ── Contradiction + high SE (medium risk — ambiguous transcript) ──────
    ("medication", "Metformin 1000mg twice daily",
     "Patient takes Metformin 1000mg BID for diabetes.",
     "contradiction", 0.65, 0.75, 0.50, "STANDARD"),
    ("vital_sign", "Heart rate recorded as 55 bpm",
     "Resting heart rate was measured at 55 bpm.",
     "contradiction", 0.70, 0.80, 0.45, "STANDARD"),
    ("diagnosis", "Mild aortic stenosis noted",
     "Echocardiogram shows mild aortic stenosis.",
     "contradiction", 0.60, 0.70, 0.50, "STANDARD"),

    # ── Contradiction + low SE + faithful (high risk) ─────────────────────
    ("medication", "Amlodipine 10mg daily",
     "Patient is on Amlodipine 10mg daily for hypertension.",
     "contradiction", 0.85, 0.20, 0.80, "DETAILED"),
    ("diagnosis", "Stage 3 chronic kidney disease",
     "Patient diagnosed with Stage 3 CKD based on GFR.",
     "contradiction", 0.82, 0.25, 0.75, "DETAILED"),
    ("vital_sign", "Oxygen saturation 88% on room air",
     "SpO2 recorded at 88% on room air.",
     "contradiction", 0.90, 0.18, 0.78, "DETAILED"),

    # ── Contradiction + low SE + unfaithful (CRITICAL) ────────────────────
    ("medication", "Digoxin 0.25mg daily",
     "Patient takes Digoxin 0.25mg once daily.",
     "contradiction", 0.15, 0.10, 0.95, "CRITICAL"),
    ("allergy", "No known drug allergies",
     "Patient denies any drug allergies.",
     "contradiction", 0.20, 0.12, 0.92, "CRITICAL"),
    ("medication", "Insulin glargine 20 units at bedtime",
     "Patient administers Insulin glargine 20 units nightly.",
     "contradiction", 0.18, 0.08, 0.95, "CRITICAL"),
    ("diagnosis", "No history of myocardial infarction",
     "Patient has no prior history of MI.",
     "contradiction", 0.22, 0.15, 0.90, "CRITICAL"),
    ("vital_sign", "Temperature 37.0°C",
     "Patient is afebrile with temperature at 37.0°C.",
     "contradiction", 0.10, 0.05, 0.95, "CRITICAL"),
]


def generate_mock_dataset(
    num_examples: int = 100,
    seed: int = 42,
) -> List[TrainingExample]:
    """
    Generate a deterministic synthetic dataset by cycling through scenarios
    and adding Gaussian noise to continuous signals for diversity.
    """
    rng = random.Random(seed)
    examples: List[TrainingExample] = []

    for i in range(num_examples):
        template = SCENARIOS[i % len(SCENARIOS)]
        claim_type, claim_text, source_sentence, ehr_status, hhem, se, risk, review = template

        # Add noise to scores (clamp to valid ranges)
        noisy_hhem = None
        if hhem is not None:
            noisy_hhem = max(0.0, min(1.0, hhem + rng.gauss(0, 0.03)))

        noisy_se = None
        if se is not None:
            noisy_se = max(0.0, min(1.0, se + rng.gauss(0, 0.02)))

        noisy_risk = max(0.0, min(1.0, risk + rng.gauss(0, 0.01)))

        examples.append(TrainingExample(
            claim_type=claim_type,
            claim_text=claim_text,
            source_sentence=source_sentence,
            ehr_status=ehr_status,
            hhem_score=noisy_hhem,
            semantic_entropy=noisy_se,
            risk_score=round(noisy_risk, 4),
            review_level=review,
        ))

    return examples


def write_ndjson(examples: List[TrainingExample], path: str) -> None:
    """Serialize training examples to newline-delimited JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for ex in examples:
            f.write(json.dumps(asdict(ex)) + "\n")
    logger.info("Wrote %d examples to %s", len(examples), path)
