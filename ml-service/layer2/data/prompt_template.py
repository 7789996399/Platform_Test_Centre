"""
Prompt templates for encoding Layer 1 signals as structured text input
and parsing Layer 2 model JSON output.

The meta-classifier receives EHR status, HHEM score, and semantic entropy
as a structured markdown prompt and outputs a JSON risk assessment.
"""

import json
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a clinical risk assessor. Given verification signals for a "
    "medical claim, output a JSON risk assessment with risk_score (0.0-1.0) "
    "and review_level (BRIEF, STANDARD, DETAILED, or CRITICAL)."
)

CLAIM_TEMPLATE = """\
## Claim Assessment

**Claim type:** {claim_type}
**Claim:** {claim_text}
**Source sentence:** {source_sentence}

### Signal 1 — EHR Verification
Status: {ehr_status}

### Signal 2 — HHEM Faithfulness
Score: {hhem_score}

### Signal 3 — Semantic Entropy
Value: {semantic_entropy}

Provide your risk assessment as JSON."""


def format_input_prompt(
    claim_type: str,
    claim_text: str,
    source_sentence: str,
    ehr_status: str,
    hhem_score: Optional[float] = None,
    semantic_entropy: Optional[float] = None,
) -> str:
    """Format Layer 1 signals into the structured prompt for the model."""
    return CLAIM_TEMPLATE.format(
        claim_type=claim_type,
        claim_text=claim_text,
        source_sentence=source_sentence,
        ehr_status=ehr_status,
        hhem_score=f"{hhem_score:.4f}" if hhem_score is not None else "N/A",
        semantic_entropy=f"{semantic_entropy:.4f}" if semantic_entropy is not None else "N/A",
    )


def format_target_response(risk_score: float, review_level: str) -> str:
    """Format the target (label) as a JSON string for training."""
    return json.dumps(
        {"risk_score": round(risk_score, 4), "review_level": review_level},
        separators=(",", ":"),
    )


def format_training_example(
    claim_type: str,
    claim_text: str,
    source_sentence: str,
    ehr_status: str,
    hhem_score: Optional[float],
    semantic_entropy: Optional[float],
    risk_score: float,
    review_level: str,
) -> str:
    """Build a complete training string: system + input + target."""
    system = f"<|system|>\n{SYSTEM_PROMPT}\n"
    user = f"<|user|>\n{format_input_prompt(claim_type, claim_text, source_sentence, ehr_status, hhem_score, semantic_entropy)}\n"
    assistant = f"<|assistant|>\n{format_target_response(risk_score, review_level)}"
    return system + user + assistant


def parse_model_response(text: str) -> dict:
    """
    Extract risk_score and review_level from model output.

    Attempts JSON parsing first, then falls back to regex extraction.
    Returns ``{"risk_score": 0.5, "review_level": "STANDARD"}`` on failure.
    """
    # Try to find a JSON object in the text
    json_match = re.search(r"\{[^}]+\}", text)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            score = parsed.get("risk_score")
            level = parsed.get("review_level")
            if isinstance(score, (int, float)) and isinstance(level, str):
                level = level.upper()
                if level in ("BRIEF", "STANDARD", "DETAILED", "CRITICAL"):
                    return {
                        "risk_score": max(0.0, min(1.0, float(score))),
                        "review_level": level,
                    }
        except (json.JSONDecodeError, ValueError):
            pass

    # Regex fallback for score
    score_match = re.search(r"risk_score[\"']?\s*[:=]\s*([0-9]*\.?[0-9]+)", text)
    level_match = re.search(
        r"review_level[\"']?\s*[:=]\s*[\"']?(BRIEF|STANDARD|DETAILED|CRITICAL)",
        text,
        re.IGNORECASE,
    )

    if score_match and level_match:
        return {
            "risk_score": max(0.0, min(1.0, float(score_match.group(1)))),
            "review_level": level_match.group(1).upper(),
        }

    logger.warning("Could not parse model response, using defaults: %s", text[:200])
    return {"risk_score": 0.5, "review_level": "STANDARD"}
