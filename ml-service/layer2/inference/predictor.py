"""
Inference module for the Layer 2 meta-classifier.

Provides ``Layer2Predictor`` for production (loads base model + LoRA adapters)
and ``MockPredictor`` for CPU testing (deterministic rules mirroring
the hand-crafted RiskAssessor decision matrix).

Usage:
    predictor = create_predictor(use_mock=True)
    result = predictor.predict(signals)
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Protocol, runtime_checkable

from layer2.data.prompt_template import (
    SYSTEM_PROMPT,
    format_input_prompt,
    parse_model_response,
)

logger = logging.getLogger(__name__)


@dataclass
class ClaimSignals:
    """Input signals from Layer 1 matching ExtractedClaim fields."""
    claim_type: str
    claim_text: str
    source_sentence: str
    ehr_status: str
    hhem_score: Optional[float] = None
    semantic_entropy: Optional[float] = None


@dataclass
class PredictionResult:
    """Output of the meta-classifier."""
    risk_score: float
    review_level: str
    confidence: float
    raw_output: str


@runtime_checkable
class Predictor(Protocol):
    """Interface for Layer 2 predictors."""

    def predict(self, signals: ClaimSignals) -> PredictionResult: ...

    def predict_batch(self, signals_list: List[ClaimSignals]) -> List[PredictionResult]: ...


class Layer2Predictor:
    """
    Production predictor that loads the base model + LoRA adapters
    and generates JSON risk assessments.
    """

    def __init__(self, adapter_path: str, base_model: Optional[str] = None):
        self._adapter_path = adapter_path
        self._base_model = base_model
        self._model = None
        self._tokenizer = None

    def _load(self):
        """Lazy-load model and tokenizer on first prediction."""
        if self._model is not None:
            return

        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel

        if self._base_model:
            base = AutoModelForCausalLM.from_pretrained(self._base_model, device_map="auto")
        else:
            # Infer base model from adapter config
            base = AutoModelForCausalLM.from_pretrained(
                self._adapter_path, device_map="auto"
            )

        self._model = PeftModel.from_pretrained(base, self._adapter_path)
        self._model.eval()

        self._tokenizer = AutoTokenizer.from_pretrained(self._adapter_path)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        logger.info("Loaded Layer2 model from %s", self._adapter_path)

    def predict(self, signals: ClaimSignals) -> PredictionResult:
        import torch

        self._load()

        prompt = format_input_prompt(
            claim_type=signals.claim_type,
            claim_text=signals.claim_text,
            source_sentence=signals.source_sentence,
            ehr_status=signals.ehr_status,
            hhem_score=signals.hhem_score,
            semantic_entropy=signals.semantic_entropy,
        )
        full_prompt = f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{prompt}\n<|assistant|>\n"

        inputs = self._tokenizer(full_prompt, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        generated = self._tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        parsed = parse_model_response(generated)

        return PredictionResult(
            risk_score=parsed["risk_score"],
            review_level=parsed["review_level"],
            confidence=1.0 - abs(parsed["risk_score"] - 0.5) * 0.2,
            raw_output=generated,
        )

    def predict_batch(self, signals_list: List[ClaimSignals]) -> List[PredictionResult]:
        return [self.predict(s) for s in signals_list]


# ── High-risk keywords (same pattern as MockHHEM) ────────────────────────
_HIGH_RISK_KEYWORDS = [
    "digoxin", "warfarin", "insulin", "morphine", "heparin",
    "chemotherapy", "blood transfusion", "anaphylaxis",
]


class MockPredictor:
    """
    Deterministic mock predictor mirroring the RiskAssessor decision matrix.

    Decision paths:
    1. VERIFIED → 0.05 / BRIEF
    2. NOT_CHECKABLE → 0.10 / BRIEF
    3. NOT_FOUND + faithful (hhem >= 0.5) → 0.15 / STANDARD
    4. NOT_FOUND + unfaithful (hhem < 0.5) → 0.85 / DETAILED
    5. CONTRADICTION + high SE (>= 0.6) → 0.50 / STANDARD
    6. CONTRADICTION + low SE + faithful (hhem >= 0.5) → 0.80 / DETAILED
    7. CONTRADICTION + low SE + unfaithful (hhem < 0.5) → 0.95 / CRITICAL
    8. Keyword override: high-risk drug detected → bump score by 0.05
    """

    def predict(self, signals: ClaimSignals) -> PredictionResult:
        status = signals.ehr_status.lower()
        hhem = signals.hhem_score
        se = signals.semantic_entropy
        claim_lower = signals.claim_text.lower()

        # Default
        risk_score = 0.5
        review_level = "STANDARD"

        if status == "verified":
            risk_score = 0.05
            review_level = "BRIEF"

        elif status == "not_checkable":
            risk_score = 0.10
            review_level = "BRIEF"

        elif status == "not_found":
            if hhem is not None and hhem >= 0.5:
                risk_score = 0.15
                review_level = "STANDARD"
            else:
                risk_score = 0.85
                review_level = "DETAILED"

        elif status == "contradiction":
            if se is not None and se >= 0.6:
                # High SE — transcript was ambiguous
                risk_score = 0.50
                review_level = "STANDARD"
            elif hhem is not None and hhem >= 0.5:
                # Low SE + faithful
                risk_score = 0.80
                review_level = "DETAILED"
            else:
                # Low SE + unfaithful → CRITICAL
                risk_score = 0.95
                review_level = "CRITICAL"

        # Keyword override
        if any(kw in claim_lower for kw in _HIGH_RISK_KEYWORDS):
            risk_score = min(1.0, risk_score + 0.05)

        confidence = 0.95  # Mock is always confident

        return PredictionResult(
            risk_score=round(risk_score, 4),
            review_level=review_level,
            confidence=confidence,
            raw_output=f"mock:{status}",
        )

    def predict_batch(self, signals_list: List[ClaimSignals]) -> List[PredictionResult]:
        return [self.predict(s) for s in signals_list]


def create_predictor(
    use_mock: bool = True,
    adapter_path: Optional[str] = None,
    base_model: Optional[str] = None,
) -> Predictor:
    """
    Factory function following the ``create_scorer()`` pattern.

    Parameters
    ----------
    use_mock : bool
        If True, returns a MockPredictor (no model loading).
    adapter_path : str, optional
        Path to saved LoRA adapter (required if use_mock=False).
    base_model : str, optional
        Base model name for Layer2Predictor.
    """
    if use_mock:
        logger.info("Using MockPredictor (deterministic rules)")
        return MockPredictor()

    if adapter_path is None:
        raise ValueError("adapter_path is required when use_mock=False")

    logger.info("Using Layer2Predictor with adapter at %s", adapter_path)
    return Layer2Predictor(adapter_path=adapter_path, base_model=base_model)
