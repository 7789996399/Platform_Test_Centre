"""
TRUST Platform - HHEM Faithfulness Scoring
==========================================
Scores claim faithfulness against source transcript using the
Vectara Hallucination Evaluation Model (HHEM).

HHEM is a CrossEncoder fine-tuned on NLI data to detect hallucination.
Given a (premise, hypothesis) pair it returns a score from 0.0 to 1.0:
    0.0 = hallucinated (not supported by source)
    1.0 = faithful (fully supported by source)

This module provides:
    - HHEMFaithfulnessScorer: real model inference via HuggingFace
    - MockHHEM: deterministic scorer for local testing (no GPU required)
    - create_scorer(): factory that picks the right implementation

Design follows source_verification.py patterns (enums, dataclasses,
normalize_text, batch processing with summary statistics).
"""

import re
import logging
from typing import List, Dict, Optional, Protocol
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

# Conditional import — graceful fallback when transformers not installed
try:
    from sentence_transformers import CrossEncoder
    HHEM_AVAILABLE = True
except ImportError:
    HHEM_AVAILABLE = False
    logger.info("sentence-transformers not installed — using MockHHEM only")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL_NAME = "vectara/hallucination_evaluation_model"

FAITHFULNESS_THRESHOLD_HIGH = 0.8
FAITHFULNESS_THRESHOLD_MED = 0.5
FAITHFULNESS_THRESHOLD_LOW = 0.2


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class FaithfulnessLevel(Enum):
    """Classified faithfulness level derived from raw HHEM score."""
    FAITHFUL = "faithful"                       # score >= 0.8
    PARTIALLY_FAITHFUL = "partially_faithful"   # 0.5 <= score < 0.8
    LIKELY_HALLUCINATED = "likely_hallucinated" # 0.2 <= score < 0.5
    HALLUCINATED = "hallucinated"               # score < 0.2


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FaithfulnessResult:
    """Result of scoring a single claim against a source transcript."""
    claim: str
    source_text: str
    score: float                        # 0.0 – 1.0
    level: FaithfulnessLevel
    needs_review: bool                  # True when score < FAITHFUL threshold
    explanation: str


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Normalize text for comparison — matches source_verification.py."""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def classify_score(score: float) -> FaithfulnessLevel:
    """Map a raw HHEM score to a FaithfulnessLevel."""
    if score >= FAITHFULNESS_THRESHOLD_HIGH:
        return FaithfulnessLevel.FAITHFUL
    elif score >= FAITHFULNESS_THRESHOLD_MED:
        return FaithfulnessLevel.PARTIALLY_FAITHFUL
    elif score >= FAITHFULNESS_THRESHOLD_LOW:
        return FaithfulnessLevel.LIKELY_HALLUCINATED
    else:
        return FaithfulnessLevel.HALLUCINATED


# ---------------------------------------------------------------------------
# Scorer protocol (interface)
# ---------------------------------------------------------------------------

class HHEMScorer(Protocol):
    """Interface that both real and mock scorers implement."""

    def score_claim(self, claim: str, source_text: str) -> FaithfulnessResult:
        ...

    def score_claims_batch(
        self, claims: List[str], source_text: str
    ) -> List[FaithfulnessResult]:
        ...


# ---------------------------------------------------------------------------
# Real implementation
# ---------------------------------------------------------------------------

class HHEMFaithfulnessScorer:
    """
    Scores faithfulness using the Vectara HHEM CrossEncoder.

    The model takes (source_text, claim) pairs and returns a probability
    that the claim is consistent with the source.  Internally it is an
    NLI model: premise = source_text, hypothesis = claim.

    Usage:
        scorer = HHEMFaithfulnessScorer()
        result = scorer.score_claim("Patient has penicillin allergy",
                                     "Doctor: Any allergies? Patient: Yes, penicillin.")
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        if not HHEM_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required for HHEMFaithfulnessScorer. "
                "Install it with: pip install sentence-transformers"
            )
        self.model_name = model_name
        self._model: Optional[CrossEncoder] = None

    def _load_model(self) -> "CrossEncoder":
        """Lazy-load the model on first use."""
        if self._model is None:
            logger.info("Loading HHEM model: %s", self.model_name)
            self._model = CrossEncoder(self.model_name)
            logger.info("HHEM model loaded successfully")
        return self._model

    def score_claim(self, claim: str, source_text: str) -> FaithfulnessResult:
        """
        Score a single claim against a source transcript.

        HHEM expects (premise, hypothesis) = (source_text, claim).
        Returns a FaithfulnessResult with score in [0.0, 1.0].
        """
        model = self._load_model()
        # HHEM input format: [[premise, hypothesis]]
        raw_scores = model.predict([(source_text, claim)])
        score = float(raw_scores[0])
        # Clamp to [0, 1] for safety
        score = max(0.0, min(1.0, score))
        level = classify_score(score)

        return FaithfulnessResult(
            claim=claim,
            source_text=source_text,
            score=score,
            level=level,
            needs_review=level != FaithfulnessLevel.FAITHFUL,
            explanation=_build_explanation(claim, score, level),
        )

    def score_claims_batch(
        self, claims: List[str], source_text: str
    ) -> List[FaithfulnessResult]:
        """
        Score multiple claims against the same source transcript.

        Uses batch prediction for efficiency — single model forward pass.
        """
        if not claims:
            return []

        model = self._load_model()
        pairs = [(source_text, claim) for claim in claims]
        raw_scores = model.predict(pairs)

        results: List[FaithfulnessResult] = []
        for claim, raw in zip(claims, raw_scores):
            score = max(0.0, min(1.0, float(raw)))
            level = classify_score(score)
            results.append(FaithfulnessResult(
                claim=claim,
                source_text=source_text,
                score=score,
                level=level,
                needs_review=level != FaithfulnessLevel.FAITHFUL,
                explanation=_build_explanation(claim, score, level),
            ))

        return results


# ---------------------------------------------------------------------------
# Mock implementation (for testing without model — per CLAUDE.md rules)
# ---------------------------------------------------------------------------

class MockHHEM:
    """
    Deterministic mock scorer for local testing.

    Scoring logic mirrors source_verification.py's substring approach:
    - Claim text found in source  → high score (0.95)
    - Partial keyword overlap     → medium score (0.60)
    - No overlap                  → low score (0.15)

    Keyword overrides allow test scenarios for specific claims:
        mock = MockHHEM(keyword_overrides={"penicillin": 0.10})
    """

    def __init__(
        self,
        default_score: float = 0.85,
        keyword_overrides: Optional[Dict[str, float]] = None,
    ):
        self.default_score = default_score
        self.keyword_overrides = keyword_overrides or {}

    def score_claim(self, claim: str, source_text: str) -> FaithfulnessResult:
        """Score using substring matching and keyword overrides."""
        score = self._compute_mock_score(claim, source_text)
        level = classify_score(score)

        return FaithfulnessResult(
            claim=claim,
            source_text=source_text,
            score=score,
            level=level,
            needs_review=level != FaithfulnessLevel.FAITHFUL,
            explanation=_build_explanation(claim, score, level, mock=True),
        )

    def score_claims_batch(
        self, claims: List[str], source_text: str
    ) -> List[FaithfulnessResult]:
        """Score a batch of claims using mock logic."""
        return [self.score_claim(claim, source_text) for claim in claims]

    def _compute_mock_score(self, claim: str, source_text: str) -> float:
        """
        Deterministic scoring for testing.

        Priority:
        1. Keyword overrides (exact control for test scenarios)
        2. Full claim substring match → 0.95
        3. Partial word overlap → scaled between 0.4 and 0.85
        4. No overlap → 0.15
        """
        claim_norm = normalize_text(claim)
        source_norm = normalize_text(source_text)

        # 1. Check keyword overrides first
        for keyword, override_score in self.keyword_overrides.items():
            if keyword.lower() in claim_norm:
                return override_score

        # 2. Full substring match
        if claim_norm in source_norm:
            return 0.95

        # 3. Partial word overlap
        claim_words = set(claim_norm.split()) - _STOP_WORDS
        source_words = set(source_norm.split()) - _STOP_WORDS
        if claim_words:
            overlap = len(claim_words & source_words) / len(claim_words)
            if overlap > 0:
                return 0.4 + (overlap * 0.45)  # Range: 0.4 – 0.85

        # 4. No overlap
        return 0.15


# Common stop words to ignore in overlap calculation
_STOP_WORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'been', 'be',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'shall', 'can', 'to', 'of',
    'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
    'that', 'which', 'who', 'whom', 'this', 'these', 'those',
    'it', 'its', 'and', 'but', 'or', 'nor', 'not', 'no', 'so',
    'if', 'then', 'than', 'too', 'very', 'just', 'about',
    'patient', 'noted', 'reported', 'states', 'denies',
}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_scorer(use_mock: bool = True) -> HHEMScorer:
    """
    Create the appropriate HHEM scorer.

    Defaults to MockHHEM per CLAUDE.md rules (no real API calls in test repo).
    Set use_mock=False to load the real Vectara HHEM model.
    """
    if use_mock:
        logger.info("Using MockHHEM scorer (test mode)")
        return MockHHEM()

    if not HHEM_AVAILABLE:
        logger.warning(
            "Real HHEM requested but sentence-transformers not installed — "
            "falling back to MockHHEM"
        )
        return MockHHEM()

    return HHEMFaithfulnessScorer()


# ---------------------------------------------------------------------------
# Summary statistics (matches verify_all_claims pattern)
# ---------------------------------------------------------------------------

def get_faithfulness_summary(results: List[FaithfulnessResult]) -> Dict:
    """
    Summarize batch scoring results.

    Mirrors the Dict structure returned by source_verification.verify_all_claims().
    """
    if not results:
        return {
            "total_claims": 0,
            "faithful": 0,
            "partially_faithful": 0,
            "likely_hallucinated": 0,
            "hallucinated": 0,
            "needs_review": 0,
            "mean_score": 0.0,
            "min_score": 0.0,
            "faithfulness_rate": 0.0,
            "results": [],
            "flagged_claims": [],
        }

    faithful = [r for r in results if r.level == FaithfulnessLevel.FAITHFUL]
    partial = [r for r in results if r.level == FaithfulnessLevel.PARTIALLY_FAITHFUL]
    likely_h = [r for r in results if r.level == FaithfulnessLevel.LIKELY_HALLUCINATED]
    hallucinated = [r for r in results if r.level == FaithfulnessLevel.HALLUCINATED]
    needs_review = [r for r in results if r.needs_review]

    scores = [r.score for r in results]

    return {
        "total_claims": len(results),
        "faithful": len(faithful),
        "partially_faithful": len(partial),
        "likely_hallucinated": len(likely_h),
        "hallucinated": len(hallucinated),
        "needs_review": len(needs_review),
        "mean_score": sum(scores) / len(scores),
        "min_score": min(scores),
        "faithfulness_rate": len(faithful) / len(results),
        "results": results,
        "flagged_claims": likely_h + hallucinated,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_explanation(
    claim: str, score: float, level: FaithfulnessLevel, mock: bool = False
) -> str:
    """Build a human-readable explanation string."""
    prefix = "[MOCK] " if mock else ""
    if level == FaithfulnessLevel.FAITHFUL:
        return f"{prefix}Claim is well-supported by source (score={score:.2f})"
    elif level == FaithfulnessLevel.PARTIALLY_FAITHFUL:
        return f"{prefix}Claim partially supported — review recommended (score={score:.2f})"
    elif level == FaithfulnessLevel.LIKELY_HALLUCINATED:
        return f"{prefix}Claim weakly supported — likely hallucination (score={score:.2f})"
    else:
        return f"{prefix}Claim not supported by source — hallucination detected (score={score:.2f})"
