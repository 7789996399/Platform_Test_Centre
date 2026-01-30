"""
TRUST Platform - HHEM Faithfulness Scorer Tests
================================================
Tests for backend/app/core/hhem_faithfulness.py

All tests use MockHHEM — no real model calls per CLAUDE.md rules.

Usage:
    pytest backend/tests/test_hhem_faithfulness.py -v
"""

import sys
import os
import pytest

# Ensure backend is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.core.hhem_faithfulness import (
    MockHHEM,
    FaithfulnessLevel,
    FaithfulnessResult,
    classify_score,
    create_scorer,
    get_faithfulness_summary,
    normalize_text,
    FAITHFULNESS_THRESHOLD_HIGH,
    FAITHFULNESS_THRESHOLD_MED,
    FAITHFULNESS_THRESHOLD_LOW,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_scorer():
    """Default MockHHEM scorer."""
    return MockHHEM()


@pytest.fixture
def transcript():
    """Sample clinical transcript for testing."""
    return (
        "Doctor: Good morning. Any allergies? "
        "Patient: Yes, I'm allergic to penicillin. "
        "Doctor: Current medications? "
        "Patient: I take metoprolol 50mg twice daily and lisinopril 10mg daily. "
        "Doctor: Any chest pain or shortness of breath? "
        "Patient: No chest pain, but I do get short of breath when climbing stairs."
    )


# =============================================================================
# Test 1: High score when claim is substring of source
# =============================================================================

class TestSubstringMatch:
    """MockHHEM should return high scores for claims found in source."""

    def test_exact_substring_scores_high(self, mock_scorer, transcript):
        result = mock_scorer.score_claim("allergic to penicillin", transcript)
        assert result.score >= 0.9
        assert result.level == FaithfulnessLevel.FAITHFUL
        assert result.needs_review is False

    def test_full_medication_substring(self, mock_scorer, transcript):
        result = mock_scorer.score_claim("metoprolol 50mg twice daily", transcript)
        assert result.score >= 0.9
        assert result.level == FaithfulnessLevel.FAITHFUL

    def test_symptom_substring(self, mock_scorer, transcript):
        result = mock_scorer.score_claim("short of breath when climbing stairs", transcript)
        assert result.score >= 0.9

    def test_denied_symptom_substring(self, mock_scorer, transcript):
        result = mock_scorer.score_claim("no chest pain", transcript)
        assert result.score >= 0.9
        assert result.level == FaithfulnessLevel.FAITHFUL


# =============================================================================
# Test 2: Low score when claim has no overlap with source
# =============================================================================

class TestNoOverlap:
    """MockHHEM should return low scores for claims absent from source."""

    def test_fabricated_medication(self, mock_scorer, transcript):
        # No words overlap with transcript (avoid "daily" which appears there)
        result = mock_scorer.score_claim(
            "ondansetron 4mg sublingual prn", transcript
        )
        assert result.score < FAITHFULNESS_THRESHOLD_LOW
        assert result.level == FaithfulnessLevel.HALLUCINATED
        assert result.needs_review is True

    def test_fabricated_diagnosis(self, mock_scorer, transcript):
        result = mock_scorer.score_claim(
            "diagnosed hepatitis cirrhosis", transcript
        )
        assert result.score < FAITHFULNESS_THRESHOLD_LOW
        assert result.needs_review is True

    def test_completely_unrelated(self, mock_scorer, transcript):
        result = mock_scorer.score_claim(
            "underwent appendectomy cholecystectomy", transcript
        )
        assert result.score < FAITHFULNESS_THRESHOLD_LOW

    def test_explanation_mentions_hallucination(self, mock_scorer, transcript):
        result = mock_scorer.score_claim(
            "ondansetron 4mg sublingual prn", transcript
        )
        assert "hallucination" in result.explanation.lower()


# =============================================================================
# Test 3: Keyword overrides work correctly
# =============================================================================

class TestKeywordOverrides:
    """Keyword overrides should take priority over substring matching."""

    def test_override_forces_low_score(self, transcript):
        scorer = MockHHEM(keyword_overrides={"penicillin": 0.10})
        result = scorer.score_claim("allergic to penicillin", transcript)
        assert result.score == 0.10
        assert result.level == FaithfulnessLevel.HALLUCINATED

    def test_override_forces_high_score(self, transcript):
        scorer = MockHHEM(keyword_overrides={"gabapentin": 0.95})
        result = scorer.score_claim("gabapentin 300mg daily", transcript)
        assert result.score == 0.95
        assert result.level == FaithfulnessLevel.FAITHFUL

    def test_override_case_insensitive(self, transcript):
        scorer = MockHHEM(keyword_overrides={"Penicillin": 0.05})
        result = scorer.score_claim("PENICILLIN allergy", transcript)
        assert result.score == 0.05

    def test_non_matching_override_ignored(self, transcript):
        scorer = MockHHEM(keyword_overrides={"codeine": 0.10})
        # "penicillin" claim should not be affected by "codeine" override
        result = scorer.score_claim("allergic to penicillin", transcript)
        assert result.score >= 0.9


# =============================================================================
# Test 4: Batch processing returns correct number of results
# =============================================================================

class TestBatchProcessing:
    """score_claims_batch should handle multiple claims correctly."""

    def test_batch_returns_correct_count(self, mock_scorer, transcript):
        claims = [
            "allergic to penicillin",
            "takes metoprolol 50mg",
            "gabapentin 300mg daily",
            "no chest pain",
        ]
        results = mock_scorer.score_claims_batch(claims, transcript)
        assert len(results) == 4

    def test_batch_preserves_order(self, mock_scorer, transcript):
        claims = [
            "allergic to penicillin",
            "gabapentin 300mg daily",
        ]
        results = mock_scorer.score_claims_batch(claims, transcript)
        assert results[0].claim == claims[0]
        assert results[1].claim == claims[1]

    def test_batch_empty_list(self, mock_scorer, transcript):
        results = mock_scorer.score_claims_batch([], transcript)
        assert results == []

    def test_batch_single_item(self, mock_scorer, transcript):
        results = mock_scorer.score_claims_batch(["no chest pain"], transcript)
        assert len(results) == 1

    def test_batch_results_are_faithfulness_results(self, mock_scorer, transcript):
        claims = ["allergic to penicillin", "gabapentin 300mg daily"]
        results = mock_scorer.score_claims_batch(claims, transcript)
        for r in results:
            assert isinstance(r, FaithfulnessResult)
            assert 0.0 <= r.score <= 1.0
            assert isinstance(r.level, FaithfulnessLevel)


# =============================================================================
# Test 5: Summary statistics calculate correctly
# =============================================================================

class TestSummaryStatistics:
    """get_faithfulness_summary should aggregate results correctly."""

    def test_summary_counts(self, mock_scorer, transcript):
        claims = [
            "allergic to penicillin",             # substring → FAITHFUL
            "metoprolol 50mg twice daily",         # substring → FAITHFUL
            "ondansetron 4mg sublingual prn",      # zero overlap → HALLUCINATED
        ]
        results = mock_scorer.score_claims_batch(claims, transcript)
        summary = get_faithfulness_summary(results)

        assert summary["total_claims"] == 3
        assert summary["faithful"] == 2
        assert summary["hallucinated"] >= 1
        assert summary["needs_review"] >= 1

    def test_summary_mean_score(self, mock_scorer, transcript):
        claims = ["allergic to penicillin", "no chest pain"]
        results = mock_scorer.score_claims_batch(claims, transcript)
        summary = get_faithfulness_summary(results)

        expected_mean = sum(r.score for r in results) / len(results)
        assert abs(summary["mean_score"] - expected_mean) < 1e-9

    def test_summary_min_score(self, mock_scorer, transcript):
        claims = [
            "allergic to penicillin",    # high score
            "gabapentin 300mg daily",    # low score
        ]
        results = mock_scorer.score_claims_batch(claims, transcript)
        summary = get_faithfulness_summary(results)

        assert summary["min_score"] == min(r.score for r in results)

    def test_summary_faithfulness_rate(self, mock_scorer, transcript):
        claims = [
            "allergic to penicillin",       # FAITHFUL
            "metoprolol 50mg twice daily",   # FAITHFUL
            "gabapentin 300mg daily",        # HALLUCINATED
        ]
        results = mock_scorer.score_claims_batch(claims, transcript)
        summary = get_faithfulness_summary(results)

        faithful_count = sum(1 for r in results if r.level == FaithfulnessLevel.FAITHFUL)
        expected_rate = faithful_count / len(results)
        assert abs(summary["faithfulness_rate"] - expected_rate) < 1e-9

    def test_summary_flagged_claims(self, mock_scorer, transcript):
        claims = [
            "allergic to penicillin",             # FAITHFUL → not flagged
            "ondansetron 4mg sublingual prn",      # HALLUCINATED → flagged
        ]
        results = mock_scorer.score_claims_batch(claims, transcript)
        summary = get_faithfulness_summary(results)

        assert len(summary["flagged_claims"]) >= 1
        for flagged in summary["flagged_claims"]:
            assert flagged.level in (
                FaithfulnessLevel.LIKELY_HALLUCINATED,
                FaithfulnessLevel.HALLUCINATED,
            )

    def test_summary_empty_input(self):
        summary = get_faithfulness_summary([])
        assert summary["total_claims"] == 0
        assert summary["mean_score"] == 0.0
        assert summary["faithfulness_rate"] == 0.0
        assert summary["flagged_claims"] == []


# =============================================================================
# Supporting unit tests
# =============================================================================

class TestClassifyScore:
    """classify_score should map thresholds correctly."""

    def test_faithful(self):
        assert classify_score(1.0) == FaithfulnessLevel.FAITHFUL
        assert classify_score(0.8) == FaithfulnessLevel.FAITHFUL
        assert classify_score(0.95) == FaithfulnessLevel.FAITHFUL

    def test_partially_faithful(self):
        assert classify_score(0.79) == FaithfulnessLevel.PARTIALLY_FAITHFUL
        assert classify_score(0.5) == FaithfulnessLevel.PARTIALLY_FAITHFUL

    def test_likely_hallucinated(self):
        assert classify_score(0.49) == FaithfulnessLevel.LIKELY_HALLUCINATED
        assert classify_score(0.2) == FaithfulnessLevel.LIKELY_HALLUCINATED

    def test_hallucinated(self):
        assert classify_score(0.19) == FaithfulnessLevel.HALLUCINATED
        assert classify_score(0.0) == FaithfulnessLevel.HALLUCINATED


class TestNormalizeText:
    """normalize_text should standardize whitespace and casing."""

    def test_lowercase(self):
        assert normalize_text("HELLO World") == "hello world"

    def test_collapse_whitespace(self):
        assert normalize_text("too   many    spaces") == "too many spaces"

    def test_strip(self):
        assert normalize_text("  padded  ") == "padded"


class TestCreateScorer:
    """Factory function should return correct implementation."""

    def test_default_returns_mock(self):
        scorer = create_scorer()
        assert isinstance(scorer, MockHHEM)

    def test_explicit_mock(self):
        scorer = create_scorer(use_mock=True)
        assert isinstance(scorer, MockHHEM)
