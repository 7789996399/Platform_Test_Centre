"""
TRUST Platform - Uncertainty Quantification
============================================
Implements Paper 2 methodology for calibrated confidence.

Key insight: Semantic entropy catches "confidently wrong" (hallucinations).
Uncertainty quantification catches "doesn't know" (epistemic uncertainty).

These are ORTHOGONAL failure modes - we need both!
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from ..models.scribe import ReviewTier





@dataclass
class UncertaintyResult:
    """Result of uncertainty quantification."""
    confidence: float          # 0-1, model's stated confidence
    consistency: float         # 0-1, agreement across samples
    calibrated_confidence: float  # Adjusted confidence
    review_tier: ReviewTier
    flags: List[str]


def calculate_consistency(responses: List[str]) -> float:
    """
    Calculate consistency across multiple responses.
    
    Higher consistency = model is more certain.
    
    Args:
        responses: Multiple sampled responses
        
    Returns:
        Consistency score 0-1 (1 = all identical)
    """
    if len(responses) <= 1:
        return 1.0
    
    # Simple approach: check how many match the most common response
    from collections import Counter
    normalized = [r.strip().lower() for r in responses]
    counts = Counter(normalized)
    most_common_count = counts.most_common(1)[0][1]
    
    return most_common_count / len(responses)


def extract_verbalized_confidence(response: str) -> Optional[float]:
    """
    Extract confidence from model's verbalized statement.
    
    Looks for patterns like:
    - "I am 85% confident..."
    - "Confidence: high"
    - "I'm fairly certain..."
    
    Args:
        response: Model's response text
        
    Returns:
        Confidence 0-1, or None if not found
    """
    import re
    
    # Look for percentage patterns
    percentage_pattern = r'(\d{1,3})%?\s*(?:confident|certain|sure)'
    match = re.search(percentage_pattern, response.lower())
    if match:
        return min(int(match.group(1)) / 100, 1.0)
    
    # Look for word patterns
    high_confidence_words = ['certain', 'definitely', 'clearly', 'absolutely']
    medium_confidence_words = ['likely', 'probably', 'appears', 'seems']
    low_confidence_words = ['possibly', 'might', 'uncertain', 'unsure', 'unclear']
    
    response_lower = response.lower()
    
    for word in high_confidence_words:
        if word in response_lower:
            return 0.9
    
    for word in low_confidence_words:
        if word in response_lower:
            return 0.4
            
    for word in medium_confidence_words:
        if word in response_lower:
            return 0.7
    
    return None


def calibrate_confidence(
    verbalized: float,
    consistency: float,
    entropy: float
) -> float:
    """
    Combine signals into calibrated confidence score.
    
    Paper 2 insight: Verbalized confidence alone is often overconfident.
    We adjust using consistency and semantic entropy.
    
    Args:
        verbalized: Model's stated confidence (0-1)
        consistency: Response consistency (0-1)
        entropy: Semantic entropy (0+, higher = worse)
        
    Returns:
        Calibrated confidence (0-1)
    """
    # Entropy penalty: high entropy reduces confidence
    entropy_factor = max(0, 1 - (entropy / 2))  # 0 at entropy=2+
    
    # Weighted combination
    # - Consistency is most reliable signal (weight 0.4)
    # - Verbalized adjusted by entropy (weight 0.3)
    # - Direct entropy penalty (weight 0.3)
    
    calibrated = (
        0.4 * consistency +
        0.3 * verbalized * entropy_factor +
        0.3 * entropy_factor
    )
    
    return round(calibrated, 3)


def assign_review_tier(
    calibrated_confidence: float,
    claim_risk_category: str = "standard"
) -> ReviewTier:
    """
    Assign review tier based on confidence and claim risk.
    
    This is the key "physician in the loop" optimization:
    - High confidence, low risk → Brief review (save time)
    - Low confidence OR high risk → Detailed review (ensure safety)
    
    Args:
        calibrated_confidence: Adjusted confidence score
        claim_risk_category: 'low', 'standard', 'high' (medication, allergy, etc.)
        
    Returns:
        ReviewTier enum
    """
    # High-risk claims always get more scrutiny
    if claim_risk_category == "high":
        if calibrated_confidence >= 0.9:
            return ReviewTier.STANDARD
        else:
            return ReviewTier.DETAILED
    
    # Standard claims
    if calibrated_confidence >= 0.85:
        return ReviewTier.BRIEF
    elif calibrated_confidence >= 0.6:
        return ReviewTier.STANDARD
    else:
        return ReviewTier.DETAILED


def categorize_claim_risk(claim_type: str) -> str:
    """
    Categorize claim by inherent risk level.
    
    High-risk claims need more careful review even if AI is confident.
    """
    high_risk_types = {
        'medication', 'allergy', 'dosage', 'procedure',
        'diagnosis', 'contraindication', 'drug_interaction'
    }
    
    low_risk_types = {
        'demographics', 'visit_date', 'provider_name',
        'chief_complaint_timing'
    }
    
    claim_lower = claim_type.lower()
    
    if claim_lower in high_risk_types:
        return "high"
    elif claim_lower in low_risk_types:
        return "low"
    else:
        return "standard"


def quantify_uncertainty(
    claim: str,
    claim_type: str,
    responses: List[str],
    entropy: float,
    verbalized_confidence: Optional[float] = None
) -> UncertaintyResult:
    """
    Full uncertainty quantification for a claim.
    
    This is the main entry point for Paper 2 methodology.
    
    Args:
        claim: The clinical claim text
        claim_type: Type of claim (medication, allergy, etc.)
        responses: Multiple sampled responses about this claim
        entropy: Semantic entropy from Paper 1 analysis
        verbalized_confidence: Optional pre-extracted confidence
        
    Returns:
        UncertaintyResult with review tier assignment
    """
    # Calculate consistency
    consistency = calculate_consistency(responses)
    
    # Get verbalized confidence
    if verbalized_confidence is None and responses:
        verbalized_confidence = extract_verbalized_confidence(responses[0])
    if verbalized_confidence is None:
        verbalized_confidence = 0.7  # Default assumption
    
    # Calibrate
    calibrated = calibrate_confidence(verbalized_confidence, consistency, entropy)
    
    # Risk category
    risk_category = categorize_claim_risk(claim_type)
    
    # Assign review tier
    review_tier = assign_review_tier(calibrated, risk_category)
    
    # Generate flags
    flags = []
    if entropy > 1.0:
        flags.append("HIGH_ENTROPY")
    if consistency < 0.6:
        flags.append("LOW_CONSISTENCY")
    if risk_category == "high":
        flags.append("HIGH_RISK_CLAIM")
    if calibrated < 0.5:
        flags.append("LOW_CONFIDENCE")
    
    return UncertaintyResult(
        confidence=verbalized_confidence,
        consistency=consistency,
        calibrated_confidence=calibrated,
        review_tier=review_tier,
        flags=flags
    )


# =============================================================
# REVIEW BURDEN CALCULATION (Paper 2 key metric)
# =============================================================

def calculate_review_burden(results: List[UncertaintyResult]) -> Dict:
    """
    Calculate physician review burden statistics.
    
    Paper 2 showed 87% reduction in review burden using this approach.
    
    Args:
        results: List of uncertainty results for all claims
        
    Returns:
        Dict with burden statistics
    """
    if not results:
        return {
            "total_claims": 0,
            "brief_review": 0,
            "standard_review": 0,
            "detailed_review": 0,
            "estimated_review_seconds": 0,
            "time_saved_percent": 0.0,
            "brief_percent": 0.0,
        }
    
    tier_counts = {
        ReviewTier.BRIEF: 0,
        ReviewTier.STANDARD: 0,
        ReviewTier.DETAILED: 0
    }
    
    # Estimated review times (seconds)
    tier_times = {
        ReviewTier.BRIEF: 20,
        ReviewTier.STANDARD: 90,
        ReviewTier.DETAILED: 240
    }
    
    for result in results:
        tier_counts[result.review_tier] += 1
    
    total_claims = len(results)
    
    # Calculate time savings vs reviewing everything in detail
    actual_time = sum(
        tier_counts[tier] * tier_times[tier] 
        for tier in tier_counts
    )
    max_time = total_claims * tier_times[ReviewTier.DETAILED]
    
    time_saved_percent = ((max_time - actual_time) / max_time * 100) if max_time > 0 else 0
    
    return {
        "total_claims": total_claims,
        "brief_review": tier_counts[ReviewTier.BRIEF],
        "standard_review": tier_counts[ReviewTier.STANDARD],
        "detailed_review": tier_counts[ReviewTier.DETAILED],
        "estimated_review_seconds": actual_time,
        "time_saved_percent": round(time_saved_percent, 1),
        "brief_percent": round(tier_counts[ReviewTier.BRIEF] / total_claims * 100, 1),
    }
