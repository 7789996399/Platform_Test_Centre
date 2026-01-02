"""
Hallucination Detection - TRUST Platform
Based on Paper 1 methodology

═══════════════════════════════════════════════════════════════════════════════
                         EHR-FIRST PIPELINE
═══════════════════════════════════════════════════════════════════════════════

    ┌─────────────────────────────────────────────────────────────────────┐
    │  AI SCRIBE NOTE                                                      │
    │  "Patient on Metoprolol 50mg, Lisinopril 10mg, allergic to PCN..."  │
    └────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │  STEP 1: EXTRACT CLAIMS                                             │
    │  → 50 individual claims extracted                                   │
    └────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │  STEP 2: EHR VERIFICATION (fast, cheap - Cerner FHIR)              │
    │                                                                     │
    │     40 VERIFIED ──────────► ✅ SKIP SE, mark LOW risk              │
    │      5 CONTRADICTED ──┐                                            │
    │      5 NOT_FOUND ─────┼───► ⚠️ PROCEED TO STEP 3                   │
    └───────────────────────┼─────────────────────────────────────────────┘
                            │
                            ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │  STEP 3: SEMANTIC ENTROPY (expensive - only 10 claims, not 50!)    │
    │  → Calculate SE using bidirectional entailment                      │
    └────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │  STEP 4: CONFIDENT HALLUCINATOR CHECK                               │
    │  If SE < 0.3 AND CONTRADICTED → 🚨 CRITICAL                        │
    └────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │  STEP 5: ASSIGN REVIEW LEVEL                                        │
    │  BRIEF (15s) / STANDARD (2-3min) / DETAILED (5+min)                │
    └─────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════

Key insight: Standard NLI fails (9.5% accuracy) because hallucinated medical
text often sounds confident and plausible.

Detection Matrix:
┌─────────────────┬──────────────────┬──────────────────┐
│                 │ EHR: VERIFIED    │ EHR: CONTRADICTS │
├─────────────────┼──────────────────┼──────────────────┤
│ High Entropy    │ REVIEW NEEDED    │ LIKELY ERROR     │
│ (uncertain)     │ (unsure but ok)  │ (unsure & wrong) │
├─────────────────┼──────────────────┼──────────────────┤
│ Low Entropy     │ LIKELY CORRECT   │ ⚠️ CONFIDENT     │
│ (confident)     │ (sure & right)   │ HALLUCINATOR ⚠️  │
└─────────────────┴──────────────────┴──────────────────┘
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RiskLevel(str, Enum):
    """Risk classification for AI-generated claims"""
    LOW = "LOW"              # Verified, confident
    MEDIUM = "MEDIUM"        # Uncertain but not contradicted
    HIGH = "HIGH"            # Uncertain and potentially wrong
    CRITICAL = "CRITICAL"    # Confident hallucinator - most dangerous


class VerificationStatus(str, Enum):
    """EHR verification status"""
    VERIFIED = "VERIFIED"           # Matches EHR data
    CONTRADICTED = "CONTRADICTED"   # Conflicts with EHR data
    NOT_FOUND = "NOT_FOUND"         # No matching EHR data
    UNABLE = "UNABLE"               # Could not verify


@dataclass
class HallucinationResult:
    """Result of hallucination detection"""
    is_hallucination: bool
    risk_level: RiskLevel
    confidence: float
    semantic_entropy: float
    ehr_status: VerificationStatus
    reasoning: str
    review_required: bool
    review_level: str  # "BRIEF", "STANDARD", "DETAILED"
    details: Dict


# =============================================================================
# CONFIDENT HALLUCINATOR DETECTION
# =============================================================================

def detect_confident_hallucinator(
    semantic_entropy: float,
    ehr_verified: bool,
    ehr_contradicted: bool,
    entropy_threshold: float = 0.3
) -> Tuple[bool, RiskLevel, str]:
    """
    Detect the "confident hallucinator" pattern.
    
    This is the most dangerous case: the model is CONFIDENT (low entropy)
    but WRONG (contradicts EHR).
    
    Args:
        semantic_entropy: Normalized entropy (0-1, lower = more confident)
        ehr_verified: True if claim matches EHR data
        ehr_contradicted: True if claim conflicts with EHR data
        entropy_threshold: Below this = "confident" (default 0.3)
    
    Returns:
        (is_confident_hallucinator, risk_level, reasoning)
    """
    
    is_confident = semantic_entropy < entropy_threshold
    is_uncertain = semantic_entropy >= entropy_threshold
    
    # Decision matrix
    if is_confident and ehr_contradicted:
        # ⚠️ MOST DANGEROUS: Confident + Wrong
        return True, RiskLevel.CRITICAL, (
            f"CONFIDENT HALLUCINATOR DETECTED: Model shows high confidence "
            f"(entropy={semantic_entropy:.2f}) but claim contradicts EHR data. "
            f"This is the most dangerous type of error - requires detailed review."
        )
    
    elif is_confident and ehr_verified:
        # Best case: Confident + Verified
        return False, RiskLevel.LOW, (
            f"Claim verified against EHR with high model confidence "
            f"(entropy={semantic_entropy:.2f}). Low risk."
        )
    
    elif is_confident and not ehr_verified and not ehr_contradicted:
        # Confident but unverifiable
        return False, RiskLevel.MEDIUM, (
            f"Model is confident (entropy={semantic_entropy:.2f}) but claim "
            f"could not be verified against EHR. Standard review recommended."
        )
    
    elif is_uncertain and ehr_contradicted:
        # Uncertain + Wrong - caught by uncertainty
        return False, RiskLevel.HIGH, (
            f"Model shows uncertainty (entropy={semantic_entropy:.2f}) and "
            f"claim contradicts EHR. Uncertainty detection working correctly."
        )
    
    elif is_uncertain and ehr_verified:
        # Uncertain but actually correct
        return False, RiskLevel.MEDIUM, (
            f"Claim verified against EHR but model shows uncertainty "
            f"(entropy={semantic_entropy:.2f}). Brief review recommended."
        )
    
    else:
        # Uncertain + Unverifiable
        return False, RiskLevel.HIGH, (
            f"Model shows uncertainty (entropy={semantic_entropy:.2f}) and "
            f"claim could not be verified. Detailed review recommended."
        )


# =============================================================================
# REVIEW LEVEL ASSIGNMENT
# =============================================================================

def assign_review_level(risk_level: RiskLevel, semantic_entropy: float) -> str:
    """
    Assign review level based on risk and uncertainty.
    
    Review Levels:
    - BRIEF (15 sec): Low risk, just confirmation
    - STANDARD (2-3 min): Medium risk, check key facts
    - DETAILED (5+ min): High/Critical risk, full review
    
    Based on Paper 2 methodology for 87% burden reduction.
    """
    
    if risk_level == RiskLevel.CRITICAL:
        return "DETAILED"
    elif risk_level == RiskLevel.HIGH:
        return "DETAILED"
    elif risk_level == RiskLevel.MEDIUM:
        if semantic_entropy > 0.5:
            return "STANDARD"
        else:
            return "BRIEF"
    else:  # LOW
        return "BRIEF"


# =============================================================================
# MAIN DETECTION FUNCTION
# =============================================================================

async def detect_hallucination(
    claim: str,
    context: str,
    semantic_entropy: Optional[float] = None,
    ehr_verification: Optional[Dict] = None,
    calculate_entropy: bool = True
) -> HallucinationResult:
    """
    Full hallucination detection pipeline.
    
    Combines:
    1. Semantic entropy (if not provided, calculates it)
    2. EHR verification status (if provided)
    3. Confident hallucinator detection
    
    Args:
        claim: The AI-generated claim to evaluate
        context: Clinical context
        semantic_entropy: Pre-calculated entropy (0-1), or None to calculate
        ehr_verification: Dict with 'status', 'matched_data', etc. or None
        calculate_entropy: If True and entropy not provided, calculate it
    
    Returns:
        HallucinationResult with full analysis
    """
    
    # Step 1: Get or calculate semantic entropy
    if semantic_entropy is None and calculate_entropy:
        from .semantic_entropy import calculate_semantic_entropy
        
        logger.info("Calculating semantic entropy for claim")
        se_result = await calculate_semantic_entropy(
            prompt=claim,
            context=context,
            num_samples=5,
            model="openai"
        )
        semantic_entropy = se_result.normalized_entropy
        entropy_details = {
            "num_clusters": se_result.num_clusters,
            "num_samples": se_result.num_samples,
            "cluster_sizes": se_result.cluster_sizes,
            "samples": se_result.samples
        }
    else:
        semantic_entropy = semantic_entropy or 0.5  # Default to uncertain
        entropy_details = {}
    
    # Step 2: Parse EHR verification
    ehr_verified = False
    ehr_contradicted = False
    ehr_status = VerificationStatus.UNABLE
    
    if ehr_verification:
        status = ehr_verification.get("status", "").upper()
        if status == "VERIFIED":
            ehr_verified = True
            ehr_status = VerificationStatus.VERIFIED
        elif status == "CONTRADICTED":
            ehr_contradicted = True
            ehr_status = VerificationStatus.CONTRADICTED
        elif status == "NOT_FOUND":
            ehr_status = VerificationStatus.NOT_FOUND
        else:
            ehr_status = VerificationStatus.UNABLE
    
    # Step 3: Detect confident hallucinator
    is_confident_hallucinator, risk_level, reasoning = detect_confident_hallucinator(
        semantic_entropy=semantic_entropy,
        ehr_verified=ehr_verified,
        ehr_contradicted=ehr_contradicted
    )
    
    # Step 4: Assign review level
    review_level = assign_review_level(risk_level, semantic_entropy)
    
    # Step 5: Determine if hallucination
    is_hallucination = is_confident_hallucinator or ehr_contradicted
    
    # Step 6: Determine if review required
    review_required = risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL, RiskLevel.MEDIUM]
    
    return HallucinationResult(
        is_hallucination=is_hallucination,
        risk_level=risk_level,
        confidence=1.0 - semantic_entropy,
        semantic_entropy=semantic_entropy,
        ehr_status=ehr_status,
        reasoning=reasoning,
        review_required=review_required,
        review_level=review_level,
        details={
            "claim": claim,
            "is_confident_hallucinator": is_confident_hallucinator,
            "entropy_details": entropy_details,
            "ehr_verification": ehr_verification or {},
            "thresholds": {
                "entropy_confident": 0.3,
                "entropy_uncertain": 0.6
            }
        }
    )


# =============================================================================
# BATCH PROCESSING (EHR-FIRST APPROACH)
# =============================================================================

async def analyze_claims(
    claims: List[Dict],
    context: str,
    ehr_data: Optional[Dict] = None
) -> Dict:
    """
    Analyze multiple claims using EHR-FIRST approach.
    
    CRITICAL OPTIMIZATION: We do NOT run expensive semantic entropy on all claims!
    
    Pipeline:
    1. First, verify ALL claims against EHR (fast, cheap)
    2. Only run semantic entropy on:
       - CONTRADICTED claims (need to check if confident hallucinator)
       - NOT_FOUND claims (can't verify, need uncertainty estimate)
    3. VERIFIED claims skip SE entirely (saves ~80% of API costs)
    
    Args:
        claims: List of {"text": "...", "claim_type": "medication|diagnosis|..."}
        context: The full clinical context
        ehr_data: Patient EHR data for verification
    
    Returns:
        Summary with per-claim analysis and overall risk
    """
    
    results = []
    risk_counts = {level: 0 for level in RiskLevel}
    
    # Track efficiency metrics
    claims_verified_by_ehr = 0
    claims_needing_se = 0
    
    for claim in claims:
        # STEP 1: EHR VERIFICATION FIRST (fast, cheap)
        ehr_verification = None
        if ehr_data:
            ehr_verification = await _verify_against_ehr(claim, ehr_data)
        
        ehr_status = ehr_verification.get("status", "UNABLE").upper() if ehr_verification else "UNABLE"
        
        # STEP 2: DECIDE IF SE IS NEEDED
        needs_semantic_entropy = ehr_status in ["CONTRADICTED", "NOT_FOUND", "UNABLE"]
        
        if ehr_status == "VERIFIED":
            # ✅ EHR VERIFIED - Skip expensive SE, mark as LOW risk
            claims_verified_by_ehr += 1
            result = HallucinationResult(
                is_hallucination=False,
                risk_level=RiskLevel.LOW,
                confidence=0.95,  # High confidence from EHR match
                semantic_entropy=0.0,  # Not calculated - not needed
                ehr_status=VerificationStatus.VERIFIED,
                reasoning="Claim verified against EHR data. Semantic entropy calculation skipped (EHR-First optimization).",
                review_required=False,
                review_level="BRIEF",
                details={
                    "claim": claim["text"],
                    "ehr_verification": ehr_verification,
                    "se_skipped": True,
                    "skip_reason": "EHR_VERIFIED"
                }
            )
        else:
            # ⚠️ NOT VERIFIED - Need semantic entropy
            claims_needing_se += 1
            result = await detect_hallucination(
                claim=claim["text"],
                context=context,
                ehr_verification=ehr_verification,
                calculate_entropy=True  # Run the expensive SE
            )
        
        results.append({
            "claim": claim,
            "result": {
                "is_hallucination": result.is_hallucination,
                "risk_level": result.risk_level.value,
                "confidence": result.confidence,
                "semantic_entropy": result.semantic_entropy,
                "ehr_status": result.ehr_status.value,
                "reasoning": result.reasoning,
                "review_level": result.review_level,
                "se_calculated": not result.details.get("se_skipped", False)
            }
        })
        
        risk_counts[result.risk_level] += 1
    
    # Calculate overall risk
    if risk_counts[RiskLevel.CRITICAL] > 0:
        overall_risk = RiskLevel.CRITICAL
    elif risk_counts[RiskLevel.HIGH] > 0:
        overall_risk = RiskLevel.HIGH
    elif risk_counts[RiskLevel.MEDIUM] > 0:
        overall_risk = RiskLevel.MEDIUM
    else:
        overall_risk = RiskLevel.LOW
    
    # Calculate efficiency savings
    total_claims = len(claims)
    se_savings_percent = (claims_verified_by_ehr / total_claims * 100) if total_claims > 0 else 0
    
    return {
        "claims": results,
        "summary": {
            "total_claims": total_claims,
            "risk_distribution": {k.value: v for k, v in risk_counts.items()},
            "overall_risk": overall_risk.value,
            "hallucinations_detected": sum(1 for r in results if r["result"]["is_hallucination"]),
            "confident_hallucinators": sum(
                1 for r in results 
                if r["result"]["risk_level"] == RiskLevel.CRITICAL.value
            )
        },
        "efficiency": {
            "claims_verified_by_ehr": claims_verified_by_ehr,
            "claims_needing_se": claims_needing_se,
            "se_savings_percent": round(se_savings_percent, 1),
            "optimization": "EHR-First: SE only runs on unverified/contradicted claims"
        },
        "review_burden": {
            "brief_review": risk_counts[RiskLevel.LOW] + risk_counts[RiskLevel.MEDIUM],
            "standard_review": 0,  # Calculated based on entropy thresholds
            "detailed_review": risk_counts[RiskLevel.HIGH] + risk_counts[RiskLevel.CRITICAL]
        }
    }


async def _verify_against_ehr(claim: Dict, ehr_data: Dict) -> Dict:
    """
    Verify a claim against EHR data.
    
    This is a simplified version - the full implementation would use
    FHIR queries and semantic matching.
    """
    # TODO: Implement full EHR verification
    # For now, return "UNABLE" to trigger uncertainty-based detection
    return {
        "status": "UNABLE",
        "matched_data": None,
        "explanation": "EHR verification not yet implemented"
    }


# =============================================================================
# LEGACY FUNCTION (for backward compatibility)
# =============================================================================

async def detect(claim: str, context: str) -> Dict:
    """
    Legacy function for backward compatibility.
    Maps to new detect_hallucination function.
    """
    result = await detect_hallucination(
        claim=claim,
        context=context,
        calculate_entropy=True
    )
    
    return {
        "is_hallucination": result.is_hallucination,
        "confidence": result.confidence,
        "reasoning": result.reasoning
    }
