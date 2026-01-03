"""
TRUST Platform - Multi-Layer Routing
=====================================
Integrates Paper 1 + Paper 2 into unified governance pipeline.

This is the "orchestrator" that:
1. Extracts claims from AI note
2. Runs source verification (fast)
3. Runs semantic entropy on unverified claims (via ML service)
4. Calculates uncertainty and assigns review tiers
5. Returns prioritized review queue for physician
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import httpx
import os

from .claim_extraction import Claim, extract_claims_from_note, ClaimType
from .source_verification import verify_all_claims, VerificationResult, VerificationStatus
from .semantic_entropy import EntropyResult
from .uncertainty import UncertaintyResult, calculate_review_burden
from ..models.scribe import ReviewTier

# ML Service URL
ML_SERVICE_URL = os.getenv("ML_SERVICE_URL", "https://trust-ml-service.azurewebsites.net")


@dataclass
class ClaimAnalysis:
    """Complete analysis of a single claim."""
    claim: Claim
    verification: VerificationResult
    entropy: Optional[EntropyResult]
    uncertainty: UncertaintyResult
    priority_score: float


@dataclass  
class NoteAnalysis:
    """Complete analysis of an AI-generated note."""
    note_id: str
    patient_id: str
    analyzed_at: datetime
    total_claims: int
    claim_analyses: List[ClaimAnalysis]
    review_queue: List[ClaimAnalysis]
    summary: Dict
    review_burden: Dict
    overall_risk: str


# =============================================================
# ML SERVICE CALLS
# =============================================================

async def call_ml_service_entropy(text: str, num_samples: int = 5) -> dict:
    """Call ML microservice for real semantic entropy calculation."""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{ML_SERVICE_URL}/analyze/semantic-entropy",
                json={"text": text, "num_samples": num_samples}
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        print(f"ML service entropy error: {e}")
        return {"entropy": 0.5, "confidence": 0.5, "error": str(e)}


async def call_ml_service_uncertainty(text: str) -> dict:
    """Call ML microservice for calibrated uncertainty."""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{ML_SERVICE_URL}/analyze/uncertainty",
                json={"text": text, "method": "calibrated"}
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        print(f"ML service uncertainty error: {e}")
        return {"uncertainty": 0.5, "calibrated_confidence": 0.5, "error": str(e)}


# =============================================================
# SCORING FUNCTIONS
# =============================================================

def calculate_priority_score(
    verification: VerificationResult,
    entropy: Optional[EntropyResult],
    uncertainty: UncertaintyResult
) -> float:
    """
    Calculate priority score for review queue ordering.
    Higher score = needs physician attention sooner.
    """
    score = 0.0
    
   # Verification status
    if verification.status == VerificationStatus.CONTRADICTED:
        score += 50
        
        # CONFIDENT HALLUCINATOR DETECTION
        # Low SE + contradiction = AI is confident but WRONG (most dangerous!)
        if entropy and entropy.entropy < 0.3:
            score += 100  # CRITICAL: Push to top of queue
        # High SE + contradiction = transcript was ambiguous (less dangerous)
        elif entropy and entropy.entropy > 0.7:
            score += 20  # Still needs review, but AI was uncertain
            
    elif verification.status == VerificationStatus.NOT_FOUND:
        score += 30
    elif verification.status == VerificationStatus.PARTIAL:
        score += 10
    
    # Entropy (if calculated) - for non-contradictions
    if entropy and verification.status != VerificationStatus.CONTRADICTED:
        score += entropy.entropy * 15 
    
    # Uncertainty
    score += (1 - uncertainty.calibrated_confidence) * 20
    
    # Review tier
    if uncertainty.review_tier == ReviewTier.DETAILED:
        score += 15
    elif uncertainty.review_tier == ReviewTier.STANDARD:
        score += 5
    
    # Claim risk level
    if verification.claim.risk_level == "high":
        score += 10
    
    # Flags
    score += len(uncertainty.flags) * 3
    
    return round(score, 2)


def determine_overall_risk(claim_analyses: List[ClaimAnalysis]) -> str:
    """Determine overall note risk level."""
    if not claim_analyses:
        return "LOW"
    
    # CRITICAL: Confident hallucinator (low SE + contradiction)
    confident_hallucinators = [
        ca for ca in claim_analyses 
        if ca.verification.status == VerificationStatus.CONTRADICTED
        and ca.entropy and ca.entropy.entropy < 0.3
    ]
    if confident_hallucinators:
        return "CRITICAL"
    
    contradictions = [ca for ca in claim_analyses 
                      if ca.verification.status == VerificationStatus.CONTRADICTED]
    if contradictions:
        return "HIGH"
    
    high_entropy = [ca for ca in claim_analyses 
                    if ca.entropy and ca.entropy.risk_level == "HIGH"]
    if len(high_entropy) >= 2:
        return "HIGH"
    
    unverified = [ca for ca in claim_analyses
                  if ca.verification.status == VerificationStatus.NOT_FOUND]
    if high_entropy or len(unverified) >= 3:
        return "MEDIUM"
    
    return "LOW"


# =============================================================
# MAIN ANALYSIS PIPELINE
# =============================================================

async def analyze_note(
    note: Dict,
    transcript: str,
    run_entropy: bool = True,
    mock_responses: Optional[Dict[str, List[str]]] = None
) -> NoteAnalysis:
    """
    Main entry point: Full analysis pipeline for an AI-generated note.
    Now calls ML microservice for real entropy and uncertainty.
    """
    # Step 1: Extract claims
    claims = extract_claims_from_note(note)
    
    # Step 2: Source verification (fast)
    verification_results = verify_all_claims(claims, transcript)
    
    # Step 2b: EHR verification (the TRUST advantage!)
    patient_id = note.get('patient', {}).get('id')
    if patient_id:
        from .ehr_verification import verify_note_against_ehr
        ehr_results = verify_note_against_ehr(claims, patient_id)
        # Merge EHR results into verification results
        for i, result in enumerate(verification_results['results']):
            for ehr_result in ehr_results.get('results', []):
                if result.claim.id == ehr_result.claim.id:
                    # EHR verification overrides source verification
                    if ehr_result.status.value == 'contradicted':
                        result.status = VerificationStatus.CONTRADICTED
                        result.explanation = f"EHR CONTRADICTION: {ehr_result.explanation}"
                    elif ehr_result.status.value == 'not_in_ehr':
                        result.status = VerificationStatus.NOT_FOUND
                        result.explanation = f"NOT IN EHR: {ehr_result.explanation}"
                    elif ehr_result.status.value == 'verified':
                        result.status = VerificationStatus.VERIFIED
                        result.explanation = f"EHR VERIFIED: {ehr_result.explanation}"
        verification_results['contradicted'] = ehr_results.get('contradicted', 0)
        # Recalculate counts after EHR verification updates
        verification_results['verified'] = len([r for r in verification_results['results'] if r.status == VerificationStatus.VERIFIED])
        verification_results['needs_entropy_check'] = len([r for r in verification_results['results'] if r.status != VerificationStatus.VERIFIED])
        verification_results['compute_saved_percent'] = (verification_results['verified'] / len(verification_results['results']) * 100) if verification_results['results'] else 0
    
    # Step 3: Analyze each claim
    
    # Step 3: Analyze each claim
    claim_analyses = []
    
    for result in verification_results['results']:
        claim = result.claim
        
        # Step 3a: Semantic entropy via ML service (ONLY for EHR contradictions!)
        # EHR-First approach: SE is expensive, so only run on contradictions
        entropy_result = None
        if run_entropy and result.status == VerificationStatus.CONTRADICTED:
            ml_entropy = await call_ml_service_entropy(claim.text, num_samples=5)
            entropy_val = ml_entropy.get("entropy", 0.5)
            
            if entropy_val < 0.3:
                risk_level = "LOW"
            elif entropy_val < 0.7:
                risk_level = "MEDIUM"
            else:
                risk_level = "HIGH"
            
            entropy_result = EntropyResult(
                entropy=entropy_val,
                n_clusters=ml_entropy.get("n_clusters", 1),
                n_responses=ml_entropy.get("num_samples", 5),
                cluster_sizes=[1],
                risk_level=risk_level
            )
        
        # Step 3b: Uncertainty - skip ML service for verified claims (EHR-First!)
        if result.status == VerificationStatus.VERIFIED:
            # EHR verified = high confidence, brief review
            calibrated_conf = 0.95
            review_tier = ReviewTier.BRIEF
            flags = []
        else:
            ml_uncertainty = await call_ml_service_uncertainty(claim.text)
            calibrated_conf = ml_uncertainty.get("calibrated_confidence", 0.5)
            
            if calibrated_conf >= 0.8:
                review_tier = ReviewTier.BRIEF
            elif calibrated_conf >= 0.5:
                review_tier = ReviewTier.STANDARD
            else:
                review_tier = ReviewTier.DETAILED
            
            interpretation = ml_uncertainty.get("details", {}).get("interpretation", "")
            flags = [interpretation.split(" - ")[0]] if interpretation else []
        
        uncertainty_result = UncertaintyResult(
            confidence=calibrated_conf,
            consistency=1.0,
            calibrated_confidence=calibrated_conf,
            review_tier=review_tier,
            flags=flags
        )
        
        # Step 3c: Calculate priority
        priority = calculate_priority_score(result, entropy_result, uncertainty_result)
        
        claim_analyses.append(ClaimAnalysis(
            claim=claim,
            verification=result,
            entropy=entropy_result,
            uncertainty=uncertainty_result,
            priority_score=priority
        ))
    
    # Step 4: Sort by priority for review queue
    review_queue = sorted(claim_analyses, key=lambda x: x.priority_score, reverse=True)
    
    # Step 5: Calculate review burden
    uncertainty_results = [ca.uncertainty for ca in claim_analyses]
    review_burden = calculate_review_burden(uncertainty_results)
    
    # Step 6: Summary
    summary = {
        "total_claims": len(claims),
        "verified": verification_results['verified'],
        "needs_review": verification_results['needs_entropy_check'],
        "contradictions": verification_results['contradicted'],
        "confident_hallucinators": len([ca for ca in claim_analyses 
            if ca.verification.status == VerificationStatus.CONTRADICTED 
            and ca.entropy and ca.entropy.entropy < 0.3]),
        "compute_saved_percent": verification_results['compute_saved_percent'],
        "high_priority_count": len([ca for ca in claim_analyses if ca.priority_score > 40])
    }
    
    return NoteAnalysis(
        note_id=note.get('note_id', 'unknown'),
        patient_id=note.get('patient', {}).get('id', 'unknown'),
        analyzed_at=datetime.now(),
        total_claims=len(claims),
        claim_analyses=claim_analyses,
        review_queue=review_queue,
        summary=summary,
        review_burden=review_burden,
        overall_risk=determine_overall_risk(claim_analyses)
    )


# =============================================================
# OUTPUT FORMATTING
# =============================================================

def format_review_queue_for_display(analysis: NoteAnalysis) -> List[Dict]:
    """Format review queue for frontend display."""
    display_items = []
    
    for i, ca in enumerate(analysis.review_queue[:20]):
        display_items.append({
            "rank": i + 1,
            "claim_text": ca.claim.text,
            "claim_type": ca.claim.claim_type.value,
            "risk_level": ca.claim.risk_level,
            "verification_status": ca.verification.status.value,
            "verification_note": ca.verification.explanation,
            "entropy": ca.entropy.entropy if ca.entropy else None,
            "entropy_risk": ca.entropy.risk_level if ca.entropy else None,
            "confidence": ca.uncertainty.calibrated_confidence,
            "review_tier": ca.uncertainty.review_tier.value,
            "flags": ca.uncertainty.flags,
            "priority_score": ca.priority_score
        })
    
    return display_items


def generate_audit_log(analysis: NoteAnalysis) -> Dict:
    """Generate audit log entry for compliance."""
    return {
        "timestamp": analysis.analyzed_at.isoformat(),
        "note_id": analysis.note_id,
        "patient_id": analysis.patient_id,
        "trust_version": "0.1.0",
        "analysis_summary": {
            "total_claims": analysis.total_claims,
            "overall_risk": analysis.overall_risk,
            "review_burden": analysis.review_burden,
            "claims_requiring_review": len([ca for ca in analysis.claim_analyses 
                                            if ca.uncertainty.review_tier != ReviewTier.BRIEF])
        },
        "flags_generated": [
            {
                "claim": ca.claim.text,
                "flags": ca.uncertainty.flags,
                "priority": ca.priority_score
            }
            for ca in analysis.claim_analyses if ca.uncertainty.flags
        ],
        "compliance_frameworks": [
            "FDA_GMLP",
            "EU_AI_Act", 
            "Health_Canada_SaMD",
            "WHO_AI_Ethics"
        ]
    }