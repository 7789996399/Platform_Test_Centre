"""
TRUST Platform - Multi-Layer Routing
=====================================
Integrates Paper 1 + Paper 2 into unified governance pipeline.

This is the "orchestrator" that:
1. Extracts claims from AI note
2. Runs source verification (fast)
3. Runs semantic entropy on unverified claims (expensive)
4. Calculates uncertainty and assigns review tiers
5. Returns prioritized review queue for physician
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json

from .claim_extraction import Claim, extract_claims_from_note, ClaimType
from .source_verification import verify_all_claims, VerificationResult, VerificationStatus
from .semantic_entropy import EntropyResult, analyze_claim, cluster_by_exact_match, calculate_entropy
from .uncertainty import (
    UncertaintyResult, quantify_uncertainty, 
    calculate_review_burden
)
from ..models.scribe import ReviewTier


@dataclass
class ClaimAnalysis:
    """Complete analysis of a single claim."""
    claim: Claim
    verification: VerificationResult
    entropy: Optional[EntropyResult]
    uncertainty: UncertaintyResult
    priority_score: float  # Higher = needs more attention
    

@dataclass  
class NoteAnalysis:
    """Complete analysis of an AI-generated note."""
    note_id: str
    patient_id: str
    analyzed_at: datetime
    total_claims: int
    claim_analyses: List[ClaimAnalysis]
    review_queue: List[ClaimAnalysis]  # Sorted by priority
    summary: Dict
    review_burden: Dict
    overall_risk: str  # LOW, MEDIUM, HIGH


def calculate_priority_score(
    verification: VerificationResult,
    entropy: Optional[EntropyResult],
    uncertainty: UncertaintyResult
) -> float:
    """
    Calculate priority score for review queue ordering.
    
    Higher score = needs physician attention sooner.
    
    Factors:
    - Contradicted claims: highest priority
    - High entropy: hallucination risk
    - Low confidence: uncertainty
    - High-risk claim types: medications, allergies
    """
    score = 0.0
    
    # Verification status
    if verification.status == VerificationStatus.CONTRADICTED:
        score += 50  # Top priority!
    elif verification.status == VerificationStatus.NOT_FOUND:
        score += 30
    elif verification.status == VerificationStatus.PARTIAL:
        score += 10
    
    # Entropy (if calculated)
    if entropy:
        score += entropy.entropy * 15  # 0-2+ range
    
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
    
    # Any contradictions = HIGH risk
    contradictions = [ca for ca in claim_analyses 
                      if ca.verification.status == VerificationStatus.CONTRADICTED]
    if contradictions:
        return "HIGH"
    
    # Multiple high-entropy claims = HIGH risk
    high_entropy = [ca for ca in claim_analyses 
                    if ca.entropy and ca.entropy.risk_level == "HIGH"]
    if len(high_entropy) >= 2:
        return "HIGH"
    
    # Any single high-entropy OR multiple unverified = MEDIUM
    unverified = [ca for ca in claim_analyses
                  if ca.verification.status == VerificationStatus.NOT_FOUND]
    if high_entropy or len(unverified) >= 3:
        return "MEDIUM"
    
    return "LOW"


def analyze_note(
    note: Dict,
    transcript: str,
    run_entropy: bool = True,
    mock_responses: Optional[Dict[str, List[str]]] = None
) -> NoteAnalysis:
    """
    Main entry point: Full analysis pipeline for an AI-generated note.
    
    Args:
        note: Parsed AI scribe note (JSON structure)
        transcript: Source transcript text
        run_entropy: Whether to run semantic entropy (expensive)
        mock_responses: For testing - pre-generated responses per claim
        
    Returns:
        Complete NoteAnalysis with prioritized review queue
    """
    # Step 1: Extract claims
    claims = extract_claims_from_note(note)
    
    # Step 2: Source verification (fast)
    verification_results = verify_all_claims(claims, transcript)
    
    # Step 3: Analyze each claim
    claim_analyses = []
    
    for result in verification_results['results']:
        claim = result.claim
        
        # Step 3a: Semantic entropy (only for unverified claims)
        entropy_result = None
        if run_entropy and result.needs_entropy_check:
            # In production, we'd generate multiple LLM responses here
            # For now, use mock responses or simple clustering
            if mock_responses and claim.id in mock_responses:
                responses = mock_responses[claim.id]
            else:
                # Placeholder: single response = 1 cluster = 0 entropy
                responses = [claim.text]
            
            clusters = cluster_by_exact_match(responses)
            cluster_sizes = [len(c) for c in clusters]
            entropy_val = calculate_entropy(cluster_sizes, len(responses))
            
            entropy_result = EntropyResult(
                entropy=entropy_val,
                n_clusters=len(clusters),
                n_responses=len(responses),
                cluster_sizes=cluster_sizes,
                risk_level="LOW" if entropy_val < 0.5 else "MEDIUM" if entropy_val < 1.0 else "HIGH"
            )
        
        # Step 3b: Uncertainty quantification
        uncertainty_result = quantify_uncertainty(
            claim=claim.text,
            claim_type=claim.claim_type.value,
            responses=[claim.text],  # Simplified for now
            entropy=entropy_result.entropy if entropy_result else 0.0
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


def format_review_queue_for_display(analysis: NoteAnalysis) -> List[Dict]:
    """
    Format review queue for frontend display.
    """
    display_items = []
    
    for i, ca in enumerate(analysis.review_queue[:20]):  # Top 20
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
    """
    Generate audit log entry for compliance.
    
    Required for FDA GMLP, EU AI Act, Health Canada compliance.
    """
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
