"""
TRUST Platform - Source Verification
=====================================
Verifies AI claims against the source transcript.

This is the "fast first pass" - checking if claims appear in the
original conversation before running expensive semantic entropy.

Key insight from Paper 1: Most claims CAN be verified by source matching.
Only unverified claims need semantic entropy analysis.
This gives us 97% compute reduction.
"""

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# Import our claim types
from .claim_extraction import Claim, ClaimType


class VerificationStatus(Enum):
    """Result of source verification."""
    VERIFIED = "verified"           # Found in transcript
    NOT_FOUND = "not_found"         # Not in transcript - needs SE analysis
    CONTRADICTED = "contradicted"   # Transcript says opposite - flag!
    PARTIAL = "partial"             # Partially matches


@dataclass
class VerificationResult:
    """Result of verifying a single claim."""
    claim: Claim
    status: VerificationStatus
    transcript_match: Optional[str]    # What we found in transcript
    match_score: float                 # 0-1 similarity score
    needs_entropy_check: bool          # Should we run semantic entropy?
    explanation: str


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    # Lowercase, remove extra whitespace, standardize numbers
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def extract_medication_components(med_text: str) -> Dict:
    """
    Parse medication string into components.
    
    "Metoprolol 50mg PO BID" -> {name: metoprolol, dose: 50, unit: mg, ...}
    """
    components = {
        'name': None,
        'dose': None,
        'unit': None,
        'route': None,
        'frequency': None
    }
    
    text = med_text.lower()
    
    # Extract dose number and unit
    dose_match = re.search(r'(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|units?)', text)
    if dose_match:
        components['dose'] = float(dose_match.group(1))
        components['unit'] = dose_match.group(2)
    
    # Extract route
    route_match = re.search(r'\b(po|iv|im|sc|sq|pr|sl|topical)\b', text)
    if route_match:
        components['route'] = route_match.group(1)
    
    # Extract frequency
    freq_match = re.search(r'\b(daily|bid|tid|qid|prn|once|twice|weekly|q\d+h?)\b', text)
    if freq_match:
        components['frequency'] = freq_match.group(1)
    
    # Name is typically the first word
    name_match = re.match(r'^([a-z]+)', text)
    if name_match:
        components['name'] = name_match.group(1)
    
    return components


def verify_medication_claim(claim: Claim, transcript: str) -> VerificationResult:
    """
    Verify a medication claim against transcript.
    
    Checks:
    1. Is the medication name mentioned?
    2. Does the dose match?
    3. Is there a contradiction?
    """
    transcript_lower = normalize_text(transcript)
    claim_components = extract_medication_components(claim.text)
    
    med_name = claim_components.get('name', '')
    
    if not med_name:
        return VerificationResult(
            claim=claim,
            status=VerificationStatus.NOT_FOUND,
            transcript_match=None,
            match_score=0.0,
            needs_entropy_check=True,
            explanation="Could not parse medication name"
        )
    
    # Check if medication name appears in transcript
    if med_name not in transcript_lower:
        return VerificationResult(
            claim=claim,
            status=VerificationStatus.NOT_FOUND,
            transcript_match=None,
            match_score=0.0,
            needs_entropy_check=True,
            explanation=f"Medication '{med_name}' not found in transcript"
        )
    
    # Medication name found - now check dose
    claimed_dose = claim_components.get('dose')
    
    # Look for dose in transcript near medication name
    # Find context around medication mention
    med_index = transcript_lower.find(med_name)
    context_start = max(0, med_index - 50)
    context_end = min(len(transcript_lower), med_index + 100)
    context = transcript_lower[context_start:context_end]
    
    # Extract dose from transcript context
    dose_in_transcript = re.search(r'(\d+(?:\.\d+)?)\s*(?:mg|milligrams?)?', context)
    
    if dose_in_transcript and claimed_dose:
        transcript_dose = float(dose_in_transcript.group(1))
        if transcript_dose == claimed_dose:
            return VerificationResult(
                claim=claim,
                status=VerificationStatus.VERIFIED,
                transcript_match=context.strip(),
                match_score=1.0,
                needs_entropy_check=False,
                explanation=f"Medication and dose verified: {med_name} {claimed_dose}"
            )
        else:
            return VerificationResult(
                claim=claim,
                status=VerificationStatus.CONTRADICTED,
                transcript_match=context.strip(),
                match_score=0.3,
                needs_entropy_check=True,
                explanation=f"DOSE MISMATCH: Claim says {claimed_dose}, transcript says {transcript_dose}"
            )
    
    # Medication found but dose not confirmed
    return VerificationResult(
        claim=claim,
        status=VerificationStatus.PARTIAL,
        transcript_match=context.strip(),
        match_score=0.6,
        needs_entropy_check=True,
        explanation=f"Medication '{med_name}' found but dose not confirmed"
    )


def verify_allergy_claim(claim: Claim, transcript: str) -> VerificationResult:
    """Verify an allergy claim against transcript."""
    transcript_lower = normalize_text(transcript)
    claim_lower = normalize_text(claim.text)
    
    # Check for NKDA / no allergies
    if 'nkda' in claim_lower or 'no known' in claim_lower:
        if 'no' in transcript_lower and 'allerg' in transcript_lower:
            return VerificationResult(
                claim=claim,
                status=VerificationStatus.VERIFIED,
                transcript_match="Patient denied allergies",
                match_score=1.0,
                needs_entropy_check=False,
                explanation="No allergies confirmed in transcript"
            )
        if 'none' in transcript_lower:
            return VerificationResult(
                claim=claim,
                status=VerificationStatus.VERIFIED,
                transcript_match="None mentioned",
                match_score=0.9,
                needs_entropy_check=False,
                explanation="No allergies indicated"
            )
    
    # Check for specific allergens
    common_allergens = ['penicillin', 'sulfa', 'codeine', 'morphine', 'latex', 
                        'iodine', 'contrast', 'aspirin', 'nsaid', 'ace inhibitor']
    
    for allergen in common_allergens:
        if allergen in claim_lower:
            if allergen in transcript_lower:
                return VerificationResult(
                    claim=claim,
                    status=VerificationStatus.VERIFIED,
                    transcript_match=f"{allergen} allergy mentioned",
                    match_score=1.0,
                    needs_entropy_check=False,
                    explanation=f"Allergy to {allergen} confirmed in transcript"
                )
            else:
                return VerificationResult(
                    claim=claim,
                    status=VerificationStatus.NOT_FOUND,
                    transcript_match=None,
                    match_score=0.0,
                    needs_entropy_check=True,
                    explanation=f"Allergy to {allergen} NOT mentioned in transcript - VERIFY!"
                )
    
    return VerificationResult(
        claim=claim,
        status=VerificationStatus.NOT_FOUND,
        transcript_match=None,
        match_score=0.0,
        needs_entropy_check=True,
        explanation="Allergy claim needs verification"
    )


def verify_vital_sign_claim(claim: Claim, transcript: str) -> VerificationResult:
    """Verify vital sign claims - these often come from EHR, not transcript."""
    # Vitals are usually measured, not discussed - partial verification
    return VerificationResult(
        claim=claim,
        status=VerificationStatus.PARTIAL,
        transcript_match=None,
        match_score=0.5,
        needs_entropy_check=False,  # Vitals are typically reliable
        explanation="Vital signs typically from direct measurement"
    )


def verify_exam_finding_claim(claim: Claim, transcript: str) -> VerificationResult:
    """Verify physical exam findings."""
    transcript_lower = normalize_text(transcript)
    claim_lower = normalize_text(claim.text)
    
    # Check for common exam terms
    if 'murmur' in claim_lower:
        if 'murmur' in transcript_lower:
            return VerificationResult(
                claim=claim,
                status=VerificationStatus.VERIFIED,
                transcript_match="Murmur discussed",
                match_score=0.8,
                needs_entropy_check=False,
                explanation="Murmur finding mentioned in transcript"
            )
    
    if 'edema' in claim_lower:
        if 'edema' in transcript_lower or 'swelling' in transcript_lower:
            return VerificationResult(
                claim=claim,
                status=VerificationStatus.VERIFIED,
                transcript_match="Edema/swelling discussed",
                match_score=0.8,
                needs_entropy_check=False,
                explanation="Edema finding in transcript"
            )
        elif 'no swelling' in transcript_lower:
            return VerificationResult(
                claim=claim,
                status=VerificationStatus.CONTRADICTED,
                transcript_match="'No swelling' in transcript",
                match_score=0.0,
                needs_entropy_check=True,
                explanation="CONTRADICTION: Note says edema, transcript says no swelling"
            )
    
    return VerificationResult(
        claim=claim,
        status=VerificationStatus.PARTIAL,
        transcript_match=None,
        match_score=0.5,
        needs_entropy_check=True,
        explanation="Exam finding needs verification"
    )


def verify_claim(claim: Claim, transcript: str) -> VerificationResult:
    """
    Main entry point: Verify any claim against transcript.
    
    Routes to appropriate verification function based on claim type.
    """
    if claim.claim_type == ClaimType.MEDICATION:
        return verify_medication_claim(claim, transcript)
    elif claim.claim_type == ClaimType.ALLERGY:
        return verify_allergy_claim(claim, transcript)
    elif claim.claim_type == ClaimType.VITAL_SIGN:
        return verify_vital_sign_claim(claim, transcript)
    elif claim.claim_type == ClaimType.EXAM_FINDING:
        return verify_exam_finding_claim(claim, transcript)
    else:
        # Default: simple text search
        transcript_lower = normalize_text(transcript)
        claim_lower = normalize_text(claim.text)
        
        if claim_lower in transcript_lower:
            return VerificationResult(
                claim=claim,
                status=VerificationStatus.VERIFIED,
                transcript_match=claim.text,
                match_score=1.0,
                needs_entropy_check=False,
                explanation="Direct match found"
            )
        else:
            return VerificationResult(
                claim=claim,
                status=VerificationStatus.NOT_FOUND,
                transcript_match=None,
                match_score=0.0,
                needs_entropy_check=True,
                explanation="No match found - needs verification"
            )


def verify_all_claims(claims: List[Claim], transcript: str) -> Dict:
    """
    Verify all claims and summarize results.
    
    Returns summary with claims needing semantic entropy analysis.
    """
    results = [verify_claim(claim, transcript) for claim in claims]
    
    verified = [r for r in results if r.status == VerificationStatus.VERIFIED]
    not_found = [r for r in results if r.status == VerificationStatus.NOT_FOUND]
    contradicted = [r for r in results if r.status == VerificationStatus.CONTRADICTED]
    partial = [r for r in results if r.status == VerificationStatus.PARTIAL]
    
    needs_entropy = [r for r in results if r.needs_entropy_check]
    
    return {
        "total_claims": len(claims),
        "verified": len(verified),
        "not_found": len(not_found),
        "contradicted": len(contradicted),
        "partial": len(partial),
        "needs_entropy_check": len(needs_entropy),
        "verification_rate": len(verified) / len(claims) if claims else 0,
        "compute_saved_percent": (1 - len(needs_entropy) / len(claims)) * 100 if claims else 0,
        "results": results,
        "flagged_claims": contradicted + not_found
    }
