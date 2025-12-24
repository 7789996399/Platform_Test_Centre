"""
TRUST Platform - EHR-Based Verification
========================================
Verifies AI scribe claims against real Cerner EHR data.

This is the power of TRUST: comparing AI output against
the actual medical record, not just the transcript!
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum

from ..services.cerner import get_patient_context, PatientContext
from .claim_extraction import Claim, ClaimType


class EHRVerificationStatus(Enum):
    VERIFIED = "verified"          # Matches EHR data
    NOT_IN_EHR = "not_in_ehr"      # Not found in EHR
    CONTRADICTED = "contradicted"  # Conflicts with EHR
    EHR_UNAVAILABLE = "ehr_unavailable"  # Couldn't fetch EHR


@dataclass
class EHRVerificationResult:
    """Result of verifying a claim against EHR."""
    claim: Claim
    status: EHRVerificationStatus
    ehr_match: Optional[str]       # What we found in EHR
    confidence: float              # 0-1 match confidence
    explanation: str


def normalize(text: str) -> str:
    """Normalize text for comparison."""
    return text.lower().strip()


def medication_matches(claim_med: str, ehr_meds: List[str]) -> tuple[bool, Optional[str]]:
    """
    Check if a medication claim matches any EHR medication.
    
    Returns (matches, matched_medication)
    """
    claim_lower = normalize(claim_med)
    
    # Extract just the drug name (first word usually)
    claim_drug = claim_lower.split()[0] if claim_lower else ""
    
    for ehr_med in ehr_meds:
        ehr_lower = normalize(ehr_med)
        
        # Exact match
        if claim_drug in ehr_lower or ehr_lower in claim_lower:
            return True, ehr_med
        
        # Common drug name variations
        drug_aliases = {
            'tylenol': 'acetaminophen',
            'advil': 'ibuprofen',
            'motrin': 'ibuprofen',
            'lipitor': 'atorvastatin',
            'zocor': 'simvastatin',
            'norvasc': 'amlodipine',
            'lasix': 'furosemide',
            'plavix': 'clopidogrel',
        }
        
        for brand, generic in drug_aliases.items():
            if brand in claim_lower and generic in ehr_lower:
                return True, ehr_med
            if generic in claim_lower and brand in ehr_lower:
                return True, ehr_med
    
    return False, None


def allergy_matches(claim_allergy: str, ehr_allergies: List[str]) -> tuple[bool, Optional[str]]:
    """
    Check if an allergy claim matches any EHR allergy.
    """
    claim_lower = normalize(claim_allergy)
    
    # Handle NKDA
    if 'nkda' in claim_lower or 'no known' in claim_lower:
        for ehr_allergy in ehr_allergies:
            if 'no known' in normalize(ehr_allergy):
                return True, ehr_allergy
        # If patient has allergies listed, NKDA is wrong!
        if ehr_allergies and not any('no known' in normalize(a) for a in ehr_allergies):
            return False, None
    
    # Check specific allergies
    for ehr_allergy in ehr_allergies:
        ehr_lower = normalize(ehr_allergy)
        
        # Common allergens
        common_allergens = ['penicillin', 'sulfa', 'aspirin', 'ibuprofen', 
                          'latex', 'peanut', 'egg', 'milk', 'codeine']
        
        for allergen in common_allergens:
            if allergen in claim_lower and allergen in ehr_lower:
                return True, ehr_allergy
    
    return False, None


def verify_claim_against_ehr(
    claim: Claim, 
    patient_context: PatientContext
) -> EHRVerificationResult:
    """
    Verify a single claim against EHR data.
    """
    if claim.claim_type == ClaimType.MEDICATION:
        matches, matched = medication_matches(claim.text, patient_context.medications)
        
        if matches:
            return EHRVerificationResult(
                claim=claim,
                status=EHRVerificationStatus.VERIFIED,
                ehr_match=matched,
                confidence=0.9,
                explanation=f"Medication verified in EHR: {matched}"
            )
        else:
            return EHRVerificationResult(
                claim=claim,
                status=EHRVerificationStatus.NOT_IN_EHR,
                ehr_match=None,
                confidence=0.0,
                explanation=f"Medication '{claim.text}' not found in patient's EHR medication list"
            )
    
    elif claim.claim_type == ClaimType.ALLERGY:
        matches, matched = allergy_matches(claim.text, patient_context.allergies)
        
        if matches:
            return EHRVerificationResult(
                claim=claim,
                status=EHRVerificationStatus.VERIFIED,
                ehr_match=matched,
                confidence=0.9,
                explanation=f"Allergy verified in EHR: {matched}"
            )
        else:
            # Check if claiming NKDA but patient has allergies
            if 'nkda' in normalize(claim.text) or 'no known' in normalize(claim.text):
                if patient_context.allergies:
                    return EHRVerificationResult(
                        claim=claim,
                        status=EHRVerificationStatus.CONTRADICTED,
                        ehr_match=f"Patient has {len(patient_context.allergies)} allergies on file",
                        confidence=0.0,
                        explanation=f"CONTRADICTION: Note says NKDA but patient has allergies in EHR!"
                    )
            
            return EHRVerificationResult(
                claim=claim,
                status=EHRVerificationStatus.NOT_IN_EHR,
                ehr_match=None,
                confidence=0.0,
                explanation=f"Allergy '{claim.text}' not found in patient's EHR"
            )
    
    # For other claim types, we can't verify against EHR directly
    return EHRVerificationResult(
        claim=claim,
        status=EHRVerificationStatus.EHR_UNAVAILABLE,
        ehr_match=None,
        confidence=0.5,
        explanation=f"Claim type '{claim.claim_type.value}' cannot be verified against EHR"
    )


def verify_note_against_ehr(
    claims: List[Claim],
    patient_id: str
) -> Dict:
    """
    Verify all claims in a note against Cerner EHR data.
    
    This is the TRUST advantage: real EHR verification!
    """
    # Fetch patient context from Cerner
    patient_context = get_patient_context(patient_id)
    
    if not patient_context:
        return {
            "status": "error",
            "error": f"Could not fetch patient {patient_id} from Cerner",
            "results": []
        }
    
    results = []
    verified_count = 0
    contradicted_count = 0
    not_in_ehr_count = 0
    
    for claim in claims:
        result = verify_claim_against_ehr(claim, patient_context)
        results.append(result)
        
        if result.status == EHRVerificationStatus.VERIFIED:
            verified_count += 1
        elif result.status == EHRVerificationStatus.CONTRADICTED:
            contradicted_count += 1
        elif result.status == EHRVerificationStatus.NOT_IN_EHR:
            not_in_ehr_count += 1
    
    return {
        "status": "success",
        "patient_id": patient_id,
        "patient_name": patient_context.name,
        "total_claims": len(claims),
        "verified": verified_count,
        "contradicted": contradicted_count,
        "not_in_ehr": not_in_ehr_count,
        "ehr_medications_count": len(patient_context.medications),
        "ehr_allergies_count": len(patient_context.allergies),
        "results": results
    }
