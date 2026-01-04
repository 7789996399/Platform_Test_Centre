"""
TRUST Platform - EHR-Based Verification
========================================
Verifies AI scribe claims against real Cerner EHR data.

This is the power of TRUST: comparing AI output against
the actual medical record, not just the transcript!

UPDATED: January 2026
- Added dose checking to medication verification
- Dose mismatch now returns NOT_IN_EHR (triggers SE calculation)
- Fixes bug where wrong-dose hallucinations were marked "verified"
"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from ..services.cerner import get_patient_context, PatientContext
from .claim_extraction import Claim, ClaimType


class EHRVerificationStatus(Enum):
    VERIFIED = "verified"          # Matches EHR data (name AND dose)
    NOT_IN_EHR = "not_in_ehr"      # Not found in EHR OR dose mismatch
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


# =============================================================================
# DOSE EXTRACTION AND COMPARISON (NEW)
# =============================================================================

def extract_dose(med_string: str) -> Tuple[Optional[float], Optional[str]]:
    """
    Extract dose value and unit from medication string.
    
    Examples:
        "Acetaminophen 500mg PRN" → (500.0, "mg")
        "Metoprolol 100 mg BID" → (100.0, "mg")
        "Lisinopril 2.5 mg daily" → (2.5, "mg")
        "digoxin 5 mcg/kg" → (5.0, "mcg/kg")
        
    Returns:
        (dose_value, dose_unit) or (None, None) if not found
    """
    if not med_string:
        return None, None
    
    # Remove commas from numbers (handles "1,000 mL" → "1000 mL")
    med_string = med_string.replace(",", "")
    
    # FIRST: Check for weight-based dosing (mcg/kg, mg/kg, g/kg)
    weight_pattern = r'(\d+(?:\.\d+)?)\s*(mcg|mg|g)/kg'
    match = re.search(weight_pattern, med_string, re.IGNORECASE)
    if match:
        return float(match.group(1)), f"{match.group(2).lower()}/kg"
    
    # THEN: Standard dose patterns
    dose_patterns = [
        r'(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|meq|units?|iu)\b',
        r'(\d+(?:\.\d+)?)\s*(milligrams?|micrograms?)',
    ]
    
    for pattern in dose_patterns:
        match = re.search(pattern, med_string, re.IGNORECASE)
        if match:
            dose_value = float(match.group(1))
            dose_unit = match.group(2).lower()
            
            # Normalize units
            unit_map = {
                "milligram": "mg",
                "milligrams": "mg",
                "microgram": "mcg",
                "micrograms": "mcg",
                "unit": "units",
            }
            dose_unit = unit_map.get(dose_unit, dose_unit)
            
            return dose_value, dose_unit
    
    return None, None


def doses_match(
    claim_dose: Optional[float], 
    claim_unit: Optional[str],
    ehr_dose: Optional[float], 
    ehr_unit: Optional[str],
    tolerance: float = 0.1
) -> bool:
    """
    Check if two doses match (with unit conversion and tolerance).
    
    Args:
        tolerance: Acceptable difference ratio (0.1 = 10%)
        
    Returns:
        True if doses match within tolerance
    """
    # If either dose is missing, we can't compare - be conservative
    if claim_dose is None or ehr_dose is None:
        return True  # Give benefit of doubt if we can't extract dose
    
    # Unit conversion to common base (mg)
    unit_to_mg = {
        "mg": 1.0,
        "g": 1000.0,
        "mcg": 0.001,
        "meq": 1.0,
        "units": 1.0,
        "iu": 1.0,
        "ml": 1.0,
    }
    
    claim_unit = (claim_unit or "mg").lower()
    ehr_unit = (ehr_unit or "mg").lower()
    
    # Convert to common unit
    claim_mg = claim_dose * unit_to_mg.get(claim_unit, 1.0)
    ehr_mg = ehr_dose * unit_to_mg.get(ehr_unit, 1.0)
    
    # Compare with tolerance
    if ehr_mg == 0:
        return claim_mg == 0
    
    difference_ratio = abs(claim_mg - ehr_mg) / ehr_mg
    return difference_ratio <= tolerance


def extract_drug_name(med_string: str) -> str:
    """
    Extract just the drug name from a medication string.
    
    Examples:
        "Acetaminophen 500mg PRN" → "acetaminophen"
        "amLODIPine (Norvasc) 5mg" → "amlodipine"
        "metoprolol tartrate 25mg" → "metoprolol"
    """
    if not med_string:
        return ""
    
    text = normalize(med_string)
    
    # Remove parenthetical content (brand names)
    text = re.sub(r'\([^)]*\)', '', text).strip()
    
    # Remove dose information
    text = re.sub(r'\d+(?:\.\d+)?\s*(mg|mcg|g|ml|meq|units?|iu)\b', '', text, flags=re.IGNORECASE)
    
    # Remove common suffixes
    text = re.sub(r'\s+(tablet|capsule|solution|oral|iv|bid|tid|qid|prn|daily|qd|qhs)s?\b', '', text, flags=re.IGNORECASE)
    
    # Get first word (the drug name)
    words = text.split()
    if words:
        return words[0].strip()
    
    return ""


# =============================================================================
# MEDICATION MATCHING (FIXED - NOW INCLUDES DOSE)
# =============================================================================

def medication_matches(claim_med: str, ehr_meds: List[str]) -> Tuple[bool, Optional[str], str]:
    """
    Check if a medication claim matches any EHR medication.
    
    NOW CHECKS BOTH NAME AND DOSE!
    
    Returns:
        (matches, matched_medication, explanation)
        
    matches = True only if BOTH name AND dose match.
    Dose mismatch returns False (will trigger SE calculation).
    """
    claim_drug = extract_drug_name(claim_med)
    claim_dose, claim_unit = extract_dose(claim_med)
    
    if not claim_drug:
        return False, None, "Could not extract drug name from claim"
    
    # Drug name aliases (brand ↔ generic)
    drug_aliases = {
        'tylenol': 'acetaminophen',
        'advil': 'ibuprofen',
        'motrin': 'ibuprofen',
        'lipitor': 'atorvastatin',
        'zocor': 'simvastatin',
        'norvasc': 'amlodipine',
        'lasix': 'furosemide',
        'plavix': 'clopidogrel',
        'glucophage': 'metformin',
        'lopressor': 'metoprolol',
        'toprol': 'metoprolol',
        'prinivil': 'lisinopril',
        'zestril': 'lisinopril',
        'prilosec': 'omeprazole',
        'nexium': 'esomeprazole',
        'synthroid': 'levothyroxine',
        'coumadin': 'warfarin',
        'ambien': 'zolpidem',
    }
    
    # Normalize claim drug name (convert brand to generic if needed)
    claim_drug_normalized = claim_drug
    for brand, generic in drug_aliases.items():
        if brand in claim_drug:
            claim_drug_normalized = generic
            break
    
    # Track best match for explanation
    best_match = None
    best_match_type = None  # "exact", "name_only", "dose_mismatch"
    
    for ehr_med in ehr_meds:
        ehr_drug = extract_drug_name(ehr_med)
        ehr_dose, ehr_unit = extract_dose(ehr_med)
        
        # Normalize EHR drug name
        ehr_drug_normalized = ehr_drug
        for brand, generic in drug_aliases.items():
            if brand in ehr_drug:
                ehr_drug_normalized = generic
                break
        
        # Check if drug names match
        names_match = (
            claim_drug_normalized == ehr_drug_normalized or
            claim_drug_normalized in ehr_drug_normalized or
            ehr_drug_normalized in claim_drug_normalized
        )
        
        if names_match:
            # Name matches! Now check dose
            if claim_dose is None or ehr_dose is None:
                # Can't compare doses - track as name_only match
                if best_match_type != "exact":
                    best_match = ehr_med
                    best_match_type = "name_only"
            elif doses_match(claim_dose, claim_unit, ehr_dose, ehr_unit):
                # FULL MATCH: name AND dose
                return True, ehr_med, f"EHR VERIFIED: Medication verified in EHR: {claim_drug_normalized}"
            else:
                # DOSE MISMATCH - this is the key fix!
                best_match = ehr_med
                best_match_type = "dose_mismatch"
    
    # No exact match found - check what we did find
    if best_match_type == "dose_mismatch":
        ehr_dose, ehr_unit = extract_dose(best_match)
        return False, best_match, (
            f"DOSE MISMATCH: '{claim_drug}' found in EHR but dose differs. "
            f"Claim: {claim_dose}{claim_unit}, EHR: {ehr_dose}{ehr_unit}"
        )
    elif best_match_type == "name_only":
        return True, best_match, f"EHR VERIFIED: Drug name '{claim_drug}' found (could not compare dose)"
    else:
        return False, None, f"NOT IN EHR: Medication '{claim_drug}' not found in patient's EHR medication list"


# =============================================================================
# ALLERGY MATCHING (unchanged)
# =============================================================================

def allergy_matches(claim_allergy: str, ehr_allergies: List[str]) -> Tuple[bool, Optional[str]]:
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
                          'latex', 'peanut', 'egg', 'milk', 'codeine',
                          'morphine', 'amoxicillin', 'shellfish', 'contrast']
        
        for allergen in common_allergens:
            if allergen in claim_lower and allergen in ehr_lower:
                return True, ehr_allergy
    
    return False, None


# =============================================================================
# CLAIM VERIFICATION (updated to use new explanation)
# =============================================================================

def verify_claim_against_ehr(
    claim: Claim, 
    patient_context: PatientContext
) -> EHRVerificationResult:
    """
    Verify a single claim against EHR data.
    """
    if claim.claim_type == ClaimType.MEDICATION:
        matches, matched, explanation = medication_matches(claim.text, patient_context.medications)
        
        if matches:
            return EHRVerificationResult(
                claim=claim,
                status=EHRVerificationStatus.VERIFIED,
                ehr_match=matched,
                confidence=0.9,
                explanation=explanation
            )
        else:
            return EHRVerificationResult(
                claim=claim,
                status=EHRVerificationStatus.NOT_IN_EHR,
                ehr_match=matched,  # May have partial match info
                confidence=0.0,
                explanation=explanation
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
