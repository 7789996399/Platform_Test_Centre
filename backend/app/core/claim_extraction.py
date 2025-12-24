"""
TRUST Platform - Clinical Claim Extraction
==========================================
Extracts verifiable claims from AI-generated clinical notes.

A "claim" is any factual assertion that can be verified against
the source transcript or medical record.
"""

import re
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum


class ClaimType(Enum):
    """Types of clinical claims with associated risk levels."""
    MEDICATION = "medication"           # High risk
    ALLERGY = "allergy"                 # High risk
    DOSAGE = "dosage"                   # High risk
    DIAGNOSIS = "diagnosis"             # High risk
    PROCEDURE = "procedure"             # High risk
    VITAL_SIGN = "vital_sign"           # Medium risk
    LAB_VALUE = "lab_value"             # Medium risk
    EXAM_FINDING = "exam_finding"       # Medium risk
    SYMPTOM = "symptom"                 # Medium risk
    HISTORY = "history"                 # Medium risk
    DEMOGRAPHIC = "demographic"         # Low risk
    ADMINISTRATIVE = "administrative"   # Low risk


@dataclass
class Claim:
    """A single verifiable claim extracted from a note."""
    id: str
    text: str
    claim_type: ClaimType
    section: str                    # Which note section it came from
    risk_level: str                 # high, medium, low
    source_span: Optional[str]      # Where in transcript (if found)
    verified: Optional[bool]        # None = not yet checked
    confidence: Optional[float]     # AI confidence in this claim


def get_risk_level(claim_type: ClaimType) -> str:
    """Map claim type to risk level."""
    high_risk = {
        ClaimType.MEDICATION,
        ClaimType.ALLERGY,
        ClaimType.DOSAGE,
        ClaimType.DIAGNOSIS,
        ClaimType.PROCEDURE
    }
    low_risk = {
        ClaimType.DEMOGRAPHIC,
        ClaimType.ADMINISTRATIVE
    }
    
    if claim_type in high_risk:
        return "high"
    elif claim_type in low_risk:
        return "low"
    else:
        return "medium"


def extract_medications(text: str, section: str) -> List[Claim]:
    """
    Extract medication claims from text.
    
    Looks for patterns like:
    - "Metoprolol 50mg PO BID"
    - "aspirin 81mg daily"
    - "on lisinopril"
    """
    claims = []
    
    # Common medication pattern: Name + optional dose + optional route + optional frequency
    med_pattern = r'\b([A-Z][a-z]+(?:in|ol|il|ide|ate|one|ax|ex|ib|mab)?)\s*(\d+\s*(?:mg|mcg|g|ml|units?))?(?:\s*(PO|IV|IM|SC|SQ|PR|SL|topical))?(?:\s*(daily|BID|TID|QID|PRN|once|twice|weekly))?\b'
    
    # Common medication names to look for
    common_meds = [
        'metoprolol', 'lisinopril', 'aspirin', 'atorvastatin', 'warfarin',
        'clopidogrel', 'plavix', 'amiodarone', 'digoxin', 'furosemide',
        'lasix', 'heparin', 'insulin', 'morphine', 'fentanyl', 'propofol',
        'midazolam', 'rocuronium', 'succinylcholine', 'epinephrine',
        'norepinephrine', 'vasopressin', 'dobutamine', 'milrinone'
    ]
    
    text_lower = text.lower()
    
    for med in common_meds:
        if med in text_lower:
            # Find the full context around the medication mention
            pattern = rf'({med}\s*\d*\s*(?:mg|mcg|g)?\s*(?:PO|IV|IM)?\s*(?:daily|BID|TID|QID|once|twice)?)'
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            
            for match in matches:
                claims.append(Claim(
                    id=f"med_{len(claims)}_{med}",
                    text=match.strip(),
                    claim_type=ClaimType.MEDICATION,
                    section=section,
                    risk_level="high",
                    source_span=None,
                    verified=None,
                    confidence=None
                ))
    
    return claims


def extract_allergies(text: str, section: str) -> List[Claim]:
    """Extract allergy claims from text."""
    claims = []
    
    # Patterns for allergies
    allergy_patterns = [
        r'allerg(?:y|ic|ies)\s*(?:to\s+)?:?\s*([^.,]+)',
        r'NKDA',
        r'no known (?:drug )?allergies',
        r'([A-Za-z]+)\s*\(([^)]+(?:rash|anaphylaxis|hives|swelling)[^)]*)\)'
    ]
    
    for pattern in allergy_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            match_text = match if isinstance(match, str) else ' '.join(match)
            claims.append(Claim(
                id=f"allergy_{len(claims)}",
                text=match_text.strip(),
                claim_type=ClaimType.ALLERGY,
                section=section,
                risk_level="high",
                source_span=None,
                verified=None,
                confidence=None
            ))
    
    return claims


def extract_vital_signs(text: str, section: str) -> List[Claim]:
    """Extract vital sign claims."""
    claims = []
    
    vital_patterns = {
        'blood_pressure': r'BP\s*:?\s*(\d{2,3}/\d{2,3})',
        'heart_rate': r'(?:HR|heart rate|pulse)\s*:?\s*(\d{2,3})',
        'oxygen_sat': r'(?:SpO2|O2 sat|oxygen)\s*:?\s*(\d{2,3})%?',
        'temperature': r'(?:temp|temperature)\s*:?\s*(\d{2,3}(?:\.\d)?)',
        'respiratory_rate': r'(?:RR|resp(?:iratory)? rate)\s*:?\s*(\d{1,2})'
    }
    
    for vital_type, pattern in vital_patterns.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            claims.append(Claim(
                id=f"vital_{vital_type}_{len(claims)}",
                text=f"{vital_type}: {match}",
                claim_type=ClaimType.VITAL_SIGN,
                section=section,
                risk_level="medium",
                source_span=None,
                verified=None,
                confidence=None
            ))
    
    return claims


def extract_exam_findings(text: str, section: str) -> List[Claim]:
    """Extract physical exam findings."""
    claims = []
    
    # Common exam finding patterns
    exam_patterns = [
        r'(murmur[^.]*)',
        r'(edema[^.]*)',
        r'(regular (?:rate and )?rhythm)',
        r'(irregular rhythm)',
        r'(clear to auscultation)',
        r'(no (?:murmurs|gallops|rubs))',
        r'(JVD[^.]*)',
        r'(jugular venous[^.]*)'
    ]
    
    for pattern in exam_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            claims.append(Claim(
                id=f"exam_{len(claims)}",
                text=match.strip(),
                claim_type=ClaimType.EXAM_FINDING,
                section=section,
                risk_level="medium",
                source_span=None,
                verified=None,
                confidence=None
            ))
    
    return claims


def extract_claims_from_note(note: Dict) -> List[Claim]:
    """
    Main entry point: Extract all claims from an AI scribe note.
    
    Args:
        note: Parsed AI scribe note with sections
        
    Returns:
        List of all extracted claims
    """
    all_claims = []
    
    sections = note.get('ai_scribe_output', {}).get('sections', {})
    
    for section_name, section_text in sections.items():
        if isinstance(section_text, str):
            # Extract different claim types from each section
            all_claims.extend(extract_medications(section_text, section_name))
            all_claims.extend(extract_allergies(section_text, section_name))
            all_claims.extend(extract_vital_signs(section_text, section_name))
            all_claims.extend(extract_exam_findings(section_text, section_name))
        
        elif isinstance(section_text, list):
            # Handle medication lists
            if section_name == 'medications':
                for med in section_text:
                    if isinstance(med, dict):
                        med_text = f"{med.get('name', '')} {med.get('dose', '')} {med.get('route', '')} {med.get('frequency', '')}"
                        all_claims.append(Claim(
                            id=f"med_list_{len(all_claims)}",
                            text=med_text.strip(),
                            claim_type=ClaimType.MEDICATION,
                            section=section_name,
                            risk_level="high",
                            source_span=None,
                            verified=None,
                            confidence=None
                        ))
    
    # Assign unique IDs
    for i, claim in enumerate(all_claims):
        claim.id = f"claim_{i:03d}_{claim.claim_type.value}"
    
    return all_claims


def summarize_claims(claims: List[Claim]) -> Dict:
    """
    Summarize extracted claims for dashboard display.
    """
    by_type = {}
    by_risk = {"high": 0, "medium": 0, "low": 0}
    
    for claim in claims:
        type_name = claim.claim_type.value
        by_type[type_name] = by_type.get(type_name, 0) + 1
        by_risk[claim.risk_level] += 1
    
    return {
        "total_claims": len(claims),
        "by_type": by_type,
        "by_risk": by_risk,
        "high_risk_claims": [c for c in claims if c.risk_level == "high"]
    }
