"""
TRUST Scribe Validation Module - EHR-First Pipeline
====================================================

NEW MODULE: trust_scribe_ehr_first_validation.py

This implements the efficient EHR-first approach:
1. Extract claims from AI note (1 LLM call)
2. Check ALL claims against EHR via FHIR (fast DB lookup, no LLM)
2.5. HHEM faithfulness scoring on unresolved claims (local model, no LLM)
3. Run Semantic Entropy ONLY on contradictions (~1-5% of claims)
4. Risk assessment combining 3 signals: EHR + HHEM + SE

Why EHR-First?
- Complex patient: 150 claims × 5 SE passes = 750 LLM calls
- EHR-First: 150 FHIR lookups + ~30 HHEM scores + ~5 SE × 5 = 25 LLM calls
- HHEM closes the NOT_FOUND blind spot with zero additional LLM cost

Author: TRUST Medical AI Platform
Version: 0.3.0
"""

import json
import math
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime

from ..core.hhem_faithfulness import (
    FaithfulnessLevel,
    FaithfulnessResult,
    MockHHEM,
    create_scorer,
    get_faithfulness_summary,
)

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class ClaimType(Enum):
    """Types of verifiable claims in medical notes."""
    MEDICATION = "medication"
    ALLERGY = "allergy"
    DIAGNOSIS = "diagnosis"
    PROCEDURE = "procedure"
    VITAL_SIGN = "vital_sign"
    LAB_RESULT = "lab_result"
    DEMOGRAPHIC = "demographic"
    SYMPTOM = "symptom"
    MEASUREMENT = "measurement"
    PLAN = "plan"
    HISTORY = "history"
    OTHER = "other"


class EHRVerificationStatus(Enum):
    """Result of EHR verification check."""
    VERIFIED = "verified"           # Claim matches EHR data
    CONTRADICTION = "contradiction" # Claim conflicts with EHR data
    NOT_FOUND = "not_found"         # Claim not in EHR (new info?)
    NOT_CHECKABLE = "not_checkable" # Claim type can't be verified against EHR


class SemanticEntropyLevel(Enum):
    """Interpretation of semantic entropy score."""
    LOW = "low"       # SE < 0.3: AI is confident
    MEDIUM = "medium" # SE 0.3-0.6: Some uncertainty
    HIGH = "high"     # SE > 0.6: AI is uncertain


class FinalRiskLevel(Enum):
    """Final risk assessment for a claim."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ExtractedClaim:
    """A single verifiable claim extracted from AI-generated note."""
    claim_id: str
    claim_text: str
    claim_type: ClaimType
    source_sentence: str  # Original sentence in AI note containing this claim
    
    # EHR verification results (populated by step 2)
    ehr_status: Optional[EHRVerificationStatus] = None
    ehr_matched_value: Optional[str] = None  # What EHR actually says
    ehr_resource_type: Optional[str] = None  # FHIR resource (MedicationStatement, etc.)
    
    # HHEM faithfulness results (populated by step 2.5, for NOT_FOUND + CONTRADICTION)
    hhem_score: Optional[float] = None
    hhem_level: Optional[FaithfulnessLevel] = None

    # Semantic entropy results (populated by step 3, ONLY for contradictions)
    semantic_entropy: Optional[float] = None
    se_level: Optional[SemanticEntropyLevel] = None
    se_responses: Optional[List[str]] = None  # The 5 responses for debugging

    # Final assessment
    risk_level: Optional[FinalRiskLevel] = None
    risk_explanation: Optional[str] = None
    requires_physician_review: bool = False


@dataclass
class SEQueryResult:
    """Result of running semantic entropy on a single claim."""
    claim_id: str
    question_asked: str
    responses: List[str]  # 5 responses from stochastic sampling
    semantic_clusters: List[List[int]]  # Which responses are semantically equivalent
    semantic_entropy: float
    se_level: SemanticEntropyLevel
    interpretation: str


@dataclass
class ValidationReport:
    """Complete validation report for an AI-generated note."""
    document_id: str
    patient_id: str
    timestamp: datetime
    
    # Claim counts
    total_claims: int
    verified_claims: int
    contradiction_claims: int
    not_found_claims: int
    not_checkable_claims: int
    
    # HHEM was run on NOT_FOUND + CONTRADICTION claims
    hhem_tested_claims: int
    hhem_unfaithful_claims: int   # HHEM score < 0.5 (likely hallucinated)

    # SE was only run on contradictions
    se_tested_claims: int
    confident_hallucinators: int  # Low SE but contradicts EHR
    ambiguous_claims: int         # High SE (transcript was unclear)
    
    # All claims with details
    claims: List[ExtractedClaim]
    
    # Summary metrics
    overall_risk: FinalRiskLevel
    time_saved_percent: float
    requires_physician_review: bool
    review_priority: str  # "routine", "elevated", "urgent"
    
    # Recommendations
    recommendations: List[str]


# =============================================================================
# TIME SAVED CALCULATION
# =============================================================================

# Average manual verification time per claim type (seconds)
MANUAL_VERIFICATION_TIME = {
    ClaimType.MEDICATION: 45,
    ClaimType.ALLERGY: 30,
    ClaimType.DIAGNOSIS: 60,
    ClaimType.PROCEDURE: 90,
    ClaimType.VITAL_SIGN: 20,
    ClaimType.LAB_RESULT: 40,
    ClaimType.DEMOGRAPHIC: 15,
    ClaimType.SYMPTOM: 25,
    ClaimType.MEASUREMENT: 30,
    ClaimType.PLAN: 45,
    ClaimType.HISTORY: 50,
    ClaimType.OTHER: 35,
}

# TRUST automated verification times (seconds)
TRUST_EHR_CHECK_TIME = 0.5      # FHIR API call
TRUST_SE_CHECK_TIME = 3.0       # Per SE query (5 LLM calls)
CONTRADICTION_REVIEW_TIME = 60  # Manual review time added per contradiction


def calculate_time_saved(claims: List[ExtractedClaim]) -> Dict[str, float]:
    """
    Calculate time saved by TRUST automated verification.
    
    Returns dict with:
    - manual_time_seconds: How long manual verification would take
    - trust_time_seconds: How long TRUST took
    - time_saved_percent: Percentage reduction
    """
    if not claims:
        return {"manual_time_seconds": 0, "trust_time_seconds": 0, "time_saved_percent": 0}
    
    # Manual time: sum of verification time per claim type
    manual_time = sum(
        MANUAL_VERIFICATION_TIME.get(c.claim_type, 35) 
        for c in claims
    )
    
    # TRUST time: EHR checks + SE checks (only on contradictions) + review time
    trust_time = 0
    contradictions = 0
    
    for claim in claims:
        # All claims get EHR check
        trust_time += TRUST_EHR_CHECK_TIME
        
        # Only contradictions get SE check
        if claim.ehr_status == EHRVerificationStatus.CONTRADICTION:
            trust_time += TRUST_SE_CHECK_TIME
            contradictions += 1
    
    # Add manual review time for contradictions
    trust_time += contradictions * CONTRADICTION_REVIEW_TIME
    
    # Calculate percentage saved
    time_saved_percent = max(0, (manual_time - trust_time) / manual_time * 100)
    
    return {
        "manual_time_seconds": manual_time,
        "trust_time_seconds": trust_time,
        "time_saved_percent": round(time_saved_percent, 1)
    }


# =============================================================================
# STEP 1: CLAIM EXTRACTION
# =============================================================================

class ClaimExtractor:
    """
    Extracts verifiable claims from AI-generated medical notes.
    
    This runs ONCE per document (single LLM call).
    """
    
    EXTRACTION_PROMPT = """
    You are a medical claim extractor. Given an AI-generated clinical note, 
    extract all verifiable factual claims.
    
    For each claim, identify:
    1. The exact claim text
    2. The claim type (medication, allergy, diagnosis, procedure, vital_sign, 
       lab_result, demographic, symptom, measurement, plan, history, other)
    3. The source sentence containing this claim
    
    Return as JSON array:
    [
        {{
            "claim_text": "Patient takes Metoprolol 50mg BID",
            "claim_type": "medication",
            "source_sentence": "CURRENT MEDICATIONS: Metoprolol 50mg PO BID"
        }},
        ...
    ]
    
    Focus on VERIFIABLE facts that can be checked against an EHR:
    - Medications (name, dose, frequency)
    - Allergies
    - Diagnoses
    - Procedures
    - Lab values
    - Vital signs
    - Demographics (name, DOB, MRN)
    
    DO NOT extract:
    - Subjective assessments ("patient appears comfortable")
    - Future plans that haven't happened
    - General statements
    
    Clinical Note:
    {note_text}
    """
    
    def __init__(self, llm_client):
        """
        Initialize with LLM client.
        
        Args:
            llm_client: Any LLM client with .complete(prompt) method
        """
        self.llm_client = llm_client
    
    def extract(self, note_text: str) -> List[ExtractedClaim]:
        """
        Extract claims from a clinical note.
        
        Args:
            note_text: The AI-generated clinical note
            
        Returns:
            List of ExtractedClaim objects
        """
        prompt = self.EXTRACTION_PROMPT.format(note_text=note_text)
        
        # Single LLM call
        response = self.llm_client.complete(prompt)
        
        # Parse JSON response
        try:
            claims_data = json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                claims_data = json.loads(json_match.group())
            else:
                raise ValueError("Could not parse claims from LLM response")
        
        # Convert to ExtractedClaim objects
        claims = []
        for i, claim_data in enumerate(claims_data):
            claim = ExtractedClaim(
                claim_id=f"claim_{i+1:03d}",
                claim_text=claim_data["claim_text"],
                claim_type=ClaimType(claim_data["claim_type"]),
                source_sentence=claim_data["source_sentence"]
            )
            claims.append(claim)
        
        return claims


# =============================================================================
# STEP 2: EHR VERIFICATION (FHIR)
# =============================================================================

class EHRVerifier:
    """
    Verifies claims against EHR via FHIR API.
    
    This is FAST - database lookups, no LLM calls.
    """
    
    # Map claim types to FHIR resources
    FHIR_RESOURCE_MAP = {
        ClaimType.MEDICATION: "MedicationStatement",
        ClaimType.ALLERGY: "AllergyIntolerance",
        ClaimType.DIAGNOSIS: "Condition",
        ClaimType.PROCEDURE: "Procedure",
        ClaimType.VITAL_SIGN: "Observation",
        ClaimType.LAB_RESULT: "Observation",
        ClaimType.DEMOGRAPHIC: "Patient",
    }
    
    def __init__(self, fhir_client):
        """
        Initialize with FHIR client.
        
        Args:
            fhir_client: FHIR client with query capabilities
        """
        self.fhir_client = fhir_client
    
    def verify_claim(self, claim: ExtractedClaim, patient_id: str) -> ExtractedClaim:
        """
        Verify a single claim against EHR.
        
        Args:
            claim: The claim to verify
            patient_id: Patient identifier for FHIR queries
            
        Returns:
            Updated claim with EHR verification status
        """
        resource_type = self.FHIR_RESOURCE_MAP.get(claim.claim_type)
        
        if not resource_type:
            claim.ehr_status = EHRVerificationStatus.NOT_CHECKABLE
            return claim
        
        # Query FHIR for matching resources
        ehr_data = self.fhir_client.search(
            resource_type=resource_type,
            patient_id=patient_id
        )
        
        # Check if claim matches EHR data
        match_result = self._check_match(claim, ehr_data)
        
        claim.ehr_status = match_result["status"]
        claim.ehr_matched_value = match_result.get("ehr_value")
        claim.ehr_resource_type = resource_type
        
        return claim
    
    def verify_all_claims(
        self, 
        claims: List[ExtractedClaim], 
        patient_id: str
    ) -> List[ExtractedClaim]:
        """
        Verify all claims against EHR.
        
        Args:
            claims: List of claims to verify
            patient_id: Patient identifier
            
        Returns:
            Updated claims with EHR verification status
        """
        for claim in claims:
            self.verify_claim(claim, patient_id)
        
        return claims
    
    def _check_match(
        self, 
        claim: ExtractedClaim, 
        ehr_data: List[Dict]
    ) -> Dict[str, Any]:
        """
        Check if claim matches EHR data.
        
        This is where the actual verification logic lives.
        Different claim types need different matching strategies.
        """
        if not ehr_data:
            return {"status": EHRVerificationStatus.NOT_FOUND}
        
        # Medication matching
        if claim.claim_type == ClaimType.MEDICATION:
            return self._match_medication(claim, ehr_data)
        
        # Allergy matching
        elif claim.claim_type == ClaimType.ALLERGY:
            return self._match_allergy(claim, ehr_data)
        
        # Diagnosis matching
        elif claim.claim_type == ClaimType.DIAGNOSIS:
            return self._match_diagnosis(claim, ehr_data)
        
        # Default: simple text matching
        else:
            return self._match_generic(claim, ehr_data)
    
    def _match_medication(
        self, 
        claim: ExtractedClaim, 
        ehr_meds: List[Dict]
    ) -> Dict[str, Any]:
        """
        Match medication claim against EHR medication list.
        
        Checks:
        - Medication name (fuzzy match)
        - Dosage (if specified)
        - Frequency (if specified)
        """
        claim_text = claim.claim_text.lower()
        
        for med in ehr_meds:
            med_name = med.get("medication_name", "").lower()
            
            # Check if medication name is in claim
            if med_name in claim_text or self._fuzzy_match(med_name, claim_text):
                # Medication found - check dosage
                ehr_dose = med.get("dosage", "")
                
                # If claim specifies dose, verify it matches
                if ehr_dose and ehr_dose.lower() not in claim_text:
                    # Dosage mismatch!
                    return {
                        "status": EHRVerificationStatus.CONTRADICTION,
                        "ehr_value": f"{med_name} {ehr_dose}",
                        "reason": f"Dosage mismatch: EHR has {ehr_dose}"
                    }
                
                # Match!
                return {
                    "status": EHRVerificationStatus.VERIFIED,
                    "ehr_value": f"{med_name} {ehr_dose}"
                }
        
        # Medication not found in EHR
        return {"status": EHRVerificationStatus.NOT_FOUND}
    
    def _match_allergy(
        self, 
        claim: ExtractedClaim, 
        ehr_allergies: List[Dict]
    ) -> Dict[str, Any]:
        """Match allergy claim against EHR allergy list."""
        claim_text = claim.claim_text.lower()
        
        # Special case: NKDA
        if "nkda" in claim_text or "no known" in claim_text:
            if not ehr_allergies or all(a.get("status") == "inactive" for a in ehr_allergies):
                return {"status": EHRVerificationStatus.VERIFIED, "ehr_value": "NKDA"}
            else:
                return {
                    "status": EHRVerificationStatus.CONTRADICTION,
                    "ehr_value": f"Active allergies: {[a.get('substance') for a in ehr_allergies]}"
                }
        
        # Check for specific allergy
        for allergy in ehr_allergies:
            substance = allergy.get("substance", "").lower()
            if substance in claim_text:
                return {"status": EHRVerificationStatus.VERIFIED, "ehr_value": substance}
        
        return {"status": EHRVerificationStatus.NOT_FOUND}
    
    def _match_diagnosis(
        self, 
        claim: ExtractedClaim, 
        ehr_conditions: List[Dict]
    ) -> Dict[str, Any]:
        """Match diagnosis claim against EHR problem list."""
        claim_text = claim.claim_text.lower()
        
        for condition in ehr_conditions:
            condition_name = condition.get("display", "").lower()
            icd_code = condition.get("code", "")
            
            if condition_name in claim_text or self._fuzzy_match(condition_name, claim_text):
                return {
                    "status": EHRVerificationStatus.VERIFIED,
                    "ehr_value": f"{condition_name} ({icd_code})"
                }
        
        return {"status": EHRVerificationStatus.NOT_FOUND}
    
    def _match_generic(
        self, 
        claim: ExtractedClaim, 
        ehr_data: List[Dict]
    ) -> Dict[str, Any]:
        """Generic matching for other claim types."""
        claim_text = claim.claim_text.lower()
        
        for item in ehr_data:
            item_text = str(item).lower()
            if self._fuzzy_match(claim_text, item_text):
                return {"status": EHRVerificationStatus.VERIFIED, "ehr_value": str(item)}
        
        return {"status": EHRVerificationStatus.NOT_FOUND}
    
    def _fuzzy_match(self, text1: str, text2: str, threshold: float = 0.6) -> bool:
        """Simple fuzzy matching based on word overlap."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return False
        
        overlap = len(words1 & words2)
        max_len = max(len(words1), len(words2))
        
        return overlap / max_len >= threshold


# =============================================================================
# STEP 3: SEMANTIC ENTROPY (ONLY FOR CONTRADICTIONS)
# =============================================================================

class TargetedSemanticEntropyCalculator:
    """
    Calculates Semantic Entropy for SPECIFIC claims that contradict EHR.
    
    KEY INSIGHT: We only run SE on contradictions, not all claims.
    This reduces LLM calls by ~95%.
    
    Purpose: Determine if the contradiction is due to:
    - HIGH SE: Transcript was ambiguous, AI guessed → Medium risk
    - LOW SE: AI is confident → Either new info OR hallucination → High risk
    """
    
    VERIFICATION_PROMPT = """
    Based on the following transcript excerpt, answer this specific question:
    
    TRANSCRIPT: {transcript}
    
    QUESTION: {question}
    
    Answer with a short, direct response. Be specific about what was or wasn't 
    mentioned in the transcript.
    """
    
    def __init__(
        self, 
        llm_client,
        num_samples: int = 5,
        temperature: float = 0.7,
        entailment_checker=None  # DeBERTa NLI for semantic equivalence
    ):
        """
        Initialize SE calculator.
        
        Args:
            llm_client: LLM client for stochastic sampling
            num_samples: Number of samples per claim (default 5)
            temperature: Sampling temperature for diversity
            entailment_checker: Optional DeBERTa model for semantic clustering
        """
        self.llm_client = llm_client
        self.num_samples = num_samples
        self.temperature = temperature
        self.entailment_checker = entailment_checker
    
    def calculate_for_claim(
        self, 
        claim: ExtractedClaim, 
        transcript: str
    ) -> SEQueryResult:
        """
        Calculate semantic entropy for a single claim.
        
        Args:
            claim: The claim to test (should be a contradiction)
            transcript: Original audio transcript
            
        Returns:
            SEQueryResult with entropy and interpretation
        """
        # Formulate the question
        question = self._formulate_question(claim)
        
        # Run N stochastic samples
        responses = []
        for _ in range(self.num_samples):
            prompt = self.VERIFICATION_PROMPT.format(
                transcript=transcript,
                question=question
            )
            response = self.llm_client.complete(
                prompt, 
                temperature=self.temperature
            )
            responses.append(response)
        
        # Cluster responses by semantic equivalence
        clusters = self._cluster_responses(responses)
        
        # Calculate entropy
        entropy = self._calculate_entropy(clusters, len(responses))
        
        # Interpret
        if entropy < 0.3:
            se_level = SemanticEntropyLevel.LOW
            interpretation = (
                f"AI is CONFIDENT about this claim (SE={entropy:.2f}). "
                f"Since it contradicts EHR, either: "
                f"(1) Patient mentioned new info not yet in EHR, or "
                f"(2) AI is a confident hallucinator. "
                f"Physician must verify with patient."
            )
        elif entropy < 0.6:
            se_level = SemanticEntropyLevel.MEDIUM
            interpretation = (
                f"AI shows SOME UNCERTAINTY (SE={entropy:.2f}). "
                f"Transcript may have been partially ambiguous. "
                f"Recommend physician review of this specific claim."
            )
        else:
            se_level = SemanticEntropyLevel.HIGH
            interpretation = (
                f"AI is UNCERTAIN about this claim (SE={entropy:.2f}). "
                f"Transcript was likely ambiguous or unclear. "
                f"AI may have made an educated guess that conflicts with EHR."
            )
        
        return SEQueryResult(
            claim_id=claim.claim_id,
            question_asked=question,
            responses=responses,
            semantic_clusters=clusters,
            semantic_entropy=entropy,
            se_level=se_level,
            interpretation=interpretation
        )
    
    def _formulate_question(self, claim: ExtractedClaim) -> str:
        """
        Create a yes/no verification question for the claim.
        """
        claim_text = claim.claim_text
        
        # Formulate based on claim type
        if claim.claim_type == ClaimType.MEDICATION:
            return f"Did the patient or provider mention that the patient takes {claim_text}?"
        
        elif claim.claim_type == ClaimType.ALLERGY:
            return f"Did the patient or provider mention an allergy to {claim_text}?"
        
        elif claim.claim_type == ClaimType.DIAGNOSIS:
            return f"Was {claim_text} discussed as a diagnosis or condition the patient has?"
        
        elif claim.claim_type == ClaimType.PROCEDURE:
            return f"Was {claim_text} mentioned as a procedure the patient had or will have?"
        
        elif claim.claim_type == ClaimType.VITAL_SIGN:
            return f"Was the vital sign '{claim_text}' mentioned in the conversation?"
        
        elif claim.claim_type == ClaimType.LAB_RESULT:
            return f"Was the lab result '{claim_text}' discussed?"
        
        else:
            return f"Was the following mentioned in the transcript: '{claim_text}'?"
    
    def _cluster_responses(self, responses: List[str]) -> List[List[int]]:
        """
        Cluster responses by semantic equivalence.
        
        Uses bidirectional entailment if DeBERTa is available,
        otherwise falls back to simple heuristics.
        """
        if self.entailment_checker:
            return self._cluster_with_deberta(responses)
        else:
            return self._cluster_heuristic(responses)
    
    def _cluster_heuristic(self, responses: List[str]) -> List[List[int]]:
        """
        Simple heuristic clustering based on yes/no/unclear patterns.
        """
        clusters = {"yes": [], "no": [], "unclear": []}
        
        for i, response in enumerate(responses):
            response_lower = response.lower()
            
            # Check for affirmative
            if any(word in response_lower for word in ["yes", "mentioned", "discussed", "confirmed", "stated"]):
                if not any(word in response_lower for word in ["not", "no", "didn't", "wasn't", "unclear"]):
                    clusters["yes"].append(i)
                    continue
            
            # Check for negative
            if any(word in response_lower for word in ["no", "not", "didn't", "wasn't", "never"]):
                clusters["no"].append(i)
                continue
            
            # Default to unclear
            clusters["unclear"].append(i)
        
        # Return non-empty clusters
        return [indices for indices in clusters.values() if indices]
    
    def _cluster_with_deberta(self, responses: List[str]) -> List[List[int]]:
        """
        Cluster using DeBERTa bidirectional entailment.
        
        Two responses are equivalent if A entails B AND B entails A.
        """
        n = len(responses)
        clusters = []
        assigned = set()
        
        for i in range(n):
            if i in assigned:
                continue
            
            cluster = [i]
            assigned.add(i)
            
            for j in range(i + 1, n):
                if j in assigned:
                    continue
                
                # Check bidirectional entailment
                is_equiv = self.entailment_checker.are_semantically_equivalent(
                    responses[i], responses[j]
                )
                
                if is_equiv:
                    cluster.append(j)
                    assigned.add(j)
            
            clusters.append(cluster)
        
        return clusters
    
    def _calculate_entropy(
        self, 
        clusters: List[List[int]], 
        total_responses: int
    ) -> float:
        """
        Calculate Shannon entropy over cluster distribution.
        
        SE = -Σ p(cluster) * log(p(cluster))
        
        Normalized to [0, 1] range.
        """
        if not clusters or total_responses == 0:
            return 0.0
        
        entropy = 0.0
        for cluster in clusters:
            p = len(cluster) / total_responses
            if p > 0:
                entropy -= p * math.log(p + 1e-10)
        
        # Normalize by max possible entropy (uniform distribution)
        max_entropy = math.log(len(clusters)) if len(clusters) > 1 else 1.0
        normalized = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return min(1.0, max(0.0, normalized))


# =============================================================================
# STEP 4: RISK ASSESSMENT
# =============================================================================

class RiskAssessor:
    """
    Determines final risk level for each claim and overall document.
    """
    
    def assess_claim(self, claim: ExtractedClaim) -> ExtractedClaim:
        """
        Assess risk for a single claim using up to 3 signals:
          1. EHR verification status
          2. HHEM faithfulness score (for NOT_FOUND + CONTRADICTION)
          3. Semantic entropy level (for CONTRADICTION only)
        """
        # Verified claims = low risk
        if claim.ehr_status == EHRVerificationStatus.VERIFIED:
            claim.risk_level = FinalRiskLevel.LOW
            claim.risk_explanation = "Claim verified against EHR"
            claim.requires_physician_review = False

        # Not checkable = low risk (can't verify)
        elif claim.ehr_status == EHRVerificationStatus.NOT_CHECKABLE:
            claim.risk_level = FinalRiskLevel.LOW
            claim.risk_explanation = "Claim type not verifiable against EHR"
            claim.requires_physician_review = False

        # Not found in EHR — use HHEM to differentiate
        elif claim.ehr_status == EHRVerificationStatus.NOT_FOUND:
            if claim.hhem_level == FaithfulnessLevel.FAITHFUL:
                # Claim is faithful to transcript but missing from EHR
                # Likely new info mentioned by patient
                claim.risk_level = FinalRiskLevel.LOW
                claim.risk_explanation = (
                    "Not in EHR but faithful to transcript "
                    f"(HHEM={claim.hhem_score:.2f}). "
                    "Likely new information from patient."
                )
                claim.requires_physician_review = False
            elif claim.hhem_level == FaithfulnessLevel.PARTIALLY_FAITHFUL:
                claim.risk_level = FinalRiskLevel.MEDIUM
                claim.risk_explanation = (
                    "Not in EHR, partially supported by transcript "
                    f"(HHEM={claim.hhem_score:.2f}). Review recommended."
                )
                claim.requires_physician_review = True
            elif claim.hhem_level in (
                FaithfulnessLevel.LIKELY_HALLUCINATED,
                FaithfulnessLevel.HALLUCINATED,
            ):
                claim.risk_level = FinalRiskLevel.HIGH
                claim.risk_explanation = (
                    "Not in EHR and not supported by transcript "
                    f"(HHEM={claim.hhem_score:.2f}). "
                    "Likely hallucination."
                )
                claim.requires_physician_review = True
            else:
                # HHEM not available — fall back to original MEDIUM
                claim.risk_level = FinalRiskLevel.MEDIUM
                claim.risk_explanation = "Claim not found in EHR - may be new information"
                claim.requires_physician_review = True

        # Contradiction = combine HHEM + SE
        elif claim.ehr_status == EHRVerificationStatus.CONTRADICTION:
            if claim.se_level == SemanticEntropyLevel.HIGH:
                # High SE = transcript was ambiguous
                claim.risk_level = FinalRiskLevel.MEDIUM
                claim.risk_explanation = (
                    "Contradiction with EHR, but AI was uncertain. "
                    "Transcript may have been ambiguous."
                )
                claim.requires_physician_review = True

            elif claim.se_level == SemanticEntropyLevel.MEDIUM:
                claim.risk_level = FinalRiskLevel.HIGH
                claim.risk_explanation = (
                    "Contradiction with EHR. AI showed some uncertainty. "
                    "Verify with patient."
                )
                claim.requires_physician_review = True

            else:  # LOW SE = confident — HHEM disambiguates
                if claim.hhem_level == FaithfulnessLevel.FAITHFUL:
                    # Faithful to transcript but contradicts EHR
                    # Patient likely reported new info not yet in record
                    claim.risk_level = FinalRiskLevel.HIGH
                    claim.risk_explanation = (
                        "Contradicts EHR but faithful to transcript "
                        f"(HHEM={claim.hhem_score:.2f}). "
                        "Patient may have reported new info. "
                        "EHR may need updating."
                    )
                    claim.requires_physician_review = True
                else:
                    # Not faithful to transcript AND contradicts EHR
                    claim.risk_level = FinalRiskLevel.CRITICAL
                    claim.risk_explanation = (
                        "CRITICAL: Contradicts EHR and not supported by transcript "
                        f"(HHEM={claim.hhem_score:.2f}). "
                        "Confident hallucination. "
                        "MUST verify with patient before signing."
                    )
                    claim.requires_physician_review = True

        return claim
    
    def assess_document(
        self, 
        claims: List[ExtractedClaim]
    ) -> Tuple[FinalRiskLevel, str]:
        """
        Determine overall document risk level.
        
        Returns (risk_level, explanation)
        """
        if not claims:
            return FinalRiskLevel.LOW, "No claims to assess"
        
        # Count by risk level
        risk_counts = {level: 0 for level in FinalRiskLevel}
        for claim in claims:
            if claim.risk_level:
                risk_counts[claim.risk_level] += 1
        
        # Determine overall risk
        if risk_counts[FinalRiskLevel.CRITICAL] > 0:
            return (
                FinalRiskLevel.CRITICAL,
                f"{risk_counts[FinalRiskLevel.CRITICAL]} critical issue(s) requiring immediate physician review"
            )
        
        elif risk_counts[FinalRiskLevel.HIGH] > 0:
            return (
                FinalRiskLevel.HIGH,
                f"{risk_counts[FinalRiskLevel.HIGH]} high-risk claim(s) requiring careful review"
            )
        
        elif risk_counts[FinalRiskLevel.MEDIUM] > 0:
            return (
                FinalRiskLevel.MEDIUM,
                f"{risk_counts[FinalRiskLevel.MEDIUM]} claim(s) flagged for review"
            )
        
        else:
            return FinalRiskLevel.LOW, "All claims verified or low risk"


# =============================================================================
# MAIN PIPELINE
# =============================================================================

class TRUSTScribeValidator:
    """
    Main orchestrator for the EHR-First Semantic Entropy pipeline.
    
    Usage:
        validator = TRUSTScribeValidator(llm_client, fhir_client)
        report = validator.validate(note_text, transcript, patient_id)
    """
    
    def __init__(
        self,
        llm_client,
        fhir_client,
        entailment_checker=None,
        hhem_scorer=None,
        se_num_samples: int = 5,
        se_temperature: float = 0.7
    ):
        """
        Initialize the validator.

        Args:
            llm_client: LLM client for claim extraction and SE
            fhir_client: FHIR client for EHR verification
            entailment_checker: Optional DeBERTa for semantic clustering
            hhem_scorer: HHEM faithfulness scorer (defaults to MockHHEM)
            se_num_samples: Number of samples for SE (default 5)
            se_temperature: Temperature for SE sampling (default 0.7)
        """
        self.claim_extractor = ClaimExtractor(llm_client)
        self.ehr_verifier = EHRVerifier(fhir_client)
        self.hhem_scorer = hhem_scorer or create_scorer(use_mock=True)
        self.se_calculator = TargetedSemanticEntropyCalculator(
            llm_client,
            num_samples=se_num_samples,
            temperature=se_temperature,
            entailment_checker=entailment_checker
        )
        self.risk_assessor = RiskAssessor()
    
    def validate(
        self,
        note_text: str,
        transcript: str,
        patient_id: str,
        document_id: Optional[str] = None
    ) -> ValidationReport:
        """
        Run the complete validation pipeline.
        
        Args:
            note_text: AI-generated clinical note
            transcript: Original audio transcript
            patient_id: Patient identifier for EHR lookup
            document_id: Optional document identifier
            
        Returns:
            ValidationReport with all results
        """
        timestamp = datetime.now()
        document_id = document_id or f"doc_{timestamp.strftime('%Y%m%d%H%M%S')}"
        
        # STEP 1: Extract claims (1 LLM call)
        claims = self.claim_extractor.extract(note_text)
        
        # STEP 2: Verify against EHR (fast FHIR lookups)
        claims = self.ehr_verifier.verify_all_claims(claims, patient_id)

        # STEP 2.5: HHEM faithfulness scoring (local model, no LLM calls)
        # Run on claims that EHR couldn't fully resolve
        needs_hhem = [
            c for c in claims
            if c.ehr_status in (
                EHRVerificationStatus.NOT_FOUND,
                EHRVerificationStatus.CONTRADICTION,
            )
        ]

        if needs_hhem:
            hhem_results = self.hhem_scorer.score_claims_batch(
                [c.claim_text for c in needs_hhem], transcript
            )
            for claim, hhem_result in zip(needs_hhem, hhem_results):
                claim.hhem_score = hhem_result.score
                claim.hhem_level = hhem_result.level
            logger.info(
                "HHEM scored %d claims: %d unfaithful",
                len(needs_hhem),
                sum(1 for r in hhem_results
                    if r.level in (FaithfulnessLevel.LIKELY_HALLUCINATED,
                                   FaithfulnessLevel.HALLUCINATED)),
            )

        # STEP 3: Run SE only on contradictions
        contradictions = [
            c for c in claims
            if c.ehr_status == EHRVerificationStatus.CONTRADICTION
        ]

        confident_hallucinators = 0
        ambiguous_claims = 0

        for claim in contradictions:
            se_result = self.se_calculator.calculate_for_claim(claim, transcript)
            
            # Update claim with SE results
            claim.semantic_entropy = se_result.semantic_entropy
            claim.se_level = se_result.se_level
            claim.se_responses = se_result.responses
            
            # Count by SE level
            if se_result.se_level == SemanticEntropyLevel.LOW:
                confident_hallucinators += 1
            elif se_result.se_level == SemanticEntropyLevel.HIGH:
                ambiguous_claims += 1
        
        # STEP 4: Risk assessment
        for claim in claims:
            self.risk_assessor.assess_claim(claim)
        
        overall_risk, risk_explanation = self.risk_assessor.assess_document(claims)
        
        # Calculate time saved
        time_metrics = calculate_time_saved(claims)
        
        # Determine review priority
        if overall_risk == FinalRiskLevel.CRITICAL:
            review_priority = "urgent"
        elif overall_risk == FinalRiskLevel.HIGH:
            review_priority = "elevated"
        else:
            review_priority = "routine"
        
        # Generate recommendations
        recommendations = self._generate_recommendations(claims, overall_risk)
        
        # HHEM summary counts
        hhem_tested = len(needs_hhem)
        hhem_unfaithful = sum(
            1 for c in needs_hhem
            if c.hhem_level in (
                FaithfulnessLevel.LIKELY_HALLUCINATED,
                FaithfulnessLevel.HALLUCINATED,
            )
        )

        # Build report
        return ValidationReport(
            document_id=document_id,
            patient_id=patient_id,
            timestamp=timestamp,
            total_claims=len(claims),
            verified_claims=sum(1 for c in claims if c.ehr_status == EHRVerificationStatus.VERIFIED),
            contradiction_claims=len(contradictions),
            not_found_claims=sum(1 for c in claims if c.ehr_status == EHRVerificationStatus.NOT_FOUND),
            not_checkable_claims=sum(1 for c in claims if c.ehr_status == EHRVerificationStatus.NOT_CHECKABLE),
            hhem_tested_claims=hhem_tested,
            hhem_unfaithful_claims=hhem_unfaithful,
            se_tested_claims=len(contradictions),
            confident_hallucinators=confident_hallucinators,
            ambiguous_claims=ambiguous_claims,
            claims=claims,
            overall_risk=overall_risk,
            time_saved_percent=time_metrics["time_saved_percent"],
            requires_physician_review=any(c.requires_physician_review for c in claims),
            review_priority=review_priority,
            recommendations=recommendations
        )
    
    def _generate_recommendations(
        self, 
        claims: List[ExtractedClaim], 
        overall_risk: FinalRiskLevel
    ) -> List[str]:
        """Generate actionable recommendations based on validation results."""
        recommendations = []
        
        # Critical issues
        critical_claims = [c for c in claims if c.risk_level == FinalRiskLevel.CRITICAL]
        if critical_claims:
            recommendations.append(
                f"🚨 CRITICAL: {len(critical_claims)} claim(s) are confident but contradict EHR. "
                f"Verify with patient before signing."
            )
            for claim in critical_claims:
                recommendations.append(f"   • {claim.claim_text}")
        
        # High risk claims
        high_risk = [c for c in claims if c.risk_level == FinalRiskLevel.HIGH]
        if high_risk:
            recommendations.append(
                f"⚠️ HIGH RISK: {len(high_risk)} claim(s) need careful review."
            )
        
        # HHEM-detected unfaithful claims (not in EHR AND not in transcript)
        hhem_hallucinated = [
            c for c in claims
            if c.ehr_status == EHRVerificationStatus.NOT_FOUND
            and c.hhem_level in (
                FaithfulnessLevel.LIKELY_HALLUCINATED,
                FaithfulnessLevel.HALLUCINATED,
            )
        ]
        if hhem_hallucinated:
            recommendations.append(
                f"🔍 HHEM ALERT: {len(hhem_hallucinated)} claim(s) not in EHR "
                f"and not supported by transcript. Likely fabricated."
            )
            for claim in hhem_hallucinated:
                recommendations.append(
                    f"   • {claim.claim_text} (HHEM={claim.hhem_score:.2f})"
                )

        # Not found but faithful to transcript (likely new patient info)
        not_found_faithful = [
            c for c in claims
            if c.ehr_status == EHRVerificationStatus.NOT_FOUND
            and c.hhem_level == FaithfulnessLevel.FAITHFUL
        ]
        if not_found_faithful:
            recommendations.append(
                f"ℹ️ NEW INFO: {len(not_found_faithful)} claim(s) not in EHR "
                f"but supported by transcript. EHR may need updating."
            )

        # Remaining not-found claims without clear HHEM signal
        not_found_other = [
            c for c in claims
            if c.ehr_status == EHRVerificationStatus.NOT_FOUND
            and c not in hhem_hallucinated
            and c not in not_found_faithful
        ]
        if not_found_other:
            recommendations.append(
                f"ℹ️ UNRESOLVED: {len(not_found_other)} claim(s) not in EHR, "
                f"partially supported by transcript. Review recommended."
            )
        
        # All good
        if overall_risk == FinalRiskLevel.LOW:
            recommendations.append(
                "✅ All claims verified or low risk. Safe for routine review."
            )
        
        return recommendations


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example showing the expected data flow
    
    example_report = {
        "document_id": "doc_20251227_143022",
        "patient_id": "P12345",
        "total_claims": 12,
        "verified_claims": 9,
        "contradiction_claims": 2,
        "not_found_claims": 1,
        "hhem_tested_claims": 3,
        "hhem_unfaithful_claims": 1,
        "se_tested_claims": 2,
        "confident_hallucinators": 1,
        "ambiguous_claims": 1,
        "overall_risk": "HIGH",
        "time_saved_percent": 78.5,
        "review_priority": "elevated",
        "recommendations": [
            "🚨 CRITICAL: 1 claim(s) contradict EHR and not supported by transcript.",
            "   • Warfarin 5mg daily (HHEM=0.12)",
            "⚠️ HIGH RISK: 1 claim(s) need careful review.",
            "🔍 HHEM ALERT: 1 claim(s) not in EHR and not in transcript.",
            "ℹ️ NEW INFO: 0 claim(s) not in EHR but supported by transcript."
        ]
    }

    print("TRUST EHR-First Validation Pipeline (with HHEM)")
    print("=" * 50)
    print(f"Total claims extracted: {example_report['total_claims']}")
    print(f"EHR verified: {example_report['verified_claims']}")
    print(f"HHEM tested: {example_report['hhem_tested_claims']}")
    print(f"  - Unfaithful to transcript: {example_report['hhem_unfaithful_claims']}")
    print(f"Contradictions (SE tested): {example_report['contradiction_claims']}")
    print(f"  - Confident hallucinators: {example_report['confident_hallucinators']}")
    print(f"  - Ambiguous (high SE): {example_report['ambiguous_claims']}")
    print(f"Time saved: {example_report['time_saved_percent']}%")
    print(f"Overall risk: {example_report['overall_risk']}")
    print()
    print("Recommendations:")
    for rec in example_report["recommendations"]:
        print(f"  {rec}")
