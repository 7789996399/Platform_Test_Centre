"""
Healthcare Industry Adapter
===========================

Implements the IndustryAdapter interface for healthcare AI governance.

This adapter provides healthcare-specific logic for:
- Clinical claim extraction (medications, allergies, diagnoses, vitals)
- EHR-based verification
- Risk classification for medical content
- HIPAA compliance checking
- Dosage and drug interaction verification

Based on patterns from the existing TRUST Platform backend, specifically
adapted from claim_extraction.py and source_verification.py.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from ..base_adapter import (
    AdapterConfig,
    Claim as BaseClaim,
    IndustryAdapter,
    RiskLevel,
    VerificationContext,
    VerificationResult,
)

# Import core engine types for protocol implementations
from ...core_engine.faithfulness import (
    Claim,
    ClaimCategory,
    SourceDocument,
    VerificationResult as CoreVerificationResult,
    VerificationStatus,
)
from ...core_engine.semantic_entropy import EntropyThresholds


# =============================================================================
# HEALTHCARE-SPECIFIC ENUMS
# =============================================================================

class HealthcareClaimType(str, Enum):
    """
    Types of clinical claims with associated risk levels.

    Ported from backend/app/core/claim_extraction.py
    """
    # High risk - require strict verification
    MEDICATION = "medication"
    ALLERGY = "allergy"
    DOSAGE = "dosage"
    DIAGNOSIS = "diagnosis"
    PROCEDURE = "procedure"
    DRUG_INTERACTION = "drug_interaction"
    CONTRAINDICATION = "contraindication"

    # Medium risk - standard verification
    VITAL_SIGN = "vital_sign"
    LAB_VALUE = "lab_value"
    EXAM_FINDING = "exam_finding"
    SYMPTOM = "symptom"
    HISTORY = "history"

    # Low risk - basic verification
    DEMOGRAPHIC = "demographic"
    ADMINISTRATIVE = "administrative"
    LIFESTYLE = "lifestyle"

    def to_category(self) -> ClaimCategory:
        """Map healthcare claim type to generic risk category."""
        high_risk = {
            self.MEDICATION, self.ALLERGY, self.DOSAGE,
            self.DIAGNOSIS, self.PROCEDURE, self.DRUG_INTERACTION,
            self.CONTRAINDICATION
        }
        low_risk = {self.DEMOGRAPHIC, self.ADMINISTRATIVE, self.LIFESTYLE}

        if self in high_risk:
            return ClaimCategory.HIGH_RISK
        elif self in low_risk:
            return ClaimCategory.LOW_RISK
        else:
            return ClaimCategory.MEDIUM_RISK


class ReviewLevel(str, Enum):
    """Review level for healthcare responses."""
    BRIEF = "brief"           # 15 seconds - quick confirmation
    STANDARD = "standard"     # 2-3 minutes - check key facts
    DETAILED = "detailed"     # 5+ minutes - full clinical review


# =============================================================================
# HEALTHCARE-SPECIFIC DATA CLASSES
# =============================================================================

@dataclass
class MedicationComponents:
    """Parsed components of a medication claim."""
    name: Optional[str] = None
    dose: Optional[float] = None
    unit: Optional[str] = None
    route: Optional[str] = None
    frequency: Optional[str] = None
    raw_text: str = ""


@dataclass
class HealthcareVerificationResult:
    """Extended verification result with healthcare-specific fields."""
    claim: Claim
    status: VerificationStatus
    confidence: float
    risk_level: RiskLevel
    review_level: ReviewLevel
    matched_source: Optional[SourceDocument] = None
    matched_text: Optional[str] = None
    explanation: str = ""
    needs_entropy_check: bool = True
    is_confident_hallucinator: bool = False
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# AUTHORITATIVE SOURCES
# =============================================================================

AUTHORITATIVE_DRUG_SOURCES = [
    "fda.gov",
    "drugbank.com",
    "rxnorm",
    "dailymed.nlm.nih.gov",
    "drugs.com",
    "epocrates",
    "micromedex",
    "lexicomp",
]

AUTHORITATIVE_CLINICAL_SOURCES = [
    "uptodate.com",
    "pubmed.ncbi.nlm.nih.gov",
    "cochranelibrary.com",
    "nice.org.uk",
    "ahrq.gov",
    "cdc.gov",
    "who.int",
    "nejm.org",
    "jamanetwork.com",
    "thelancet.com",
]

COMMON_ALLERGENS = [
    "penicillin", "amoxicillin", "sulfa", "sulfamethoxazole",
    "codeine", "morphine", "hydrocodone", "oxycodone",
    "latex", "iodine", "contrast", "shellfish",
    "aspirin", "nsaid", "ibuprofen", "naproxen",
    "ace inhibitor", "lisinopril", "enalapril",
    "statin", "atorvastatin", "simvastatin",
    "metformin", "insulin",
]


# =============================================================================
# REGEX PATTERNS FOR CLAIM EXTRACTION
# =============================================================================

MEDICATION_PATTERNS = {
    # Dose with unit: "50mg", "100 mg", "0.5 g"
    "dose": re.compile(r'(\d+(?:[.,]\d+)?)\s*(mg|mcg|g|ml|units?|iu|meq)', re.IGNORECASE),

    # Weight-based dose: "5 mcg/kg", "10mg/kg/day"
    "weight_dose": re.compile(r'(\d+(?:[.,]\d+)?)\s*(mg|mcg|g)/kg(?:/(?:day|dose))?', re.IGNORECASE),

    # Route: PO, IV, IM, SC, etc.
    "route": re.compile(r'\b(po|oral|iv|intravenous|im|intramuscular|sc|sq|subcutaneous|pr|rectal|sl|sublingual|topical|inhaled|nebulized)\b', re.IGNORECASE),

    # Frequency: BID, TID, daily, etc.
    "frequency": re.compile(r'\b(daily|once daily|bid|twice daily|tid|three times daily|qid|four times daily|prn|as needed|weekly|monthly|q\d+h?|every \d+ hours?)\b', re.IGNORECASE),
}

VITAL_SIGN_PATTERNS = {
    "blood_pressure": re.compile(r'(?:BP|blood pressure)\s*:?\s*(\d{2,3})\s*/\s*(\d{2,3})', re.IGNORECASE),
    "heart_rate": re.compile(r'(?:HR|heart rate|pulse)\s*:?\s*(\d{2,3})\s*(?:bpm)?', re.IGNORECASE),
    "oxygen_sat": re.compile(r'(?:SpO2|O2 sat|oxygen saturation?|sat)\s*:?\s*(\d{2,3})\s*%?', re.IGNORECASE),
    "temperature": re.compile(r'(?:temp|temperature)\s*:?\s*(\d{2,3}(?:\.\d)?)\s*(?:°?[FC])?', re.IGNORECASE),
    "respiratory_rate": re.compile(r'(?:RR|resp(?:iratory)? rate)\s*:?\s*(\d{1,2})', re.IGNORECASE),
    "weight": re.compile(r'(?:weight|wt)\s*:?\s*(\d{2,3}(?:\.\d)?)\s*(?:kg|lbs?|pounds?)?', re.IGNORECASE),
    "height": re.compile(r'(?:height|ht)\s*:?\s*(\d{1,3}(?:\.\d)?)\s*(?:cm|in|inches|feet|ft)?', re.IGNORECASE),
}

ALLERGY_PATTERNS = [
    re.compile(r'allerg(?:y|ic|ies)\s*(?:to\s+)?:?\s*([^.,\n]+)', re.IGNORECASE),
    re.compile(r'NKDA', re.IGNORECASE),
    re.compile(r'no known (?:drug )?allergies', re.IGNORECASE),
    re.compile(r'NKA', re.IGNORECASE),
    # Allergen with reaction: "Penicillin (rash)", "Sulfa (anaphylaxis)"
    re.compile(r'([A-Za-z]+)\s*\(([^)]*(?:rash|anaphylaxis|hives|swelling|itching|reaction)[^)]*)\)', re.IGNORECASE),
]

EXAM_FINDING_PATTERNS = [
    re.compile(r'((?:grade\s+)?[I-V]+/(?:VI|6)\s+(?:systolic|diastolic|holosystolic)?\s*murmur[^.]*)', re.IGNORECASE),
    re.compile(r'(\d\+\s*(?:pitting\s+)?edema[^.]*)', re.IGNORECASE),
    re.compile(r'(regular (?:rate and )?rhythm)', re.IGNORECASE),
    re.compile(r'(irregularly irregular)', re.IGNORECASE),
    re.compile(r'(clear to auscultation(?:\s+bilaterally)?)', re.IGNORECASE),
    re.compile(r'(no (?:murmurs|gallops|rubs|wheezes|crackles))', re.IGNORECASE),
    re.compile(r'(JVD[^.]*)', re.IGNORECASE),
    re.compile(r'(jugular venous (?:distension|pressure)[^.]*)', re.IGNORECASE),
    re.compile(r'(hepatomegaly|splenomegaly)', re.IGNORECASE),
]


# =============================================================================
# HEALTHCARE ADAPTER IMPLEMENTATION
# =============================================================================

class HealthcareAdapter(IndustryAdapter):
    """
    Healthcare-specific adapter for TRUST verification.

    Provides domain-specific logic for:
    - Clinical terminology validation
    - Drug interaction checking
    - Dosage verification
    - HIPAA compliance
    - Medical claim source attribution

    Example:
        >>> adapter = HealthcareAdapter()
        >>> await adapter.initialize()
        >>>
        >>> claims = adapter.extract_claims("Patient on Metoprolol 50mg BID")
        >>> for claim in claims:
        ...     result = await adapter.verify_claim(claim, context)
        ...     print(f"{claim.text}: {result.status}")
    """

    def __init__(self):
        config = AdapterConfig(
            industry_name="healthcare",
            version="1.0.0",
            entropy_threshold=0.2,  # Stricter for medical content
            faithfulness_threshold=0.9,  # High faithfulness required
            ensemble_agreement_threshold=0.8,  # Strong consensus needed
            max_risk_level_auto_approve=RiskLevel.MINIMAL,
            custom_settings={
                "require_source_citation": True,
                "drug_interaction_check": True,
                "dosage_verification": True,
                "phi_detection": True,
            }
        )
        super().__init__(config)
        self._claim_counter = 0

    # =========================================================================
    # Lifecycle Methods
    # =========================================================================

    async def initialize(self) -> None:
        """Initialize healthcare-specific resources."""
        # In production, this would:
        # - Load medical ontologies (SNOMED, ICD-10, RxNorm)
        # - Initialize drug interaction database connection
        # - Load clinical guidelines knowledge base
        # - Initialize NER models for medical entity extraction
        self._initialized = True

    async def shutdown(self) -> None:
        """Clean up healthcare adapter resources."""
        # In production, this would:
        # - Close database connections
        # - Flush audit logs
        # - Release model resources
        self._initialized = False

    # =========================================================================
    # Risk Assessment Methods
    # =========================================================================

    def classify_risk(self, context: VerificationContext) -> RiskLevel:
        """
        Classify risk level for healthcare responses.

        Risk Classification:
        - CRITICAL: Dosage recommendations, drug interactions, contraindications
        - HIGH: Diagnostic suggestions, treatment plans, procedures
        - MEDIUM: General medical information, symptoms, history
        - LOW: Wellness/lifestyle information, demographics
        """
        # Extract claims to analyze
        claims = self.extract_claims(context.response)

        if not claims:
            return RiskLevel.LOW

        # Check for critical claim types
        critical_types = {
            HealthcareClaimType.DOSAGE,
            HealthcareClaimType.DRUG_INTERACTION,
            HealthcareClaimType.CONTRAINDICATION,
        }

        high_types = {
            HealthcareClaimType.MEDICATION,
            HealthcareClaimType.ALLERGY,
            HealthcareClaimType.DIAGNOSIS,
            HealthcareClaimType.PROCEDURE,
        }

        for claim in claims:
            claim_type = self._get_healthcare_claim_type(claim)
            if claim_type in critical_types:
                return RiskLevel.CRITICAL

        for claim in claims:
            claim_type = self._get_healthcare_claim_type(claim)
            if claim_type in high_types:
                return RiskLevel.HIGH

        # Check for medium risk types
        medium_types = {
            HealthcareClaimType.VITAL_SIGN,
            HealthcareClaimType.LAB_VALUE,
            HealthcareClaimType.EXAM_FINDING,
            HealthcareClaimType.SYMPTOM,
            HealthcareClaimType.HISTORY,
        }

        for claim in claims:
            claim_type = self._get_healthcare_claim_type(claim)
            if claim_type in medium_types:
                return RiskLevel.MEDIUM

        return RiskLevel.LOW

    def get_risk_thresholds(self) -> Dict[str, float]:
        """Get healthcare-specific risk thresholds."""
        return {
            "entropy": self._config.entropy_threshold,
            "faithfulness": self._config.faithfulness_threshold,
            "ensemble_agreement": self._config.ensemble_agreement_threshold,
        }

    def get_entropy_thresholds(self) -> EntropyThresholds:
        """Get entropy thresholds configured for healthcare."""
        return EntropyThresholds(
            low_medium_boundary=0.2,  # Stricter than default
            medium_high_boundary=0.5,
        )

    # =========================================================================
    # Claim Extraction Methods
    # =========================================================================

    def extract_claims(self, response: str) -> List[Claim]:
        """
        Extract medical claims from an AI response.

        Extracts:
        - Medications with dosages
        - Allergies
        - Vital signs
        - Diagnoses/conditions
        - Exam findings
        """
        all_claims: List[Claim] = []

        # Extract different claim types
        all_claims.extend(self._extract_medications(response))
        all_claims.extend(self._extract_allergies(response))
        all_claims.extend(self._extract_vital_signs(response))
        all_claims.extend(self._extract_exam_findings(response))
        all_claims.extend(self._extract_conditions(response))

        # Assign unique IDs
        for i, claim in enumerate(all_claims):
            claim.id = f"hc_{i:03d}_{claim.claim_type}"

        return all_claims

    def _extract_medications(self, text: str, section: str = "general") -> List[Claim]:
        """Extract medication claims from text."""
        claims = []

        # Look for bullet points in medication sections
        lines = text.split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith(('- ', '• ', '* ')):
                med_text = line[2:].strip()
                # Check if it looks like a medication (has dose pattern)
                if med_text and MEDICATION_PATTERNS["dose"].search(med_text):
                    components = self._parse_medication(med_text)
                    claims.append(Claim(
                        id="",  # Will be assigned later
                        text=med_text,
                        category=ClaimCategory.HIGH_RISK,
                        claim_type=HealthcareClaimType.MEDICATION.value,
                        section=section,
                        metadata={
                            "components": {
                                "name": components.name,
                                "dose": components.dose,
                                "unit": components.unit,
                                "route": components.route,
                                "frequency": components.frequency,
                            }
                        }
                    ))

        return claims

    def _extract_allergies(self, text: str, section: str = "general") -> List[Claim]:
        """Extract allergy claims from text."""
        claims = []

        for pattern in ALLERGY_PATTERNS:
            matches = pattern.findall(text)
            for match in matches:
                if isinstance(match, tuple):
                    match_text = ' '.join(match).strip()
                else:
                    match_text = match.strip()

                if match_text:
                    # Determine if it's NKDA
                    is_nkda = 'nkda' in match_text.lower() or 'no known' in match_text.lower()

                    claims.append(Claim(
                        id="",
                        text=match_text,
                        category=ClaimCategory.HIGH_RISK,
                        claim_type=HealthcareClaimType.ALLERGY.value,
                        section=section,
                        metadata={"is_nkda": is_nkda}
                    ))

        return claims

    def _extract_vital_signs(self, text: str, section: str = "general") -> List[Claim]:
        """Extract vital sign claims from text."""
        claims = []

        for vital_type, pattern in VITAL_SIGN_PATTERNS.items():
            matches = pattern.findall(text)
            for match in matches:
                if isinstance(match, tuple):
                    value = '/'.join(match)
                else:
                    value = match

                claims.append(Claim(
                    id="",
                    text=f"{vital_type}: {value}",
                    category=ClaimCategory.MEDIUM_RISK,
                    claim_type=HealthcareClaimType.VITAL_SIGN.value,
                    section=section,
                    metadata={"vital_type": vital_type, "value": value}
                ))

        return claims

    def _extract_exam_findings(self, text: str, section: str = "general") -> List[Claim]:
        """Extract physical exam findings from text."""
        claims = []

        for pattern in EXAM_FINDING_PATTERNS:
            matches = pattern.findall(text)
            for match in matches:
                claims.append(Claim(
                    id="",
                    text=match.strip(),
                    category=ClaimCategory.MEDIUM_RISK,
                    claim_type=HealthcareClaimType.EXAM_FINDING.value,
                    section=section,
                ))

        return claims

    def _extract_conditions(self, text: str, section: str = "general") -> List[Claim]:
        """Extract diagnosis/condition claims from text."""
        claims = []

        # Look for conditions in bullet points (typically in assessment/diagnosis sections)
        condition_keywords = ['diagnosis', 'assessment', 'condition', 'problem', 'history']
        in_condition_section = any(kw in section.lower() for kw in condition_keywords)

        if in_condition_section or 'diagnos' in text.lower():
            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith(('- ', '• ', '* ', '1.', '2.', '3.')):
                    # Remove bullet/number prefix
                    condition = re.sub(r'^[-•*\d.]+\s*', '', line).strip()
                    if condition and len(condition) > 3:
                        claims.append(Claim(
                            id="",
                            text=condition,
                            category=ClaimCategory.HIGH_RISK,
                            claim_type=HealthcareClaimType.DIAGNOSIS.value,
                            section=section,
                        ))

        return claims

    def _parse_medication(self, med_text: str) -> MedicationComponents:
        """Parse medication string into components."""
        components = MedicationComponents(raw_text=med_text)
        text = med_text.lower()

        # Extract dose number and unit
        dose_match = MEDICATION_PATTERNS["dose"].search(text)
        if dose_match:
            # Handle comma as decimal separator
            dose_str = dose_match.group(1).replace(',', '.')
            components.dose = float(dose_str)
            components.unit = dose_match.group(2).lower()

        # Extract route
        route_match = MEDICATION_PATTERNS["route"].search(text)
        if route_match:
            components.route = route_match.group(1).lower()

        # Extract frequency
        freq_match = MEDICATION_PATTERNS["frequency"].search(text)
        if freq_match:
            components.frequency = freq_match.group(1).lower()

        # Name is typically the first word (before numbers/doses)
        name_match = re.match(r'^([a-z]+)', text)
        if name_match:
            components.name = name_match.group(1)

        return components

    def classify_claim(self, claim: Claim) -> str:
        """Classify a claim into a healthcare-specific category."""
        return claim.claim_type

    def _get_healthcare_claim_type(self, claim: Claim) -> HealthcareClaimType:
        """Get the HealthcareClaimType enum for a claim."""
        try:
            return HealthcareClaimType(claim.claim_type)
        except ValueError:
            return HealthcareClaimType.ADMINISTRATIVE

    # =========================================================================
    # Verification Methods
    # =========================================================================

    async def verify_claim(
        self,
        claim: Claim,
        context: VerificationContext
    ) -> VerificationResult:
        """
        Verify a medical claim against authoritative sources.

        Routes to appropriate verification method based on claim type.
        """
        claim_type = self._get_healthcare_claim_type(claim)

        # Get transcript/source text from context
        transcript = context.session_metadata.get("transcript", "")
        if not transcript and context.source_documents:
            transcript = " ".join(
                str(doc.get("content", ""))
                for doc in context.source_documents
            )

        # Route to appropriate verifier
        if claim_type == HealthcareClaimType.MEDICATION:
            return await self._verify_medication_claim(claim, transcript)
        elif claim_type == HealthcareClaimType.ALLERGY:
            return await self._verify_allergy_claim(claim, transcript)
        elif claim_type == HealthcareClaimType.VITAL_SIGN:
            return await self._verify_vital_sign_claim(claim, transcript)
        elif claim_type == HealthcareClaimType.EXAM_FINDING:
            return await self._verify_exam_finding_claim(claim, transcript)
        else:
            return await self._verify_generic_claim(claim, transcript)

    async def _verify_medication_claim(
        self,
        claim: Claim,
        transcript: str
    ) -> VerificationResult:
        """Verify a medication claim against transcript/EHR."""
        transcript_lower = transcript.lower()
        components = self._parse_medication(claim.text)

        med_name = components.name

        if not med_name:
            return VerificationResult(
                passed=False,
                confidence=0.0,
                risk_level=RiskLevel.HIGH,
                details={"status": "not_found"},
                warnings=["Could not parse medication name"],
            )

        # Check if medication name appears in transcript
        if med_name not in transcript_lower:
            return VerificationResult(
                passed=False,
                confidence=0.0,
                risk_level=RiskLevel.HIGH,
                details={
                    "status": "not_found",
                    "medication": med_name,
                },
                warnings=[f"Medication '{med_name}' not found in transcript"],
            )

        # Medication name found - now check dose
        claimed_dose = components.dose

        # Find context around medication mention
        med_index = transcript_lower.find(med_name)
        context_start = max(0, med_index - 50)
        context_end = min(len(transcript_lower), med_index + 100)
        context_text = transcript_lower[context_start:context_end]

        # Extract dose from transcript context
        dose_match = MEDICATION_PATTERNS["dose"].search(context_text)

        if dose_match and claimed_dose:
            transcript_dose = float(dose_match.group(1).replace(',', '.'))

            if transcript_dose == claimed_dose:
                return VerificationResult(
                    passed=True,
                    confidence=1.0,
                    risk_level=RiskLevel.LOW,
                    details={
                        "status": "verified",
                        "medication": med_name,
                        "dose": claimed_dose,
                        "matched_text": context_text.strip(),
                    },
                )
            else:
                return VerificationResult(
                    passed=False,
                    confidence=0.3,
                    risk_level=RiskLevel.CRITICAL,
                    details={
                        "status": "contradicted",
                        "medication": med_name,
                        "claimed_dose": claimed_dose,
                        "transcript_dose": transcript_dose,
                    },
                    warnings=[
                        f"DOSE MISMATCH: Claim says {claimed_dose}, "
                        f"transcript says {transcript_dose}"
                    ],
                )

        # Medication found but dose not confirmed
        return VerificationResult(
            passed=False,
            confidence=0.6,
            risk_level=RiskLevel.MEDIUM,
            details={
                "status": "partial",
                "medication": med_name,
                "matched_text": context_text.strip(),
            },
            warnings=[f"Medication '{med_name}' found but dose not confirmed"],
        )

    async def _verify_allergy_claim(
        self,
        claim: Claim,
        transcript: str
    ) -> VerificationResult:
        """Verify an allergy claim against transcript/EHR."""
        transcript_lower = transcript.lower()
        claim_lower = claim.text.lower()

        # Check for NKDA / no allergies
        is_nkda = claim.metadata.get("is_nkda", False)
        if is_nkda or 'nkda' in claim_lower or 'no known' in claim_lower:
            if 'no' in transcript_lower and 'allerg' in transcript_lower:
                return VerificationResult(
                    passed=True,
                    confidence=1.0,
                    risk_level=RiskLevel.LOW,
                    details={"status": "verified", "type": "nkda"},
                )
            if 'none' in transcript_lower:
                return VerificationResult(
                    passed=True,
                    confidence=0.9,
                    risk_level=RiskLevel.LOW,
                    details={"status": "verified", "type": "nkda"},
                )

        # Check for specific allergens
        for allergen in COMMON_ALLERGENS:
            if allergen in claim_lower:
                if allergen in transcript_lower:
                    return VerificationResult(
                        passed=True,
                        confidence=1.0,
                        risk_level=RiskLevel.LOW,
                        details={
                            "status": "verified",
                            "allergen": allergen,
                        },
                    )
                else:
                    return VerificationResult(
                        passed=False,
                        confidence=0.0,
                        risk_level=RiskLevel.CRITICAL,
                        details={
                            "status": "not_found",
                            "allergen": allergen,
                        },
                        warnings=[
                            f"Allergy to {allergen} NOT mentioned in transcript - VERIFY!"
                        ],
                    )

        return VerificationResult(
            passed=False,
            confidence=0.0,
            risk_level=RiskLevel.HIGH,
            details={"status": "not_found"},
            warnings=["Allergy claim needs verification"],
        )

    async def _verify_vital_sign_claim(
        self,
        claim: Claim,
        transcript: str
    ) -> VerificationResult:
        """Verify vital sign claims - often from direct measurement, not transcript."""
        # Vitals are usually measured, not discussed - partial verification
        return VerificationResult(
            passed=True,
            confidence=0.5,
            risk_level=RiskLevel.LOW,
            details={
                "status": "partial",
                "note": "Vital signs typically from direct measurement",
            },
        )

    async def _verify_exam_finding_claim(
        self,
        claim: Claim,
        transcript: str
    ) -> VerificationResult:
        """Verify physical exam findings."""
        transcript_lower = transcript.lower()
        claim_lower = claim.text.lower()

        # Check for common exam terms
        if 'murmur' in claim_lower:
            if 'murmur' in transcript_lower:
                return VerificationResult(
                    passed=True,
                    confidence=0.8,
                    risk_level=RiskLevel.LOW,
                    details={"status": "verified", "finding": "murmur"},
                )

        if 'edema' in claim_lower:
            if 'edema' in transcript_lower or 'swelling' in transcript_lower:
                return VerificationResult(
                    passed=True,
                    confidence=0.8,
                    risk_level=RiskLevel.LOW,
                    details={"status": "verified", "finding": "edema"},
                )
            elif 'no swelling' in transcript_lower:
                return VerificationResult(
                    passed=False,
                    confidence=0.0,
                    risk_level=RiskLevel.HIGH,
                    details={"status": "contradicted"},
                    warnings=["CONTRADICTION: Note says edema, transcript says no swelling"],
                )

        return VerificationResult(
            passed=False,
            confidence=0.5,
            risk_level=RiskLevel.MEDIUM,
            details={"status": "partial"},
            warnings=["Exam finding needs verification"],
        )

    async def _verify_generic_claim(
        self,
        claim: Claim,
        transcript: str
    ) -> VerificationResult:
        """Default verification using simple text search."""
        transcript_lower = transcript.lower()
        claim_lower = claim.text.lower()

        if claim_lower in transcript_lower:
            return VerificationResult(
                passed=True,
                confidence=1.0,
                risk_level=RiskLevel.LOW,
                details={"status": "verified"},
            )

        # Check word overlap
        claim_words = set(claim_lower.split())
        transcript_words = set(transcript_lower.split())
        overlap = len(claim_words & transcript_words) / len(claim_words) if claim_words else 0

        if overlap > 0.7:
            return VerificationResult(
                passed=True,
                confidence=overlap,
                risk_level=RiskLevel.LOW,
                details={"status": "partial", "word_overlap": overlap},
            )

        return VerificationResult(
            passed=False,
            confidence=0.0,
            risk_level=RiskLevel.MEDIUM,
            details={"status": "not_found"},
            warnings=["No match found - needs verification"],
        )

    async def verify_response(
        self,
        context: VerificationContext
    ) -> VerificationResult:
        """Verify complete healthcare response."""
        claims = context.claims or self.extract_claims(context.response)

        if not claims:
            return VerificationResult(
                passed=True,
                confidence=1.0,
                risk_level=RiskLevel.LOW,
                details={"note": "No verifiable claims extracted"},
            )

        # Verify each claim
        results = []
        for claim in claims:
            result = await self.verify_claim(claim, context)
            results.append(result)

        # Aggregate results
        passed_count = sum(1 for r in results if r.passed)
        total = len(results)

        # Determine overall risk (highest among claims)
        risk_levels = [r.risk_level for r in results]
        if RiskLevel.CRITICAL in risk_levels:
            overall_risk = RiskLevel.CRITICAL
        elif RiskLevel.HIGH in risk_levels:
            overall_risk = RiskLevel.HIGH
        elif RiskLevel.MEDIUM in risk_levels:
            overall_risk = RiskLevel.MEDIUM
        else:
            overall_risk = RiskLevel.LOW

        # Aggregate warnings
        all_warnings = []
        for r in results:
            all_warnings.extend(r.warnings)

        return VerificationResult(
            passed=passed_count == total,
            confidence=passed_count / total if total > 0 else 1.0,
            risk_level=overall_risk,
            details={
                "total_claims": total,
                "verified_claims": passed_count,
                "claim_results": [
                    {"claim": c.text, "passed": r.passed, "risk": r.risk_level.value}
                    for c, r in zip(claims, results)
                ],
            },
            warnings=all_warnings,
        )

    # =========================================================================
    # Source Attribution Methods
    # =========================================================================

    def get_authoritative_sources(self, claim: Claim) -> List[str]:
        """Get authoritative medical sources for claim verification."""
        claim_type = self._get_healthcare_claim_type(claim)

        if claim_type in (HealthcareClaimType.MEDICATION,
                          HealthcareClaimType.DOSAGE,
                          HealthcareClaimType.DRUG_INTERACTION,
                          HealthcareClaimType.ALLERGY):
            return AUTHORITATIVE_DRUG_SOURCES
        else:
            return AUTHORITATIVE_CLINICAL_SOURCES

    def validate_source(self, source_id: str) -> bool:
        """Validate that a source is authoritative for healthcare."""
        source_lower = source_id.lower()
        all_sources = AUTHORITATIVE_DRUG_SOURCES + AUTHORITATIVE_CLINICAL_SOURCES
        return any(src in source_lower for src in all_sources)

    # =========================================================================
    # Compliance Methods
    # =========================================================================

    def get_compliance_requirements(self) -> List[str]:
        """Get healthcare compliance requirements."""
        return ["HIPAA", "FDA_GUIDANCE", "CLINICAL_GUIDELINES", "HITECH"]

    async def check_compliance(
        self,
        context: VerificationContext
    ) -> VerificationResult:
        """Check HIPAA and regulatory compliance."""
        warnings = []

        # Check for PHI indicators
        phi_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{10}\b',  # Phone numbers
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',  # Dates
        ]

        response = context.response
        for pattern in phi_patterns:
            if re.search(pattern, response):
                warnings.append("Potential PHI detected - review for HIPAA compliance")
                break

        return VerificationResult(
            passed=len(warnings) == 0,
            confidence=1.0 if not warnings else 0.5,
            risk_level=RiskLevel.HIGH if warnings else RiskLevel.LOW,
            details={"compliance_check": "hipaa"},
            warnings=warnings,
        )

    # =========================================================================
    # Escalation Methods
    # =========================================================================

    def should_escalate(
        self,
        verification_result: VerificationResult,
        context: VerificationContext
    ) -> bool:
        """Determine if healthcare response needs human review."""
        # Always escalate CRITICAL and HIGH risk
        if verification_result.risk_level in (RiskLevel.CRITICAL, RiskLevel.HIGH):
            return True

        # Escalate if confidence is low
        if verification_result.confidence < 0.7:
            return True

        # Escalate if verification failed
        if not verification_result.passed:
            return True

        # Escalate if there are warnings
        if verification_result.warnings:
            return True

        return False

    def get_escalation_reason(
        self,
        verification_result: VerificationResult,
        context: VerificationContext
    ) -> str:
        """Get explanation for healthcare escalation."""
        reasons = []

        if verification_result.risk_level == RiskLevel.CRITICAL:
            reasons.append("Critical medical content requires clinician review")
        if verification_result.risk_level == RiskLevel.HIGH:
            reasons.append("High-risk medical information detected")
        if verification_result.confidence < 0.7:
            reasons.append(f"Low confidence score: {verification_result.confidence:.2f}")
        if not verification_result.passed:
            reasons.append("Verification checks failed")
        if verification_result.warnings:
            reasons.append(f"Warnings: {', '.join(verification_result.warnings)}")

        return "; ".join(reasons) if reasons else "Manual review requested"

    def get_review_level(
        self,
        verification_result: VerificationResult,
        entropy: float = 0.0
    ) -> ReviewLevel:
        """
        Assign review level based on risk and uncertainty.

        Based on Paper 2 methodology for 87% burden reduction.
        """
        if verification_result.risk_level == RiskLevel.CRITICAL:
            return ReviewLevel.DETAILED
        elif verification_result.risk_level == RiskLevel.HIGH:
            return ReviewLevel.DETAILED
        elif verification_result.risk_level == RiskLevel.MEDIUM:
            if entropy > 0.5:
                return ReviewLevel.STANDARD
            else:
                return ReviewLevel.BRIEF
        else:
            return ReviewLevel.BRIEF

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_domain_terminology(self) -> Dict[str, str]:
        """Get medical terminology mappings."""
        return {
            # Frequency abbreviations
            "PRN": "as needed",
            "BID": "twice daily",
            "TID": "three times daily",
            "QID": "four times daily",
            "QD": "once daily",
            "QHS": "at bedtime",
            "QAM": "every morning",
            "QPM": "every evening",
            "Q4H": "every 4 hours",
            "Q6H": "every 6 hours",
            "Q8H": "every 8 hours",
            "Q12H": "every 12 hours",

            # Route abbreviations
            "PO": "by mouth",
            "IV": "intravenous",
            "IM": "intramuscular",
            "SC": "subcutaneous",
            "SQ": "subcutaneous",
            "SL": "sublingual",
            "PR": "per rectum",
            "TOP": "topical",
            "INH": "inhaled",
            "NEB": "nebulized",

            # Common medical abbreviations
            "HTN": "hypertension",
            "DM": "diabetes mellitus",
            "T2DM": "type 2 diabetes mellitus",
            "CHF": "congestive heart failure",
            "COPD": "chronic obstructive pulmonary disease",
            "CAD": "coronary artery disease",
            "MI": "myocardial infarction",
            "CVA": "cerebrovascular accident (stroke)",
            "DVT": "deep vein thrombosis",
            "PE": "pulmonary embolism",
            "UTI": "urinary tract infection",
            "GERD": "gastroesophageal reflux disease",
            "CKD": "chronic kidney disease",
            "ESRD": "end-stage renal disease",
            "NKDA": "no known drug allergies",
            "NKA": "no known allergies",
        }

    def normalize_response(self, response: str) -> str:
        """Normalize medical response text for consistent processing."""
        # Expand common abbreviations for better matching
        text = response
        terminology = self.get_domain_terminology()

        # Only expand if the abbreviation is standalone (not part of a word)
        for abbrev, expansion in terminology.items():
            pattern = rf'\b{re.escape(abbrev)}\b'
            # Add expansion in parentheses after abbreviation
            text = re.sub(pattern, f'{abbrev} ({expansion})', text, flags=re.IGNORECASE)

        return text
