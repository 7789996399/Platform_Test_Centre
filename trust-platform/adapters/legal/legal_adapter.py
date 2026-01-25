"""
Legal Industry Adapter
======================

Implements the IndustryAdapter interface for legal AI governance.

This adapter provides legal-specific logic for:
- Case citation verification (critical - fabricated citations are a known LLM problem)
- Statute reference validation
- Contract clause extraction and verification
- Jurisdiction determination
- Limitation period checking

Key Risk: LLMs are notorious for fabricating legal citations that sound plausible
but reference non-existent cases. This adapter applies stricter thresholds for
citation claims.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from adapters.base_adapter import (
    AdapterConfig,
    Claim as BaseClaim,
    IndustryAdapter,
    RiskLevel,
    VerificationContext,
    VerificationResult,
)

from core_engine.faithfulness import (
    Claim,
    ClaimCategory,
    SourceDocument,
    VerificationResult as CoreVerificationResult,
    VerificationStatus,
)
from core_engine.semantic_entropy import EntropyThresholds


# =============================================================================
# LEGAL-SPECIFIC ENUMS
# =============================================================================

class LegalClaimType(str, Enum):
    """
    Types of legal claims with associated risk levels.

    Citation claims are highest risk due to known LLM hallucination patterns.
    """
    # Critical risk - fabrication is common and dangerous
    CASE_CITATION = "case_citation"           # Case law references
    STATUTE_REFERENCE = "statute_reference"   # Statutory citations
    REGULATION_REFERENCE = "regulation_reference"  # Regulatory citations

    # High risk - material to legal advice
    CONTRACT_CLAUSE = "contract_clause"       # Contract terms/provisions
    LEGAL_HOLDING = "legal_holding"           # Court holdings/rulings
    PRECEDENT = "precedent"                   # Legal precedent claims
    JURISDICTION = "jurisdiction"             # Jurisdictional assertions

    # Medium risk - important but less likely fabricated
    LIMITATION_PERIOD = "limitation_period"   # Statute of limitations
    LEGAL_STANDARD = "legal_standard"         # Standards of review, burdens
    PROCEDURAL_REQUIREMENT = "procedural_requirement"  # Filing requirements
    DAMAGES_CALCULATION = "damages_calculation"  # Damage computations

    # Lower risk - general legal information
    LEGAL_DEFINITION = "legal_definition"     # Term definitions
    GENERAL_PRINCIPLE = "general_principle"   # General legal principles
    ADMINISTRATIVE = "administrative"         # Case management info

    def to_category(self) -> ClaimCategory:
        """Map legal claim type to generic risk category."""
        critical_risk = {
            self.CASE_CITATION,
            self.STATUTE_REFERENCE,
            self.REGULATION_REFERENCE,
        }
        high_risk = {
            self.CONTRACT_CLAUSE,
            self.LEGAL_HOLDING,
            self.PRECEDENT,
            self.JURISDICTION,
        }
        low_risk = {
            self.LEGAL_DEFINITION,
            self.GENERAL_PRINCIPLE,
            self.ADMINISTRATIVE,
        }

        if self in critical_risk:
            return ClaimCategory.HIGH_RISK  # Map to HIGH_RISK (most severe)
        elif self in high_risk:
            return ClaimCategory.HIGH_RISK
        elif self in low_risk:
            return ClaimCategory.LOW_RISK
        else:
            return ClaimCategory.MEDIUM_RISK


class LegalReviewLevel(str, Enum):
    """Review level for legal responses."""
    BRIEF = "brief"           # 30 seconds - citation spot check
    STANDARD = "standard"     # 2 minutes - clause/holding review
    DETAILED = "detailed"     # 10+ minutes - full legal reasoning review


# =============================================================================
# LEGAL-SPECIFIC DATA CLASSES
# =============================================================================

@dataclass
class CaseCitationComponents:
    """Parsed components of a case citation."""
    case_name: Optional[str] = None
    volume: Optional[str] = None
    reporter: Optional[str] = None
    page: Optional[str] = None
    court: Optional[str] = None
    year: Optional[int] = None
    parallel_citations: List[str] = field(default_factory=list)
    raw_text: str = ""


@dataclass
class StatuteCitationComponents:
    """Parsed components of a statute citation."""
    title: Optional[str] = None
    code: Optional[str] = None
    section: Optional[str] = None
    subsection: Optional[str] = None
    jurisdiction: Optional[str] = None
    raw_text: str = ""


@dataclass
class LegalVerificationResult:
    """Extended verification result with legal-specific fields."""
    claim: Claim
    status: VerificationStatus
    confidence: float
    risk_level: RiskLevel
    review_level: LegalReviewLevel
    matched_source: Optional[SourceDocument] = None
    matched_text: Optional[str] = None
    explanation: str = ""
    needs_entropy_check: bool = True
    is_fabricated_citation: bool = False
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# AUTHORITATIVE SOURCES
# =============================================================================

AUTHORITATIVE_CASE_SOURCES = [
    "westlaw.com",
    "lexisnexis.com",
    "courtlistener.com",
    "casetext.com",
    "google.com/scholar",
    "law.justia.com",
    "findlaw.com",
    "oyez.org",
    "supremecourt.gov",
    "uscourts.gov",
]

AUTHORITATIVE_STATUTE_SOURCES = [
    "law.cornell.edu",
    "congress.gov",
    "govinfo.gov",
    "uscode.house.gov",
    "law.justia.com",
    "legis.state",  # State legislature sites
    "westlaw.com",
    "lexisnexis.com",
]

AUTHORITATIVE_REGULATORY_SOURCES = [
    "ecfr.gov",
    "federalregister.gov",
    "regulations.gov",
    "sec.gov",
    "ftc.gov",
    "dol.gov",
]

# Common federal reporters
FEDERAL_REPORTERS = [
    "U.S.", "S. Ct.", "S.Ct.", "L. Ed.", "L.Ed.", "L. Ed. 2d", "L.Ed.2d",
    "F.", "F.2d", "F.3d", "F.4th",
    "F. Supp.", "F.Supp.", "F. Supp. 2d", "F.Supp.2d", "F. Supp. 3d", "F.Supp.3d",
    "Fed. Cl.", "Fed.Cl.",
    "B.R.",  # Bankruptcy Reporter
]

# Common state reporters
STATE_REPORTERS = [
    "Cal.", "Cal.2d", "Cal.3d", "Cal.4th", "Cal.5th",
    "Cal. App.", "Cal.App.2d", "Cal.App.3d", "Cal.App.4th", "Cal.App.5th",
    "N.Y.", "N.Y.2d", "N.Y.3d",
    "A.D.", "A.D.2d", "A.D.3d",
    "Ill.", "Ill.2d",
    "Tex.", "S.W.", "S.W.2d", "S.W.3d",
    "N.E.", "N.E.2d", "N.E.3d",
    "So.", "So.2d", "So.3d",
    "P.", "P.2d", "P.3d",
    "A.", "A.2d", "A.3d",
    "N.W.", "N.W.2d",
    "S.E.", "S.E.2d",
]

ALL_REPORTERS = FEDERAL_REPORTERS + STATE_REPORTERS


# =============================================================================
# REGEX PATTERNS FOR CLAIM EXTRACTION
# =============================================================================

# Case citation patterns
CASE_CITATION_PATTERNS = [
    # Standard format: Case Name, Volume Reporter Page (Court Year)
    # e.g., "Brown v. Board of Education, 347 U.S. 483 (1954)"
    re.compile(
        r'([A-Z][A-Za-z\'\-\.\s]+(?:\s+v\.?\s+[A-Z][A-Za-z\'\-\.\s]+)?)'
        r',?\s*(\d+)\s+(' + '|'.join(re.escape(r) for r in ALL_REPORTERS) + r')\s+(\d+)'
        r'(?:\s*\(([^)]+)\s*(\d{4})\))?',
        re.IGNORECASE
    ),
    # Supra/infra references
    re.compile(r'([A-Z][a-z]+(?:\s+v\.?\s+[A-Z][a-z]+)?),?\s+supra', re.IGNORECASE),
    # Id. references
    re.compile(r'\bId\.\s+at\s+(\d+)', re.IGNORECASE),
]

# Statute citation patterns
STATUTE_CITATION_PATTERNS = [
    # U.S. Code: 42 U.S.C. § 1983
    re.compile(r'(\d+)\s+U\.?S\.?C\.?\s*§?\s*(\d+[a-z]?)(?:\(([^)]+)\))?', re.IGNORECASE),
    # Code of Federal Regulations: 17 C.F.R. § 240.10b-5
    re.compile(r'(\d+)\s+C\.?F\.?R\.?\s*§?\s*([\d\.]+[a-z\-]*)', re.IGNORECASE),
    # State statutes: Cal. Civ. Code § 1542
    re.compile(r'([A-Z][a-z]+\.?)\s+([A-Z][a-z]+\.?)\s+(?:Code|Law)\s*§?\s*(\d+[a-z]?)', re.IGNORECASE),
    # Generic section references
    re.compile(r'[Ss]ection\s+(\d+(?:\.\d+)?[a-z]?)\s+of\s+(?:the\s+)?([A-Z][A-Za-z\s]+(?:Act|Code|Law))', re.IGNORECASE),
]

# Contract clause patterns
CONTRACT_CLAUSE_PATTERNS = [
    re.compile(r'(?:Section|Article|Clause|Paragraph)\s+(\d+(?:\.\d+)?[a-z]?)', re.IGNORECASE),
    re.compile(r'(?:pursuant to|under|in accordance with)\s+(?:Section|Article)\s+(\d+)', re.IGNORECASE),
    re.compile(r'indemnif(?:y|ication|ies)', re.IGNORECASE),
    re.compile(r'limitation of liability', re.IGNORECASE),
    re.compile(r'force majeure', re.IGNORECASE),
    re.compile(r'governing law', re.IGNORECASE),
    re.compile(r'arbitration\s+(?:clause|provision|agreement)', re.IGNORECASE),
    re.compile(r'non-?compete', re.IGNORECASE),
    re.compile(r'confidentiality', re.IGNORECASE),
    re.compile(r'termination\s+(?:clause|provision|for cause)', re.IGNORECASE),
]

# Limitation period patterns
LIMITATION_PATTERNS = [
    re.compile(r'statute of limitations?\s+(?:is|of)\s+(\d+)\s+years?', re.IGNORECASE),
    re.compile(r'limitations?\s+period\s+(?:is|of)\s+(\d+)\s+(?:years?|months?|days?)', re.IGNORECASE),
    re.compile(r'must (?:be filed|file|commence|bring)\s+within\s+(\d+)\s+(?:years?|months?|days?)', re.IGNORECASE),
    re.compile(r'(?:time-?barred|barred by limitations?)\s+after\s+(\d+)', re.IGNORECASE),
]

# Jurisdiction patterns
JURISDICTION_PATTERNS = [
    re.compile(r'(?:subject matter|personal|in rem|quasi in rem)\s+jurisdiction', re.IGNORECASE),
    re.compile(r'diversity\s+(?:jurisdiction|of citizenship)', re.IGNORECASE),
    re.compile(r'federal\s+question\s+jurisdiction', re.IGNORECASE),
    re.compile(r'(?:proper|appropriate)\s+venue', re.IGNORECASE),
    re.compile(r'forum\s+(?:selection|non conveniens)', re.IGNORECASE),
    re.compile(r'(?:state|federal)\s+court\s+(?:has|lacks)\s+jurisdiction', re.IGNORECASE),
]


# =============================================================================
# LEGAL ADAPTER IMPLEMENTATION
# =============================================================================

class LegalAdapter(IndustryAdapter):
    """
    Legal-specific adapter for TRUST verification.

    Provides domain-specific logic for:
    - Case citation verification (critical - LLMs often fabricate citations)
    - Statute reference validation
    - Contract clause extraction
    - Jurisdiction analysis
    - Limitation period verification

    Key Risk Mitigation:
    This adapter uses VERY strict entropy thresholds for citations because
    LLMs frequently generate plausible-sounding but completely fabricated
    case citations. A "confident hallucinator" in legal context could result
    in sanctions, malpractice, or case dismissal.

    Example:
        >>> adapter = LegalAdapter()
        >>> await adapter.initialize()
        >>>
        >>> claims = adapter.extract_claims("In Smith v. Jones, 500 F.3d 100 (2023)...")
        >>> for claim in claims:
        ...     result = await adapter.verify_claim(claim, context)
        ...     if result.details.get("is_fabricated_citation"):
        ...         print(f"WARNING: Potentially fabricated citation: {claim.text}")
    """

    def __init__(self):
        config = AdapterConfig(
            industry_name="legal",
            version="1.0.0",
            entropy_threshold=0.15,  # VERY strict for legal - citations must be consistent
            faithfulness_threshold=0.95,  # Near-perfect source matching required
            ensemble_agreement_threshold=0.9,  # Strong consensus for legal claims
            max_risk_level_auto_approve=RiskLevel.MINIMAL,
            custom_settings={
                "citation_verification_required": True,
                "jurisdiction_check": True,
                "precedent_validation": True,
                "fabrication_detection": True,
            }
        )
        super().__init__(config)
        self._claim_counter = 0

    # =========================================================================
    # Lifecycle Methods
    # =========================================================================

    async def initialize(self) -> None:
        """Initialize legal-specific resources."""
        # In production, this would:
        # - Connect to case law databases (Westlaw, LexisNexis APIs)
        # - Load statute databases
        # - Initialize legal NER models
        # - Load jurisdiction mappings
        self._initialized = True

    async def shutdown(self) -> None:
        """Clean up legal adapter resources."""
        self._initialized = False

    # =========================================================================
    # Risk Assessment Methods
    # =========================================================================

    def classify_risk(self, context: VerificationContext) -> RiskLevel:
        """
        Classify risk level for legal responses.

        Risk Classification:
        - CRITICAL: Case citations, statute references (fabrication risk)
        - HIGH: Legal holdings, contract clauses, jurisdictional claims
        - MEDIUM: Procedural requirements, limitation periods
        - LOW: General legal principles, definitions
        """
        claims = self.extract_claims(context.response)

        if not claims:
            return RiskLevel.LOW

        # Citation claims are always critical risk
        critical_types = {
            LegalClaimType.CASE_CITATION,
            LegalClaimType.STATUTE_REFERENCE,
            LegalClaimType.REGULATION_REFERENCE,
        }

        high_types = {
            LegalClaimType.CONTRACT_CLAUSE,
            LegalClaimType.LEGAL_HOLDING,
            LegalClaimType.PRECEDENT,
            LegalClaimType.JURISDICTION,
        }

        for claim in claims:
            claim_type = self._get_legal_claim_type(claim)
            if claim_type in critical_types:
                return RiskLevel.CRITICAL

        for claim in claims:
            claim_type = self._get_legal_claim_type(claim)
            if claim_type in high_types:
                return RiskLevel.HIGH

        medium_types = {
            LegalClaimType.LIMITATION_PERIOD,
            LegalClaimType.LEGAL_STANDARD,
            LegalClaimType.PROCEDURAL_REQUIREMENT,
            LegalClaimType.DAMAGES_CALCULATION,
        }

        for claim in claims:
            claim_type = self._get_legal_claim_type(claim)
            if claim_type in medium_types:
                return RiskLevel.MEDIUM

        return RiskLevel.LOW

    def get_risk_thresholds(self) -> Dict[str, float]:
        """Get legal-specific risk thresholds."""
        return {
            "entropy": self._config.entropy_threshold,
            "faithfulness": self._config.faithfulness_threshold,
            "ensemble_agreement": self._config.ensemble_agreement_threshold,
        }

    def get_entropy_thresholds(self) -> EntropyThresholds:
        """Get entropy thresholds configured for legal domain."""
        return EntropyThresholds(
            low_medium_boundary=0.15,  # Very strict - citations must be consistent
            medium_high_boundary=0.4,
        )

    # =========================================================================
    # Claim Extraction Methods
    # =========================================================================

    def extract_claims(self, response: str) -> List[Claim]:
        """
        Extract legal claims from an AI response.

        Extracts:
        - Case citations
        - Statute references
        - Contract clauses
        - Jurisdiction claims
        - Limitation periods
        """
        all_claims: List[Claim] = []

        all_claims.extend(self._extract_case_citations(response))
        all_claims.extend(self._extract_statute_citations(response))
        all_claims.extend(self._extract_contract_clauses(response))
        all_claims.extend(self._extract_jurisdiction_claims(response))
        all_claims.extend(self._extract_limitation_periods(response))

        # Assign unique IDs
        for i, claim in enumerate(all_claims):
            claim.id = f"legal_{i:03d}_{claim.claim_type}"

        return all_claims

    def _extract_case_citations(self, text: str, section: str = "general") -> List[Claim]:
        """Extract case citation claims from text."""
        claims = []
        seen_citations = set()

        for pattern in CASE_CITATION_PATTERNS:
            matches = pattern.finditer(text)
            for match in matches:
                citation_text = match.group(0).strip()

                # Deduplicate
                if citation_text.lower() in seen_citations:
                    continue
                seen_citations.add(citation_text.lower())

                # Parse citation components
                components = self._parse_case_citation(citation_text)

                claims.append(Claim(
                    id="",
                    text=citation_text,
                    category=ClaimCategory.HIGH_RISK,
                    claim_type=LegalClaimType.CASE_CITATION.value,
                    source_span=(match.start(), match.end()),
                    section=section,
                    metadata={
                        "components": {
                            "case_name": components.case_name,
                            "volume": components.volume,
                            "reporter": components.reporter,
                            "page": components.page,
                            "court": components.court,
                            "year": components.year,
                        },
                        "is_citation": True,
                        "fabrication_risk": "high",
                    }
                ))

        return claims

    def _extract_statute_citations(self, text: str, section: str = "general") -> List[Claim]:
        """Extract statute citation claims from text."""
        claims = []
        seen_citations = set()

        for pattern in STATUTE_CITATION_PATTERNS:
            matches = pattern.finditer(text)
            for match in matches:
                citation_text = match.group(0).strip()

                if citation_text.lower() in seen_citations:
                    continue
                seen_citations.add(citation_text.lower())

                components = self._parse_statute_citation(citation_text)

                claims.append(Claim(
                    id="",
                    text=citation_text,
                    category=ClaimCategory.HIGH_RISK,
                    claim_type=LegalClaimType.STATUTE_REFERENCE.value,
                    source_span=(match.start(), match.end()),
                    section=section,
                    metadata={
                        "components": {
                            "title": components.title,
                            "code": components.code,
                            "section": components.section,
                            "subsection": components.subsection,
                        },
                        "is_citation": True,
                    }
                ))

        return claims

    def _extract_contract_clauses(self, text: str, section: str = "general") -> List[Claim]:
        """Extract contract clause claims from text."""
        claims = []

        for pattern in CONTRACT_CLAUSE_PATTERNS:
            matches = pattern.finditer(text)
            for match in matches:
                # Get context around match
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 100)
                context_text = text[start:end].strip()

                claims.append(Claim(
                    id="",
                    text=context_text,
                    category=ClaimCategory.HIGH_RISK,
                    claim_type=LegalClaimType.CONTRACT_CLAUSE.value,
                    source_span=(match.start(), match.end()),
                    section=section,
                    metadata={"clause_type": match.group(0)}
                ))

        return claims

    def _extract_jurisdiction_claims(self, text: str, section: str = "general") -> List[Claim]:
        """Extract jurisdiction claims from text."""
        claims = []

        for pattern in JURISDICTION_PATTERNS:
            matches = pattern.finditer(text)
            for match in matches:
                start = max(0, match.start() - 30)
                end = min(len(text), match.end() + 50)
                context_text = text[start:end].strip()

                claims.append(Claim(
                    id="",
                    text=context_text,
                    category=ClaimCategory.HIGH_RISK,
                    claim_type=LegalClaimType.JURISDICTION.value,
                    source_span=(match.start(), match.end()),
                    section=section,
                ))

        return claims

    def _extract_limitation_periods(self, text: str, section: str = "general") -> List[Claim]:
        """Extract limitation period claims from text."""
        claims = []

        for pattern in LIMITATION_PATTERNS:
            matches = pattern.finditer(text)
            for match in matches:
                start = max(0, match.start() - 20)
                end = min(len(text), match.end() + 30)
                context_text = text[start:end].strip()

                # Extract the period value
                period_value = match.group(1) if match.groups() else None

                claims.append(Claim(
                    id="",
                    text=context_text,
                    category=ClaimCategory.MEDIUM_RISK,
                    claim_type=LegalClaimType.LIMITATION_PERIOD.value,
                    source_span=(match.start(), match.end()),
                    section=section,
                    metadata={"period_value": period_value}
                ))

        return claims

    def _parse_case_citation(self, citation_text: str) -> CaseCitationComponents:
        """Parse a case citation into components."""
        components = CaseCitationComponents(raw_text=citation_text)

        # Try to parse standard format
        # "Brown v. Board of Education, 347 U.S. 483 (1954)"
        main_pattern = re.compile(
            r'([A-Z][A-Za-z\'\-\.\s]+(?:\s+v\.?\s+[A-Z][A-Za-z\'\-\.\s]+)?)'
            r',?\s*(\d+)\s+([A-Za-z\.\s]+\d*[a-z]*)\s+(\d+)'
            r'(?:\s*\(([^)]*?)(\d{4})\))?',
            re.IGNORECASE
        )

        match = main_pattern.search(citation_text)
        if match:
            components.case_name = match.group(1).strip() if match.group(1) else None
            components.volume = match.group(2) if match.group(2) else None
            components.reporter = match.group(3).strip() if match.group(3) else None
            components.page = match.group(4) if match.group(4) else None
            components.court = match.group(5).strip() if match.group(5) else None
            components.year = int(match.group(6)) if match.group(6) else None

        return components

    def _parse_statute_citation(self, citation_text: str) -> StatuteCitationComponents:
        """Parse a statute citation into components."""
        components = StatuteCitationComponents(raw_text=citation_text)

        # U.S. Code pattern
        usc_pattern = re.compile(r'(\d+)\s+U\.?S\.?C\.?\s*§?\s*(\d+[a-z]?)(?:\(([^)]+)\))?', re.IGNORECASE)
        match = usc_pattern.search(citation_text)
        if match:
            components.title = match.group(1)
            components.code = "U.S.C."
            components.section = match.group(2)
            components.subsection = match.group(3)
            components.jurisdiction = "Federal"
            return components

        # CFR pattern
        cfr_pattern = re.compile(r'(\d+)\s+C\.?F\.?R\.?\s*§?\s*([\d\.]+)', re.IGNORECASE)
        match = cfr_pattern.search(citation_text)
        if match:
            components.title = match.group(1)
            components.code = "C.F.R."
            components.section = match.group(2)
            components.jurisdiction = "Federal"

        return components

    def classify_claim(self, claim: Claim) -> str:
        """Classify a claim into a legal-specific category."""
        return claim.claim_type

    def _get_legal_claim_type(self, claim: Claim) -> LegalClaimType:
        """Get the LegalClaimType enum for a claim."""
        try:
            return LegalClaimType(claim.claim_type)
        except ValueError:
            return LegalClaimType.ADMINISTRATIVE

    # =========================================================================
    # Verification Methods
    # =========================================================================

    async def verify_claim(
        self,
        claim: Claim,
        context: VerificationContext
    ) -> VerificationResult:
        """
        Verify a legal claim against authoritative sources.

        Routes to appropriate verification method based on claim type.
        """
        claim_type = self._get_legal_claim_type(claim)

        # Get source documents from context
        source_text = context.session_metadata.get("source_document", "")
        if not source_text and context.source_documents:
            source_text = " ".join(
                str(doc.get("content", ""))
                for doc in context.source_documents
            )

        # Route to appropriate verifier
        if claim_type == LegalClaimType.CASE_CITATION:
            return await self._verify_case_citation(claim, source_text, context)
        elif claim_type == LegalClaimType.STATUTE_REFERENCE:
            return await self._verify_statute_citation(claim, source_text, context)
        elif claim_type == LegalClaimType.CONTRACT_CLAUSE:
            return await self._verify_contract_clause(claim, source_text)
        elif claim_type == LegalClaimType.JURISDICTION:
            return await self._verify_jurisdiction_claim(claim, source_text)
        elif claim_type == LegalClaimType.LIMITATION_PERIOD:
            return await self._verify_limitation_period(claim, source_text)
        else:
            return await self._verify_generic_claim(claim, source_text)

    async def _verify_case_citation(
        self,
        claim: Claim,
        source_text: str,
        context: VerificationContext
    ) -> VerificationResult:
        """
        Verify a case citation against authoritative sources.

        This is CRITICAL - fabricated citations are a known LLM failure mode.
        """
        components = claim.metadata.get("components", {})
        case_name = components.get("case_name", "")
        reporter = components.get("reporter", "")
        volume = components.get("volume", "")
        page = components.get("page", "")

        # Check if citation appears in source documents
        source_lower = source_text.lower()
        claim_lower = claim.text.lower()

        # Check for case name in sources
        if case_name and case_name.lower() in source_lower:
            # Case name found - verify details
            if volume and page and f"{volume}" in source_text and f"{page}" in source_text:
                return VerificationResult(
                    passed=True,
                    confidence=0.95,
                    risk_level=RiskLevel.LOW,
                    details={
                        "status": "verified",
                        "citation_type": "case",
                        "case_name": case_name,
                    },
                )
            else:
                return VerificationResult(
                    passed=False,
                    confidence=0.5,
                    risk_level=RiskLevel.HIGH,
                    details={
                        "status": "partial",
                        "citation_type": "case",
                        "case_name": case_name,
                    },
                    warnings=["Case name found but citation details could not be verified"],
                )

        # Citation not found in sources - HIGH RISK for fabrication
        # Check for red flags
        warnings = []
        is_fabricated = False

        # Red flag: Very recent year with obscure reporter
        year = components.get("year")
        if year and year >= 2023:
            warnings.append(f"Recent case ({year}) - verify existence in legal database")

        # Red flag: Implausible volume/page combinations
        if volume and page:
            try:
                if int(volume) > 1000 or int(page) > 5000:
                    warnings.append("Unusual volume/page numbers - verify citation")
                    is_fabricated = True
            except ValueError:
                pass

        # Red flag: Case name follows common fabrication patterns
        if case_name:
            # LLMs often use generic names
            generic_patterns = ["Smith v. Jones", "Doe v. ", "ABC v. ", "XYZ Corp"]
            if any(pattern.lower() in case_name.lower() for pattern in generic_patterns):
                warnings.append("Generic case name pattern - higher fabrication risk")

        return VerificationResult(
            passed=False,
            confidence=0.0,
            risk_level=RiskLevel.CRITICAL,
            details={
                "status": "not_found",
                "citation_type": "case",
                "is_fabricated_citation": is_fabricated,
                "components": components,
            },
            warnings=warnings or ["Case citation NOT found in sources - VERIFY IN LEGAL DATABASE"],
        )

    async def _verify_statute_citation(
        self,
        claim: Claim,
        source_text: str,
        context: VerificationContext
    ) -> VerificationResult:
        """Verify a statute citation against sources."""
        components = claim.metadata.get("components", {})
        title = components.get("title", "")
        section = components.get("section", "")

        source_lower = source_text.lower()

        # Check if statute reference appears in sources
        if title and section:
            # Look for the statute in source
            patterns = [
                f"{title}.*{section}",
                f"section {section}",
                claim.text.lower(),
            ]

            for pattern in patterns:
                if re.search(pattern, source_lower):
                    return VerificationResult(
                        passed=True,
                        confidence=0.9,
                        risk_level=RiskLevel.LOW,
                        details={
                            "status": "verified",
                            "citation_type": "statute",
                            "title": title,
                            "section": section,
                        },
                    )

        return VerificationResult(
            passed=False,
            confidence=0.0,
            risk_level=RiskLevel.HIGH,
            details={
                "status": "not_found",
                "citation_type": "statute",
            },
            warnings=["Statute citation not verified - check official code"],
        )

    async def _verify_contract_clause(
        self,
        claim: Claim,
        source_text: str
    ) -> VerificationResult:
        """Verify a contract clause against source document."""
        source_lower = source_text.lower()
        claim_lower = claim.text.lower()

        # Direct text match
        if claim_lower in source_lower:
            return VerificationResult(
                passed=True,
                confidence=1.0,
                risk_level=RiskLevel.LOW,
                details={"status": "verified", "type": "contract_clause"},
            )

        # Check for clause type keywords
        clause_type = claim.metadata.get("clause_type", "").lower()
        if clause_type and clause_type in source_lower:
            return VerificationResult(
                passed=True,
                confidence=0.7,
                risk_level=RiskLevel.MEDIUM,
                details={"status": "partial", "type": "contract_clause"},
                warnings=["Clause type found but exact text not verified"],
            )

        return VerificationResult(
            passed=False,
            confidence=0.0,
            risk_level=RiskLevel.HIGH,
            details={"status": "not_found", "type": "contract_clause"},
            warnings=["Contract clause not found in source document"],
        )

    async def _verify_jurisdiction_claim(
        self,
        claim: Claim,
        source_text: str
    ) -> VerificationResult:
        """Verify jurisdiction claims."""
        # Jurisdiction claims need legal analysis - flag for review
        return VerificationResult(
            passed=False,
            confidence=0.5,
            risk_level=RiskLevel.MEDIUM,
            details={"status": "needs_review", "type": "jurisdiction"},
            warnings=["Jurisdiction claims require legal analysis - review recommended"],
        )

    async def _verify_limitation_period(
        self,
        claim: Claim,
        source_text: str
    ) -> VerificationResult:
        """Verify limitation period claims."""
        period_value = claim.metadata.get("period_value")
        source_lower = source_text.lower()

        if period_value:
            # Look for matching period in sources
            period_patterns = [
                f"{period_value} year",
                f"{period_value}-year",
                f"within {period_value}",
            ]

            for pattern in period_patterns:
                if pattern in source_lower:
                    return VerificationResult(
                        passed=True,
                        confidence=0.8,
                        risk_level=RiskLevel.LOW,
                        details={
                            "status": "verified",
                            "type": "limitation_period",
                            "period": period_value,
                        },
                    )

        return VerificationResult(
            passed=False,
            confidence=0.0,
            risk_level=RiskLevel.MEDIUM,
            details={"status": "not_found", "type": "limitation_period"},
            warnings=["Limitation period not verified - check applicable statute"],
        )

    async def _verify_generic_claim(
        self,
        claim: Claim,
        source_text: str
    ) -> VerificationResult:
        """Default verification using text matching."""
        source_lower = source_text.lower()
        claim_lower = claim.text.lower()

        if claim_lower in source_lower:
            return VerificationResult(
                passed=True,
                confidence=1.0,
                risk_level=RiskLevel.LOW,
                details={"status": "verified"},
            )

        # Check word overlap
        claim_words = set(claim_lower.split())
        source_words = set(source_lower.split())
        overlap = len(claim_words & source_words) / len(claim_words) if claim_words else 0

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
            warnings=["Claim not verified against sources"],
        )

    async def verify_response(
        self,
        context: VerificationContext
    ) -> VerificationResult:
        """Verify complete legal response."""
        claims = context.claims or self.extract_claims(context.response)

        if not claims:
            return VerificationResult(
                passed=True,
                confidence=1.0,
                risk_level=RiskLevel.LOW,
                details={"note": "No verifiable legal claims extracted"},
            )

        results = []
        for claim in claims:
            result = await self.verify_claim(claim, context)
            results.append(result)

        # Aggregate results
        passed_count = sum(1 for r in results if r.passed)
        total = len(results)

        # Check for fabricated citations (critical)
        fabricated_citations = sum(
            1 for r in results
            if r.details.get("is_fabricated_citation", False)
        )

        # Determine overall risk
        risk_levels = [r.risk_level for r in results]
        if RiskLevel.CRITICAL in risk_levels or fabricated_citations > 0:
            overall_risk = RiskLevel.CRITICAL
        elif RiskLevel.HIGH in risk_levels:
            overall_risk = RiskLevel.HIGH
        elif RiskLevel.MEDIUM in risk_levels:
            overall_risk = RiskLevel.MEDIUM
        else:
            overall_risk = RiskLevel.LOW

        all_warnings = []
        for r in results:
            all_warnings.extend(r.warnings)

        return VerificationResult(
            passed=passed_count == total and fabricated_citations == 0,
            confidence=passed_count / total if total > 0 else 1.0,
            risk_level=overall_risk,
            details={
                "total_claims": total,
                "verified_claims": passed_count,
                "fabricated_citations": fabricated_citations,
                "claim_results": [
                    {"claim": c.text[:50], "passed": r.passed, "risk": r.risk_level.value}
                    for c, r in zip(claims, results)
                ],
            },
            warnings=all_warnings,
        )

    # =========================================================================
    # Source Attribution Methods
    # =========================================================================

    def get_authoritative_sources(self, claim: Claim) -> List[str]:
        """Get authoritative legal sources for claim verification."""
        claim_type = self._get_legal_claim_type(claim)

        if claim_type in (LegalClaimType.CASE_CITATION, LegalClaimType.LEGAL_HOLDING):
            return AUTHORITATIVE_CASE_SOURCES
        elif claim_type in (LegalClaimType.STATUTE_REFERENCE,):
            return AUTHORITATIVE_STATUTE_SOURCES
        elif claim_type == LegalClaimType.REGULATION_REFERENCE:
            return AUTHORITATIVE_REGULATORY_SOURCES
        else:
            return AUTHORITATIVE_CASE_SOURCES + AUTHORITATIVE_STATUTE_SOURCES

    def validate_source(self, source_id: str) -> bool:
        """Validate that a source is authoritative for legal domain."""
        source_lower = source_id.lower()
        all_sources = (
            AUTHORITATIVE_CASE_SOURCES +
            AUTHORITATIVE_STATUTE_SOURCES +
            AUTHORITATIVE_REGULATORY_SOURCES
        )
        return any(src in source_lower for src in all_sources)

    # =========================================================================
    # Compliance Methods
    # =========================================================================

    def get_compliance_requirements(self) -> List[str]:
        """Get legal compliance requirements."""
        return [
            "MODEL_RULES_PROFESSIONAL_CONDUCT",
            "UNAUTHORIZED_PRACTICE_LAW",
            "ATTORNEY_CLIENT_PRIVILEGE",
            "WORK_PRODUCT_DOCTRINE",
        ]

    async def check_compliance(
        self,
        context: VerificationContext
    ) -> VerificationResult:
        """Check legal ethics compliance."""
        warnings = []
        response = context.response

        # Check for UPL (Unauthorized Practice of Law) indicators
        upl_patterns = [
            r'you should\s+(?:file|sue|bring action)',
            r'I advise you to',
            r'my legal (?:advice|opinion) is',
            r'as your (?:attorney|lawyer)',
        ]

        for pattern in upl_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                warnings.append("Potential UPL risk - ensure appropriate disclaimers")
                break

        # Check for privilege concerns
        if re.search(r'privileged|confidential.*communication|attorney-client', response, re.IGNORECASE):
            warnings.append("Privilege-related content - verify handling")

        return VerificationResult(
            passed=len(warnings) == 0,
            confidence=1.0 if not warnings else 0.5,
            risk_level=RiskLevel.HIGH if warnings else RiskLevel.LOW,
            details={"compliance_check": "legal_ethics"},
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
        """Determine if legal response needs human review."""
        # Always escalate CRITICAL (citation issues) and HIGH risk
        if verification_result.risk_level in (RiskLevel.CRITICAL, RiskLevel.HIGH):
            return True

        # Escalate if confidence is low
        if verification_result.confidence < 0.8:
            return True

        # Escalate if verification failed
        if not verification_result.passed:
            return True

        # Escalate if potentially fabricated citations
        if verification_result.details.get("fabricated_citations", 0) > 0:
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
        """Get explanation for legal escalation."""
        reasons = []

        if verification_result.risk_level == RiskLevel.CRITICAL:
            reasons.append("Critical legal content requires attorney review")

        fabricated = verification_result.details.get("fabricated_citations", 0)
        if fabricated > 0:
            reasons.append(f"URGENT: {fabricated} potentially fabricated citation(s) detected")

        if verification_result.risk_level == RiskLevel.HIGH:
            reasons.append("High-risk legal claims detected")

        if verification_result.confidence < 0.8:
            reasons.append(f"Low confidence score: {verification_result.confidence:.2f}")

        if not verification_result.passed:
            reasons.append("Legal verification checks failed")

        if verification_result.warnings:
            reasons.append(f"Warnings: {', '.join(verification_result.warnings[:3])}")

        return "; ".join(reasons) if reasons else "Manual legal review requested"

    def get_review_level(
        self,
        verification_result: VerificationResult,
        entropy: float = 0.0
    ) -> LegalReviewLevel:
        """
        Assign review level based on risk and uncertainty.

        Legal review levels:
        - BRIEF (30s): Citation spot check
        - STANDARD (2min): Clause/holding review
        - DETAILED (10min): Full legal reasoning review
        """
        # Fabricated citations always need detailed review
        if verification_result.details.get("fabricated_citations", 0) > 0:
            return LegalReviewLevel.DETAILED

        if verification_result.risk_level == RiskLevel.CRITICAL:
            return LegalReviewLevel.DETAILED
        elif verification_result.risk_level == RiskLevel.HIGH:
            return LegalReviewLevel.DETAILED
        elif verification_result.risk_level == RiskLevel.MEDIUM:
            if entropy > 0.4:
                return LegalReviewLevel.STANDARD
            else:
                return LegalReviewLevel.BRIEF
        else:
            return LegalReviewLevel.BRIEF

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_domain_terminology(self) -> Dict[str, str]:
        """Get legal terminology mappings."""
        return {
            # Latin terms
            "stare decisis": "let the decision stand (precedent)",
            "res judicata": "matter already judged",
            "prima facie": "on its face; at first appearance",
            "pro se": "representing oneself",
            "amicus curiae": "friend of the court",
            "habeas corpus": "produce the body",
            "certiorari": "to be made certain",
            "mandamus": "we command",
            "de novo": "anew; from the beginning",
            "sua sponte": "on its own motion",
            "inter alia": "among other things",
            "mens rea": "guilty mind",
            "actus reus": "guilty act",

            # Common abbreviations
            "v.": "versus",
            "et al.": "and others",
            "Id.": "the same (citation)",
            "Ibid.": "in the same place",
            "cf.": "compare",
            "e.g.": "for example",
            "i.e.": "that is",

            # Court abbreviations
            "SCOTUS": "Supreme Court of the United States",
            "SDNY": "Southern District of New York",
            "EDNY": "Eastern District of New York",
            "NDCA": "Northern District of California",
            "9th Cir.": "Ninth Circuit Court of Appeals",
            "2d Cir.": "Second Circuit Court of Appeals",

            # Legal standards
            "BRD": "beyond a reasonable doubt",
            "BFOQ": "bona fide occupational qualification",
            "TRO": "temporary restraining order",
            "MSJ": "motion for summary judgment",
            "MTD": "motion to dismiss",
        }

    def normalize_response(self, response: str) -> str:
        """Normalize legal response text."""
        text = response
        # Standardize citation formats
        text = re.sub(r'v\s+', 'v. ', text)  # Normalize versus
        text = re.sub(r'§\s*(\d)', r'§ \1', text)  # Space after section symbol
        return text
