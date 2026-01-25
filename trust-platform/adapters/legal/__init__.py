"""
TRUST Legal Adapter

Domain-specific adapter for legal AI governance including:
- Case citation verification (critical - fabrication is a known LLM problem)
- Statute reference validation
- Contract clause extraction
- Jurisdiction determination
- Limitation period checking
"""

from .legal_adapter import (
    LegalAdapter,
    LegalClaimType,
    LegalReviewLevel,
    LegalVerificationResult,
    CaseCitationComponents,
    StatuteCitationComponents,
    AUTHORITATIVE_CASE_SOURCES,
    AUTHORITATIVE_STATUTE_SOURCES,
    AUTHORITATIVE_REGULATORY_SOURCES,
    FEDERAL_REPORTERS,
    STATE_REPORTERS,
    ALL_REPORTERS,
    CASE_CITATION_PATTERNS,
    STATUTE_CITATION_PATTERNS,
    CONTRACT_CLAUSE_PATTERNS,
    LIMITATION_PATTERNS,
    JURISDICTION_PATTERNS,
)

__all__ = [
    'LegalAdapter',
    'LegalClaimType',
    'LegalReviewLevel',
    'LegalVerificationResult',
    'CaseCitationComponents',
    'StatuteCitationComponents',
    'AUTHORITATIVE_CASE_SOURCES',
    'AUTHORITATIVE_STATUTE_SOURCES',
    'AUTHORITATIVE_REGULATORY_SOURCES',
    'FEDERAL_REPORTERS',
    'STATE_REPORTERS',
    'ALL_REPORTERS',
    'CASE_CITATION_PATTERNS',
    'STATUTE_CITATION_PATTERNS',
    'CONTRACT_CLAUSE_PATTERNS',
    'LIMITATION_PATTERNS',
    'JURISDICTION_PATTERNS',
]
