"""
TRUST Adapters - Industry-Specific Configuration Modules

Adapters provide domain-specific logic for:
- Risk threshold configuration
- Domain terminology and ontologies
- Regulatory compliance rules
- Custom verification workflows

Available Adapters:
- HealthcareAdapter: Clinical terminology, drug interactions, HIPAA compliance
- LegalAdapter: Case citations, statutes, fabrication detection
- FinanceAdapter: Performance claims, fee disclosures, Reg BI compliance
"""

from .base_adapter import (
    IndustryAdapter,
    AdapterConfig,
    Claim,
    RiskLevel,
    VerificationContext,
    VerificationResult,
)

from .healthcare import (
    HealthcareAdapter,
    HealthcareClaimType,
    HealthcareVerificationResult,
    MedicationComponents,
    ReviewLevel as HealthcareReviewLevel,
    AUTHORITATIVE_DRUG_SOURCES,
    AUTHORITATIVE_CLINICAL_SOURCES,
)

from .legal import (
    LegalAdapter,
    LegalClaimType,
    LegalReviewLevel,
    LegalVerificationResult,
    CaseCitationComponents,
    StatuteCitationComponents,
    AUTHORITATIVE_CASE_SOURCES,
    AUTHORITATIVE_STATUTE_SOURCES,
)

from .finance import (
    FinanceAdapter,
    FinanceClaimType,
    FinanceReviewLevel,
    FinanceVerificationResult,
    PerformanceComponents,
    FeeComponents,
    AUTHORITATIVE_MARKET_SOURCES,
    AUTHORITATIVE_FUND_SOURCES,
)

__all__ = [
    # Base
    'IndustryAdapter',
    'AdapterConfig',
    'Claim',
    'RiskLevel',
    'VerificationContext',
    'VerificationResult',
    # Healthcare
    'HealthcareAdapter',
    'HealthcareClaimType',
    'HealthcareVerificationResult',
    'MedicationComponents',
    'HealthcareReviewLevel',
    'AUTHORITATIVE_DRUG_SOURCES',
    'AUTHORITATIVE_CLINICAL_SOURCES',
    # Legal
    'LegalAdapter',
    'LegalClaimType',
    'LegalReviewLevel',
    'LegalVerificationResult',
    'CaseCitationComponents',
    'StatuteCitationComponents',
    'AUTHORITATIVE_CASE_SOURCES',
    'AUTHORITATIVE_STATUTE_SOURCES',
    # Finance
    'FinanceAdapter',
    'FinanceClaimType',
    'FinanceReviewLevel',
    'FinanceVerificationResult',
    'PerformanceComponents',
    'FeeComponents',
    'AUTHORITATIVE_MARKET_SOURCES',
    'AUTHORITATIVE_FUND_SOURCES',
]
