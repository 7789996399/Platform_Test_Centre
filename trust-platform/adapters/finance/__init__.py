"""
TRUST Finance Adapter

Domain-specific adapter for financial services AI governance including:
- Performance figure verification (SEC Rule 206(4)-1 compliance)
- Fee disclosure validation (Reg BI, Form ADV)
- Risk rating verification
- Suitability assessment review
- Forward-looking statement detection
"""

from .finance_adapter import (
    FinanceAdapter,
    FinanceClaimType,
    FinanceReviewLevel,
    FinanceVerificationResult,
    PerformanceComponents,
    FeeComponents,
    AUTHORITATIVE_MARKET_SOURCES,
    AUTHORITATIVE_REGULATORY_SOURCES,
    AUTHORITATIVE_FUND_SOURCES,
    STANDARD_BENCHMARKS,
    PERFORMANCE_PATTERNS,
    FEE_PATTERNS,
    RISK_PATTERNS,
    SUITABILITY_PATTERNS,
    FORWARD_LOOKING_PATTERNS,
    GUARANTEE_PATTERNS,
)

__all__ = [
    'FinanceAdapter',
    'FinanceClaimType',
    'FinanceReviewLevel',
    'FinanceVerificationResult',
    'PerformanceComponents',
    'FeeComponents',
    'AUTHORITATIVE_MARKET_SOURCES',
    'AUTHORITATIVE_REGULATORY_SOURCES',
    'AUTHORITATIVE_FUND_SOURCES',
    'STANDARD_BENCHMARKS',
    'PERFORMANCE_PATTERNS',
    'FEE_PATTERNS',
    'RISK_PATTERNS',
    'SUITABILITY_PATTERNS',
    'FORWARD_LOOKING_PATTERNS',
    'GUARANTEE_PATTERNS',
]
