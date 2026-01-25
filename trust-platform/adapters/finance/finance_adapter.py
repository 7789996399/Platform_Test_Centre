"""
Finance Industry Adapter
========================

Implements the IndustryAdapter interface for financial services AI governance.

This adapter provides finance-specific logic for:
- Performance figure verification (critical - past performance claims)
- Risk rating validation
- Fee disclosure compliance
- Suitability assessment
- Forward-looking statement detection

Key Risks:
- Misleading performance claims can result in regulatory action (SEC, FINRA)
- Fee disclosure failures violate Reg BI and fiduciary standards
- Forward-looking statements without proper disclaimers are prohibited
- Suitability failures can result in arbitration/litigation

Regulatory Context:
- SEC Rule 206(4)-1 (Investment Adviser Advertising)
- FINRA Rules 2210-2216 (Communications)
- Regulation Best Interest (Reg BI)
- Form ADV Part 2A/2B requirements
"""

import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
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

from ...core_engine.faithfulness import (
    Claim,
    ClaimCategory,
    SourceDocument,
    VerificationResult as CoreVerificationResult,
    VerificationStatus,
)
from ...core_engine.semantic_entropy import EntropyThresholds


# =============================================================================
# FINANCE-SPECIFIC ENUMS
# =============================================================================

class FinanceClaimType(str, Enum):
    """
    Types of financial claims with associated risk levels.

    Performance and fee claims are highest risk due to regulatory scrutiny.
    """
    # Critical risk - high regulatory scrutiny
    PERFORMANCE_FIGURE = "performance_figure"     # Returns, yields, gains
    FEE_DISCLOSURE = "fee_disclosure"             # Fees, costs, expenses
    GUARANTEED_RETURN = "guaranteed_return"       # Any guarantee language

    # High risk - material to investment decisions
    RISK_RATING = "risk_rating"                   # Risk scores, ratings
    SUITABILITY = "suitability"                   # Suitability determinations
    BENCHMARK_COMPARISON = "benchmark_comparison" # Comparison to indices
    FORWARD_LOOKING = "forward_looking"           # Projections, forecasts

    # Medium risk - important disclosures
    ASSET_ALLOCATION = "asset_allocation"         # Portfolio composition
    DIVERSIFICATION = "diversification"           # Diversification claims
    TAX_IMPLICATION = "tax_implication"           # Tax-related statements
    LIQUIDITY = "liquidity"                       # Liquidity statements

    # Lower risk - general information
    MARKET_DATA = "market_data"                   # Current prices, quotes
    ACCOUNT_BALANCE = "account_balance"           # Account values
    TRANSACTION = "transaction"                   # Trade details
    REGULATORY_STATUS = "regulatory_status"       # Registration info

    def to_category(self) -> ClaimCategory:
        """Map finance claim type to generic risk category."""
        critical_risk = {
            self.PERFORMANCE_FIGURE,
            self.FEE_DISCLOSURE,
            self.GUARANTEED_RETURN,
        }
        high_risk = {
            self.RISK_RATING,
            self.SUITABILITY,
            self.BENCHMARK_COMPARISON,
            self.FORWARD_LOOKING,
        }
        low_risk = {
            self.MARKET_DATA,
            self.ACCOUNT_BALANCE,
            self.TRANSACTION,
            self.REGULATORY_STATUS,
        }

        if self in critical_risk:
            return ClaimCategory.HIGH_RISK
        elif self in high_risk:
            return ClaimCategory.HIGH_RISK
        elif self in low_risk:
            return ClaimCategory.LOW_RISK
        else:
            return ClaimCategory.MEDIUM_RISK


class FinanceReviewLevel(str, Enum):
    """Review level for financial responses."""
    BRIEF = "brief"           # 30 seconds - data verification spot check
    STANDARD = "standard"     # 2 minutes - recommendation review
    DETAILED = "detailed"     # 10+ minutes - full suitability review


# =============================================================================
# FINANCE-SPECIFIC DATA CLASSES
# =============================================================================

@dataclass
class PerformanceComponents:
    """Parsed components of a performance claim."""
    return_value: Optional[float] = None
    return_type: Optional[str] = None  # "total", "annualized", "cumulative", "YTD"
    time_period: Optional[str] = None
    asset_name: Optional[str] = None
    benchmark: Optional[str] = None
    net_or_gross: Optional[str] = None  # "net", "gross"
    raw_text: str = ""


@dataclass
class FeeComponents:
    """Parsed components of a fee disclosure."""
    fee_amount: Optional[float] = None
    fee_type: Optional[str] = None  # "expense_ratio", "management_fee", "load", etc.
    fee_basis: Optional[str] = None  # "AUM", "flat", "per_trade"
    frequency: Optional[str] = None  # "annual", "monthly", "per_transaction"
    raw_text: str = ""


@dataclass
class FinanceVerificationResult:
    """Extended verification result with finance-specific fields."""
    claim: Claim
    status: VerificationStatus
    confidence: float
    risk_level: RiskLevel
    review_level: FinanceReviewLevel
    matched_source: Optional[SourceDocument] = None
    matched_value: Optional[float] = None
    variance: Optional[float] = None  # Difference from source value
    explanation: str = ""
    needs_entropy_check: bool = True
    compliance_flags: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# AUTHORITATIVE SOURCES
# =============================================================================

AUTHORITATIVE_MARKET_SOURCES = [
    "bloomberg.com",
    "refinitiv.com",
    "morningstar.com",
    "factset.com",
    "yahoofinance.com",
    "marketwatch.com",
    "wsj.com",
    "cnbc.com",
    "reuters.com",
]

AUTHORITATIVE_REGULATORY_SOURCES = [
    "sec.gov",
    "finra.org",
    "cftc.gov",
    "fdic.gov",
    "occ.gov",
    "federalreserve.gov",
    "nasaa.org",
]

AUTHORITATIVE_FUND_SOURCES = [
    "morningstar.com",
    "fundresearch.fidelity.com",
    "vanguard.com",
    "blackrock.com",
    "schwab.com",
    "etf.com",
]

# Standard benchmarks
STANDARD_BENCHMARKS = [
    "S&P 500",
    "S&P500",
    "SPX",
    "Dow Jones",
    "DJIA",
    "Nasdaq",
    "NASDAQ-100",
    "Russell 2000",
    "Russell 1000",
    "MSCI EAFE",
    "MSCI World",
    "MSCI Emerging Markets",
    "Bloomberg Aggregate",
    "Barclays Aggregate",
    "10-Year Treasury",
    "LIBOR",
    "SOFR",
    "Fed Funds Rate",
]


# =============================================================================
# REGEX PATTERNS FOR CLAIM EXTRACTION
# =============================================================================

# Performance claim patterns
PERFORMANCE_PATTERNS = [
    # Percentage returns: "returned 12.5%", "gained 8%", "up 15%"
    re.compile(
        r'(?:return(?:ed|s)?|gain(?:ed|s)?|up|down|lost?|yield(?:ed|s)?|'
        r'perform(?:ed|ance)?|increas(?:ed?|ing)|decreas(?:ed?|ing))\s*'
        r'(?:of\s+)?(?:approximately\s+)?'
        r'([+-]?\d+(?:\.\d+)?)\s*%',
        re.IGNORECASE
    ),
    # "12.5% return", "8% gain"
    re.compile(
        r'([+-]?\d+(?:\.\d+)?)\s*%\s*'
        r'(?:return|gain|loss|yield|increase|decrease|growth)',
        re.IGNORECASE
    ),
    # Annual/annualized returns
    re.compile(
        r'(?:annual(?:ized)?|yearly|YTD|year-to-date|MTD|QTD|ITD)\s*'
        r'(?:return|performance|gain)?\s*'
        r'(?:of\s+)?([+-]?\d+(?:\.\d+)?)\s*%',
        re.IGNORECASE
    ),
    # "outperformed by X%"
    re.compile(
        r'(?:out|under)perform(?:ed|ing)?\s*(?:by\s+)?([+-]?\d+(?:\.\d+)?)\s*%',
        re.IGNORECASE
    ),
    # Dollar returns: "$1,000 gain"
    re.compile(
        r'\$\s*([\d,]+(?:\.\d{2})?)\s*(?:gain|loss|return|profit)',
        re.IGNORECASE
    ),
]

# Fee disclosure patterns
FEE_PATTERNS = [
    # Expense ratio: "expense ratio of 0.75%"
    re.compile(
        r'expense\s+ratio\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*%',
        re.IGNORECASE
    ),
    # Management fee
    re.compile(
        r'(?:management|advisory|admin(?:istrative)?)\s+fee\s*'
        r'(?:of\s+)?(\d+(?:\.\d+)?)\s*%',
        re.IGNORECASE
    ),
    # Basis points: "25 bps", "25 basis points"
    re.compile(
        r'(\d+)\s*(?:bps|basis\s+points?)',
        re.IGNORECASE
    ),
    # Flat fees
    re.compile(
        r'(?:annual|monthly|quarterly)?\s*fee\s+(?:of\s+)?\$\s*([\d,]+(?:\.\d{2})?)',
        re.IGNORECASE
    ),
    # Load fees
    re.compile(
        r'(?:front[- ]?end|back[- ]?end|deferred)?\s*load\s*'
        r'(?:of\s+)?(\d+(?:\.\d+)?)\s*%',
        re.IGNORECASE
    ),
    # Trading commissions
    re.compile(
        r'commission\s*(?:of\s+)?\$\s*([\d,]+(?:\.\d{2})?)',
        re.IGNORECASE
    ),
]

# Risk rating patterns
RISK_PATTERNS = [
    # Numeric risk ratings
    re.compile(
        r'risk\s+(?:score|rating|level)\s*(?:of\s+)?(\d+)(?:\s*/\s*(\d+))?',
        re.IGNORECASE
    ),
    # Descriptive risk levels
    re.compile(
        r'(?:low|moderate|medium|high|aggressive|conservative)\s+risk',
        re.IGNORECASE
    ),
    # Morningstar-style ratings
    re.compile(
        r'(\d)\s*(?:star|stars)\s+(?:rating|rated)',
        re.IGNORECASE
    ),
    # Volatility measures
    re.compile(
        r'(?:standard deviation|volatility|beta)\s+(?:of\s+)?(\d+(?:\.\d+)?)',
        re.IGNORECASE
    ),
    # Sharpe ratio
    re.compile(
        r'sharpe\s+ratio\s+(?:of\s+)?([+-]?\d+(?:\.\d+)?)',
        re.IGNORECASE
    ),
]

# Suitability patterns
SUITABILITY_PATTERNS = [
    re.compile(
        r'(?:suitable|appropriate|recommended)\s+for\s+'
        r'(?:you|your|investors?|clients?)',
        re.IGNORECASE
    ),
    re.compile(
        r'(?:risk\s+tolerance|investment\s+objective|time\s+horizon)\s+'
        r'(?:is|of)\s+([a-z]+)',
        re.IGNORECASE
    ),
    re.compile(
        r'(?:aggressive|moderate|conservative)\s+(?:investor|portfolio|allocation)',
        re.IGNORECASE
    ),
    re.compile(
        r'(?:growth|income|balanced|preservation)\s+(?:objective|strategy)',
        re.IGNORECASE
    ),
]

# Forward-looking patterns
FORWARD_LOOKING_PATTERNS = [
    # Explicit projections
    re.compile(
        r'(?:expect(?:ed|s)?|project(?:ed|ion|s)?|forecast(?:ed|s)?|'
        r'anticipat(?:ed?|ing)|estimat(?:ed?|ing)|predict(?:ed|ion|s)?)\s+'
        r'(?:to\s+)?(?:return|grow|increase|decrease|yield|gain)',
        re.IGNORECASE
    ),
    # Future returns
    re.compile(
        r'(?:will|should|could|may|might)\s+'
        r'(?:return|grow|increase|decrease|yield|gain|perform)',
        re.IGNORECASE
    ),
    # Target prices/returns
    re.compile(
        r'(?:target|price\s+target|projected)\s+'
        r'(?:of\s+)?\$?\s*([\d,]+(?:\.\d+)?)',
        re.IGNORECASE
    ),
    # Time-based forward statements
    re.compile(
        r'(?:in|over|within)\s+(?:the\s+)?(?:next|coming)\s+'
        r'(?:\d+\s+)?(?:year|month|quarter|week)',
        re.IGNORECASE
    ),
    # Outlook language
    re.compile(
        r'(?:outlook|view|prospects?)\s+(?:is|are|remains?)\s+'
        r'(?:positive|negative|neutral|bullish|bearish)',
        re.IGNORECASE
    ),
]

# Guaranteed return patterns (PROHIBITED without qualification)
GUARANTEE_PATTERNS = [
    re.compile(
        r'(?:guarantee(?:d|s)?|certain|assured|promise(?:d|s)?)\s+'
        r'(?:return|yield|gain|profit|income)',
        re.IGNORECASE
    ),
    re.compile(
        r'(?:risk[- ]?free|no\s+risk|safe)\s+'
        r'(?:return|yield|investment|income)',
        re.IGNORECASE
    ),
    re.compile(
        r'(?:can\'?t|cannot|won\'?t)\s+lose',
        re.IGNORECASE
    ),
]


# =============================================================================
# FINANCE ADAPTER IMPLEMENTATION
# =============================================================================

class FinanceAdapter(IndustryAdapter):
    """
    Finance-specific adapter for TRUST verification.

    Provides domain-specific logic for:
    - Performance figure verification against authoritative data
    - Fee disclosure validation
    - Risk rating verification
    - Suitability assessment review
    - Forward-looking statement detection

    Key Risk Mitigation:
    This adapter uses VERY strict thresholds for performance claims because:
    - SEC Rule 206(4)-1 prohibits misleading performance advertising
    - FINRA requires "fair and balanced" communications
    - Regulation Best Interest requires accurate fee disclosure
    - Forward-looking statements require specific disclaimers

    Example:
        >>> adapter = FinanceAdapter()
        >>> await adapter.initialize()
        >>>
        >>> claims = adapter.extract_claims("The fund returned 12.5% last year...")
        >>> for claim in claims:
        ...     result = await adapter.verify_claim(claim, context)
        ...     if not result.passed:
        ...         print(f"WARNING: Unverified performance claim: {claim.text}")
    """

    def __init__(self):
        config = AdapterConfig(
            industry_name="finance",
            version="1.0.0",
            entropy_threshold=0.15,  # Strict - financial figures must be consistent
            faithfulness_threshold=0.95,  # Near-perfect matching for numerical claims
            ensemble_agreement_threshold=0.9,  # Strong consensus required
            max_risk_level_auto_approve=RiskLevel.MINIMAL,
            custom_settings={
                "performance_verification_required": True,
                "fee_disclosure_check": True,
                "forward_looking_detection": True,
                "suitability_review": True,
                "numerical_tolerance": 0.01,  # 1% tolerance for numerical matching
            }
        )
        super().__init__(config)
        self._claim_counter = 0

    # =========================================================================
    # Lifecycle Methods
    # =========================================================================

    async def initialize(self) -> None:
        """Initialize finance-specific resources."""
        # In production, this would:
        # - Connect to market data feeds (Bloomberg, Refinitiv)
        # - Load portfolio management system APIs
        # - Initialize regulatory database connections
        # - Load fee schedules and product databases
        self._initialized = True

    async def shutdown(self) -> None:
        """Clean up finance adapter resources."""
        self._initialized = False

    # =========================================================================
    # Risk Assessment Methods
    # =========================================================================

    def classify_risk(self, context: VerificationContext) -> RiskLevel:
        """
        Classify risk level for financial responses.

        Risk Classification:
        - CRITICAL: Performance figures, fee disclosures, guarantees
        - HIGH: Risk ratings, suitability, forward-looking
        - MEDIUM: Asset allocation, tax implications
        - LOW: Market data quotes, account balances
        """
        claims = self.extract_claims(context.response)

        if not claims:
            return RiskLevel.LOW

        # Guarantee claims are always critical
        for claim in claims:
            if self._get_finance_claim_type(claim) == FinanceClaimType.GUARANTEED_RETURN:
                return RiskLevel.CRITICAL

        critical_types = {
            FinanceClaimType.PERFORMANCE_FIGURE,
            FinanceClaimType.FEE_DISCLOSURE,
            FinanceClaimType.GUARANTEED_RETURN,
        }

        high_types = {
            FinanceClaimType.RISK_RATING,
            FinanceClaimType.SUITABILITY,
            FinanceClaimType.BENCHMARK_COMPARISON,
            FinanceClaimType.FORWARD_LOOKING,
        }

        for claim in claims:
            claim_type = self._get_finance_claim_type(claim)
            if claim_type in critical_types:
                return RiskLevel.CRITICAL

        for claim in claims:
            claim_type = self._get_finance_claim_type(claim)
            if claim_type in high_types:
                return RiskLevel.HIGH

        medium_types = {
            FinanceClaimType.ASSET_ALLOCATION,
            FinanceClaimType.DIVERSIFICATION,
            FinanceClaimType.TAX_IMPLICATION,
            FinanceClaimType.LIQUIDITY,
        }

        for claim in claims:
            claim_type = self._get_finance_claim_type(claim)
            if claim_type in medium_types:
                return RiskLevel.MEDIUM

        return RiskLevel.LOW

    def get_risk_thresholds(self) -> Dict[str, float]:
        """Get finance-specific risk thresholds."""
        return {
            "entropy": self._config.entropy_threshold,
            "faithfulness": self._config.faithfulness_threshold,
            "ensemble_agreement": self._config.ensemble_agreement_threshold,
            "numerical_tolerance": self._config.custom_settings["numerical_tolerance"],
        }

    def get_entropy_thresholds(self) -> EntropyThresholds:
        """Get entropy thresholds configured for finance domain."""
        return EntropyThresholds(
            low_medium_boundary=0.15,  # Very strict - numbers must be consistent
            medium_high_boundary=0.4,
        )

    # =========================================================================
    # Claim Extraction Methods
    # =========================================================================

    def extract_claims(self, response: str) -> List[Claim]:
        """
        Extract financial claims from an AI response.

        Extracts:
        - Performance figures
        - Fee disclosures
        - Risk ratings
        - Suitability statements
        - Forward-looking statements
        """
        all_claims: List[Claim] = []

        all_claims.extend(self._extract_performance_claims(response))
        all_claims.extend(self._extract_fee_claims(response))
        all_claims.extend(self._extract_risk_claims(response))
        all_claims.extend(self._extract_suitability_claims(response))
        all_claims.extend(self._extract_forward_looking_claims(response))
        all_claims.extend(self._extract_guarantee_claims(response))

        # Assign unique IDs
        for i, claim in enumerate(all_claims):
            claim.id = f"finance_{i:03d}_{claim.claim_type}"

        return all_claims

    def _extract_performance_claims(self, text: str, section: str = "general") -> List[Claim]:
        """Extract performance figure claims from text."""
        claims = []
        seen_claims = set()

        for pattern in PERFORMANCE_PATTERNS:
            matches = pattern.finditer(text)
            for match in matches:
                claim_text = match.group(0).strip()

                if claim_text.lower() in seen_claims:
                    continue
                seen_claims.add(claim_text.lower())

                # Parse performance components
                components = self._parse_performance_claim(claim_text)

                claims.append(Claim(
                    id="",
                    text=claim_text,
                    category=ClaimCategory.HIGH_RISK,
                    claim_type=FinanceClaimType.PERFORMANCE_FIGURE.value,
                    source_span=(match.start(), match.end()),
                    section=section,
                    metadata={
                        "components": {
                            "return_value": components.return_value,
                            "return_type": components.return_type,
                            "time_period": components.time_period,
                            "net_or_gross": components.net_or_gross,
                        },
                        "is_numerical": True,
                        "regulatory_risk": "high",
                    }
                ))

        return claims

    def _extract_fee_claims(self, text: str, section: str = "general") -> List[Claim]:
        """Extract fee disclosure claims from text."""
        claims = []
        seen_claims = set()

        for pattern in FEE_PATTERNS:
            matches = pattern.finditer(text)
            for match in matches:
                claim_text = match.group(0).strip()

                if claim_text.lower() in seen_claims:
                    continue
                seen_claims.add(claim_text.lower())

                components = self._parse_fee_claim(claim_text)

                claims.append(Claim(
                    id="",
                    text=claim_text,
                    category=ClaimCategory.HIGH_RISK,
                    claim_type=FinanceClaimType.FEE_DISCLOSURE.value,
                    source_span=(match.start(), match.end()),
                    section=section,
                    metadata={
                        "components": {
                            "fee_amount": components.fee_amount,
                            "fee_type": components.fee_type,
                            "fee_basis": components.fee_basis,
                        },
                        "is_numerical": True,
                        "regulatory_risk": "high",
                    }
                ))

        return claims

    def _extract_risk_claims(self, text: str, section: str = "general") -> List[Claim]:
        """Extract risk rating claims from text."""
        claims = []

        for pattern in RISK_PATTERNS:
            matches = pattern.finditer(text)
            for match in matches:
                start = max(0, match.start() - 30)
                end = min(len(text), match.end() + 50)
                context_text = text[start:end].strip()

                claims.append(Claim(
                    id="",
                    text=context_text,
                    category=ClaimCategory.HIGH_RISK,
                    claim_type=FinanceClaimType.RISK_RATING.value,
                    source_span=(match.start(), match.end()),
                    section=section,
                    metadata={"risk_indicator": match.group(0)}
                ))

        return claims

    def _extract_suitability_claims(self, text: str, section: str = "general") -> List[Claim]:
        """Extract suitability claims from text."""
        claims = []

        for pattern in SUITABILITY_PATTERNS:
            matches = pattern.finditer(text)
            for match in matches:
                start = max(0, match.start() - 30)
                end = min(len(text), match.end() + 80)
                context_text = text[start:end].strip()

                claims.append(Claim(
                    id="",
                    text=context_text,
                    category=ClaimCategory.HIGH_RISK,
                    claim_type=FinanceClaimType.SUITABILITY.value,
                    source_span=(match.start(), match.end()),
                    section=section,
                ))

        return claims

    def _extract_forward_looking_claims(self, text: str, section: str = "general") -> List[Claim]:
        """Extract forward-looking statements from text."""
        claims = []

        for pattern in FORWARD_LOOKING_PATTERNS:
            matches = pattern.finditer(text)
            for match in matches:
                start = max(0, match.start() - 20)
                end = min(len(text), match.end() + 60)
                context_text = text[start:end].strip()

                claims.append(Claim(
                    id="",
                    text=context_text,
                    category=ClaimCategory.HIGH_RISK,
                    claim_type=FinanceClaimType.FORWARD_LOOKING.value,
                    source_span=(match.start(), match.end()),
                    section=section,
                    metadata={"requires_disclaimer": True}
                ))

        return claims

    def _extract_guarantee_claims(self, text: str, section: str = "general") -> List[Claim]:
        """Extract guarantee claims (CRITICAL - generally prohibited)."""
        claims = []

        for pattern in GUARANTEE_PATTERNS:
            matches = pattern.finditer(text)
            for match in matches:
                start = max(0, match.start() - 20)
                end = min(len(text), match.end() + 40)
                context_text = text[start:end].strip()

                claims.append(Claim(
                    id="",
                    text=context_text,
                    category=ClaimCategory.HIGH_RISK,
                    claim_type=FinanceClaimType.GUARANTEED_RETURN.value,
                    source_span=(match.start(), match.end()),
                    section=section,
                    metadata={
                        "compliance_flag": "PROHIBITED_GUARANTEE",
                        "requires_immediate_review": True,
                    }
                ))

        return claims

    def _parse_performance_claim(self, claim_text: str) -> PerformanceComponents:
        """Parse a performance claim into components."""
        components = PerformanceComponents(raw_text=claim_text)

        # Extract percentage value
        pct_match = re.search(r'([+-]?\d+(?:\.\d+)?)\s*%', claim_text)
        if pct_match:
            try:
                components.return_value = float(pct_match.group(1))
            except ValueError:
                pass

        # Determine return type
        if re.search(r'annual(?:ized)?', claim_text, re.IGNORECASE):
            components.return_type = "annualized"
        elif re.search(r'YTD|year-to-date', claim_text, re.IGNORECASE):
            components.return_type = "YTD"
        elif re.search(r'cumulative|total', claim_text, re.IGNORECASE):
            components.return_type = "cumulative"
        else:
            components.return_type = "unknown"

        # Determine net vs gross
        if re.search(r'\bnet\b', claim_text, re.IGNORECASE):
            components.net_or_gross = "net"
        elif re.search(r'\bgross\b', claim_text, re.IGNORECASE):
            components.net_or_gross = "gross"

        return components

    def _parse_fee_claim(self, claim_text: str) -> FeeComponents:
        """Parse a fee claim into components."""
        components = FeeComponents(raw_text=claim_text)

        # Extract percentage
        pct_match = re.search(r'(\d+(?:\.\d+)?)\s*%', claim_text)
        if pct_match:
            try:
                components.fee_amount = float(pct_match.group(1))
            except ValueError:
                pass

        # Extract basis points
        bps_match = re.search(r'(\d+)\s*(?:bps|basis\s+points?)', claim_text, re.IGNORECASE)
        if bps_match:
            try:
                components.fee_amount = float(bps_match.group(1)) / 100  # Convert to %
            except ValueError:
                pass

        # Determine fee type
        if re.search(r'expense\s+ratio', claim_text, re.IGNORECASE):
            components.fee_type = "expense_ratio"
        elif re.search(r'management', claim_text, re.IGNORECASE):
            components.fee_type = "management_fee"
        elif re.search(r'load', claim_text, re.IGNORECASE):
            components.fee_type = "load"
        elif re.search(r'commission', claim_text, re.IGNORECASE):
            components.fee_type = "commission"

        return components

    def classify_claim(self, claim: Claim) -> str:
        """Classify a claim into a finance-specific category."""
        return claim.claim_type

    def _get_finance_claim_type(self, claim: Claim) -> FinanceClaimType:
        """Get the FinanceClaimType enum for a claim."""
        try:
            return FinanceClaimType(claim.claim_type)
        except ValueError:
            return FinanceClaimType.MARKET_DATA

    # =========================================================================
    # Verification Methods
    # =========================================================================

    async def verify_claim(
        self,
        claim: Claim,
        context: VerificationContext
    ) -> VerificationResult:
        """
        Verify a financial claim against authoritative sources.

        Routes to appropriate verification method based on claim type.
        """
        claim_type = self._get_finance_claim_type(claim)

        # Get source data from context
        source_data = context.session_metadata.get("source_data", {})
        source_text = context.session_metadata.get("source_document", "")
        if not source_text and context.source_documents:
            source_text = " ".join(
                str(doc.get("content", ""))
                for doc in context.source_documents
            )

        # Route to appropriate verifier
        if claim_type == FinanceClaimType.PERFORMANCE_FIGURE:
            return await self._verify_performance_claim(claim, source_data, source_text, context)
        elif claim_type == FinanceClaimType.FEE_DISCLOSURE:
            return await self._verify_fee_claim(claim, source_data, source_text, context)
        elif claim_type == FinanceClaimType.RISK_RATING:
            return await self._verify_risk_claim(claim, source_text)
        elif claim_type == FinanceClaimType.SUITABILITY:
            return await self._verify_suitability_claim(claim, context)
        elif claim_type == FinanceClaimType.FORWARD_LOOKING:
            return await self._verify_forward_looking_claim(claim, context)
        elif claim_type == FinanceClaimType.GUARANTEED_RETURN:
            return await self._verify_guarantee_claim(claim, context)
        else:
            return await self._verify_generic_claim(claim, source_text)

    async def _verify_performance_claim(
        self,
        claim: Claim,
        source_data: Dict[str, Any],
        source_text: str,
        context: VerificationContext
    ) -> VerificationResult:
        """
        Verify a performance claim against authoritative data.

        This is CRITICAL - misleading performance claims can result in:
        - SEC enforcement actions
        - FINRA sanctions
        - Client arbitration
        """
        components = claim.metadata.get("components", {})
        claimed_return = components.get("return_value")
        return_type = components.get("return_type", "unknown")

        warnings = []
        compliance_flags = []

        # Check if return appears in source data
        if claimed_return is not None:
            tolerance = self._config.custom_settings.get("numerical_tolerance", 0.01)

            # Look for matching value in source data
            source_returns = source_data.get("returns", {})
            for period, value in source_returns.items():
                if isinstance(value, (int, float)):
                    if abs(value - claimed_return) <= abs(claimed_return * tolerance):
                        return VerificationResult(
                            passed=True,
                            confidence=0.95,
                            risk_level=RiskLevel.LOW,
                            details={
                                "status": "verified",
                                "claim_type": "performance",
                                "claimed_value": claimed_return,
                                "source_value": value,
                                "period": period,
                            },
                        )

            # Look for value in text
            source_lower = source_text.lower()
            return_str = f"{claimed_return}"
            if return_str in source_text or f"{claimed_return}%" in source_text:
                return VerificationResult(
                    passed=True,
                    confidence=0.85,
                    risk_level=RiskLevel.LOW,
                    details={
                        "status": "verified_text",
                        "claim_type": "performance",
                        "claimed_value": claimed_return,
                    },
                    warnings=["Verified by text match - confirm data source"],
                )

        # Performance claim not found - HIGH RISK
        if return_type == "unknown":
            warnings.append("Return type not specified (annualized vs. cumulative)")

        if components.get("net_or_gross") is None:
            warnings.append("Net/gross returns not specified - regulatory risk")
            compliance_flags.append("NET_GROSS_DISCLOSURE")

        return VerificationResult(
            passed=False,
            confidence=0.0,
            risk_level=RiskLevel.CRITICAL,
            details={
                "status": "not_found",
                "claim_type": "performance",
                "claimed_value": claimed_return,
                "compliance_flags": compliance_flags,
            },
            warnings=warnings or ["Performance claim NOT verified - REQUIRES DATA SOURCE"],
        )

    async def _verify_fee_claim(
        self,
        claim: Claim,
        source_data: Dict[str, Any],
        source_text: str,
        context: VerificationContext
    ) -> VerificationResult:
        """Verify a fee disclosure against source documents."""
        components = claim.metadata.get("components", {})
        claimed_fee = components.get("fee_amount")
        fee_type = components.get("fee_type", "unknown")

        if claimed_fee is not None:
            tolerance = self._config.custom_settings.get("numerical_tolerance", 0.01)

            # Look in source data
            source_fees = source_data.get("fees", {})
            for f_type, value in source_fees.items():
                if isinstance(value, (int, float)):
                    if abs(value - claimed_fee) <= abs(claimed_fee * tolerance):
                        return VerificationResult(
                            passed=True,
                            confidence=0.95,
                            risk_level=RiskLevel.LOW,
                            details={
                                "status": "verified",
                                "claim_type": "fee",
                                "claimed_value": claimed_fee,
                                "source_value": value,
                                "fee_type": f_type,
                            },
                        )

            # Text match
            fee_str = f"{claimed_fee}"
            if fee_str in source_text or f"{claimed_fee}%" in source_text:
                return VerificationResult(
                    passed=True,
                    confidence=0.8,
                    risk_level=RiskLevel.LOW,
                    details={
                        "status": "verified_text",
                        "claim_type": "fee",
                    },
                )

        return VerificationResult(
            passed=False,
            confidence=0.0,
            risk_level=RiskLevel.HIGH,
            details={
                "status": "not_found",
                "claim_type": "fee",
            },
            warnings=["Fee disclosure not verified - check prospectus/ADV"],
        )

    async def _verify_risk_claim(
        self,
        claim: Claim,
        source_text: str
    ) -> VerificationResult:
        """Verify risk rating claims."""
        source_lower = source_text.lower()
        claim_lower = claim.text.lower()

        # Direct text match
        if claim_lower in source_lower:
            return VerificationResult(
                passed=True,
                confidence=0.9,
                risk_level=RiskLevel.LOW,
                details={"status": "verified", "type": "risk_rating"},
            )

        # Check for risk indicator in source
        risk_indicator = claim.metadata.get("risk_indicator", "").lower()
        if risk_indicator and risk_indicator in source_lower:
            return VerificationResult(
                passed=True,
                confidence=0.7,
                risk_level=RiskLevel.MEDIUM,
                details={"status": "partial", "type": "risk_rating"},
            )

        return VerificationResult(
            passed=False,
            confidence=0.0,
            risk_level=RiskLevel.HIGH,
            details={"status": "not_found", "type": "risk_rating"},
            warnings=["Risk rating not verified - requires review"],
        )

    async def _verify_suitability_claim(
        self,
        claim: Claim,
        context: VerificationContext
    ) -> VerificationResult:
        """
        Verify suitability claims.

        Suitability claims ALWAYS require human review per Reg BI.
        """
        return VerificationResult(
            passed=False,
            confidence=0.5,
            risk_level=RiskLevel.HIGH,
            details={
                "status": "needs_review",
                "type": "suitability",
                "regulatory_requirement": "Reg BI",
            },
            warnings=["Suitability determination requires advisor review per Reg BI"],
        )

    async def _verify_forward_looking_claim(
        self,
        claim: Claim,
        context: VerificationContext
    ) -> VerificationResult:
        """
        Verify forward-looking statements.

        Forward-looking statements require specific disclaimers.
        """
        response = context.response

        # Check for required disclaimers
        disclaimer_patterns = [
            r'past performance.*(?:not|no).*guarantee',
            r'forward.looking.*subject to',
            r'(?:may|could|might).*not.*achieve',
            r'no assurance',
            r'projections.*(?:not|no).*guarantee',
        ]

        has_disclaimer = any(
            re.search(pattern, response, re.IGNORECASE)
            for pattern in disclaimer_patterns
        )

        if has_disclaimer:
            return VerificationResult(
                passed=True,
                confidence=0.7,
                risk_level=RiskLevel.MEDIUM,
                details={
                    "status": "compliant",
                    "type": "forward_looking",
                    "disclaimer_present": True,
                },
                warnings=["Verify disclaimer adequacy for regulatory compliance"],
            )

        return VerificationResult(
            passed=False,
            confidence=0.0,
            risk_level=RiskLevel.HIGH,
            details={
                "status": "missing_disclaimer",
                "type": "forward_looking",
            },
            warnings=["Forward-looking statement WITHOUT required disclaimer - ADD DISCLAIMER"],
        )

    async def _verify_guarantee_claim(
        self,
        claim: Claim,
        context: VerificationContext
    ) -> VerificationResult:
        """
        Verify guarantee claims.

        Guarantee claims are GENERALLY PROHIBITED and require immediate review.
        """
        return VerificationResult(
            passed=False,
            confidence=0.0,
            risk_level=RiskLevel.CRITICAL,
            details={
                "status": "prohibited",
                "type": "guarantee",
                "compliance_flag": "PROHIBITED_GUARANTEE",
            },
            warnings=[
                "CRITICAL: Guarantee language detected - PROHIBITED under SEC/FINRA rules",
                "Remove or qualify the guarantee claim immediately",
            ],
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
        """Verify complete financial response."""
        claims = context.claims or self.extract_claims(context.response)

        if not claims:
            return VerificationResult(
                passed=True,
                confidence=1.0,
                risk_level=RiskLevel.LOW,
                details={"note": "No verifiable financial claims extracted"},
            )

        results = []
        for claim in claims:
            result = await self.verify_claim(claim, context)
            results.append(result)

        # Aggregate results
        passed_count = sum(1 for r in results if r.passed)
        total = len(results)

        # Check for critical compliance issues
        critical_compliance = sum(
            1 for r in results
            if r.details.get("compliance_flag") == "PROHIBITED_GUARANTEE"
        )

        # Check for missing disclaimers
        missing_disclaimers = sum(
            1 for r in results
            if r.details.get("status") == "missing_disclaimer"
        )

        # Determine overall risk
        risk_levels = [r.risk_level for r in results]
        if RiskLevel.CRITICAL in risk_levels or critical_compliance > 0:
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
            passed=passed_count == total and critical_compliance == 0,
            confidence=passed_count / total if total > 0 else 1.0,
            risk_level=overall_risk,
            details={
                "total_claims": total,
                "verified_claims": passed_count,
                "critical_compliance_issues": critical_compliance,
                "missing_disclaimers": missing_disclaimers,
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
        """Get authoritative financial sources for claim verification."""
        claim_type = self._get_finance_claim_type(claim)

        if claim_type in (
            FinanceClaimType.PERFORMANCE_FIGURE,
            FinanceClaimType.BENCHMARK_COMPARISON,
            FinanceClaimType.MARKET_DATA,
        ):
            return AUTHORITATIVE_MARKET_SOURCES
        elif claim_type in (
            FinanceClaimType.FEE_DISCLOSURE,
            FinanceClaimType.RISK_RATING,
        ):
            return AUTHORITATIVE_FUND_SOURCES
        elif claim_type in (
            FinanceClaimType.REGULATORY_STATUS,
        ):
            return AUTHORITATIVE_REGULATORY_SOURCES
        else:
            return AUTHORITATIVE_MARKET_SOURCES + AUTHORITATIVE_FUND_SOURCES

    def validate_source(self, source_id: str) -> bool:
        """Validate that a source is authoritative for finance domain."""
        source_lower = source_id.lower()
        all_sources = (
            AUTHORITATIVE_MARKET_SOURCES +
            AUTHORITATIVE_FUND_SOURCES +
            AUTHORITATIVE_REGULATORY_SOURCES
        )
        return any(src in source_lower for src in all_sources)

    # =========================================================================
    # Compliance Methods
    # =========================================================================

    def get_compliance_requirements(self) -> List[str]:
        """Get financial compliance requirements."""
        return [
            "SEC_RULE_206_4_1",      # Investment Adviser Advertising
            "FINRA_RULE_2210",        # Communications with the Public
            "FINRA_RULE_2111",        # Suitability
            "REGULATION_BEST_INTEREST",
            "FORM_ADV_PART_2",
            "FORM_CRS",
        ]

    async def check_compliance(
        self,
        context: VerificationContext
    ) -> VerificationResult:
        """Check financial regulatory compliance."""
        warnings = []
        compliance_flags = []
        response = context.response

        # Check for prohibited guarantee language
        for pattern in GUARANTEE_PATTERNS:
            if pattern.search(response):
                warnings.append("CRITICAL: Prohibited guarantee language detected")
                compliance_flags.append("PROHIBITED_GUARANTEE")
                break

        # Check for forward-looking statements without disclaimers
        has_forward_looking = any(p.search(response) for p in FORWARD_LOOKING_PATTERNS)
        has_disclaimer = re.search(
            r'past performance.*(?:not|no).*guarantee',
            response,
            re.IGNORECASE
        )

        if has_forward_looking and not has_disclaimer:
            warnings.append("Forward-looking statement without required disclaimer")
            compliance_flags.append("MISSING_DISCLAIMER")

        # Check for cherry-picking indicators
        if re.search(r'(?:best|top|highest)\s+(?:performing|return)', response, re.IGNORECASE):
            if not re.search(r'(?:as of|through|ending)', response, re.IGNORECASE):
                warnings.append("Performance highlight may need date/period context")

        return VerificationResult(
            passed=len(compliance_flags) == 0,
            confidence=1.0 if not compliance_flags else 0.0,
            risk_level=RiskLevel.CRITICAL if compliance_flags else RiskLevel.LOW,
            details={
                "compliance_check": "financial_regulations",
                "compliance_flags": compliance_flags,
            },
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
        """Determine if financial response needs compliance review."""
        # Always escalate CRITICAL
        if verification_result.risk_level == RiskLevel.CRITICAL:
            return True

        # Escalate compliance flags
        if verification_result.details.get("compliance_flags"):
            return True

        # Escalate unverified performance claims
        if verification_result.details.get("claim_type") == "performance":
            if not verification_result.passed:
                return True

        # Escalate suitability determinations
        if verification_result.details.get("type") == "suitability":
            return True

        # Escalate if confidence is low
        if verification_result.confidence < 0.8:
            return True

        # Escalate warnings
        if verification_result.warnings:
            return True

        return False

    def get_escalation_reason(
        self,
        verification_result: VerificationResult,
        context: VerificationContext
    ) -> str:
        """Get explanation for financial escalation."""
        reasons = []

        if verification_result.risk_level == RiskLevel.CRITICAL:
            reasons.append("Critical compliance issue requires immediate review")

        flags = verification_result.details.get("compliance_flags", [])
        if "PROHIBITED_GUARANTEE" in flags:
            reasons.append("URGENT: Prohibited guarantee language detected")
        if "MISSING_DISCLAIMER" in flags:
            reasons.append("Required disclaimer missing for forward-looking statement")

        if verification_result.details.get("claim_type") == "performance":
            if not verification_result.passed:
                reasons.append("Unverified performance claim - potential misleading advertising")

        if verification_result.details.get("type") == "suitability":
            reasons.append("Suitability determination requires advisor review per Reg BI")

        if verification_result.confidence < 0.8:
            reasons.append(f"Low confidence score: {verification_result.confidence:.2f}")

        if verification_result.warnings:
            reasons.append(f"Warnings: {', '.join(verification_result.warnings[:2])}")

        return "; ".join(reasons) if reasons else "Manual compliance review requested"

    def get_review_level(
        self,
        verification_result: VerificationResult,
        entropy: float = 0.0
    ) -> FinanceReviewLevel:
        """
        Assign review level based on risk and uncertainty.

        Finance review levels:
        - BRIEF (30s): Data verification spot check
        - STANDARD (2min): Recommendation review
        - DETAILED (10min): Full suitability review
        """
        # Compliance issues always need detailed review
        if verification_result.details.get("compliance_flags"):
            return FinanceReviewLevel.DETAILED

        if verification_result.risk_level == RiskLevel.CRITICAL:
            return FinanceReviewLevel.DETAILED
        elif verification_result.risk_level == RiskLevel.HIGH:
            return FinanceReviewLevel.DETAILED
        elif verification_result.risk_level == RiskLevel.MEDIUM:
            if entropy > 0.4:
                return FinanceReviewLevel.STANDARD
            else:
                return FinanceReviewLevel.BRIEF
        else:
            return FinanceReviewLevel.BRIEF

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_domain_terminology(self) -> Dict[str, str]:
        """Get financial terminology mappings."""
        return {
            # Return types
            "NAV": "Net Asset Value",
            "AUM": "Assets Under Management",
            "CAGR": "Compound Annual Growth Rate",
            "IRR": "Internal Rate of Return",
            "TWR": "Time-Weighted Return",
            "MWR": "Money-Weighted Return",
            "YTD": "Year-to-Date",
            "MTD": "Month-to-Date",
            "QTD": "Quarter-to-Date",
            "ITD": "Inception-to-Date",

            # Risk metrics
            "alpha": "Excess return vs benchmark",
            "beta": "Market sensitivity",
            "Sharpe": "Risk-adjusted return",
            "Sortino": "Downside risk-adjusted return",
            "VaR": "Value at Risk",
            "CVaR": "Conditional Value at Risk",

            # Fee types
            "bps": "Basis points (0.01%)",
            "MER": "Management Expense Ratio",
            "TER": "Total Expense Ratio",
            "12b-1": "Marketing/distribution fee",

            # Regulatory
            "RIA": "Registered Investment Adviser",
            "IAR": "Investment Adviser Representative",
            "BD": "Broker-Dealer",
            "RR": "Registered Representative",
            "CRD": "Central Registration Depository",
            "ADV": "Form ADV (adviser registration)",
            "Reg BI": "Regulation Best Interest",

            # Account types
            "IRA": "Individual Retirement Account",
            "401(k)": "Employer-sponsored retirement plan",
            "403(b)": "Non-profit/education retirement plan",
            "SEP": "Simplified Employee Pension",
            "SIMPLE": "Savings Incentive Match Plan for Employees",
            "HSA": "Health Savings Account",
            "529": "Education savings plan",

            # Asset classes
            "equity": "Stocks/ownership securities",
            "fixed income": "Bonds/debt securities",
            "alternatives": "Non-traditional investments",
            "REIT": "Real Estate Investment Trust",
            "ETF": "Exchange-Traded Fund",
            "MF": "Mutual Fund",
        }

    def normalize_response(self, response: str) -> str:
        """Normalize financial response text."""
        text = response
        # Standardize percentage formats
        text = re.sub(r'(\d+)\s*percent', r'\1%', text, flags=re.IGNORECASE)
        # Standardize basis points
        text = re.sub(r'(\d+)\s*basis\s+points?', r'\1 bps', text, flags=re.IGNORECASE)
        return text
