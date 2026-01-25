#!/usr/bin/env python3
"""
🛡️ TRUST Platform Demo
========================

Demonstrates the Externalized Metacognitive Core Engine across three domains:
- 🏥 Healthcare: Medication, allergy, and dosage verification
- ⚖️ Legal: Case citation and statute verification (fabrication detection!)
- 💰 Finance: Performance claims and fee disclosure verification

This demo shows how the platform extracts claims from AI outputs and
identifies potential "confident hallucinators" - the most dangerous
type of AI error where the model sounds confident but is wrong.
"""

import sys
from pathlib import Path

# Add trust-platform to path
sys.path.insert(0, str(Path(__file__).parent))

from adapters.healthcare import HealthcareAdapter, HealthcareClaimType
from adapters.legal import LegalAdapter, LegalClaimType
from adapters.finance import FinanceAdapter, FinanceClaimType
from adapters.base_adapter import RiskLevel, VerificationContext


# =============================================================================
# SAMPLE AI OUTPUTS
# =============================================================================

HEALTHCARE_OUTPUT = """
Based on my review of the patient's medical history:

The patient takes Lisinopril 10mg daily for hypertension. They also take
Metformin 500mg twice daily for type 2 diabetes. The patient is allergic
to penicillin and reports a history of anaphylaxis.

Blood pressure at last visit was 128/82 mmHg, which is within normal limits.
Heart rate was 72 bpm. The patient's HbA1c of 7.2% indicates moderate
glycemic control.

I recommend continuing current medications and scheduling a follow-up in 3 months.
"""

LEGAL_OUTPUT = """
Regarding your breach of contract claim:

Per Smith v. Anderson, 456 F.3d 789 (2nd Cir. 2019), the defendant's motion
to dismiss must be denied when the plaintiff has adequately pleaded damages.

Additionally, under 42 U.S.C. § 1983, civil rights claims require showing
that the defendant acted under color of state law. See Monroe v. Pape,
365 U.S. 167 (1961).

The statute of limitations for breach of contract in New York is 6 years
pursuant to N.Y. C.P.L.R. § 213. Given that the breach occurred in 2021,
your claim remains timely.

The forum selection clause in Section 12.3 of the Agreement mandates
arbitration in New York.
"""

FINANCE_OUTPUT = """
Investment Portfolio Summary:

This fund achieved 15.2% returns last year, significantly outperforming
the S&P 500 benchmark by 3.4%. The fund has a 0.75% expense ratio with
no front-end load.

Based on your moderate risk tolerance and 10-year investment horizon,
this fund is suitable for your portfolio. The fund's Sharpe ratio of 1.2
indicates strong risk-adjusted performance.

Looking ahead, we expect continued growth of 8-10% annually given current
market conditions. Past performance does not guarantee future results.

The management fee is 0.50% annually, with an additional 12b-1 fee of 0.25%.
"""


# =============================================================================
# DISPLAY HELPERS
# =============================================================================

def print_header(title: str, emoji: str = "🔷"):
    """Print a formatted section header."""
    print()
    print(f"{emoji}" + "=" * 70 + f"{emoji}")
    print(f"  {title}")
    print(f"{emoji}" + "=" * 70 + f"{emoji}")
    print()


def print_subheader(title: str):
    """Print a formatted subsection header."""
    print()
    print(f"  📋 {title}")
    print(f"  " + "-" * 50)


def get_risk_emoji(risk: RiskLevel) -> str:
    """Get emoji for risk level."""
    return {
        RiskLevel.MINIMAL: "🟢",
        RiskLevel.LOW: "🟢",
        RiskLevel.MEDIUM: "🟡",
        RiskLevel.HIGH: "🟠",
        RiskLevel.CRITICAL: "🔴",
    }.get(risk, "⚪")


def get_claim_type_emoji(claim_type: str) -> str:
    """Get emoji for claim type."""
    emoji_map = {
        # Healthcare
        "medication": "💊",
        "allergy": "🤧",
        "dosage": "💉",
        "vital_sign": "❤️",
        "lab_result": "🔬",
        "diagnosis": "🩺",
        # Legal
        "case_citation": "📚",
        "statute_reference": "📜",
        "contract_clause": "📝",
        "jurisdiction": "🏛️",
        "limitation_period": "⏰",
        # Finance
        "performance_figure": "📈",
        "fee_disclosure": "💵",
        "risk_rating": "📊",
        "suitability": "✅",
        "forward_looking": "🔮",
        "guaranteed_return": "⚠️",
        "benchmark_comparison": "📉",
    }
    return emoji_map.get(claim_type, "📌")


# =============================================================================
# DEMO FUNCTIONS
# =============================================================================

def demo_healthcare():
    """Demonstrate healthcare adapter."""
    print_header("HEALTHCARE DOMAIN", "🏥")

    print("  📄 AI Output:")
    print("  " + "-" * 50)
    for line in HEALTHCARE_OUTPUT.strip().split('\n'):
        print(f"  │ {line}")
    print()

    # Initialize adapter
    adapter = HealthcareAdapter()

    # Extract claims
    claims = adapter.extract_claims(HEALTHCARE_OUTPUT)

    print_subheader(f"Extracted Claims ({len(claims)} found)")

    for i, claim in enumerate(claims, 1):
        emoji = get_claim_type_emoji(claim.claim_type)
        print(f"  {i}. {emoji} [{claim.claim_type.upper()}]")
        print(f"     \"{claim.text[:80]}{'...' if len(claim.text) > 80 else ''}\"")

        # Show metadata for medications
        if claim.claim_type == "medication" and claim.metadata.get("components"):
            components = claim.metadata["components"]
            if components.get("drug_name"):
                print(f"     💊 Drug: {components.get('drug_name')}, "
                      f"Dose: {components.get('dose', 'N/A')}, "
                      f"Frequency: {components.get('frequency', 'N/A')}")
        print()

    # Classify risk
    context = VerificationContext(
        query="Review patient medical history",
        response=HEALTHCARE_OUTPUT,
        claims=claims,
        source_documents=[],
        session_metadata={},
    )
    risk = adapter.classify_risk(context)
    risk_emoji = get_risk_emoji(risk)

    print_subheader("Risk Assessment")
    print(f"  {risk_emoji} Overall Risk Level: {risk.value.upper()}")
    print()

    # Explain risk
    print("  🔍 Risk Factors:")
    critical_types = {"medication", "dosage", "allergy", "drug_interaction"}
    found_critical = [c for c in claims if c.claim_type in critical_types]
    if found_critical:
        print(f"     • Found {len(found_critical)} high-risk clinical claims")
        print("     • Medications and dosages require strict verification")
        print("     • Allergy information is safety-critical")

    return claims, risk


def demo_legal():
    """Demonstrate legal adapter."""
    print_header("LEGAL DOMAIN", "⚖️")

    print("  📄 AI Output:")
    print("  " + "-" * 50)
    for line in LEGAL_OUTPUT.strip().split('\n'):
        print(f"  │ {line}")
    print()

    # Initialize adapter
    adapter = LegalAdapter()

    # Extract claims
    claims = adapter.extract_claims(LEGAL_OUTPUT)

    print_subheader(f"Extracted Claims ({len(claims)} found)")

    for i, claim in enumerate(claims, 1):
        emoji = get_claim_type_emoji(claim.claim_type)
        print(f"  {i}. {emoji} [{claim.claim_type.upper()}]")
        print(f"     \"{claim.text[:80]}{'...' if len(claim.text) > 80 else ''}\"")

        # Show fabrication warning for citations
        if claim.claim_type == "case_citation":
            print("     ⚠️  WARNING: Case citations are HIGH RISK for fabrication!")
            if claim.metadata.get("components"):
                comp = claim.metadata["components"]
                if comp.get("case_name"):
                    print(f"     📚 Case: {comp.get('case_name')}, "
                          f"Reporter: {comp.get('reporter', 'N/A')}, "
                          f"Year: {comp.get('year', 'N/A')}")
        print()

    # Classify risk
    context = VerificationContext(
        query="Legal analysis of breach of contract claim",
        response=LEGAL_OUTPUT,
        claims=claims,
        source_documents=[],
        session_metadata={},
    )
    risk = adapter.classify_risk(context)
    risk_emoji = get_risk_emoji(risk)

    print_subheader("Risk Assessment")
    print(f"  {risk_emoji} Overall Risk Level: {risk.value.upper()}")
    print()

    # Explain risk
    print("  🔍 Risk Factors:")
    citation_claims = [c for c in claims if c.claim_type in ("case_citation", "statute_reference")]
    if citation_claims:
        print(f"     • Found {len(citation_claims)} legal citations")
        print("     • ⚠️  LLMs frequently FABRICATE legal citations!")
        print("     • These must be verified against Westlaw/LexisNexis")
        print("     • Fabricated citations can result in court sanctions")

    return claims, risk


def demo_finance():
    """Demonstrate finance adapter."""
    print_header("FINANCE DOMAIN", "💰")

    print("  📄 AI Output:")
    print("  " + "-" * 50)
    for line in FINANCE_OUTPUT.strip().split('\n'):
        print(f"  │ {line}")
    print()

    # Initialize adapter
    adapter = FinanceAdapter()

    # Extract claims
    claims = adapter.extract_claims(FINANCE_OUTPUT)

    print_subheader(f"Extracted Claims ({len(claims)} found)")

    for i, claim in enumerate(claims, 1):
        emoji = get_claim_type_emoji(claim.claim_type)
        print(f"  {i}. {emoji} [{claim.claim_type.upper()}]")
        print(f"     \"{claim.text[:80]}{'...' if len(claim.text) > 80 else ''}\"")

        # Show regulatory warnings
        if claim.claim_type == "performance_figure":
            print("     📊 Performance claims require SEC/FINRA compliance!")
        elif claim.claim_type == "forward_looking":
            print("     ⚠️  Forward-looking statement requires disclaimer!")
        elif claim.claim_type == "suitability":
            print("     ✅ Suitability claims require Reg BI compliance!")
        print()

    # Classify risk
    context = VerificationContext(
        query="Investment portfolio summary",
        response=FINANCE_OUTPUT,
        claims=claims,
        source_documents=[],
        session_metadata={},
    )
    risk = adapter.classify_risk(context)
    risk_emoji = get_risk_emoji(risk)

    print_subheader("Risk Assessment")
    print(f"  {risk_emoji} Overall Risk Level: {risk.value.upper()}")
    print()

    # Explain risk
    print("  🔍 Risk Factors:")
    perf_claims = [c for c in claims if c.claim_type == "performance_figure"]
    forward_claims = [c for c in claims if c.claim_type == "forward_looking"]
    if perf_claims:
        print(f"     • Found {len(perf_claims)} performance claims")
        print("     • SEC Rule 206(4)-1 requires accurate advertising")
    if forward_claims:
        print(f"     • Found {len(forward_claims)} forward-looking statements")
        print("     • Must include required disclaimers")

    return claims, risk


def print_confident_hallucinator_explanation():
    """Print explanation of confident hallucinator detection."""
    print_header("CONFIDENT HALLUCINATOR DETECTION", "🎯")

    print("""
  The TRUST Platform's core insight is the "Detection Matrix":

  ┌─────────────────┬──────────────────┬───────────────────┐
  │                 │ Source: VERIFIED │ Source: CONFLICTS │
  ├─────────────────┼──────────────────┼───────────────────┤
  │ High Entropy    │ 🟡 REVIEW NEEDED │ 🟠 LIKELY ERROR   │
  │ (uncertain)     │ (unsure but ok)  │ (unsure & wrong)  │
  ├─────────────────┼──────────────────┼───────────────────┤
  │ Low Entropy     │ 🟢 LIKELY CORRECT│ 🔴 CONFIDENT      │
  │ (confident)     │ (sure & right)   │ HALLUCINATOR! 🔴  │
  └─────────────────┴──────────────────┴───────────────────┘

  The "Confident Hallucinator" (bottom-right) is the MOST DANGEROUS:

  • The AI appears confident (gives consistent answers)
  • But the content contradicts authoritative sources
  • This is exactly how LLMs fabricate legal citations!

  📚 Example: "Per Smith v. Anderson, 456 F.3d 789 (2nd Cir. 2019)..."

     - The AI sounds confident (low entropy)
     - The citation looks real (proper format)
     - BUT: The case may not exist!
     - This could result in court sanctions for the attorney

  🛡️ The TRUST Platform catches this by:

     1. Extracting verifiable claims from AI outputs
     2. Verifying claims against authoritative sources FIRST (fast)
     3. Running semantic entropy only on unverified claims
     4. Flagging the dangerous "confident + wrong" pattern
""")


def print_summary(healthcare_claims, healthcare_risk,
                  legal_claims, legal_risk,
                  finance_claims, finance_risk):
    """Print final summary."""
    print_header("SUMMARY", "📊")

    print("  Domain         │ Claims │ Risk Level  │ Key Concerns")
    print("  ───────────────┼────────┼─────────────┼──────────────────────────────")

    # Healthcare
    hc_emoji = get_risk_emoji(healthcare_risk)
    hc_concerns = "Medications, allergies"
    print(f"  🏥 Healthcare   │   {len(healthcare_claims):2d}   │ {hc_emoji} {healthcare_risk.value:9s} │ {hc_concerns}")

    # Legal
    lg_emoji = get_risk_emoji(legal_risk)
    lg_concerns = "Citation fabrication risk!"
    print(f"  ⚖️  Legal       │   {len(legal_claims):2d}   │ {lg_emoji} {legal_risk.value:9s} │ {lg_concerns}")

    # Finance
    fn_emoji = get_risk_emoji(finance_risk)
    fn_concerns = "Performance claims, disclosures"
    print(f"  💰 Finance     │   {len(finance_claims):2d}   │ {fn_emoji} {finance_risk.value:9s} │ {fn_concerns}")

    print()
    total_claims = len(healthcare_claims) + len(legal_claims) + len(finance_claims)
    print(f"  📈 Total claims extracted: {total_claims}")

    # Count high-risk claims
    high_risk_types = {
        "medication", "allergy", "dosage",  # Healthcare
        "case_citation", "statute_reference",  # Legal
        "performance_figure", "fee_disclosure", "forward_looking",  # Finance
    }
    all_claims = healthcare_claims + legal_claims + finance_claims
    high_risk_count = sum(1 for c in all_claims if c.claim_type in high_risk_types)
    print(f"  ⚠️  High-risk claims requiring verification: {high_risk_count}")

    print()
    print("  " + "=" * 60)
    print("  🛡️  TRUST Platform: Catching hallucinations before they cause harm")
    print("  " + "=" * 60)


def main():
    """Run the full demo."""
    print()
    print("🛡️" + "=" * 70 + "🛡️")
    print("  TRUST PLATFORM DEMO")
    print("  Externalized Metacognitive Core for AI Governance")
    print("🛡️" + "=" * 70 + "🛡️")

    # Run demos
    healthcare_claims, healthcare_risk = demo_healthcare()
    legal_claims, legal_risk = demo_legal()
    finance_claims, finance_risk = demo_finance()

    # Explain confident hallucinator
    print_confident_hallucinator_explanation()

    # Print summary
    print_summary(
        healthcare_claims, healthcare_risk,
        legal_claims, legal_risk,
        finance_claims, finance_risk
    )

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
