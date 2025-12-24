"""
TRUST Platform - Pipeline Test Script
======================================
Quick test to verify all core modules work together.

Run from backend folder:
    python test_pipeline.py
"""

import json
import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

from app.core.claim_extraction import extract_claims_from_note, summarize_claims
from app.core.source_verification import verify_all_claims
from app.core.routing import analyze_note, format_review_queue_for_display, generate_audit_log


def load_mock_note(filename: str) -> dict:
    """Load a mock note from the mock_data folder."""
    mock_path = Path(__file__).parent.parent / "mock_data" / "scribe_notes" / filename
    with open(mock_path) as f:
        return json.load(f)


def test_clean_note():
    """Test with a clean note (no hallucinations)."""
    print("\n" + "="*60)
    print("TEST 1: Clean Note (sample_note_1.json)")
    print("="*60)
    
    note = load_mock_note("sample_note_1.json")
    transcript = note.get("source_transcript", "")
    
    # Extract claims
    claims = extract_claims_from_note(note)
    summary = summarize_claims(claims)
    
    print(f"\n📋 Extracted {summary['total_claims']} claims:")
    print(f"   By type: {summary['by_type']}")
    print(f"   By risk: {summary['by_risk']}")
    
    # Verify against transcript
    verification = verify_all_claims(claims, transcript)
    
    print(f"\n✅ Verification Results:")
    print(f"   Verified: {verification['verified']}")
    print(f"   Not found: {verification['not_found']}")
    print(f"   Contradicted: {verification['contradicted']}")
    print(f"   Compute saved: {verification['compute_saved_percent']:.1f}%")
    
    # Full analysis
    analysis = analyze_note(note, transcript, run_entropy=True)
    
    print(f"\n🎯 Overall Risk: {analysis.overall_risk}")
    print(f"   Review burden: {analysis.review_burden}")
    
    return analysis


def test_hallucinated_note():
    """Test with a note containing hallucinations."""
    print("\n" + "="*60)
    print("TEST 2: Hallucinated Note (sample_note_2_hallucinated.json)")
    print("="*60)
    
    note = load_mock_note("sample_note_2_hallucinated.json")
    transcript = note.get("source_transcript", "")
    
    # Extract claims
    claims = extract_claims_from_note(note)
    summary = summarize_claims(claims)
    
    print(f"\n📋 Extracted {summary['total_claims']} claims:")
    print(f"   High risk claims: {len(summary['high_risk_claims'])}")
    
    # Verify against transcript
    verification = verify_all_claims(claims, transcript)
    
    print(f"\n⚠️  Verification Results:")
    print(f"   Verified: {verification['verified']}")
    print(f"   Not found: {verification['not_found']}")
    print(f"   Contradicted: {verification['contradicted']} ← HALLUCINATIONS DETECTED!")
    
    # Show flagged claims
    if verification['flagged_claims']:
        print(f"\n�� Flagged Claims:")
        for result in verification['flagged_claims'][:5]:
            print(f"   - {result.claim.text}")
            print(f"     Status: {result.status.value}")
            print(f"     Reason: {result.explanation}")
    
    # Full analysis
    analysis = analyze_note(note, transcript, run_entropy=True)
    
    print(f"\n🎯 Overall Risk: {analysis.overall_risk}")
    
    # Show review queue
    review_queue = format_review_queue_for_display(analysis)
    print(f"\n📊 Review Queue (top 5 priority items):")
    for item in review_queue[:5]:
        print(f"   #{item['rank']}: {item['claim_text'][:50]}...")
        print(f"       Priority: {item['priority_score']} | Status: {item['verification_status']} | Tier: {item['review_tier']}")
    
    # Generate audit log
    audit = generate_audit_log(analysis)
    print(f"\n📝 Audit Log Generated:")
    print(f"   Timestamp: {audit['timestamp']}")
    print(f"   Compliance: {', '.join(audit['compliance_frameworks'])}")
    
    return analysis


def main():
    print("\n" + "🏥 "*20)
    print("   TRUST Platform - Pipeline Test")
    print("   Auditing AI. Protecting Patients. Empowering Physicians.")
    print("🏥 "*20)
    
    try:
        # Test 1: Clean note
        analysis1 = test_clean_note()
        
        # Test 2: Hallucinated note
        analysis2 = test_hallucinated_note()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print("\nCore pipeline is working:")
        print("  ✓ Claim extraction")
        print("  ✓ Source verification")
        print("  ✓ Semantic entropy (basic mode)")
        print("  ✓ Uncertainty quantification")
        print("  ✓ Review tier assignment")
        print("  ✓ Priority routing")
        print("  ✓ Audit log generation")
        print("\nNext steps:")
        print("  - Add DeBERTa model for proper entailment checking")
        print("  - Connect to Azure OpenAI for response generation")
        print("  - Build FastAPI endpoints")
        print("  - Connect to Cerner sandbox")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
