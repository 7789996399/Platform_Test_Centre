"""
TRUST Platform - Layer 1 Integration Tests
==========================================
End-to-end test of the 3-signal validation pipeline:
    EHR verification → HHEM faithfulness → Semantic entropy → Risk assessment

Scenario: 10 clinical claims with known ground truth:
    - 5 verifiable in EHR → VERIFIED → LOW risk
    - 2 in transcript but not EHR → NOT_FOUND + FAITHFUL → LOW risk
    - 2 not in transcript, not in EHR → NOT_FOUND + HALLUCINATED → HIGH risk
    - 1 contradicts EHR → CONTRADICTION → tests both HIGH_SE and LOW_SE paths

All mocks, no real API calls per CLAUDE.md rules.

Usage:
    pytest backend/tests/test_layer1_integration.py -v
"""

import sys
import os
import json
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.services.trust_scribe_ehr_first_validation import (
    TRUSTScribeValidator,
    ClaimType,
    EHRVerificationStatus,
    SemanticEntropyLevel,
    FinalRiskLevel,
    ValidationReport,
)
from app.core.hhem_faithfulness import MockHHEM, FaithfulnessLevel


# =============================================================================
# MOCK CLIENTS
# =============================================================================

class MockLLMClient:
    """
    Mock LLM client for testing.

    - Calls without temperature → claim extraction (returns pre-defined JSON)
    - Calls with temperature → SE stochastic sampling (returns from list)
    """

    def __init__(self, extraction_json: str, se_responses: list):
        self.extraction_json = extraction_json
        self.se_responses = se_responses
        self._se_idx = 0
        self.extraction_call_count = 0
        self.se_call_count = 0

    def complete(self, prompt, temperature=None):
        if temperature is not None:
            resp = self.se_responses[self._se_idx % len(self.se_responses)]
            self._se_idx += 1
            self.se_call_count += 1
            return resp
        self.extraction_call_count += 1
        return self.extraction_json


class MockFHIRClient:
    """Mock FHIR client returning pre-defined data by resource type."""

    def __init__(self, data_by_resource: dict):
        self.data_by_resource = data_by_resource
        self.search_calls = []

    def search(self, resource_type, patient_id):
        self.search_calls.append((resource_type, patient_id))
        return self.data_by_resource.get(resource_type, [])


# =============================================================================
# TEST DATA
# =============================================================================

# 10 claims with known expected outcomes
CLAIMS_JSON = json.dumps([
    # --- 5 EHR-verifiable → VERIFIED → LOW risk ---
    {
        "claim_text": "Metoprolol 50mg BID",
        "claim_type": "medication",
        "source_sentence": "MEDICATIONS: Metoprolol 50mg PO BID"
    },
    {
        "claim_text": "Lisinopril 10mg daily",
        "claim_type": "medication",
        "source_sentence": "MEDICATIONS: Lisinopril 10mg PO daily"
    },
    {
        "claim_text": "Penicillin allergy",
        "claim_type": "allergy",
        "source_sentence": "ALLERGIES: Penicillin (rash)"
    },
    {
        "claim_text": "Hypertension",
        "claim_type": "diagnosis",
        "source_sentence": "PMH: Hypertension, managed with medication"
    },
    {
        "claim_text": "Atrial fibrillation",
        "claim_type": "diagnosis",
        "source_sentence": "PMH: Atrial fibrillation"
    },

    # --- 2 NOT in EHR but IN transcript → NOT_FOUND + FAITHFUL → LOW risk ---
    {
        "claim_text": "Aspirin 81mg daily",
        "claim_type": "medication",
        "source_sentence": "MEDICATIONS: Aspirin 81mg PO daily (patient-reported)"
    },
    {
        "claim_text": "Migraine headaches",
        "claim_type": "diagnosis",
        "source_sentence": "HPI: Patient reports migraine headaches"
    },

    # --- 2 NOT in EHR, NOT in transcript → NOT_FOUND + HALLUCINATED → HIGH risk ---
    {
        "claim_text": "Gabapentin 300mg TID",
        "claim_type": "medication",
        "source_sentence": "MEDICATIONS: Gabapentin 300mg PO TID"
    },
    {
        "claim_text": "Type 2 diabetes",
        "claim_type": "diagnosis",
        "source_sentence": "PMH: Type 2 diabetes mellitus"
    },

    # --- 1 contradicts EHR (dose mismatch) → CONTRADICTION → SE tested ---
    {
        "claim_text": "Warfarin 5mg daily",
        "claim_type": "medication",
        "source_sentence": "MEDICATIONS: Warfarin 5mg PO daily"
    },
])

# Transcript supports 7 of 10 claims (claims 1-7 and 10, NOT claims 8-9)
TRANSCRIPT = (
    "Doctor: Good morning. What brings you in today? "
    "Patient: I've been having some chest tightness. "
    "Doctor: Let me review your medications. You're taking "
    "metoprolol 50mg twice daily and lisinopril 10mg daily, correct? "
    "Patient: Yes. I also started taking aspirin 81mg daily on my own. "
    "Doctor: Any allergies? "
    "Patient: Yes, I'm allergic to penicillin. "
    "Doctor: Your records show hypertension and atrial fibrillation. "
    "Patient: Right. I've also been getting migraine headaches recently. "
    "Doctor: Let's discuss your warfarin. "
    "Patient: I've been taking warfarin 5mg daily."
)

MOCK_NOTE = "AI-generated clinical note (content unused — claims from mock LLM)"

# EHR data: verifies 5 claims, contradicts 1 (warfarin dose)
EHR_DATA = {
    "MedicationStatement": [
        {"medication_name": "metoprolol", "dosage": "50mg"},
        {"medication_name": "lisinopril", "dosage": "10mg"},
        {"medication_name": "warfarin", "dosage": "2.5mg"},  # claim says 5mg → CONTRADICTION
    ],
    "AllergyIntolerance": [
        {"substance": "penicillin", "status": "active"},
    ],
    "Condition": [
        {"display": "Hypertension", "code": "I10"},
        {"display": "Atrial fibrillation", "code": "I48.91"},
        # No migraine, no diabetes → NOT_FOUND for those claims
    ],
}

# SE responses: all agree → entropy ≈ 0 → LOW SE (confident)
LOW_SE_RESPONSES = [
    "Yes, the patient mentioned taking warfarin 5mg daily",
    "Yes, warfarin 5mg was discussed by the patient",
    "Yes, the patient confirmed warfarin 5mg daily",
    "Yes, warfarin was mentioned at a dose of 5mg daily",
    "Yes, the patient stated they take warfarin 5mg",
]

# SE responses: mixed → entropy high → HIGH SE (uncertain)
HIGH_SE_RESPONSES = [
    "Yes, warfarin was mentioned",
    "No, warfarin was not clearly discussed in the transcript",
    "The transcript is unclear about the warfarin dose",
    "No, the specific medication was not mentioned",
    "Yes, the patient discussed warfarin briefly",
]

PATIENT_ID = "P12345"
DOCUMENT_ID = "test_doc_001"


# =============================================================================
# HELPERS
# =============================================================================

def _run_pipeline(se_responses, hhem_overrides=None):
    """Run the full validation pipeline with given SE responses and HHEM config."""
    llm = MockLLMClient(CLAIMS_JSON, se_responses)
    fhir = MockFHIRClient(EHR_DATA)
    hhem = MockHHEM(keyword_overrides=hhem_overrides or {})

    validator = TRUSTScribeValidator(
        llm_client=llm,
        fhir_client=fhir,
        hhem_scorer=hhem,
    )
    report = validator.validate(
        MOCK_NOTE, TRANSCRIPT, PATIENT_ID, document_id=DOCUMENT_ID
    )
    return report, llm, fhir


def _get_claim(report, claim_text):
    """Look up a specific claim by its text."""
    matches = [c for c in report.claims if c.claim_text == claim_text]
    assert len(matches) == 1, f"Expected 1 claim '{claim_text}', found {len(matches)}"
    return matches[0]


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="module")
def low_se_pipeline():
    """Pipeline run with LOW SE (all responses agree → confident)."""
    return _run_pipeline(LOW_SE_RESPONSES)


@pytest.fixture(scope="module")
def high_se_pipeline():
    """Pipeline run with HIGH SE (mixed responses → uncertain)."""
    return _run_pipeline(HIGH_SE_RESPONSES)


@pytest.fixture(scope="module")
def low_se_report(low_se_pipeline):
    return low_se_pipeline[0]


@pytest.fixture(scope="module")
def low_se_llm(low_se_pipeline):
    return low_se_pipeline[1]


@pytest.fixture(scope="module")
def low_se_fhir(low_se_pipeline):
    return low_se_pipeline[2]


@pytest.fixture(scope="module")
def high_se_report(high_se_pipeline):
    return high_se_pipeline[0]


# =============================================================================
# TEST 1: CORRECT NUMBER OF CLAIMS EXTRACTED
# =============================================================================

class TestClaimExtraction:
    """Verify claim extraction produces the right count and types."""

    def test_extracts_10_claims(self, low_se_report):
        assert low_se_report.total_claims == 10
        assert len(low_se_report.claims) == 10

    def test_claim_types_breakdown(self, low_se_report):
        types = [c.claim_type for c in low_se_report.claims]
        assert types.count(ClaimType.MEDICATION) == 5
        assert types.count(ClaimType.DIAGNOSIS) == 4
        assert types.count(ClaimType.ALLERGY) == 1

    def test_single_llm_extraction_call(self, low_se_llm):
        assert low_se_llm.extraction_call_count == 1

    def test_claim_ids_sequential(self, low_se_report):
        ids = [c.claim_id for c in low_se_report.claims]
        assert ids == [f"claim_{i:03d}" for i in range(1, 11)]


# =============================================================================
# TEST 2: CORRECT RISK LEVELS (3-SIGNAL MATRIX)
# =============================================================================

class TestRiskLevelsLowSE:
    """Risk levels when SE is LOW (confident contradiction)."""

    def test_verified_claims_are_low_risk(self, low_se_report):
        verified_texts = [
            "Metoprolol 50mg BID",
            "Lisinopril 10mg daily",
            "Penicillin allergy",
            "Hypertension",
            "Atrial fibrillation",
        ]
        for text in verified_texts:
            claim = _get_claim(low_se_report, text)
            assert claim.ehr_status == EHRVerificationStatus.VERIFIED, text
            assert claim.risk_level == FinalRiskLevel.LOW, text
            assert claim.requires_physician_review is False, text

    def test_not_found_faithful_is_low_risk(self, low_se_report):
        """NOT_FOUND + FAITHFUL to transcript → LOW risk (new patient info)."""
        for text in ["Aspirin 81mg daily", "Migraine headaches"]:
            claim = _get_claim(low_se_report, text)
            assert claim.ehr_status == EHRVerificationStatus.NOT_FOUND, text
            assert claim.hhem_level == FaithfulnessLevel.FAITHFUL, text
            assert claim.risk_level == FinalRiskLevel.LOW, text
            assert claim.requires_physician_review is False, text

    def test_not_found_hallucinated_is_high_risk(self, low_se_report):
        """NOT_FOUND + HALLUCINATED → HIGH risk (fabricated claim)."""
        for text in ["Gabapentin 300mg TID", "Type 2 diabetes"]:
            claim = _get_claim(low_se_report, text)
            assert claim.ehr_status == EHRVerificationStatus.NOT_FOUND, text
            assert claim.hhem_level == FaithfulnessLevel.HALLUCINATED, text
            assert claim.risk_level == FinalRiskLevel.HIGH, text
            assert claim.requires_physician_review is True, text

    def test_contradiction_low_se_faithful_is_high_risk(self, low_se_report):
        """CONTRADICTION + LOW SE + FAITHFUL → HIGH (patient reported new info)."""
        warfarin = _get_claim(low_se_report, "Warfarin 5mg daily")
        assert warfarin.ehr_status == EHRVerificationStatus.CONTRADICTION
        assert warfarin.se_level == SemanticEntropyLevel.LOW
        assert warfarin.hhem_level == FaithfulnessLevel.FAITHFUL
        assert warfarin.risk_level == FinalRiskLevel.HIGH
        assert warfarin.requires_physician_review is True
        assert "faithful" in warfarin.risk_explanation.lower()


class TestRiskLevelsHighSE:
    """Risk levels when SE is HIGH (ambiguous transcript)."""

    def test_contradiction_high_se_is_medium_risk(self, high_se_report):
        """CONTRADICTION + HIGH SE → MEDIUM risk (transcript was ambiguous)."""
        warfarin = _get_claim(high_se_report, "Warfarin 5mg daily")
        assert warfarin.ehr_status == EHRVerificationStatus.CONTRADICTION
        assert warfarin.se_level == SemanticEntropyLevel.HIGH
        assert warfarin.risk_level == FinalRiskLevel.MEDIUM
        assert warfarin.requires_physician_review is True

    def test_non_contradiction_claims_unchanged(self, high_se_report):
        """Changing SE responses should not affect non-contradiction claims."""
        # Verified claims still LOW
        metoprolol = _get_claim(high_se_report, "Metoprolol 50mg BID")
        assert metoprolol.risk_level == FinalRiskLevel.LOW

        # NOT_FOUND + FAITHFUL still LOW
        aspirin = _get_claim(high_se_report, "Aspirin 81mg daily")
        assert aspirin.risk_level == FinalRiskLevel.LOW

        # NOT_FOUND + HALLUCINATED still HIGH
        gabapentin = _get_claim(high_se_report, "Gabapentin 300mg TID")
        assert gabapentin.risk_level == FinalRiskLevel.HIGH


class TestCriticalPath:
    """CONTRADICTION + LOW SE + HALLUCINATED HHEM → CRITICAL risk."""

    def test_contradiction_low_se_hallucinated_is_critical(self):
        """When claim contradicts EHR AND isn't in transcript → CRITICAL."""
        report, _, _ = _run_pipeline(
            LOW_SE_RESPONSES,
            hhem_overrides={"warfarin": 0.10},
        )
        warfarin = _get_claim(report, "Warfarin 5mg daily")
        assert warfarin.ehr_status == EHRVerificationStatus.CONTRADICTION
        assert warfarin.se_level == SemanticEntropyLevel.LOW
        assert warfarin.hhem_level == FaithfulnessLevel.HALLUCINATED
        assert warfarin.risk_level == FinalRiskLevel.CRITICAL
        assert warfarin.requires_physician_review is True


# =============================================================================
# TEST 3: HHEM ONLY RUNS ON NOT_FOUND + CONTRADICTION
# =============================================================================

class TestHHEMTargeting:
    """HHEM scoring should only run on claims EHR couldn't resolve."""

    def test_verified_claims_have_no_hhem_score(self, low_se_report):
        for text in ["Metoprolol 50mg BID", "Lisinopril 10mg daily",
                      "Penicillin allergy", "Hypertension", "Atrial fibrillation"]:
            claim = _get_claim(low_se_report, text)
            assert claim.hhem_score is None, f"HHEM should not score VERIFIED claim: {text}"
            assert claim.hhem_level is None, text

    def test_not_found_claims_have_hhem_score(self, low_se_report):
        for text in ["Aspirin 81mg daily", "Migraine headaches",
                      "Gabapentin 300mg TID", "Type 2 diabetes"]:
            claim = _get_claim(low_se_report, text)
            assert claim.hhem_score is not None, f"HHEM should score NOT_FOUND claim: {text}"
            assert claim.hhem_level is not None, text

    def test_contradiction_has_hhem_score(self, low_se_report):
        warfarin = _get_claim(low_se_report, "Warfarin 5mg daily")
        assert warfarin.hhem_score is not None
        assert warfarin.hhem_level is not None

    def test_hhem_tested_count(self, low_se_report):
        assert low_se_report.hhem_tested_claims == 5  # 4 NOT_FOUND + 1 CONTRADICTION


# =============================================================================
# TEST 4: SE ONLY RUNS ON CONTRADICTION CLAIMS
# =============================================================================

class TestSETargeting:
    """Semantic entropy should only run on contradictions."""

    def test_only_contradiction_has_se(self, low_se_report):
        for claim in low_se_report.claims:
            if claim.ehr_status == EHRVerificationStatus.CONTRADICTION:
                assert claim.semantic_entropy is not None, (
                    f"SE should run on CONTRADICTION: {claim.claim_text}"
                )
                assert claim.se_level is not None
            else:
                assert claim.semantic_entropy is None, (
                    f"SE should NOT run on {claim.ehr_status.value}: {claim.claim_text}"
                )
                assert claim.se_level is None

    def test_se_call_count(self, low_se_llm):
        # 1 contradiction × 5 samples = 5 LLM calls
        assert low_se_llm.se_call_count == 5

    def test_low_se_entropy_value(self, low_se_report):
        warfarin = _get_claim(low_se_report, "Warfarin 5mg daily")
        # All responses agree → 1 cluster → entropy ≈ 0
        assert warfarin.semantic_entropy < 0.3
        assert warfarin.se_level == SemanticEntropyLevel.LOW

    def test_high_se_entropy_value(self, high_se_report):
        warfarin = _get_claim(high_se_report, "Warfarin 5mg daily")
        # Mixed responses → 3 clusters → entropy > 0.6
        assert warfarin.semantic_entropy > 0.6
        assert warfarin.se_level == SemanticEntropyLevel.HIGH


# =============================================================================
# TEST 5: VALIDATION REPORT STATISTICS
# =============================================================================

class TestReportStatistics:
    """ValidationReport fields should accurately reflect pipeline results."""

    def test_ehr_counts(self, low_se_report):
        assert low_se_report.verified_claims == 5
        assert low_se_report.not_found_claims == 4
        assert low_se_report.contradiction_claims == 1
        assert low_se_report.not_checkable_claims == 0

    def test_hhem_counts(self, low_se_report):
        assert low_se_report.hhem_tested_claims == 5
        # Gabapentin + Type 2 diabetes
        assert low_se_report.hhem_unfaithful_claims == 2

    def test_se_counts_low_se(self, low_se_report):
        assert low_se_report.se_tested_claims == 1
        assert low_se_report.confident_hallucinators == 1
        assert low_se_report.ambiguous_claims == 0

    def test_se_counts_high_se(self, high_se_report):
        assert high_se_report.se_tested_claims == 1
        assert high_se_report.confident_hallucinators == 0
        assert high_se_report.ambiguous_claims == 1

    def test_overall_risk_low_se(self, low_se_report):
        # Gabapentin and diabetes are HIGH risk → overall HIGH
        assert low_se_report.overall_risk == FinalRiskLevel.HIGH

    def test_overall_risk_high_se(self, high_se_report):
        # Gabapentin and diabetes still HIGH → overall HIGH
        assert high_se_report.overall_risk == FinalRiskLevel.HIGH

    def test_overall_risk_critical_path(self):
        report, _, _ = _run_pipeline(
            LOW_SE_RESPONSES,
            hhem_overrides={"warfarin": 0.10},
        )
        # Warfarin is CRITICAL → overall CRITICAL
        assert report.overall_risk == FinalRiskLevel.CRITICAL

    def test_requires_physician_review(self, low_se_report):
        assert low_se_report.requires_physician_review is True

    def test_review_priority(self, low_se_report):
        # Overall HIGH → "elevated"
        assert low_se_report.review_priority == "elevated"

    def test_document_id_preserved(self, low_se_report):
        assert low_se_report.document_id == DOCUMENT_ID

    def test_patient_id_preserved(self, low_se_report):
        assert low_se_report.patient_id == PATIENT_ID

    def test_time_saved_positive(self, low_se_report):
        assert low_se_report.time_saved_percent > 0

    def test_recommendations_present(self, low_se_report):
        assert len(low_se_report.recommendations) > 0
