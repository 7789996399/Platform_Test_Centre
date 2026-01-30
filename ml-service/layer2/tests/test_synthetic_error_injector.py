"""
Tests for synthetic_error_injector.py

Covers:
- ErrorType enum completeness
- MockEHRContext defaults
- Each individual error injection type
- Determinism / seed reproducibility
- inject_random_errors rate and labelling
- Claim-type-to-error-type compatibility
- generate_training_set_from_notes end-to-end
- Edge cases (no dose found, no frequency found, no drug match, empty inputs)
"""

import sys
from pathlib import Path

# Ensure layer2 is importable when running from ml-service/
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from layer2.data.synthetic_error_injector import (
    ErrorType,
    ErrorInjector,
    MockEHRContext,
    InjectedClaim,
    NoteWithClaims,
    DEFAULT_EHR_CONTEXT,
    DOSE_MULTIPLIERS,
    FREQUENCY_SWAPS,
    SIMILAR_DRUGS,
    ABNORMAL_VITALS,
    FAKE_ALLERGIES,
    FAKE_SYMPTOMS,
    _COMPATIBLE_ERRORS,
    generate_training_set_from_notes,
)
from layer2.data.mock_data_generator import TrainingExample


# ═════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def injector():
    return ErrorInjector(seed=42)


@pytest.fixture
def ehr():
    return DEFAULT_EHR_CONTEXT


@pytest.fixture
def custom_ehr():
    return MockEHRContext(
        medications=["Warfarin 5mg daily", "Insulin glargine 20 units at bedtime"],
        allergies=["Penicillin"],
        vitals={"BP": "130/85", "HR": "80", "temp": "36.8°C"},
    )


# ═════════════════════════════════════════════════════════════════════════════
# ERROR TYPE ENUM
# ═════════════════════════════════════════════════════════════════════════════

class TestErrorType:
    def test_all_seven_types_exist(self):
        assert len(ErrorType) == 7

    def test_values_are_snake_case(self):
        for et in ErrorType:
            assert et.value == et.value.lower()
            assert "_" in et.value or et.value.isalpha()

    def test_string_enum(self):
        assert ErrorType.WRONG_DOSE == "wrong_dose"
        assert isinstance(ErrorType.WRONG_DOSE, str)


# ═════════════════════════════════════════════════════════════════════════════
# MOCK EHR CONTEXT
# ═════════════════════════════════════════════════════════════════════════════

class TestMockEHRContext:
    def test_default_has_medications(self):
        assert len(DEFAULT_EHR_CONTEXT.medications) > 0

    def test_default_has_allergies(self):
        assert len(DEFAULT_EHR_CONTEXT.allergies) > 0

    def test_default_has_vitals(self):
        assert "BP" in DEFAULT_EHR_CONTEXT.vitals
        assert "HR" in DEFAULT_EHR_CONTEXT.vitals

    def test_empty_context(self):
        ctx = MockEHRContext()
        assert ctx.medications == []
        assert ctx.allergies == []
        assert ctx.vitals == {}


# ═════════════════════════════════════════════════════════════════════════════
# SUBSTITUTION TABLE SANITY
# ═════════════════════════════════════════════════════════════════════════════

class TestSubstitutionTables:
    def test_dose_multipliers_nonzero(self):
        for m in DOSE_MULTIPLIERS:
            assert m > 0

    def test_frequency_swaps_are_bidirectional_coverage(self):
        # Every swap target should differ from source
        for src, dst in FREQUENCY_SWAPS.items():
            assert src.lower() != dst.lower()

    def test_similar_drugs_are_bidirectional(self):
        for drug_a, drug_b in SIMILAR_DRUGS.items():
            assert drug_b in SIMILAR_DRUGS, f"{drug_b} missing reverse mapping"
            assert SIMILAR_DRUGS[drug_b] == drug_a

    def test_abnormal_vitals_all_have_values(self):
        for vital, values in ABNORMAL_VITALS.items():
            assert len(values) >= 2

    def test_fake_allergies_nonempty(self):
        assert len(FAKE_ALLERGIES) >= 10

    def test_fake_symptoms_nonempty(self):
        assert len(FAKE_SYMPTOMS) >= 10


# ═════════════════════════════════════════════════════════════════════════════
# INDIVIDUAL ERROR INJECTIONS
# ═════════════════════════════════════════════════════════════════════════════

class TestWrongDose:
    def test_modifies_numeric_dose(self, injector, ehr):
        text = "Patient takes Metoprolol 50mg daily"
        modified, details = injector.inject_error(
            text, "medication", ehr, ErrorType.WRONG_DOSE,
        )
        assert modified != text
        assert "mg" in modified
        assert "dose changed" in details

    def test_handles_decimal_dose(self, injector, ehr):
        text = "Digoxin 0.25mg once daily"
        modified, details = injector.inject_error(
            text, "medication", ehr, ErrorType.WRONG_DOSE,
        )
        assert modified != text
        assert "dose changed" in details

    def test_no_dose_returns_unchanged(self, injector, ehr):
        text = "Patient takes Metoprolol daily"
        modified, details = injector.inject_error(
            text, "medication", ehr, ErrorType.WRONG_DOSE,
        )
        assert modified == text
        assert "no dose found" in details

    def test_units_preserved(self, injector, ehr):
        text = "Insulin glargine 20 units at bedtime"
        modified, details = injector.inject_error(
            text, "medication", ehr, ErrorType.WRONG_DOSE,
        )
        assert "units" in modified


class TestWrongFrequency:
    def test_swaps_daily(self, injector, ehr):
        text = "Metoprolol 50mg daily"
        modified, details = injector.inject_error(
            text, "medication", ehr, ErrorType.WRONG_FREQUENCY,
        )
        assert "daily" not in modified.lower() or "BID" in modified
        assert "frequency changed" in details

    def test_swaps_bid(self, injector, ehr):
        text = "Metformin 500mg BID"
        modified, details = injector.inject_error(
            text, "medication", ehr, ErrorType.WRONG_FREQUENCY,
        )
        assert "TID" in modified
        assert "frequency changed" in details

    def test_no_frequency_returns_unchanged(self, injector, ehr):
        text = "Patient takes Aspirin"
        modified, details = injector.inject_error(
            text, "medication", ehr, ErrorType.WRONG_FREQUENCY,
        )
        assert modified == text
        assert "no frequency found" in details


class TestWrongMedication:
    def test_swaps_known_drug(self, injector, ehr):
        text = "Patient takes Metformin 500mg BID"
        modified, details = injector.inject_error(
            text, "medication", ehr, ErrorType.WRONG_MEDICATION,
        )
        assert "metoprolol" in modified.lower()
        assert "drug swapped" in details

    def test_preserves_case(self, injector, ehr):
        text = "Lisinopril 10mg daily"
        modified, details = injector.inject_error(
            text, "medication", ehr, ErrorType.WRONG_MEDICATION,
        )
        # Original starts uppercase, so replacement should too
        assert modified[0].isupper()

    def test_unknown_drug_returns_unchanged(self, injector, ehr):
        text = "Patient takes ZZZdrug 100mg"
        modified, details = injector.inject_error(
            text, "medication", ehr, ErrorType.WRONG_MEDICATION,
        )
        assert modified == text
        assert "no known drug found" in details


class TestContradictsEHR:
    def test_modifies_medication_dose(self, injector, ehr):
        text = "Metoprolol 50mg daily"
        modified, details = injector.inject_error(
            text, "medication", ehr, ErrorType.CONTRADICTS_EHR,
        )
        assert "contradicts EHR" in details

    def test_uses_ehr_medications(self, injector, custom_ehr):
        text = "Warfarin 5mg daily"
        modified, details = injector.inject_error(
            text, "medication", custom_ehr, ErrorType.CONTRADICTS_EHR,
        )
        assert "contradicts EHR" in details

    def test_empty_ehr_returns_unchanged(self, injector):
        empty_ehr = MockEHRContext()
        text = "Metoprolol 50mg daily"
        modified, details = injector.inject_error(
            text, "medication", empty_ehr, ErrorType.CONTRADICTS_EHR,
        )
        assert "no EHR contradiction possible" in details


class TestHallucinatedAllergy:
    def test_generates_allergy(self, injector, ehr):
        text = "No known allergies"
        modified, details = injector.inject_error(
            text, "allergy", ehr, ErrorType.HALLUCINATED_ALLERGY,
        )
        assert "Patient allergic to" in modified
        assert "hallucinated allergy" in details

    def test_avoids_existing_allergies(self, injector, ehr):
        # Run many times — the chosen allergy should never be one already in EHR
        known_lower = {a.lower() for a in ehr.allergies}
        for seed in range(50):
            inj = ErrorInjector(seed=seed)
            modified, _ = inj.inject_error(
                "No allergies", "allergy", ehr, ErrorType.HALLUCINATED_ALLERGY,
            )
            # Extract the allergy name
            allergy = modified.replace("Patient allergic to ", "")
            assert allergy.lower() not in known_lower, \
                f"seed={seed} picked existing allergy: {allergy}"


class TestHallucinatedSymptom:
    def test_generates_symptom(self, injector, ehr):
        text = "Patient doing well"
        modified, details = injector.inject_error(
            text, "diagnosis", ehr, ErrorType.HALLUCINATED_SYMPTOM,
        )
        assert "Patient reports" in modified
        assert "hallucinated symptom" in details

    def test_symptom_from_list(self, injector, ehr):
        _, details = injector.inject_error(
            "text", "diagnosis", ehr, ErrorType.HALLUCINATED_SYMPTOM,
        )
        symptom = details.replace("hallucinated symptom: ", "")
        assert symptom in FAKE_SYMPTOMS


class TestWrongVital:
    def test_modifies_bp(self, injector, ehr):
        text = "Blood pressure 120/80 mmHg"
        modified, details = injector.inject_error(
            text, "vital_sign", ehr, ErrorType.WRONG_VITAL,
        )
        assert modified != text
        assert "vital" in details.lower()

    def test_modifies_hr(self, injector, ehr):
        text = "Heart rate of 72 bpm"
        modified, details = injector.inject_error(
            text, "vital_sign", ehr, ErrorType.WRONG_VITAL,
        )
        assert modified != text

    def test_no_vital_detected_uses_fallback(self, injector, ehr):
        text = "Patient stable"
        modified, details = injector.inject_error(
            text, "vital_sign", ehr, ErrorType.WRONG_VITAL,
        )
        # Should still produce a vital
        assert "vital replaced" in details or "vital changed" in details


# ═════════════════════════════════════════════════════════════════════════════
# DETERMINISM
# ═════════════════════════════════════════════════════════════════════════════

class TestDeterminism:
    def test_same_seed_same_results(self, ehr):
        claims = [
            ("Metoprolol 50mg daily", "medication"),
            ("Patient allergic to Latex", "allergy"),
            ("Blood pressure 130/85 mmHg", "vital_sign"),
            ("Type 2 diabetes mellitus", "diagnosis"),
        ]
        inj1 = ErrorInjector(seed=123)
        inj2 = ErrorInjector(seed=123)

        r1 = inj1.inject_random_errors(claims, ehr, error_rate=0.50)
        r2 = inj2.inject_random_errors(claims, ehr, error_rate=0.50)

        assert len(r1) == len(r2)
        for a, b in zip(r1, r2):
            assert a.is_error == b.is_error
            assert a.modified_text == b.modified_text
            assert a.error_type == b.error_type
            assert a.error_details == b.error_details

    def test_different_seed_different_results(self, ehr):
        claims = [("Metoprolol 50mg daily", "medication")] * 10
        r1 = ErrorInjector(seed=1).inject_random_errors(claims, ehr, error_rate=0.50)
        r2 = ErrorInjector(seed=2).inject_random_errors(claims, ehr, error_rate=0.50)

        # At least one difference (statistically near-certain with 10 claims)
        diffs = sum(1 for a, b in zip(r1, r2)
                    if a.is_error != b.is_error or a.modified_text != b.modified_text)
        assert diffs > 0


# ═════════════════════════════════════════════════════════════════════════════
# INJECT RANDOM ERRORS
# ═════════════════════════════════════════════════════════════════════════════

class TestInjectRandomErrors:
    def test_returns_correct_count(self, injector, ehr):
        claims = [
            ("Metoprolol 50mg daily", "medication"),
            ("Patient allergic to Latex", "allergy"),
            ("Blood pressure 130/85 mmHg", "vital_sign"),
        ]
        results = injector.inject_random_errors(claims, ehr, error_rate=0.50)
        assert len(results) == 3

    def test_all_results_are_injected_claims(self, injector, ehr):
        claims = [("Metoprolol 50mg daily", "medication")]
        results = injector.inject_random_errors(claims, ehr, error_rate=1.0)
        assert all(isinstance(r, InjectedClaim) for r in results)

    def test_error_rate_zero_means_no_errors(self, ehr):
        claims = [("claim text", "medication")] * 20
        results = ErrorInjector(seed=0).inject_random_errors(
            claims, ehr, error_rate=0.0,
        )
        assert all(not r.is_error for r in results)

    def test_error_rate_one_means_all_errors(self, ehr):
        claims = [("Metoprolol 50mg daily", "medication")] * 10
        results = ErrorInjector(seed=0).inject_random_errors(
            claims, ehr, error_rate=1.0,
        )
        assert all(r.is_error for r in results)

    def test_approximate_error_rate(self, ehr):
        claims = [("Metoprolol 50mg daily", "medication")] * 200
        results = ErrorInjector(seed=42).inject_random_errors(
            claims, ehr, error_rate=0.30,
        )
        actual_rate = sum(r.is_error for r in results) / len(results)
        assert 0.15 <= actual_rate <= 0.45, f"actual_rate={actual_rate}"

    def test_non_error_claims_unchanged(self, injector, ehr):
        claims = [("Metoprolol 50mg daily", "medication")] * 10
        results = injector.inject_random_errors(claims, ehr, error_rate=0.0)
        for r in results:
            assert r.modified_text == r.original_text
            assert r.error_type is None
            assert r.error_details == ""

    def test_error_claims_have_details(self, ehr):
        claims = [("Metoprolol 50mg daily", "medication")] * 10
        results = ErrorInjector(seed=0).inject_random_errors(
            claims, ehr, error_rate=1.0,
        )
        for r in results:
            assert r.is_error
            assert r.error_type is not None
            assert r.error_details != ""


# ═════════════════════════════════════════════════════════════════════════════
# CLAIM TYPE COMPATIBILITY
# ═════════════════════════════════════════════════════════════════════════════

class TestCompatibility:
    def test_medication_gets_medication_errors(self, ehr):
        for seed in range(20):
            inj = ErrorInjector(seed=seed)
            results = inj.inject_random_errors(
                [("Metoprolol 50mg daily", "medication")],
                ehr, error_rate=1.0,
            )
            assert results[0].error_type in _COMPATIBLE_ERRORS["medication"]

    def test_allergy_gets_allergy_errors(self, ehr):
        for seed in range(20):
            inj = ErrorInjector(seed=seed)
            results = inj.inject_random_errors(
                [("Allergic to Penicillin", "allergy")],
                ehr, error_rate=1.0,
            )
            assert results[0].error_type in _COMPATIBLE_ERRORS["allergy"]

    def test_vital_sign_gets_vital_errors(self, ehr):
        for seed in range(20):
            inj = ErrorInjector(seed=seed)
            results = inj.inject_random_errors(
                [("Blood pressure 120/80", "vital_sign")],
                ehr, error_rate=1.0,
            )
            assert results[0].error_type in _COMPATIBLE_ERRORS["vital_sign"]

    def test_diagnosis_gets_diagnosis_errors(self, ehr):
        for seed in range(20):
            inj = ErrorInjector(seed=seed)
            results = inj.inject_random_errors(
                [("Type 2 diabetes mellitus", "diagnosis")],
                ehr, error_rate=1.0,
            )
            assert results[0].error_type in _COMPATIBLE_ERRORS["diagnosis"]

    def test_unknown_claim_type_uses_all_errors(self, ehr):
        """Unknown claim types should fall back to any ErrorType."""
        inj = ErrorInjector(seed=42)
        results = inj.inject_random_errors(
            [("some claim", "unknown_type")],
            ehr, error_rate=1.0,
        )
        assert results[0].error_type in ErrorType


# ═════════════════════════════════════════════════════════════════════════════
# GENERATE TRAINING SET FROM NOTES
# ═════════════════════════════════════════════════════════════════════════════

class TestGenerateTrainingSet:
    @pytest.fixture
    def sample_notes(self):
        return [
            NoteWithClaims(
                note_text="Patient presents with controlled hypertension. "
                          "Currently on Metoprolol 50mg daily and Lisinopril 10mg daily.",
                claims=[
                    ("Metoprolol 50mg daily", "medication"),
                    ("Lisinopril 10mg daily", "medication"),
                    ("Blood pressure 120/80 mmHg", "vital_sign"),
                    ("Patient allergic to Penicillin", "allergy"),
                ],
            ),
            NoteWithClaims(
                note_text="Follow-up visit for diabetes management. "
                          "Metformin 500mg BID, A1c improved.",
                claims=[
                    ("Metformin 500mg BID", "medication"),
                    ("Type 2 diabetes mellitus", "diagnosis"),
                    ("Heart rate of 72 bpm", "vital_sign"),
                ],
            ),
        ]

    @pytest.fixture
    def ehr_contexts(self):
        return [DEFAULT_EHR_CONTEXT]

    def test_returns_training_examples(self, sample_notes, ehr_contexts):
        examples = generate_training_set_from_notes(
            sample_notes, ehr_contexts, error_rate=0.30, seed=42,
        )
        assert len(examples) == 7  # 4 + 3 claims
        assert all(isinstance(e, TrainingExample) for e in examples)

    def test_training_examples_have_valid_fields(self, sample_notes, ehr_contexts):
        examples = generate_training_set_from_notes(
            sample_notes, ehr_contexts, error_rate=0.50, seed=42,
        )
        for ex in examples:
            assert ex.claim_type in ("medication", "allergy", "diagnosis", "vital_sign")
            assert ex.ehr_status in ("verified", "contradiction", "not_found", "not_checkable")
            assert 0.0 <= ex.risk_score <= 1.0
            assert ex.review_level in ("BRIEF", "STANDARD", "DETAILED", "CRITICAL")

    def test_error_rate_zero_all_verified(self, sample_notes, ehr_contexts):
        examples = generate_training_set_from_notes(
            sample_notes, ehr_contexts, error_rate=0.0, seed=42,
        )
        for ex in examples:
            assert ex.ehr_status == "verified"
            assert ex.review_level == "BRIEF"
            assert ex.risk_score < 0.15

    def test_error_rate_one_all_errored(self, sample_notes, ehr_contexts):
        examples = generate_training_set_from_notes(
            sample_notes, ehr_contexts, error_rate=1.0, seed=42,
        )
        for ex in examples:
            assert ex.ehr_status in ("contradiction", "not_found")
            assert ex.risk_score > 0.50

    def test_deterministic(self, sample_notes, ehr_contexts):
        e1 = generate_training_set_from_notes(
            sample_notes, ehr_contexts, error_rate=0.30, seed=99,
        )
        e2 = generate_training_set_from_notes(
            sample_notes, ehr_contexts, error_rate=0.30, seed=99,
        )
        assert len(e1) == len(e2)
        for a, b in zip(e1, e2):
            assert a.claim_text == b.claim_text
            assert a.risk_score == b.risk_score
            assert a.ehr_status == b.ehr_status

    def test_source_sentence_truncated(self, ehr_contexts):
        long_note = "A" * 200 + " with claims."
        notes = [NoteWithClaims(
            note_text=long_note,
            claims=[("Metoprolol 50mg daily", "medication")],
        )]
        examples = generate_training_set_from_notes(
            notes, ehr_contexts, error_rate=0.0, seed=42,
        )
        assert len(examples[0].source_sentence) <= 120

    def test_empty_notes_returns_empty(self, ehr_contexts):
        examples = generate_training_set_from_notes(
            [], ehr_contexts, error_rate=0.30, seed=42,
        )
        assert examples == []

    def test_ehr_context_cycling(self):
        """When there are more notes than ehr_contexts, contexts cycle."""
        notes = [
            NoteWithClaims(note_text="Note 1", claims=[("claim1", "medication")]),
            NoteWithClaims(note_text="Note 2", claims=[("claim2", "medication")]),
            NoteWithClaims(note_text="Note 3", claims=[("claim3", "medication")]),
        ]
        contexts = [DEFAULT_EHR_CONTEXT]
        examples = generate_training_set_from_notes(
            notes, contexts, error_rate=0.0, seed=42,
        )
        assert len(examples) == 3


# ═════════════════════════════════════════════════════════════════════════════
# EDGE CASES
# ═════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_empty_claim_list(self, injector, ehr):
        results = injector.inject_random_errors([], ehr, error_rate=0.50)
        assert results == []

    def test_single_claim(self, injector, ehr):
        results = injector.inject_random_errors(
            [("Metoprolol 50mg daily", "medication")],
            ehr, error_rate=1.0,
        )
        assert len(results) == 1
        assert results[0].is_error

    def test_very_large_dose(self, injector, ehr):
        text = "Patient takes 99999mg of something"
        modified, details = injector.inject_error(
            text, "medication", ehr, ErrorType.WRONG_DOSE,
        )
        assert "dose changed" in details

    def test_units_variant_mcg(self, injector, ehr):
        text = "Levothyroxine 75mcg daily"
        modified, details = injector.inject_error(
            text, "medication", ehr, ErrorType.WRONG_DOSE,
        )
        assert "mcg" in modified
        assert "dose changed" in details

    def test_inject_error_with_unknown_type(self, injector, ehr):
        """Handler dispatch returns unchanged for unknown error type (defensive)."""
        # We can't really pass an unknown ErrorType since it's an enum,
        # but we can verify the handler table covers all types
        from layer2.data.synthetic_error_injector import _ERROR_HANDLERS
        for et in ErrorType:
            assert et in _ERROR_HANDLERS, f"missing handler for {et}"
