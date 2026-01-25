#!/usr/bin/env python3
"""
Test Cases Generator
====================

Generates 100 deterministic medical claim test cases for comparing
OLD platform (SE-only) vs NEW platform (conformal prediction).

All cases are generated with seed=42 for reproducibility.
"""

import random
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from enum import Enum
import json


class ClaimType(str, Enum):
    """Types of verifiable claims."""
    MEDICATION = "medication"
    ALLERGY = "allergy"
    DIAGNOSIS = "diagnosis"
    VITAL_SIGN = "vital_sign"
    LAB_RESULT = "lab_result"
    PROCEDURE = "procedure"
    HISTORY = "history"
    DEMOGRAPHIC = "demographic"


@dataclass
class TestCase:
    """A single test case for platform comparison."""
    case_id: str
    claim_text: str
    claim_type: ClaimType
    ehr_record: Dict[str, Any]  # What's in the "EHR"
    ehr_data_available: bool  # Whether EHR has data for this claim
    ground_truth_correct: bool  # Is the claim factually correct?
    multiple_responses: List[str]  # 5 simulated LLM responses for SE
    expected_ehr_status: str  # verified/contradiction/not_found
    difficulty: str  # easy/medium/hard - affects SE variance

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['claim_type'] = self.claim_type.value
        return d


# =============================================================================
# MEDICAL DATA TEMPLATES
# =============================================================================

MEDICATIONS = [
    ("Metoprolol", "50mg", "BID", "beta blocker"),
    ("Lisinopril", "10mg", "daily", "ACE inhibitor"),
    ("Metformin", "500mg", "BID", "diabetes medication"),
    ("Atorvastatin", "20mg", "daily", "statin"),
    ("Aspirin", "81mg", "daily", "antiplatelet"),
    ("Omeprazole", "20mg", "daily", "PPI"),
    ("Amlodipine", "5mg", "daily", "calcium channel blocker"),
    ("Gabapentin", "300mg", "TID", "neuropathic pain"),
    ("Levothyroxine", "75mcg", "daily", "thyroid hormone"),
    ("Sertraline", "50mg", "daily", "SSRI"),
    ("Furosemide", "40mg", "daily", "diuretic"),
    ("Warfarin", "5mg", "daily", "anticoagulant"),
    ("Prednisone", "10mg", "daily", "corticosteroid"),
    ("Albuterol", "2 puffs", "PRN", "bronchodilator"),
    ("Insulin glargine", "20 units", "at bedtime", "basal insulin"),
]

ALLERGIES = [
    ("Penicillin", "rash"),
    ("Sulfa drugs", "anaphylaxis"),
    ("Codeine", "nausea"),
    ("Latex", "contact dermatitis"),
    ("Iodine contrast", "hives"),
    ("NSAIDs", "GI bleeding"),
    ("Shellfish", "anaphylaxis"),
    ("Peanuts", "throat swelling"),
]

DIAGNOSES = [
    ("Hypertension", "I10"),
    ("Type 2 Diabetes", "E11.9"),
    ("Hyperlipidemia", "E78.5"),
    ("Atrial fibrillation", "I48.91"),
    ("COPD", "J44.9"),
    ("Heart failure", "I50.9"),
    ("Chronic kidney disease stage 3", "N18.3"),
    ("Osteoarthritis", "M19.90"),
    ("Depression", "F32.9"),
    ("Hypothyroidism", "E03.9"),
]

VITAL_SIGNS = [
    ("blood pressure", "mmHg", (90, 180), (60, 110)),
    ("heart rate", "bpm", (50, 120), None),
    ("temperature", "°F", (97.0, 102.0), None),
    ("respiratory rate", "breaths/min", (12, 24), None),
    ("oxygen saturation", "%", (88, 100), None),
    ("weight", "kg", (50, 150), None),
]

LAB_RESULTS = [
    ("hemoglobin", "g/dL", (8.0, 18.0)),
    ("WBC", "K/uL", (3.0, 15.0)),
    ("creatinine", "mg/dL", (0.5, 3.0)),
    ("glucose", "mg/dL", (70, 400)),
    ("potassium", "mEq/L", (3.0, 6.0)),
    ("sodium", "mEq/L", (130, 150)),
    ("BUN", "mg/dL", (8, 40)),
    ("HbA1c", "%", (5.0, 12.0)),
]


# =============================================================================
# RESPONSE GENERATION FOR SEMANTIC ENTROPY
# =============================================================================

def generate_consistent_responses(claim_text: str, correct: bool, difficulty: str) -> List[str]:
    """
    Generate 5 simulated LLM responses for semantic entropy calculation.

    For SE calculation:
    - Consistent responses (all agree) → LOW entropy
    - Mixed responses → MEDIUM entropy
    - Highly varied responses → HIGH entropy

    difficulty affects how consistent the responses are:
    - easy: responses are very consistent
    - medium: some variation
    - hard: high variation
    """
    random_state = random.getstate()

    if difficulty == "easy":
        # Easy cases: high agreement
        if correct:
            responses = [
                f"Verified: {claim_text}",
                f"Confirmed: {claim_text}",
                f"The claim '{claim_text}' is accurate",
                f"This matches the record: {claim_text}",
                f"Correct: {claim_text}",
            ]
        else:
            responses = [
                f"Incorrect: {claim_text}",
                f"Contradiction found: {claim_text}",
                f"The claim '{claim_text}' conflicts with records",
                f"This does not match: {claim_text}",
                f"Error: {claim_text}",
            ]
    elif difficulty == "medium":
        # Medium cases: some variation
        base = "verified" if correct else "contradiction"
        variations = [
            f"{base}: {claim_text}",
            f"Likely {base}: {claim_text}",
            f"Appears to be {base}: {claim_text}",
            f"Review needed but probably {base}: {claim_text}",
            f"Mostly {base}: {claim_text}" if random.random() > 0.3 else f"Uncertain: {claim_text}",
        ]
        responses = variations
    else:  # hard
        # Hard cases: high uncertainty, mixed responses
        if correct:
            responses = [
                f"Verified: {claim_text}",
                f"Uncertain: {claim_text}",
                f"Possibly correct: {claim_text}",
                f"Cannot confirm: {claim_text}",
                f"Likely correct: {claim_text}",
            ]
        else:
            responses = [
                f"Contradiction: {claim_text}",
                f"Uncertain: {claim_text}",
                f"Possibly incorrect: {claim_text}",
                f"Cannot determine: {claim_text}",
                f"Review needed: {claim_text}",
            ]

    random.setstate(random_state)
    return responses


# =============================================================================
# CASE GENERATORS BY TYPE
# =============================================================================

def generate_medication_case(case_num: int, rng: random.Random) -> TestCase:
    """Generate a medication-related test case."""
    med_name, dose, freq, med_class = rng.choice(MEDICATIONS)

    # Decide if claim is correct or has an error
    is_correct = rng.random() > 0.3  # 70% correct claims
    ehr_available = rng.random() > 0.15  # 85% have EHR data

    if is_correct:
        claim_text = f"Patient takes {med_name} {dose} {freq}"
        ehr_dose = dose
        expected_status = "verified" if ehr_available else "not_found"
    else:
        # Introduce an error
        error_type = rng.choice(["dose", "frequency", "wrong_med"])
        if error_type == "dose":
            wrong_dose = f"{int(dose.split('m')[0]) * 2}mg"
            claim_text = f"Patient takes {med_name} {wrong_dose} {freq}"
            ehr_dose = dose
        elif error_type == "frequency":
            wrong_freq = "TID" if freq == "BID" else "BID"
            claim_text = f"Patient takes {med_name} {dose} {wrong_freq}"
            ehr_dose = dose
        else:
            other_med = rng.choice([m[0] for m in MEDICATIONS if m[0] != med_name])
            claim_text = f"Patient takes {other_med} {dose} {freq}"
            ehr_dose = dose
        expected_status = "contradiction" if ehr_available else "not_found"

    difficulty = rng.choice(["easy", "easy", "medium", "medium", "hard"])

    return TestCase(
        case_id=f"MED_{case_num:03d}",
        claim_text=claim_text,
        claim_type=ClaimType.MEDICATION,
        ehr_record={
            "medication_name": med_name,
            "dose": ehr_dose,
            "frequency": freq,
            "class": med_class,
        } if ehr_available else {},
        ehr_data_available=ehr_available,
        ground_truth_correct=is_correct,
        multiple_responses=generate_consistent_responses(claim_text, is_correct, difficulty),
        expected_ehr_status=expected_status,
        difficulty=difficulty,
    )


def generate_allergy_case(case_num: int, rng: random.Random) -> TestCase:
    """Generate an allergy-related test case."""
    allergen, reaction = rng.choice(ALLERGIES)

    is_correct = rng.random() > 0.25  # 75% correct
    ehr_available = rng.random() > 0.1  # 90% have allergy data

    if is_correct:
        claim_text = f"Patient is allergic to {allergen} causing {reaction}"
        expected_status = "verified" if ehr_available else "not_found"
    else:
        # Wrong reaction or wrong allergen
        if rng.random() > 0.5:
            wrong_reaction = rng.choice(["rash", "nausea", "hives"])
            claim_text = f"Patient is allergic to {allergen} causing {wrong_reaction}"
        else:
            other_allergen = rng.choice([a[0] for a in ALLERGIES if a[0] != allergen])
            claim_text = f"Patient is allergic to {other_allergen}"
        expected_status = "contradiction" if ehr_available else "not_found"

    difficulty = rng.choice(["easy", "medium", "hard"])

    return TestCase(
        case_id=f"ALG_{case_num:03d}",
        claim_text=claim_text,
        claim_type=ClaimType.ALLERGY,
        ehr_record={
            "allergen": allergen,
            "reaction": reaction,
        } if ehr_available else {},
        ehr_data_available=ehr_available,
        ground_truth_correct=is_correct,
        multiple_responses=generate_consistent_responses(claim_text, is_correct, difficulty),
        expected_ehr_status=expected_status,
        difficulty=difficulty,
    )


def generate_diagnosis_case(case_num: int, rng: random.Random) -> TestCase:
    """Generate a diagnosis-related test case."""
    diagnosis, icd_code = rng.choice(DIAGNOSES)

    is_correct = rng.random() > 0.2  # 80% correct
    ehr_available = rng.random() > 0.1

    if is_correct:
        claim_text = f"Patient has history of {diagnosis}"
        expected_status = "verified" if ehr_available else "not_found"
    else:
        other_dx = rng.choice([d[0] for d in DIAGNOSES if d[0] != diagnosis])
        claim_text = f"Patient has history of {other_dx}"
        expected_status = "contradiction" if ehr_available else "not_found"

    difficulty = rng.choice(["easy", "medium", "medium", "hard"])

    return TestCase(
        case_id=f"DX_{case_num:03d}",
        claim_text=claim_text,
        claim_type=ClaimType.DIAGNOSIS,
        ehr_record={
            "diagnosis": diagnosis,
            "icd_code": icd_code,
        } if ehr_available else {},
        ehr_data_available=ehr_available,
        ground_truth_correct=is_correct,
        multiple_responses=generate_consistent_responses(claim_text, is_correct, difficulty),
        expected_ehr_status=expected_status,
        difficulty=difficulty,
    )


def generate_vital_sign_case(case_num: int, rng: random.Random) -> TestCase:
    """Generate a vital sign-related test case."""
    vital_name, unit, systolic_range, diastolic_range = rng.choice(VITAL_SIGNS)

    is_correct = rng.random() > 0.25
    ehr_available = rng.random() > 0.05  # 95% have vitals

    if vital_name == "blood pressure":
        true_systolic = rng.randint(*systolic_range)
        true_diastolic = rng.randint(*diastolic_range)
        true_value = f"{true_systolic}/{true_diastolic}"

        if is_correct:
            claim_value = true_value
        else:
            claim_value = f"{true_systolic + rng.randint(10, 30)}/{true_diastolic + rng.randint(5, 15)}"
    else:
        if vital_name == "temperature":
            true_value = round(rng.uniform(*systolic_range), 1)
        else:
            true_value = rng.randint(int(systolic_range[0]), int(systolic_range[1]))

        if is_correct:
            claim_value = true_value
        else:
            if isinstance(true_value, float):
                claim_value = round(true_value + rng.uniform(1, 3), 1)
            else:
                claim_value = true_value + rng.randint(10, 30)

    claim_text = f"{vital_name.title()} is {claim_value} {unit}"
    expected_status = "verified" if (is_correct and ehr_available) else (
        "contradiction" if ehr_available else "not_found"
    )

    difficulty = rng.choice(["easy", "easy", "medium"])

    return TestCase(
        case_id=f"VS_{case_num:03d}",
        claim_text=claim_text,
        claim_type=ClaimType.VITAL_SIGN,
        ehr_record={
            "vital_name": vital_name,
            "value": true_value,
            "unit": unit,
        } if ehr_available else {},
        ehr_data_available=ehr_available,
        ground_truth_correct=is_correct,
        multiple_responses=generate_consistent_responses(claim_text, is_correct, difficulty),
        expected_ehr_status=expected_status,
        difficulty=difficulty,
    )


def generate_lab_result_case(case_num: int, rng: random.Random) -> TestCase:
    """Generate a lab result-related test case."""
    lab_name, unit, value_range = rng.choice(LAB_RESULTS)

    is_correct = rng.random() > 0.3
    ehr_available = rng.random() > 0.1

    true_value = round(rng.uniform(*value_range), 1)

    if is_correct:
        claim_value = true_value
    else:
        # Introduce error
        claim_value = round(true_value * rng.uniform(1.2, 1.5), 1)

    claim_text = f"{lab_name.title()} is {claim_value} {unit}"
    expected_status = "verified" if (is_correct and ehr_available) else (
        "contradiction" if ehr_available else "not_found"
    )

    difficulty = rng.choice(["easy", "medium", "medium", "hard"])

    return TestCase(
        case_id=f"LAB_{case_num:03d}",
        claim_text=claim_text,
        claim_type=ClaimType.LAB_RESULT,
        ehr_record={
            "lab_name": lab_name,
            "value": true_value,
            "unit": unit,
        } if ehr_available else {},
        ehr_data_available=ehr_available,
        ground_truth_correct=is_correct,
        multiple_responses=generate_consistent_responses(claim_text, is_correct, difficulty),
        expected_ehr_status=expected_status,
        difficulty=difficulty,
    )


# =============================================================================
# MAIN GENERATOR
# =============================================================================

def generate_test_cases(n_cases: int = 100, seed: int = 42) -> List[TestCase]:
    """
    Generate n_cases deterministic test cases.

    Distribution:
    - 30% medication claims
    - 15% allergy claims
    - 20% diagnosis claims
    - 20% vital sign claims
    - 15% lab result claims
    """
    rng = random.Random(seed)
    cases = []

    # Distribution
    n_medication = int(n_cases * 0.30)
    n_allergy = int(n_cases * 0.15)
    n_diagnosis = int(n_cases * 0.20)
    n_vital = int(n_cases * 0.20)
    n_lab = n_cases - n_medication - n_allergy - n_diagnosis - n_vital

    case_num = 0

    # Generate medication cases
    for i in range(n_medication):
        cases.append(generate_medication_case(case_num, rng))
        case_num += 1

    # Generate allergy cases
    for i in range(n_allergy):
        cases.append(generate_allergy_case(case_num, rng))
        case_num += 1

    # Generate diagnosis cases
    for i in range(n_diagnosis):
        cases.append(generate_diagnosis_case(case_num, rng))
        case_num += 1

    # Generate vital sign cases
    for i in range(n_vital):
        cases.append(generate_vital_sign_case(case_num, rng))
        case_num += 1

    # Generate lab result cases
    for i in range(n_lab):
        cases.append(generate_lab_result_case(case_num, rng))
        case_num += 1

    # Shuffle deterministically
    rng.shuffle(cases)

    # Re-assign case IDs after shuffle
    for i, case in enumerate(cases):
        case.case_id = f"CASE_{i:03d}"

    return cases


def save_test_cases(cases: List[TestCase], filepath: str) -> None:
    """Save test cases to JSON file."""
    data = [case.to_dict() for case in cases]
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_test_cases(filepath: str) -> List[TestCase]:
    """Load test cases from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    cases = []
    for d in data:
        d['claim_type'] = ClaimType(d['claim_type'])
        cases.append(TestCase(**d))
    return cases


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys

    print("Generating 100 test cases with seed=42...")
    cases = generate_test_cases(n_cases=100, seed=42)

    # Summary
    print(f"\nGenerated {len(cases)} test cases:")

    # Count by type
    type_counts = {}
    for case in cases:
        t = case.claim_type.value
        type_counts[t] = type_counts.get(t, 0) + 1

    for claim_type, count in sorted(type_counts.items()):
        print(f"  {claim_type}: {count}")

    # Count correct vs incorrect
    n_correct = sum(1 for c in cases if c.ground_truth_correct)
    n_incorrect = len(cases) - n_correct
    print(f"\nGround truth: {n_correct} correct, {n_incorrect} incorrect")

    # Count EHR availability
    n_ehr = sum(1 for c in cases if c.ehr_data_available)
    print(f"EHR data available: {n_ehr}/{len(cases)}")

    # Count expected statuses
    status_counts = {}
    for case in cases:
        s = case.expected_ehr_status
        status_counts[s] = status_counts.get(s, 0) + 1

    print(f"\nExpected EHR statuses:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")

    # Save
    output_path = "experiments/conformal_comparison/test_cases.json"
    save_test_cases(cases, output_path)
    print(f"\nSaved to {output_path}")

    # Show first 3 cases
    print("\nFirst 3 cases:")
    for case in cases[:3]:
        print(f"\n{case.case_id}:")
        print(f"  Claim: {case.claim_text}")
        print(f"  Type: {case.claim_type.value}")
        print(f"  Correct: {case.ground_truth_correct}")
        print(f"  EHR available: {case.ehr_data_available}")
        print(f"  Expected status: {case.expected_ehr_status}")
