"""
Synthetic error injector for generating training data when IHID
doesn't have AI scribe data yet.

Takes correct clinical claims and introduces realistic errors at a
controlled rate, producing labelled (claim, is_error, error_type) tuples
suitable for training the Layer 2 meta-classifier.

All operations are deterministic given a seed.
"""

import logging
import random
import re
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Tuple

from .mock_data_generator import TrainingExample

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# ERROR TYPES
# ═════════════════════════════════════════════════════════════════════════════

class ErrorType(str, Enum):
    """Categories of synthetic errors that can be injected into claims."""
    WRONG_DOSE = "wrong_dose"
    WRONG_FREQUENCY = "wrong_frequency"
    WRONG_MEDICATION = "wrong_medication"
    CONTRADICTS_EHR = "contradicts_ehr"
    HALLUCINATED_ALLERGY = "hallucinated_allergy"
    HALLUCINATED_SYMPTOM = "hallucinated_symptom"
    WRONG_VITAL = "wrong_vital"


# ═════════════════════════════════════════════════════════════════════════════
# SUBSTITUTION TABLES
# ═════════════════════════════════════════════════════════════════════════════

DOSE_MULTIPLIERS = [0.5, 2.0, 5.0, 10.0]

FREQUENCY_SWAPS: Dict[str, str] = {
    "daily": "BID",
    "once daily": "BID",
    "BID": "TID",
    "twice daily": "TID",
    "TID": "QID",
    "three times daily": "QID",
    "QID": "daily",
    "four times daily": "daily",
    "at bedtime": "BID",
    "nightly": "BID",
    "every 12 hours": "every 8 hours",
    "every 8 hours": "every 6 hours",
    "every 6 hours": "every 4 hours",
    "every 4 hours": "every 12 hours",
    "weekly": "daily",
    "once weekly": "daily",
    "PRN": "TID",
    "as needed": "TID",
}

SIMILAR_DRUGS: Dict[str, str] = {
    "metformin": "metoprolol",
    "metoprolol": "metformin",
    "lisinopril": "losartan",
    "losartan": "lisinopril",
    "amlodipine": "amiodarone",
    "amiodarone": "amlodipine",
    "atorvastatin": "rosuvastatin",
    "rosuvastatin": "atorvastatin",
    "omeprazole": "esomeprazole",
    "esomeprazole": "omeprazole",
    "sertraline": "citalopram",
    "citalopram": "sertraline",
    "gabapentin": "pregabalin",
    "pregabalin": "gabapentin",
    "warfarin": "apixaban",
    "apixaban": "warfarin",
    "hydrochlorothiazide": "furosemide",
    "furosemide": "hydrochlorothiazide",
    "prednisone": "prednisolone",
    "prednisolone": "prednisone",
    "insulin glargine": "insulin lispro",
    "insulin lispro": "insulin glargine",
}

ABNORMAL_VITALS: Dict[str, List[str]] = {
    "BP": ["185/110", "210/120", "85/50", "220/140"],
    "HR": ["145", "38", "180", "28"],
    "temp": ["39.5°C", "40.2°C", "34.8°C", "41.0°C"],
    "SpO2": ["78%", "82%", "70%", "65%"],
    "RR": ["32", "6", "40", "4"],
}

# Canonical vital sign regex patterns for detection
_VITAL_PATTERNS = {
    "BP": re.compile(r"\b\d{2,3}/\d{2,3}\b"),
    "HR": re.compile(r"\b(?:HR|heart rate|pulse)\s*(?:of|at|:)?\s*(\d{2,3})\s*(?:bpm)?", re.IGNORECASE),
    "temp": re.compile(r"\b(\d{2}\.\d)°?[CF]?\b"),
    "SpO2": re.compile(r"\b(?:SpO2|O2 sat|oxygen sat(?:uration)?)\s*(?:of|at|:)?\s*(\d{2,3})%?", re.IGNORECASE),
    "RR": re.compile(r"\b(?:RR|respiratory rate)\s*(?:of|at|:)?\s*(\d{1,2})\b", re.IGNORECASE),
}

FAKE_ALLERGIES = [
    "penicillin", "sulfa", "codeine", "aspirin", "iodine",
    "latex", "cephalosporins", "fluoroquinolones", "NSAIDs",
    "tetracycline", "erythromycin", "vancomycin",
]

FAKE_SYMPTOMS = [
    "chest pain", "shortness of breath", "syncope",
    "hemoptysis", "severe headache", "vision changes",
    "numbness in left arm", "palpitations", "diaphoresis",
    "altered mental status", "abdominal rigidity", "melena",
]


# ═════════════════════════════════════════════════════════════════════════════
# MOCK EHR CONTEXT
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class MockEHRContext:
    """Simulated EHR record for testing error injection."""
    medications: List[str] = field(default_factory=list)
    allergies: List[str] = field(default_factory=list)
    vitals: Dict[str, str] = field(default_factory=dict)


DEFAULT_EHR_CONTEXT = MockEHRContext(
    medications=[
        "Metoprolol 50mg daily",
        "Lisinopril 10mg daily",
        "Metformin 500mg BID",
        "Atorvastatin 20mg at bedtime",
        "Aspirin 81mg daily",
    ],
    allergies=["Penicillin", "Sulfa"],
    vitals={
        "BP": "120/80",
        "HR": "72",
        "temp": "37.0°C",
        "SpO2": "98%",
        "RR": "16",
    },
)


# ═════════════════════════════════════════════════════════════════════════════
# INJECTED CLAIM RESULT
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class InjectedClaim:
    """Result of error injection on a single claim."""
    original_text: str
    modified_text: str
    claim_type: str
    is_error: bool
    error_type: Optional[ErrorType]
    error_details: str


# ═════════════════════════════════════════════════════════════════════════════
# ERROR INJECTOR
# ═════════════════════════════════════════════════════════════════════════════

class ErrorInjector:
    """
    Injects realistic clinical errors into claims for training data generation.

    All randomness is controlled by the internal RNG seeded at construction.
    """

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    # ── Public API ───────────────────────────────────────────────────────

    def inject_error(
        self,
        claim_text: str,
        claim_type: str,
        ehr_context: MockEHRContext,
        error_type: ErrorType,
    ) -> Tuple[str, str]:
        """
        Inject a specific error type into a claim.

        Returns (modified_claim_text, error_details_string).
        """
        handler = _ERROR_HANDLERS.get(error_type)
        if handler is None:
            return claim_text, "unknown error type"
        return handler(self, claim_text, claim_type, ehr_context)

    def inject_random_errors(
        self,
        claims: List[Tuple[str, str]],
        ehr_context: MockEHRContext,
        error_rate: float = 0.30,
    ) -> List[InjectedClaim]:
        """
        Inject errors into a list of (claim_text, claim_type) tuples.

        Each claim has an ``error_rate`` probability of being corrupted.
        The error type is chosen based on claim_type compatibility.

        Returns a list of ``InjectedClaim`` with labels.
        """
        results: List[InjectedClaim] = []

        for claim_text, claim_type in claims:
            if self._rng.random() < error_rate:
                error_type = self._pick_error_type(claim_type)
                modified, details = self.inject_error(
                    claim_text, claim_type, ehr_context, error_type,
                )
                results.append(InjectedClaim(
                    original_text=claim_text,
                    modified_text=modified,
                    claim_type=claim_type,
                    is_error=True,
                    error_type=error_type,
                    error_details=details,
                ))
            else:
                results.append(InjectedClaim(
                    original_text=claim_text,
                    modified_text=claim_text,
                    claim_type=claim_type,
                    is_error=False,
                    error_type=None,
                    error_details="",
                ))

        return results

    # ── Error type selection ─────────────────────────────────────────────

    def _pick_error_type(self, claim_type: str) -> ErrorType:
        """Choose a compatible error type for the given claim type."""
        compatible = _COMPATIBLE_ERRORS.get(claim_type, list(ErrorType))
        return self._rng.choice(compatible)

    # ── Individual error handlers ────────────────────────────────────────

    def _inject_wrong_dose(
        self, claim_text: str, claim_type: str, ehr_context: MockEHRContext,
    ) -> Tuple[str, str]:
        dose_match = re.search(
            r"(\d+(?:\.\d+)?)\s*(mg|mcg|units?|ml|g)\b",
            claim_text, re.IGNORECASE,
        )
        if not dose_match:
            return claim_text, "no dose found"

        original_dose = float(dose_match.group(1))
        unit = dose_match.group(2)
        multiplier = self._rng.choice(DOSE_MULTIPLIERS)
        new_dose = original_dose * multiplier

        # Format: keep integer if no decimal needed
        if new_dose == int(new_dose):
            dose_str = str(int(new_dose))
        else:
            dose_str = f"{new_dose:.1f}"

        modified = claim_text[:dose_match.start(1)] + dose_str + claim_text[dose_match.end(1):]
        details = f"dose changed: {original_dose}{unit} -> {dose_str}{unit} (x{multiplier})"
        return modified, details

    def _inject_wrong_frequency(
        self, claim_text: str, claim_type: str, ehr_context: MockEHRContext,
    ) -> Tuple[str, str]:
        text_lower = claim_text.lower()
        for original, replacement in FREQUENCY_SWAPS.items():
            idx = text_lower.find(original.lower())
            if idx != -1:
                end = idx + len(original)
                modified = claim_text[:idx] + replacement + claim_text[end:]
                return modified, f"frequency changed: {original} -> {replacement}"
        return claim_text, "no frequency found"

    def _inject_wrong_medication(
        self, claim_text: str, claim_type: str, ehr_context: MockEHRContext,
    ) -> Tuple[str, str]:
        text_lower = claim_text.lower()
        for original, replacement in SIMILAR_DRUGS.items():
            idx = text_lower.find(original.lower())
            if idx != -1:
                end = idx + len(original)
                # Preserve original casing style
                if claim_text[idx].isupper():
                    replacement = replacement.capitalize()
                modified = claim_text[:idx] + replacement + claim_text[end:]
                return modified, f"drug swapped: {original} -> {replacement}"
        return claim_text, "no known drug found"

    def _inject_contradicts_ehr(
        self, claim_text: str, claim_type: str, ehr_context: MockEHRContext,
    ) -> Tuple[str, str]:
        if claim_type == "medication" and ehr_context.medications:
            # Pick a random EHR medication and change the dose
            ehr_med = self._rng.choice(ehr_context.medications)
            dose_match = re.search(r"(\d+(?:\.\d+)?)\s*(mg|mcg|units?|ml|g)\b",
                                   ehr_med, re.IGNORECASE)
            if dose_match:
                original_dose = float(dose_match.group(1))
                wrong_dose = original_dose * self._rng.choice([2.0, 5.0, 10.0])
                if wrong_dose == int(wrong_dose):
                    dose_str = str(int(wrong_dose))
                else:
                    dose_str = f"{wrong_dose:.1f}"
                modified = re.sub(
                    r"\d+(?:\.\d+)?(?=\s*(?:mg|mcg|units?|ml|g)\b)",
                    dose_str,
                    ehr_med,
                    count=1,
                    flags=re.IGNORECASE,
                )
                return modified, f"contradicts EHR: {ehr_med} -> {modified}"

        return claim_text, "no EHR contradiction possible"

    def _inject_hallucinated_allergy(
        self, claim_text: str, claim_type: str, ehr_context: MockEHRContext,
    ) -> Tuple[str, str]:
        # Pick an allergy NOT in the EHR
        known = {a.lower() for a in ehr_context.allergies}
        candidates = [a for a in FAKE_ALLERGIES if a.lower() not in known]
        if not candidates:
            candidates = FAKE_ALLERGIES
        allergy = self._rng.choice(candidates)
        modified = f"Patient allergic to {allergy}"
        return modified, f"hallucinated allergy: {allergy}"

    def _inject_hallucinated_symptom(
        self, claim_text: str, claim_type: str, ehr_context: MockEHRContext,
    ) -> Tuple[str, str]:
        symptom = self._rng.choice(FAKE_SYMPTOMS)
        modified = f"Patient reports {symptom}"
        return modified, f"hallucinated symptom: {symptom}"

    def _inject_wrong_vital(
        self, claim_text: str, claim_type: str, ehr_context: MockEHRContext,
    ) -> Tuple[str, str]:
        # Try to detect which vital is in the claim
        for vital_name, pattern in _VITAL_PATTERNS.items():
            if pattern.search(claim_text):
                abnormal_values = ABNORMAL_VITALS.get(vital_name, [])
                if abnormal_values:
                    new_val = self._rng.choice(abnormal_values)
                    match = pattern.search(claim_text)
                    modified = claim_text[:match.start()] + _format_vital(vital_name, new_val) + claim_text[match.end():]
                    return modified, f"vital changed: {vital_name} -> {new_val}"

        # No vital pattern matched — substitute with a random abnormal vital
        vital_name = self._rng.choice(list(ABNORMAL_VITALS.keys()))
        new_val = self._rng.choice(ABNORMAL_VITALS[vital_name])
        modified = f"{_VITAL_LABELS[vital_name]} {new_val}"
        return modified, f"vital replaced: {vital_name} -> {new_val}"


def _format_vital(vital_name: str, value: str) -> str:
    """Format a vital sign value with its label prefix."""
    return f"{_VITAL_LABELS[vital_name]} {value}"


_VITAL_LABELS = {
    "BP": "Blood pressure",
    "HR": "Heart rate",
    "temp": "Temperature",
    "SpO2": "SpO2",
    "RR": "Respiratory rate",
}


# Handler dispatch table
_ERROR_HANDLERS = {
    ErrorType.WRONG_DOSE: ErrorInjector._inject_wrong_dose,
    ErrorType.WRONG_FREQUENCY: ErrorInjector._inject_wrong_frequency,
    ErrorType.WRONG_MEDICATION: ErrorInjector._inject_wrong_medication,
    ErrorType.CONTRADICTS_EHR: ErrorInjector._inject_contradicts_ehr,
    ErrorType.HALLUCINATED_ALLERGY: ErrorInjector._inject_hallucinated_allergy,
    ErrorType.HALLUCINATED_SYMPTOM: ErrorInjector._inject_hallucinated_symptom,
    ErrorType.WRONG_VITAL: ErrorInjector._inject_wrong_vital,
}

# Which error types are compatible with which claim types
_COMPATIBLE_ERRORS: Dict[str, List[ErrorType]] = {
    "medication": [
        ErrorType.WRONG_DOSE,
        ErrorType.WRONG_FREQUENCY,
        ErrorType.WRONG_MEDICATION,
        ErrorType.CONTRADICTS_EHR,
    ],
    "allergy": [
        ErrorType.HALLUCINATED_ALLERGY,
        ErrorType.CONTRADICTS_EHR,
    ],
    "diagnosis": [
        ErrorType.HALLUCINATED_SYMPTOM,
        ErrorType.CONTRADICTS_EHR,
    ],
    "vital_sign": [
        ErrorType.WRONG_VITAL,
    ],
}


# ═════════════════════════════════════════════════════════════════════════════
# TRAINING SET GENERATION
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class NoteWithClaims:
    """A clinical note broken into extractable claims."""
    note_text: str
    claims: List[Tuple[str, str]]  # (claim_text, claim_type)


def generate_training_set_from_notes(
    notes: List[NoteWithClaims],
    ehr_contexts: List[MockEHRContext],
    error_rate: float = 0.30,
    seed: int = 42,
) -> List[TrainingExample]:
    """
    Generate labelled training examples from clinical notes and EHR contexts.

    For each note, claims are extracted, errors are injected at the specified
    rate, and Layer 1 signals are simulated to produce ``TrainingExample``
    instances compatible with the existing data pipeline.

    Parameters
    ----------
    notes : list[NoteWithClaims]
        Clinical notes with pre-extracted claims.
    ehr_contexts : list[MockEHRContext]
        Matching EHR contexts (one per note, or cycled).
    error_rate : float
        Probability of injecting an error into each claim.
    seed : int
        RNG seed for reproducibility.
    """
    injector = ErrorInjector(seed=seed)
    rng = random.Random(seed + 1)  # Separate RNG for signal noise
    training_examples: List[TrainingExample] = []

    for i, note in enumerate(notes):
        ehr = ehr_contexts[i % len(ehr_contexts)]
        injected = injector.inject_random_errors(
            note.claims, ehr, error_rate=error_rate,
        )

        for result in injected:
            if result.is_error:
                ehr_status, hhem, se, risk, review = _signals_for_error(
                    result.error_type, rng,
                )
            else:
                ehr_status, hhem, se, risk, review = _signals_for_correct(rng)

            training_examples.append(TrainingExample(
                claim_type=result.claim_type,
                claim_text=result.modified_text,
                source_sentence=note.note_text[:120],  # Truncate for source
                ehr_status=ehr_status,
                hhem_score=hhem,
                semantic_entropy=se,
                risk_score=round(risk, 4),
                review_level=review,
            ))

    return training_examples


def _signals_for_error(
    error_type: Optional[ErrorType], rng: random.Random,
) -> Tuple[str, Optional[float], Optional[float], float, str]:
    """Simulate Layer 1 signals for an injected error."""
    if error_type in (ErrorType.CONTRADICTS_EHR, ErrorType.WRONG_DOSE,
                      ErrorType.WRONG_FREQUENCY, ErrorType.WRONG_MEDICATION):
        # These would be caught by EHR verification
        ehr_status = "contradiction"
        hhem = max(0.0, min(1.0, 0.20 + rng.gauss(0, 0.05)))
        se = max(0.0, min(1.0, 0.15 + rng.gauss(0, 0.05)))
        risk = max(0.0, min(1.0, 0.90 + rng.gauss(0, 0.03)))
        review = "CRITICAL"
    elif error_type in (ErrorType.HALLUCINATED_ALLERGY,
                        ErrorType.HALLUCINATED_SYMPTOM):
        # Not in EHR — flagged as not_found
        ehr_status = "not_found"
        hhem = max(0.0, min(1.0, 0.30 + rng.gauss(0, 0.05)))
        se = max(0.0, min(1.0, 0.50 + rng.gauss(0, 0.05)))
        risk = max(0.0, min(1.0, 0.80 + rng.gauss(0, 0.03)))
        review = "DETAILED"
    elif error_type == ErrorType.WRONG_VITAL:
        # Vitals contradiction
        ehr_status = "contradiction"
        hhem = max(0.0, min(1.0, 0.25 + rng.gauss(0, 0.05)))
        se = max(0.0, min(1.0, 0.12 + rng.gauss(0, 0.04)))
        risk = max(0.0, min(1.0, 0.92 + rng.gauss(0, 0.02)))
        review = "CRITICAL"
    else:
        ehr_status = "contradiction"
        hhem = 0.25
        se = 0.15
        risk = 0.90
        review = "CRITICAL"
    return ehr_status, hhem, se, risk, review


def _signals_for_correct(
    rng: random.Random,
) -> Tuple[str, Optional[float], Optional[float], float, str]:
    """Simulate Layer 1 signals for a correct (unmodified) claim."""
    ehr_status = "verified"
    hhem = None  # Verified claims skip HHEM
    se = None    # Verified claims skip SE
    risk = max(0.0, min(1.0, 0.05 + rng.gauss(0, 0.01)))
    review = "BRIEF"
    return ehr_status, hhem, se, risk, review
