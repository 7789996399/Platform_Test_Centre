"""
TRUST Test Scenario Generator
==============================

MedHallu-style benchmark for AI Scribe validation.

Ground Truth: Cerner Sandbox EHR data
Hallucinations: Injected with known labels
Test: Does TRUST catch them?

Usage:
    python test_scenario_generator.py --patient SANDBOX_PATIENT_ID --num-scenarios 10
"""

import asyncio
import random
import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from enum import Enum


# =============================================================================
# HALLUCINATION TYPES (matching MedHallu categories)
# =============================================================================

class HallucinationType(Enum):
    """Types of hallucinations we can inject."""
    MEDICATION_ADDITION = "medication_addition"      # Add fake medication
    MEDICATION_DOSE_CHANGE = "medication_dose_change"  # Wrong dose
    MEDICATION_REMOVAL = "medication_removal"        # Omit real medication
    ALLERGY_ADDITION = "allergy_addition"            # Add fake allergy
    ALLERGY_REMOVAL = "allergy_removal"              # Omit real allergy
    DIAGNOSIS_ADDITION = "diagnosis_addition"        # Add fake diagnosis
    DIAGNOSIS_REMOVAL = "diagnosis_removal"          # Omit real diagnosis
    VITAL_SIGN_ERROR = "vital_sign_error"            # Wrong vital sign value
    LAB_VALUE_ERROR = "lab_value_error"              # Wrong lab value


@dataclass
class InjectedHallucination:
    """A known hallucination we injected for testing."""
    hallucination_id: str
    hallucination_type: HallucinationType
    fake_claim: str              # What the AI will say (wrong)
    ground_truth: Optional[str]  # What EHR actually says (if applicable)
    severity: str                # "critical", "high", "medium", "low"
    detected_by_trust: Optional[bool] = None
    detection_method: Optional[str] = None  # "ehr_contradiction", "semantic_entropy", etc.


@dataclass
class TestScenario:
    """A complete test scenario with ground truth and hallucinations."""
    scenario_id: str
    patient_id: str
    created_at: datetime
    
    # Ground truth from EHR
    ehr_medications: List[str]
    ehr_allergies: List[str]
    ehr_conditions: List[str]
    ehr_vitals: Dict[str, str]
    
    # Generated content
    synthetic_transcript: str
    ai_scribe_note: str
    
    # Injected hallucinations (we know exactly what's fake)
    injected_hallucinations: List[InjectedHallucination]
    
    # TRUST results
    trust_analysis: Optional[Dict] = None
    
    # Scoring
    hallucinations_detected: int = 0
    hallucinations_missed: int = 0
    false_positives: int = 0
    detection_rate: float = 0.0


# =============================================================================
# FAKE DATA BANKS (for injection)
# =============================================================================

FAKE_MEDICATIONS = [
    ("Warfarin 5mg daily", "critical"),
    ("Digoxin 0.25mg daily", "critical"),
    ("Methotrexate 15mg weekly", "critical"),
    ("Oxycodone 10mg PRN", "high"),
    ("Gabapentin 300mg TID", "medium"),
    ("Omeprazole 20mg daily", "low"),
    ("Vitamin D 1000 IU daily", "low"),
    ("Amoxicillin 500mg TID", "high"),
    ("Prednisone 10mg daily", "high"),
    ("Atorvastatin 40mg daily", "medium"),
]

FAKE_ALLERGIES = [
    ("Penicillin - anaphylaxis", "critical"),
    ("Sulfa drugs - rash", "high"),
    ("Codeine - nausea", "medium"),
    ("Latex - hives", "high"),
    ("Iodine contrast - swelling", "high"),
    ("Aspirin - GI bleeding", "high"),
]

FAKE_DIAGNOSES = [
    ("Type 2 Diabetes Mellitus", "high"),
    ("Atrial Fibrillation", "critical"),
    ("Chronic Kidney Disease Stage 3", "high"),
    ("COPD", "high"),
    ("Heart Failure", "critical"),
    ("Hypothyroidism", "medium"),
    ("GERD", "low"),
    ("Osteoarthritis", "low"),
]

DOSE_MODIFICATIONS = {
    "metoprolol": ["25mg", "50mg", "100mg", "200mg"],
    "lisinopril": ["5mg", "10mg", "20mg", "40mg"],
    "amlodipine": ["2.5mg", "5mg", "10mg"],
    "atorvastatin": ["10mg", "20mg", "40mg", "80mg"],
    "metformin": ["500mg", "850mg", "1000mg"],
}


# =============================================================================
# TRANSCRIPT GENERATOR
# =============================================================================

class TranscriptGenerator:
    """Generates synthetic transcripts from EHR data with optional hallucinations."""
    
    TEMPLATES = {
        "greeting": [
            "Good morning {patient_name}, I'm Dr. Smith. How are you feeling today?",
            "Hello {patient_name}, thanks for coming in. What brings you here today?",
            "Hi {patient_name}, I see you're here for your follow-up. How have you been?",
        ],
        "medication_review": [
            "I see you're taking {medication}. How has that been working for you?",
            "Are you still on {medication}? Any side effects?",
            "Let's review your medications. You're currently on {medication}, correct?",
        ],
        "allergy_check": [
            "And I have here that you're allergic to {allergy}. Is that still accurate?",
            "Any new allergies since your last visit? I see {allergy} in your chart.",
            "Just confirming - you have an allergy to {allergy}, right?",
        ],
        "condition_mention": [
            "Given your history of {condition}, we should monitor that closely.",
            "How has your {condition} been? Any changes?",
            "I want to discuss your {condition} today.",
        ],
        "vital_signs": [
            "Your blood pressure today is {bp}. {interpretation}",
            "Let me check your vitals. BP is {bp}, heart rate {hr}.",
        ],
        "closing": [
            "Any other concerns before we wrap up?",
            "Do you have any questions for me?",
            "Let's schedule a follow-up in {weeks} weeks.",
        ],
    }
    
    def generate(
        self,
        patient_name: str,
        medications: List[str],
        allergies: List[str],
        conditions: List[str],
        vitals: Dict[str, str],
        hallucinations: List[InjectedHallucination]
    ) -> str:
        """Generate a synthetic transcript."""
        
        lines = []
        
        # Greeting
        lines.append(f"Doctor: {random.choice(self.TEMPLATES['greeting']).format(patient_name=patient_name)}")
        lines.append(f"Patient: I'm doing okay, thanks for asking.")
        lines.append("")
        
        # Medication review (include real + hallucinated)
        all_meds = medications.copy()
        for h in hallucinations:
            if h.hallucination_type == HallucinationType.MEDICATION_ADDITION:
                all_meds.append(h.fake_claim)
            elif h.hallucination_type == HallucinationType.MEDICATION_DOSE_CHANGE:
                # Replace the real dose with fake dose
                all_meds = [h.fake_claim if h.ground_truth and h.ground_truth.split()[0].lower() in m.lower() else m for m in all_meds]
        
        # Remove medications if MEDICATION_REMOVAL hallucination
        for h in hallucinations:
            if h.hallucination_type == HallucinationType.MEDICATION_REMOVAL:
                all_meds = [m for m in all_meds if h.ground_truth.lower() not in m.lower()]
        
        if all_meds:
            lines.append("Doctor: Let's review your medications.")
            for med in all_meds[:5]:  # Limit to 5
                lines.append(f"Doctor: {random.choice(self.TEMPLATES['medication_review']).format(medication=med)}")
                lines.append(f"Patient: Yes, I've been taking it as prescribed.")
            lines.append("")
        
        # Allergy check (include real + hallucinated)
        all_allergies = allergies.copy()
        for h in hallucinations:
            if h.hallucination_type == HallucinationType.ALLERGY_ADDITION:
                all_allergies.append(h.fake_claim)
            elif h.hallucination_type == HallucinationType.ALLERGY_REMOVAL:
                all_allergies = [a for a in all_allergies if h.ground_truth.lower() not in a.lower()]
        
        if all_allergies:
            lines.append(f"Doctor: I want to confirm your allergies. You're allergic to {', '.join(all_allergies)}?")
            lines.append(f"Patient: That's correct.")
            lines.append("")
        elif not allergies:
            lines.append("Doctor: Any allergies to medications?")
            lines.append("Patient: No, none that I know of.")
            lines.append("")
        
        # Conditions (include real + hallucinated)
        all_conditions = conditions.copy()
        for h in hallucinations:
            if h.hallucination_type == HallucinationType.DIAGNOSIS_ADDITION:
                all_conditions.append(h.fake_claim)
        
        if all_conditions:
            for condition in all_conditions[:3]:
                lines.append(f"Doctor: {random.choice(self.TEMPLATES['condition_mention']).format(condition=condition)}")
                lines.append(f"Patient: It's been manageable.")
            lines.append("")
        
        # Vitals
        if vitals:
            bp = vitals.get("blood_pressure", "120/80")
            hr = vitals.get("heart_rate", "72")
            
            # Inject vital sign errors
            for h in hallucinations:
                if h.hallucination_type == HallucinationType.VITAL_SIGN_ERROR:
                    if "blood_pressure" in h.fake_claim.lower() or "bp" in h.fake_claim.lower():
                        bp = h.fake_claim.split()[-1]  # Extract the fake value
            
            lines.append(f"Doctor: Your blood pressure today is {bp}, heart rate {hr}.")
            lines.append(f"Patient: Okay.")
            lines.append("")
        
        # Closing
        lines.append(f"Doctor: {random.choice(self.TEMPLATES['closing']).format(weeks=random.choice([2, 4, 6, 8]))}")
        lines.append(f"Patient: No, I think we covered everything. Thank you, doctor.")
        
        return "\n".join(lines)


# =============================================================================
# AI NOTE GENERATOR (simulates AI scribe)
# =============================================================================

class AIScribeSimulator:
    """Simulates an AI scribe generating a note from transcript."""
    
    NOTE_TEMPLATE = """
CLINICAL NOTE
=============

Patient: {patient_name}
Date: {date}
Visit Type: Follow-up

CHIEF COMPLAINT:
Routine follow-up visit.

CURRENT MEDICATIONS:
{medications}

ALLERGIES:
{allergies}

MEDICAL HISTORY:
{conditions}

VITAL SIGNS:
{vitals}

ASSESSMENT AND PLAN:
Patient is here for routine follow-up. Current medications reviewed and continued.
Follow-up scheduled in {weeks} weeks.

Electronically signed by: AI Scribe (pending physician review)
"""
    
    def generate_note(
        self,
        patient_name: str,
        transcript: str,
        medications: List[str],
        allergies: List[str],
        conditions: List[str],
        vitals: Dict[str, str]
    ) -> str:
        """Generate a clinical note from transcript."""
        
        med_list = "\n".join([f"- {m}" for m in medications]) if medications else "- None documented"
        allergy_list = "\n".join([f"- {a}" for a in allergies]) if allergies else "- NKDA (No Known Drug Allergies)"
        condition_list = "\n".join([f"- {c}" for c in conditions]) if conditions else "- None documented"
        vital_str = f"BP: {vitals.get('blood_pressure', 'N/A')}, HR: {vitals.get('heart_rate', 'N/A')}"
        
        return self.NOTE_TEMPLATE.format(
            patient_name=patient_name,
            date=datetime.now().strftime("%Y-%m-%d"),
            medications=med_list,
            allergies=allergy_list,
            conditions=condition_list,
            vitals=vital_str,
            weeks=random.choice([2, 4, 6, 8])
        )


# =============================================================================
# HALLUCINATION INJECTOR
# =============================================================================

class HallucinationInjector:
    """Injects known hallucinations into test scenarios."""
    
    def __init__(self, seed: int = None):
        if seed:
            random.seed(seed)
    
    def inject(
        self,
        ehr_medications: List[str],
        ehr_allergies: List[str],
        ehr_conditions: List[str],
        num_hallucinations: int = 2,
        hallucination_types: List[HallucinationType] = None
    ) -> Tuple[List[str], List[str], List[str], List[InjectedHallucination]]:
        """
        Inject hallucinations and return modified lists + tracking.
        
        Returns:
            (modified_meds, modified_allergies, modified_conditions, hallucinations)
        """
        if hallucination_types is None:
            hallucination_types = [
                HallucinationType.MEDICATION_ADDITION,
                HallucinationType.ALLERGY_ADDITION,
                HallucinationType.DIAGNOSIS_ADDITION,
            ]
        
        hallucinations = []
        modified_meds = ehr_medications.copy()
        modified_allergies = ehr_allergies.copy()
        modified_conditions = ehr_conditions.copy()
        
        for i in range(num_hallucinations):
            h_type = random.choice(hallucination_types)
            h_id = f"HAL_{i+1:03d}"
            
            if h_type == HallucinationType.MEDICATION_ADDITION:
                fake_med, severity = random.choice(FAKE_MEDICATIONS)
                # Make sure it's not already in EHR
                if not any(fake_med.split()[0].lower() in m.lower() for m in ehr_medications):
                    modified_meds.append(fake_med)
                    hallucinations.append(InjectedHallucination(
                        hallucination_id=h_id,
                        hallucination_type=h_type,
                        fake_claim=fake_med,
                        ground_truth=None,
                        severity=severity
                    ))
            
            elif h_type == HallucinationType.MEDICATION_DOSE_CHANGE and ehr_medications:
                # Pick a real medication and change its dose
                real_med = random.choice(ehr_medications)
                drug_name = real_med.split()[0].lower()
                if drug_name in DOSE_MODIFICATIONS:
                    current_dose = next((d for d in DOSE_MODIFICATIONS[drug_name] if d in real_med), None)
                    other_doses = [d for d in DOSE_MODIFICATIONS[drug_name] if d != current_dose]
                    if other_doses:
                        fake_dose = random.choice(other_doses)
                        fake_med = real_med.replace(current_dose, fake_dose) if current_dose else f"{drug_name} {fake_dose}"
                        hallucinations.append(InjectedHallucination(
                            hallucination_id=h_id,
                            hallucination_type=h_type,
                            fake_claim=fake_med,
                            ground_truth=real_med,
                            severity="high"
                        ))
            
            elif h_type == HallucinationType.ALLERGY_ADDITION:
                fake_allergy, severity = random.choice(FAKE_ALLERGIES)
                if not any(fake_allergy.split()[0].lower() in a.lower() for a in ehr_allergies):
                    modified_allergies.append(fake_allergy)
                    hallucinations.append(InjectedHallucination(
                        hallucination_id=h_id,
                        hallucination_type=h_type,
                        fake_claim=fake_allergy,
                        ground_truth=None,
                        severity=severity
                    ))
            
            elif h_type == HallucinationType.DIAGNOSIS_ADDITION:
                fake_dx, severity = random.choice(FAKE_DIAGNOSES)
                if not any(fake_dx.lower() in c.lower() for c in ehr_conditions):
                    modified_conditions.append(fake_dx)
                    hallucinations.append(InjectedHallucination(
                        hallucination_id=h_id,
                        hallucination_type=h_type,
                        fake_claim=fake_dx,
                        ground_truth=None,
                        severity=severity
                    ))
        
        return modified_meds, modified_allergies, modified_conditions, hallucinations


# =============================================================================
# MAIN TEST SCENARIO GENERATOR
# =============================================================================

class TestScenarioGenerator:
    """
    Main class to generate test scenarios using Cerner sandbox as ground truth.
    """
    
    def __init__(self, cerner_client=None):
        """
        Initialize with Cerner client.
        
        Args:
            cerner_client: Your existing Cerner/FHIR client
        """
        self.cerner_client = cerner_client
        self.transcript_generator = TranscriptGenerator()
        self.scribe_simulator = AIScribeSimulator()
        self.hallucination_injector = HallucinationInjector()
    
    async def generate_scenario(
        self,
        patient_id: str,
        num_hallucinations: int = 2,
        hallucination_types: List[HallucinationType] = None
    ) -> TestScenario:
        """
        Generate a complete test scenario for a Cerner sandbox patient.
        
        Args:
            patient_id: Cerner sandbox patient ID
            num_hallucinations: How many hallucinations to inject
            hallucination_types: Types of hallucinations to inject
        
        Returns:
            TestScenario with ground truth, hallucinations, and generated content
        """
        
        # Step 1: Fetch REAL patient data from Cerner sandbox (ground truth)
        print(f"📋 Fetching patient {patient_id} from Cerner sandbox...")
        
        if self.cerner_client:
            patient_context = await self.cerner_client.get_patient_context(patient_id)
            patient_name = patient_context.name
            ehr_medications = patient_context.medications
            ehr_allergies = patient_context.allergies
            ehr_conditions = patient_context.conditions
            ehr_vitals = patient_context.vitals or {"blood_pressure": "120/80", "heart_rate": "72"}
        else:
            # Mock data for testing without Cerner
            patient_name = "Test Patient"
            ehr_medications = ["Metoprolol 50mg BID", "Lisinopril 10mg daily", "Aspirin 81mg daily"]
            ehr_allergies = ["Penicillin"]
            ehr_conditions = ["Hypertension", "Hyperlipidemia"]
            ehr_vitals = {"blood_pressure": "128/82", "heart_rate": "76"}
        
        print(f"   ✓ Patient: {patient_name}")
        print(f"   ✓ Medications: {len(ehr_medications)}")
        print(f"   ✓ Allergies: {len(ehr_allergies)}")
        print(f"   ✓ Conditions: {len(ehr_conditions)}")
        
        # Step 2: Inject hallucinations (we know exactly what's fake)
        print(f"\n💉 Injecting {num_hallucinations} hallucinations...")
        
        modified_meds, modified_allergies, modified_conditions, hallucinations = \
            self.hallucination_injector.inject(
                ehr_medications,
                ehr_allergies,
                ehr_conditions,
                num_hallucinations=num_hallucinations,
                hallucination_types=hallucination_types
            )
        
        for h in hallucinations:
            print(f"   💊 {h.hallucination_type.value}: '{h.fake_claim}' (severity: {h.severity})")
        
        # Step 3: Generate synthetic transcript (includes hallucinations)
        print(f"\n🎤 Generating synthetic transcript...")
        transcript = self.transcript_generator.generate(
            patient_name=patient_name,
            medications=modified_meds,
            allergies=modified_allergies,
            conditions=modified_conditions,
            vitals=ehr_vitals,
            hallucinations=hallucinations
        )
        
        # Step 4: Generate AI scribe note (includes hallucinations)
        print(f"📝 Generating AI scribe note...")
        ai_note = self.scribe_simulator.generate_note(
            patient_name=patient_name,
            transcript=transcript,
            medications=modified_meds,
            allergies=modified_allergies,
            conditions=modified_conditions,
            vitals=ehr_vitals
        )
        
        # Step 5: Create test scenario
        scenario = TestScenario(
            scenario_id=f"SCENARIO_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{patient_id}",
            patient_id=patient_id,
            created_at=datetime.now(),
            ehr_medications=ehr_medications,
            ehr_allergies=ehr_allergies,
            ehr_conditions=ehr_conditions,
            ehr_vitals=ehr_vitals,
            synthetic_transcript=transcript,
            ai_scribe_note=ai_note,
            injected_hallucinations=hallucinations
        )
        
        print(f"\n✅ Scenario created: {scenario.scenario_id}")
        
        return scenario
    def _parse_note_sections(self, note_text: str) -> Dict[str, str]:
        """Parse raw note text into sections."""
        sections = {}
        current_section = None
        current_content = []
        
        for line in note_text.split('\n'):
            # Check for section headers
            if 'MEDICATIONS:' in line.upper():
                if current_section:
                    sections[current_section] = '\n'.join(current_content)
                current_section = 'medications'
                current_content = []
            elif 'ALLERGIES:' in line.upper():
                if current_section:
                    sections[current_section] = '\n'.join(current_content)
                current_section = 'allergies'
                current_content = []
            elif 'MEDICAL HISTORY:' in line.upper() or 'CONDITIONS:' in line.upper():
                if current_section:
                    sections[current_section] = '\n'.join(current_content)
                current_section = 'conditions'
                current_content = []
            elif 'VITAL SIGNS:' in line.upper():
                if current_section:
                    sections[current_section] = '\n'.join(current_content)
                current_section = 'vitals'
                current_content = []
            elif 'ASSESSMENT' in line.upper() or 'PLAN:' in line.upper():
                if current_section:
                    sections[current_section] = '\n'.join(current_content)
                current_section = 'assessment'
                current_content = []
            elif current_section:
                current_content.append(line)
        
        # Don't forget the last section
        if current_section:
            sections[current_section] = '\n'.join(current_content)
        
        return sections
    
    async def run_trust_analysis(self, scenario: TestScenario) -> TestScenario:
        """
        Run TRUST analysis on a test scenario and score detection.
        """
        # Import your TRUST pipeline
        from app.core.routing import analyze_note
        from app.services import cerner as cerner_module
        from app.services.cerner import PatientContext
        
        # MOCK the Cerner service to return our ground truth EHR data
        original_get_patient_context = cerner_module.get_patient_context
        
        def mock_get_patient_context(patient_id):
            """Return our test scenario's ground truth as the EHR data."""

            return PatientContext(
                patient_id=patient_id,
                name="Test Patient",
                birth_date="1960-01-01",
                gender="male",
                conditions=scenario.ehr_conditions,
                medications=scenario.ehr_medications,
                allergies=scenario.ehr_allergies
            )   
        
        # Patch it
        cerner_module.get_patient_context = mock_get_patient_context
        
        print(f"\n🔍 Running TRUST analysis on scenario {scenario.scenario_id}...")
        
        # Run TRUST
       # Parse the AI note into sections for claim extraction
        note_dict = {
            "ai_scribe_output": {
                "sections": self._parse_note_sections(scenario.ai_scribe_note)
            },
            "patient": {"id": scenario.patient_id}
        }
        
        analysis = await analyze_note(
            note=note_dict,
            transcript=scenario.synthetic_transcript,
            run_entropy=True
        )
        
        # Store raw results
        scenario.trust_analysis = {
            "total_claims": analysis.total_claims,
            "verified": analysis.summary.get("verified", 0),
            "contradictions": analysis.summary.get("contradictions", 0),
            "confident_hallucinators": analysis.summary.get("confident_hallucinators", 0),
            "overall_risk": analysis.overall_risk,
            "review_queue": [
                {
                    "claim": ca.claim.text,
                    "status": ca.verification.status.value,
                    "priority": ca.priority_score
                }
                for ca in analysis.review_queue[:10]
            ]
        }
        
        # Score: Did TRUST catch our injected hallucinations?
        detected = 0
        missed = 0
        
        for h in scenario.injected_hallucinations:
            # Check if this hallucination was flagged
            was_detected = False
            
            for ca in analysis.claim_analyses:
                # Check if claim matches our injected hallucination
                if h.fake_claim.lower() in ca.claim.text.lower():
                    if ca.verification.status.value in ["contradicted", "not_in_ehr"]:
                        was_detected = True
                        h.detected_by_trust = True
                        h.detection_method = ca.verification.status.value
                        break
            
            if was_detected:
                detected += 1
                print(f"   ✅ CAUGHT: {h.fake_claim}")
            else:
                missed += 1
                h.detected_by_trust = False
                print(f"   ❌ MISSED: {h.fake_claim}")
        
        scenario.hallucinations_detected = detected
        scenario.hallucinations_missed = missed
        scenario.detection_rate = detected / len(scenario.injected_hallucinations) if scenario.injected_hallucinations else 1.0
        
        print(f"\n📊 DETECTION SCORE: {detected}/{len(scenario.injected_hallucinations)} ({scenario.detection_rate:.1%})")
        # Restore original
        cerner_module.get_patient_context = original_get_patient_context
        return scenario
    
    def export_scenario(self, scenario: TestScenario, filepath: str):
        """Export scenario to JSON for analysis."""
        
        data = {
            "scenario_id": scenario.scenario_id,
            "patient_id": scenario.patient_id,
            "created_at": scenario.created_at.isoformat(),
            "ground_truth": {
                "medications": scenario.ehr_medications,
                "allergies": scenario.ehr_allergies,
                "conditions": scenario.ehr_conditions,
                "vitals": scenario.ehr_vitals
            },
            "generated_content": {
                "transcript": scenario.synthetic_transcript,
                "ai_note": scenario.ai_scribe_note
            },
            "injected_hallucinations": [
                {
                    "id": h.hallucination_id,
                    "type": h.hallucination_type.value,
                    "fake_claim": h.fake_claim,
                    "ground_truth": h.ground_truth,
                    "severity": h.severity,
                    "detected": h.detected_by_trust,
                    "detection_method": h.detection_method
                }
                for h in scenario.injected_hallucinations
            ],
            "trust_analysis": scenario.trust_analysis,
            "scoring": {
                "hallucinations_detected": scenario.hallucinations_detected,
                "hallucinations_missed": scenario.hallucinations_missed,
                "false_positives": scenario.false_positives,
                "detection_rate": scenario.detection_rate
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"💾 Exported to {filepath}")


# =============================================================================
# CLI RUNNER
# =============================================================================

async def main():
    """Run test scenario generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="TRUST Test Scenario Generator")
    parser.add_argument("--patient", default="MOCK_PATIENT_001", help="Cerner sandbox patient ID")
    parser.add_argument("--num-scenarios", type=int, default=1, help="Number of scenarios to generate")
    parser.add_argument("--num-hallucinations", type=int, default=2, help="Hallucinations per scenario")
    parser.add_argument("--run-trust", action="store_true", help="Run TRUST analysis")
    parser.add_argument("--export", default="test_scenarios", help="Export directory")
    args = parser.parse_args()
    
    # Initialize generator (without Cerner client for now - uses mock data)
    generator = TestScenarioGenerator(cerner_client=None)
    
    scenarios = []
    
    for i in range(args.num_scenarios):
        print(f"\n{'='*60}")
        print(f"GENERATING SCENARIO {i+1}/{args.num_scenarios}")
        print(f"{'='*60}")
        
        scenario = await generator.generate_scenario(
            patient_id=args.patient,
            num_hallucinations=args.num_hallucinations
        )
        
        if args.run_trust:
            scenario = await generator.run_trust_analysis(scenario)
        
        scenarios.append(scenario)
        
        # Export
        import os
        os.makedirs(args.export, exist_ok=True)
        generator.export_scenario(
            scenario,
            f"{args.export}/{scenario.scenario_id}.json"
        )
    
    # Summary
    if args.run_trust and scenarios:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        
        total_injected = sum(len(s.injected_hallucinations) for s in scenarios)
        total_detected = sum(s.hallucinations_detected for s in scenarios)
        overall_rate = total_detected / total_injected if total_injected else 0
        
        print(f"Total scenarios: {len(scenarios)}")
        print(f"Total hallucinations injected: {total_injected}")
        print(f"Total detected by TRUST: {total_detected}")
        print(f"Overall detection rate: {overall_rate:.1%}")


if __name__ == "__main__":
    asyncio.run(main())
