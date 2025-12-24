"""
TRUST Platform - Cerner FHIR Service
=====================================
Fetches patient data from Cerner sandbox for verification.
"""

import requests
from typing import Dict, List, Optional
from dataclasses import dataclass


# Cerner Open Sandbox (public, no auth needed for testing)
FHIR_BASE_URL = "https://fhir-open.cerner.com/r4/ec2458f2-1e24-41c8-b71b-0e701af7583d"

HEADERS = {"Accept": "application/fhir+json"}


@dataclass
class PatientContext:
    """Patient context for AI scribe verification."""
    patient_id: str
    name: str
    birth_date: str
    gender: str
    conditions: List[str]
    medications: List[str]
    allergies: List[str]


def get_patient(patient_id: str) -> Optional[Dict]:
    """Fetch patient demographics."""
    url = f"{FHIR_BASE_URL}/Patient/{patient_id}"
    response = requests.get(url, headers=HEADERS)
    
    if response.status_code == 200:
        return response.json()
    return None


def get_conditions(patient_id: str) -> List[str]:
    """Fetch patient conditions/diagnoses."""
    url = f"{FHIR_BASE_URL}/Condition"
    params = {"patient": patient_id, "_count": 20}
    response = requests.get(url, headers=HEADERS, params=params)
    
    conditions = []
    if response.status_code == 200:
        bundle = response.json()
        for entry in bundle.get("entry", []):
            condition = entry.get("resource", {})
            text = condition.get("code", {}).get("text", "Unknown")
            conditions.append(text)
    return conditions


def get_medications(patient_id: str) -> List[str]:
    """Fetch patient medications."""
    url = f"{FHIR_BASE_URL}/MedicationRequest"
    params = {"patient": patient_id, "_count": 20}
    response = requests.get(url, headers=HEADERS, params=params)
    
    medications = []
    if response.status_code == 200:
        bundle = response.json()
        for entry in bundle.get("entry", []):
            med = entry.get("resource", {})
            text = med.get("medicationCodeableConcept", {}).get("text", "Unknown")
            medications.append(text)
    return medications


def get_allergies(patient_id: str) -> List[str]:
    """Fetch patient allergies."""
    url = f"{FHIR_BASE_URL}/AllergyIntolerance"
    params = {"patient": patient_id, "_count": 20}
    response = requests.get(url, headers=HEADERS, params=params)
    
    allergies = []
    if response.status_code == 200:
        bundle = response.json()
        for entry in bundle.get("entry", []):
            allergy = entry.get("resource", {})
            text = allergy.get("code", {}).get("text", "Unknown")
            allergies.append(text)
    return allergies


def get_patient_context(patient_id: str) -> Optional[PatientContext]:
    """
    Get complete patient context for AI verification.
    
    This is what we compare AI scribe output against!
    """
    patient = get_patient(patient_id)
    if not patient:
        return None
    
    # Extract name
    name_data = patient.get("name", [{}])[0]
    given = " ".join(name_data.get("given", ["Unknown"]))
    family = name_data.get("family", "Unknown")
    name = f"{given} {family}"
    
    return PatientContext(
        patient_id=patient_id,
        name=name,
        birth_date=patient.get("birthDate", "Unknown"),
        gender=patient.get("gender", "Unknown"),
        conditions=get_conditions(patient_id),
        medications=get_medications(patient_id),
        allergies=get_allergies(patient_id)
    )


def search_patients(count: int = 10) -> List[Dict]:
    """Search for available test patients."""
    url = f"{FHIR_BASE_URL}/Patient"
    params = {"_count": count}
    response = requests.get(url, headers=HEADERS, params=params)
    
    patients = []
    if response.status_code == 200:
        bundle = response.json()
        for entry in bundle.get("entry", []):
            p = entry.get("resource", {})
            name_data = p.get("name", [{}])[0]
            given = " ".join(name_data.get("given", ["?"]))
            family = name_data.get("family", "?")
            patients.append({
                "id": p.get("id"),
                "name": f"{given} {family}",
                "birthDate": p.get("birthDate"),
                "gender": p.get("gender")
            })
    return patients
