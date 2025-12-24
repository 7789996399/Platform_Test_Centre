import requests

FHIR_BASE_URL = "https://fhir-open.cerner.com/r4/ec2458f2-1e24-41c8-b71b-0e701af7583d"

def search_patients():
    print("\nSearching for Test Patients...")
    url = f"{FHIR_BASE_URL}/Patient"
    response = requests.get(url, headers={'Accept': 'application/fhir+json'}, params={'_count': 5})
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        bundle = response.json()
        entries = bundle.get('entry', [])
        print(f"Found {len(entries)} patients!\n")
        for entry in entries[:5]:
            patient = entry.get('resource', {})
            patient_id = patient.get('id')
            name = patient.get('name', [{}])[0]
            given = ' '.join(name.get('given', ['Unknown']))
            family = name.get('family', 'Unknown')
            print(f"  - {given} {family} (ID: {patient_id})")
        return entries[0]['resource']['id'] if entries else None
    return None

if __name__ == "__main__":
    print("TRUST Platform - Cerner Sandbox Test")
    patient_id = search_patients()
    if patient_id:
        print(f"\nSUCCESS! Found patient: {patient_id}")

