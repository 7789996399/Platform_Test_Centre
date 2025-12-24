"""
TRUST Platform - Cerner Sandbox Connection Test
================================================
Tests OAuth2 authentication and FHIR API access.
"""

import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Cerner Sandbox Configuration
CERNER_CLIENT_ID = os.getenv('CERNER_CLIENT_ID', 'b6105cd0-4ceb-4b04-bb87-33568c91c1f0')
CERNER_CLIENT_SECRET = os.getenv('CERNER_CLIENT_SECRET', 'Ekhs-rJntlodAeP1tOhEM-NiBdXCEtuE')

# Cerner Open Sandbox URLs
TOKEN_URL = "https://authorization.cerner.com/tenants/ec2458f2-1e24-41c8-b71b-0e701af7583d/protocols/oauth2/profiles/smart-v1/token"
FHIR_BASE_URL = "https://fhir-ehr-code.cerner.com/r4/ec2458f2-1e24-41c8-b71b-0e701af7583d"


def get_access_token():
    """
    Get OAuth2 access token using client credentials.
    """
    print("\n" + "="*60)
    print("STEP 1: Getting Access Token")
    print("="*60)
    
    data = {
        'grant_type': 'client_credentials',
        'scope': 'system/Patient.read system/DocumentReference.read'
    }
    
    try:
        response = requests.post(
            TOKEN_URL,
            data=data,
            auth=(CERNER_CLIENT_ID, CERNER_CLIENT_SECRET),
            headers={'Accept': 'application/json'}
        )
        
        print(f"Token URL: {TOKEN_URL}")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            token_data = response.json()
            print("✅ Access token obtained!")
            print(f"   Token type: {token_data.get('token_type')}")
            print(f"   Expires in: {token_data.get('expires_in')} seconds")
            print(f"   Scope: {token_data.get('scope', 'N/A')}")
            return token_data.get('access_token')
        else:
            print(f"❌ Failed to get token")
            print(f"   Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def test_fhir_metadata(token):
    """
    Test FHIR server capability statement (no auth required).
    """
    print("\n" + "="*60)
    print("STEP 2: Testing FHIR Server Metadata")
    print("="*60)
    
    url = f"{FHIR_BASE_URL}/metadata"
    
    try:
        response = requests.get(url, headers={'Accept': 'application/fhir+json'})
        
        print(f"URL: {url}")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            metadata = response.json()
            print("✅ FHIR server is accessible!")
            print(f"   FHIR Version: {metadata.get('fhirVersion')}")
            print(f"   Software: {metadata.get('software', {}).get('name', 'N/A')}")
            
            # List available resources
            resources = [r['type'] for r in metadata.get('rest', [{}])[0].get('resource', [])]
            print(f"   Available resources: {len(resources)}")
            if 'Patient' in resources:
                print("   ✅ Patient resource available")
            if 'DocumentReference' in resources:
                print("   ✅ DocumentReference resource available")
            return True
        else:
            print(f"❌ Failed to get metadata: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def search_patients(token):
    """
    Search for test patients in the sandbox.
    """
    print("\n" + "="*60)
    print("STEP 3: Searching for Test Patients")
    print("="*60)
    
    if not token:
        print("⚠️  No token available, skipping authenticated request")
        return
    
    url = f"{FHIR_BASE_URL}/Patient"
    headers = {
        'Authorization': f'Bearer {token}',
        'Accept': 'application/fhir+json'
    }
    params = {
        '_count': 5  # Limit to 5 patients
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        
        print(f"URL: {url}")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            bundle = response.json()
            total = bundle.get('total', 0)
            entries = bundle.get('entry', [])
            
            print(f"✅ Found {total} patients (showing first {len(entries)})")
            
            for entry in entries[:5]:
                patient = entry.get('resource', {})
                patient_id = patient.get('id')
                name = patient.get('name', [{}])[0]
                given = ' '.join(name.get('given', ['Unknown']))
                family = name.get('family', 'Unknown')
                print(f"   - {given} {family} (ID: {patient_id})")
            
            return entries[0]['resource']['id'] if entries else None
        else:
            print(f"❌ Failed to search patients")
            print(f"   Response: {response.text[:500]}")
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def get_document_references(token, patient_id):
    """
    Get DocumentReferences for a patient (where AI scribe notes would live).
    """
    print("\n" + "="*60)
    print("STEP 4: Checking DocumentReferences")
    print("="*60)
    
    if not token or not patient_id:
        print("⚠️  Missing token or patient_id, skipping")
        return
    
    url = f"{FHIR_BASE_URL}/DocumentReference"
    headers = {
        'Authorization': f'Bearer {token}',
        'Accept': 'application/fhir+json'
    }
    params = {
        'patient': patient_id,
        '_count': 5
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        
        print(f"URL: {url}?patient={patient_id}")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            bundle = response.json()
            total = bundle.get('total', 0)
            entries = bundle.get('entry', [])
            
            print(f"✅ Found {total} documents for patient {patient_id}")
            
            for entry in entries[:5]:
                doc = entry.get('resource', {})
                doc_id = doc.get('id')
                doc_type = doc.get('type', {}).get('text', 'Unknown type')
                status = doc.get('status', 'unknown')
                print(f"   - {doc_type} (ID: {doc_id}, Status: {status})")
                
            if total == 0:
                print("   (No documents yet - this is normal for test patients)")
        else:
            print(f"❌ Failed to get documents")
            print(f"   Response: {response.text[:500]}")
            
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    print("\n" + "🏥 "*15)
    print("  TRUST Platform - Cerner Sandbox Connection Test")
    print("🏥 "*15)
    
    print(f"\nClient ID: {CERNER_CLIENT_ID[:8]}...{CERNER_CLIENT_ID[-4:]}")
    print(f"FHIR Base: {FHIR_BASE_URL}")
    
    # Step 1: Get access token
    token = get_access_token()
    
    # Step 2: Test FHIR metadata (works without auth)
    test_fhir_metadata(token)
    
    # Step 3: Search patients
    patient_id = search_patients(token)
    
    # Step 4: Get documents for patient
    if patient_id:
        get_document_references(token, patient_id)
    
    print("\n" + "="*60)
    print("CONNECTION TEST COMPLETE")
    print("="*60)
    
    if token:
        print("✅ OAuth2 authentication: WORKING")
    else:
        print("❌ OAuth2 authentication: FAILED")
    
    print("\nNext steps:")
    print("  - If token failed: Check client credentials")
    print("  - If metadata worked: FHIR server is accessible")
    print("  - If patients found: Full read access confirmed")


if __name__ == "__main__":
    main()
