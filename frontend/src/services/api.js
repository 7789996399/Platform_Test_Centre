/**
 * TRUST Platform - API Service
 * Handles all communication with the backend
 */

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api/v1';

/**
 * Analyze a clinical note for hallucinations/uncertainty
 */
export async function analyzeNote(noteData) {
  const response = await fetch(`${API_BASE_URL}/scribe/analyze`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(noteData),
  });
  
  if (!response.ok) {
    throw new Error(`Analysis failed: ${response.statusText}`);
  }
  
  return response.json();
}

/**
 * Quick analysis (less thorough, faster)
 */
export async function analyzeNoteQuick(noteData) {
  const response = await fetch(`${API_BASE_URL}/scribe/analyze/quick`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(noteData),
  });
  
  if (!response.ok) {
    throw new Error(`Quick analysis failed: ${response.statusText}`);
  }
  
  return response.json();
}

/**
 * Check if the API is healthy
 */
export async function checkHealth() {
  const baseUrl = API_BASE_URL.replace('/api/v1', '');
  const response = await fetch(`${baseUrl}/health`);
  return response.json();
}

/**
 * Get patient data from Cerner
 */
export async function getCernerPatient(patientId) {
  const response = await fetch(`${API_BASE_URL}/cerner/patient/${patientId}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch patient: ${response.statusText}`);
  }
  return response.json();
}

/**
 * Verify note claims against EHR data
 */
export async function verifyAgainstEHR(patientId, noteData) {
  const response = await fetch(`${API_BASE_URL}/scribe/verify-ehr/${patientId}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(noteData),
  });
  if (!response.ok) {
    throw new Error(`EHR verification failed: ${response.statusText}`);
  }
  return response.json();
}

/**
 * Log an audit event
 */
export async function logAuditEvent(eventData) {
  try {
    const response = await fetch(`${API_BASE_URL}/audit/log`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(eventData),
    });
    return response.json();
  } catch (error) {
    console.error('Audit log failed:', error);
  }
}

/**
 * Get audit logs with optional filters
 */
export async function getAuditLogs(filters = {}) {
  const params = new URLSearchParams(filters);
  const response = await fetch(`${API_BASE_URL}/audit/logs?${params}`);
  return response.json();
}

/**
 * Get list of pending AI documents for review
 */
export async function getPendingDocuments() {
  const response = await fetch(`${API_BASE_URL}/documents/pending`);
  if (!response.ok) {
    throw new Error(`Failed to fetch documents: ${response.statusText}`);
  }
  return response.json();
}

/**
 * Transform API response to dashboard format
 */
export function transformApiResponse(apiResponse) {
  const riskToType = {
    'CRITICAL': 'HALLUCINATION_DETECTED',
    'HIGH': 'HIGH_UNCERTAINTY',
    'MEDIUM': 'STANDARD',
    'LOW': 'STANDARD',
  };
  
  return {
    id: apiResponse.note_id,
    patientId: apiResponse.patient_id,
    patientName: 'Loading...',
    from: 'AI Scribe',
    subject: 'Clinical Note',
    author: 'Cerner AI Scribe',
    description: 'AI Generated Note',
    createDate: new Date(apiResponse.analyzed_at).toLocaleDateString('en-GB', {
      day: '2-digit', month: 'short', year: 'numeric'
    }).replace(/ /g, '-'),
    status: apiResponse.overall_risk === 'LOW' ? 'Ready' : 'Review Required',
    type: riskToType[apiResponse.overall_risk] || 'STANDARD',
    semanticEntropy: apiResponse.review_queue?.[0]?.entropy?.entropy || 0,
    assigned: 'Raubenheimer...',
    reviewLevel: apiResponse.review_burden?.detailed_review > 0 ? 'Detailed' : 
                 apiResponse.review_burden?.standard_review > 0 ? 'Standard' : 'Brief',
    ehrVerification: {
      claims: apiResponse.review_queue?.map(item => ({
        type: item.claim.claim_type,
        text: item.claim.text,
        verified: item.verification.status === 'verified',
        status: item.verification.status,
        explanation: item.verification.explanation,
        priority: item.priority_score,
      })) || [],
      totalClaims: apiResponse.summary?.total_claims || 0,
      verified: apiResponse.summary?.verified || 0,
      contradictions: apiResponse.summary?.contradictions || 0,
      unverified: apiResponse.summary?.needs_review || 0,
    },
    reviewBurden: apiResponse.review_burden,
  };
}
