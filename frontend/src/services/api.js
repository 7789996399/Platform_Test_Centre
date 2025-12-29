/**
 * TRUST Platform - API Service
 */

const API_BASE_URL = 'https://api.trustplatform.ca/api/v1';

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

export async function checkHealth() {
  const response = await fetch(`${API_BASE_URL}/scribe/health`);
  return response.json();
}
/**
 * Get patient context from Cerner EHR.
 */
export async function getCernerPatient(patientId) {
  const response = await fetch(`${API_BASE_URL}/cerner/patient/${patientId}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch patient: ${response.statusText}`);
  }
  return response.json();
}

/**
 * Verify AI note against real EHR data.
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
 * Audit Logging Functions
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
    // Don't throw - audit logging shouldn't break the app
  }
}

export async function getAuditLogs(filters = {}) {
  const params = new URLSearchParams(filters);
  const response = await fetch(`${API_BASE_URL}/audit/logs?${params}`);
  return response.json();
}

export async function getAuditSummary(days = 7) {
  const response = await fetch(`${API_BASE_URL}/audit/summary?days=${days}`);
  return response.json();
}
