/**
 * TRUST Platform - API Service
 */

const API_BASE_URL = 'http://127.0.0.1:8000/api/v1';

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
