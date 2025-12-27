/**
 * TRUST Platform - Main Application
 * ===================================
 * Auditing AI. Protecting Patients. Empowering Physicians.
 */

import React, { useState } from 'react';
import TrustDashboard from './components/dashboard/TrustDashboard';
import EHRVerification from './components/dashboard/EHRVerification';
import { analyzeNote, verifyAgainstEHR } from './services/api';

// TRUST Brand Colors
const COLORS = {
  teal: '#0891b2',
  slate: {
    50: '#f8fafc',
    100: '#f1f5f9',
    800: '#1e293b',
    900: '#0f172a',
  }
};

// Sample test data
const SAMPLE_NOTE = {
  note_id: "demo-001",
  patient_id: "patient-12345",
  patient_name: "Scott Jackson",
  encounter_type: "Pre-operative Assessment",
  encounter_date: "2025-12-24",
  sections: {
    chief_complaint: "67-year-old male presents for pre-operative assessment prior to scheduled aortic valve replacement.",
    history_of_present_illness: "Patient reports progressive dyspnea on exertion over the past 6 months. Denies chest pain at rest.",
    medications: "Metoprolol 50mg PO BID, Lisinopril 10mg PO daily, Aspirin 81mg PO daily, Atorvastatin 40mg PO daily",
    allergies: "No known drug allergies (NKDA)",
    cardiac_history: "Moderate aortic stenosis (valve area 1.1 cm²). Hypertension controlled on current regimen.",
    physical_exam: "BP 132/78, HR 68, SpO2 98% on room air. 3/6 systolic ejection murmur at right upper sternal border.",
    assessment_plan: "Patient cleared for AVR from anesthesia standpoint. Continue beta-blocker perioperatively."
  },
  source_transcript: "Good morning Mr. Jackson, I'm Dr. Raubenheimer. I see you're here for your pre-op assessment for your aortic valve surgery. How are you feeling today? Pretty good, just get a bit winded going up stairs. Any chest pain? No, not really. Good. Let me check your medications - you're on metoprolol? Yes, 50 twice a day. And lisinopril? Yes, 10 in the morning. Aspirin? The baby aspirin, yes. And a statin? Atorvastatin 40. Any allergies to medications? None that I know of. Great. Your echo shows moderate aortic stenosis with a valve area of 1.1 cm squared. Blood pressure today is 132/78, heart rate 68, oxygen is 98%. I can hear the murmur. We'll proceed with the AVR as planned. Keep taking your metoprolol."
};

// Sample with hallucinations for testing
const SAMPLE_HALLUCINATED = {
  note_id: "demo-002",
  patient_id: "patient-67890",
  patient_name: "Maria Garcia",
  encounter_type: "Pre-operative Assessment",
  encounter_date: "2025-12-24",
  sections: {
    chief_complaint: "59-year-old female presents for pre-operative assessment prior to scheduled CABG.",
    medications: "Metoprolol 25mg BID, Lisinopril 5mg daily, Aspirin 325mg daily, Clopidogrel 75mg daily, Warfarin 5mg daily",
    allergies: "Penicillin (rash), Sulfa drugs (anaphylaxis)",
    physical_exam: "BP 145/88, HR 72. Irregular rhythm. Bilateral lower extremity edema 1+.",
    assessment_plan: "High-risk patient for CABG. Hold Clopidogrel 5 days pre-op. Consider IABP standby."
  },
  source_transcript: "Good afternoon Mrs. Garcia. Any chest pain recently? Yes, when I walk too far, goes away when I rest. What medications are you on? Metoprolol 25 twice a day, lisinopril 5 in the morning, aspirin 325 one a day, and Plavix 75. Any blood thinners like warfarin? No, just the aspirin and Plavix. Any allergies? Penicillin gives me a rash. Any other allergies? No, just that one. Blood pressure is 145/88, heart rate 72. Rhythm sounds a bit irregular. No swelling in your legs. We'll need to stop the Plavix five days before surgery."
};

function App() {
  const [analysisResult, setAnalysisResult] = useState(null);
  const [ehrResult, setEhrResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedSample, setSelectedSample] = useState('clean');
  

  const handleAnalyze = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const noteData = selectedSample === 'clean' ? SAMPLE_NOTE : SAMPLE_HALLUCINATED;
      const result = await analyzeNote(noteData);
      setAnalysisResult(result); 
      const ehrData = await verifyAgainstEHR('12724066', noteData);
      setEhrResult (ehrData);

    } catch (err) {
      setError(err.message);
      console.error('Analysis failed:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ 
      minHeight: '100vh',
      background: COLORS.slate[100]
    }}>
      {/* Header */}
      <header style={{
        background: COLORS.slate[900],
        color: 'white',
        padding: '16px 24px',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          {/* TRUST Logo Ring */}
          <svg width="40" height="40" viewBox="0 0 40 40">
            <circle cx="20" cy="20" r="15" fill="none" stroke="#334155" strokeWidth="3"/>
            <path d="M20 5 A15 15 0 1 1 8 30" fill="none" stroke={COLORS.teal} strokeWidth="3" strokeLinecap="round"/>
            <circle cx="8" cy="30" r="5" fill={COLORS.teal}/>
            <path d="M5 30 L7 32 L11 27" stroke="white" strokeWidth="1.5" fill="none" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
          <div>
            <h1 style={{ margin: 0, fontSize: '20px', fontWeight: '700' }}>
              TRUST Platform
            </h1>
            <p style={{ margin: 0, fontSize: '11px', color: '#94a3b8' }}>
              Auditing AI. Protecting Patients. Empowering Physicians.
            </p>
          </div>
        </div>
        
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          {/* Sample selector */}
          <select
            value={selectedSample}
            onChange={(e) => setSelectedSample(e.target.value)}
            style={{
              padding: '8px 12px',
              borderRadius: '6px',
              border: 'none',
              background: COLORS.slate[800],
              color: 'white',
              fontSize: '14px',
              cursor: 'pointer'
            }}
          >
            <option value="clean">Clean Note (No Errors)</option>
            <option value="hallucinated">Hallucinated Note (With Errors)</option>
          </select>
          
          {/* Analyze button */}
          <button
            onClick={handleAnalyze}
            disabled={loading}
            style={{
              background: COLORS.teal,
              color: 'white',
              border: 'none',
              padding: '10px 24px',
              borderRadius: '6px',
              fontWeight: '600',
              fontSize: '14px',
              cursor: loading ? 'wait' : 'pointer',
              opacity: loading ? 0.7 : 1
            }}
          >
            {loading ? 'Analyzing...' : 'Analyze Note'}
          </button>
        </div>
      </header>

      {/* Error display */}
      {error && (
        <div style={{
          background: '#fef2f2',
          border: '1px solid #fecaca',
          color: '#dc2626',
          padding: '12px 24px',
          margin: '20px',
          borderRadius: '8px'
        }}>
          <strong>Error:</strong> {error}
          <p style={{ margin: '8px 0 0 0', fontSize: '14px' }}>
            Make sure the backend is running: <code>cd backend && uvicorn app.main:app --reload</code>
          </p>
        </div>
      )}

      {/* Main content */}
      <main>
        <TrustDashboard analysisResult={analysisResult} />
	<EHRVerification ehrResult={ehrResult} />
      </main>

      {/* Footer */}
      <footer style={{
        textAlign: 'center',
        padding: '20px',
        color: COLORS.slate[500],
        fontSize: '13px'
      }}>
        TRUST Platform v0.1.0 | Healthcare AI Governance
      </footer>
    </div>
  );
}

export default App;
