import React, { useState, useEffect } from 'react';

/**
 * TRUST AI Governance Pipeline Dashboard (IT-facing)
 * 
 * Visual architecture showing how semantic entropy and source verification
 * work together for practical AI scribe note governance.
 * 
 * Connected to live TRUST backend API for real-time analysis.
 */

const API_BASE_URL = 'https://api.trustplatform.ca/api/v1';

// Sample data showing the pipeline in action
const sampleNote = {
  originalAudio: `"Good morning Mr. Jackson, I'm Dr. Raubenheimer. I see you're here for your pre-op 
assessment for your aortic valve surgery. How are you feeling today? Any chest pain or shortness 
of breath? Good. I see you're on metoprolol for your blood pressure. Any allergies to medications? 
No known allergies, great. Your echo from last week shows moderate aortic stenosis with a valve 
area of 1.1 cm squared. We'll proceed with the AVR as planned for next Tuesday."`,
  
  aiScribeOutput: `PRE-OPERATIVE ANESTHESIA ASSESSMENT

Patient: Scott Jackson
DOB: 03/15/1958
Date of Service: 12/27/2025
Procedure: Aortic Valve Replacement (AVR)

HISTORY OF PRESENT ILLNESS:
67-year-old male presents for pre-operative assessment prior to scheduled AVR.
Patient denies chest pain or shortness of breath.

CURRENT MEDICATIONS:
- Metoprolol 50mg PO BID
- Lisinopril 10mg PO daily
- Aspirin 81mg PO daily

ALLERGIES: NKDA

CARDIAC HISTORY:
- Moderate aortic stenosis (AVA 1.1 cm²)
- Hypertension, controlled

PLAN:
Proceed with AVR as scheduled. Patient cleared for surgery from anesthesia standpoint.

Electronically signed by: AI Scribe (pending physician review)`,

  extractedClaims: [
    { id: 1, text: 'Patient: Scott Jackson', type: 'Demographics', inSource: true, confidence: 98, seScore: null, risk: 'low', status: 'verified' },
    { id: 2, text: 'DOB: 03/15/1958', type: 'Demographics', inSource: false, confidence: null, seScore: null, risk: 'medium', status: 'unverified', note: 'DOB not mentioned in audio' },
    { id: 3, text: 'Procedure: Aortic Valve Replacement', type: 'Procedure', inSource: true, confidence: 96, seScore: 0.18, risk: 'high', status: 'verified' },
    { id: 4, text: 'Denies chest pain', type: 'Symptom', inSource: true, confidence: 94, seScore: null, risk: 'medium', status: 'verified' },
    { id: 5, text: 'Denies shortness of breath', type: 'Symptom', inSource: true, confidence: 94, seScore: null, risk: 'medium', status: 'verified' },
    { id: 6, text: 'Metoprolol 50mg PO BID', type: 'Medication', inSource: false, confidence: 45, seScore: 1.42, risk: 'high', status: 'flagged', note: 'Dosage/frequency not in source' },
    { id: 7, text: 'Lisinopril 10mg PO daily', type: 'Medication', inSource: false, confidence: 22, seScore: 1.89, risk: 'high', status: 'hallucination', note: 'Medication not mentioned in audio' },
    { id: 8, text: 'Aspirin 81mg PO daily', type: 'Medication', inSource: false, confidence: 28, seScore: 1.76, risk: 'high', status: 'hallucination', note: 'Medication not mentioned in audio' },
    { id: 9, text: 'NKDA', type: 'Allergy', inSource: true, confidence: 97, seScore: 0.12, risk: 'high', status: 'verified' },
    { id: 10, text: 'Moderate aortic stenosis', type: 'Diagnosis', inSource: true, confidence: 95, seScore: 0.22, risk: 'high', status: 'verified' },
    { id: 11, text: 'AVA 1.1 cm²', type: 'Measurement', inSource: true, confidence: 98, seScore: 0.08, risk: 'high', status: 'verified' },
    { id: 12, text: 'Hypertension, controlled', type: 'Diagnosis', inSource: false, confidence: 71, seScore: 0.68, risk: 'medium', status: 'inferred', note: 'Inferred from metoprolol mention' },
    { id: 13, text: 'Proceed with AVR as scheduled', type: 'Plan', inSource: true, confidence: 96, seScore: 0.15, risk: 'high', status: 'verified' },
  ]
};

// Status badge component
const StatusBadge = ({ status }) => {
  const styles = {
    verified: { bg: '#dcfce7', color: '#166534', label: '✓ Verified' },
    flagged: { bg: '#fef3c7', color: '#92400e', label: '⚠ Flagged' },
    hallucination: { bg: '#fee2e2', color: '#991b1b', label: '✗ Hallucination' },
    unverified: { bg: '#f3f4f6', color: '#4b5563', label: '? Unverified' },
    inferred: { bg: '#dbeafe', color: '#1e40af', label: '~ Inferred' },
  };
  const style = styles[status] || styles.unverified;
  
  return (
    <span style={{
      padding: '2px 8px',
      borderRadius: '4px',
      fontSize: '11px',
      fontWeight: '600',
      backgroundColor: style.bg,
      color: style.color,
    }}>
      {style.label}
    </span>
  );
};

// Risk indicator
const RiskBadge = ({ risk }) => {
  const colors = {
    high: '#ef4444',
    medium: '#f59e0b',
    low: '#10b981',
  };
  return (
    <span style={{
      display: 'inline-block',
      width: '8px',
      height: '8px',
      borderRadius: '50%',
      backgroundColor: colors[risk] || colors.low,
      marginRight: '4px',
    }} />
  );
};

// Semantic Entropy visual
const SEIndicator = ({ score }) => {
  if (score === null) return <span style={{ color: '#9ca3af' }}>—</span>;
  
  let color = '#10b981';
  if (score > 0.5) color = '#f59e0b';
  if (score > 1.0) color = '#ef4444';
  
  const width = Math.min(score / 2 * 100, 100);
  
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
      <div style={{ 
        width: '60px', 
        height: '6px', 
        backgroundColor: '#e5e7eb', 
        borderRadius: '3px',
        overflow: 'hidden'
      }}>
        <div style={{ 
          width: `${width}%`, 
          height: '100%', 
          backgroundColor: color,
          borderRadius: '3px'
        }} />
      </div>
      <span style={{ fontSize: '11px', color, fontWeight: '600' }}>{score.toFixed(2)}</span>
    </div>
  );
};

// Pipeline stage component
const PipelineStage = ({ number, title, subtitle, isActive, children }) => (
  <div style={{
    background: isActive ? '#f0f9ff' : 'white',
    border: `2px solid ${isActive ? '#0891b2' : '#e5e7eb'}`,
    borderRadius: '12px',
    padding: '20px',
    marginBottom: '16px',
  }}>
    <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '12px' }}>
      <div style={{
        width: '32px',
        height: '32px',
        borderRadius: '50%',
        backgroundColor: isActive ? '#0891b2' : '#9ca3af',
        color: 'white',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        fontWeight: '700',
        fontSize: '14px',
      }}>
        {number}
      </div>
      <div>
        <div style={{ fontWeight: '600', color: '#0f172a' }}>{title}</div>
        <div style={{ fontSize: '12px', color: '#64748b' }}>{subtitle}</div>
      </div>
    </div>
    {children}
  </div>
);

// EHR Verification Status Component
const EHRVerificationCard = ({ ehrResult, loading }) => {
  if (loading) {
    return (
      <div style={{
        background: '#f0f9ff',
        border: '2px solid #0891b2',
        borderRadius: '12px',
        padding: '20px',
        marginBottom: '20px',
        textAlign: 'center'
      }}>
        <div style={{ fontSize: '14px', color: '#0891b2' }}>🔄 Verifying against Cerner EHR...</div>
      </div>
    );
  }
  
  if (!ehrResult) return null;
  
  return (
    <div style={{
      background: 'linear-gradient(135deg, #0f172a 0%, #1e293b 100%)',
      borderRadius: '12px',
      padding: '24px',
      marginBottom: '20px',
      color: 'white'
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '16px' }}>
        <svg width="28" height="28" viewBox="0 0 90 90" fill="none">
          <circle cx="45" cy="45" r="32" stroke="#334155" strokeWidth="4" fill="none"/>
          <path d="M45 13 A32 32 0 1 1 17 52" stroke="#0891b2" strokeWidth="4" fill="none" strokeLinecap="round"/>
          <circle cx="17" cy="52" r="7" fill="#0891b2"/>
          <path d="M13 52 L16 55 L22 48" stroke="#ffffff" strokeWidth="2" fill="none"/>
        </svg>
        <div>
          <h3 style={{ margin: 0, fontSize: '18px' }}>Live Cerner EHR Verification</h3>
          <div style={{ fontSize: '12px', color: '#94a3b8' }}>Patient: {ehrResult.patient_name}</div>
        </div>
      </div>
      
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px' }}>
        <div style={{ background: 'rgba(16, 185, 129, 0.2)', padding: '16px', borderRadius: '8px', textAlign: 'center' }}>
          <div style={{ fontSize: '28px', fontWeight: '700', color: '#10b981' }}>{ehrResult.verified}</div>
          <div style={{ fontSize: '11px', color: '#94a3b8' }}>Verified in EHR</div>
        </div>
        <div style={{ background: 'rgba(245, 158, 11, 0.2)', padding: '16px', borderRadius: '8px', textAlign: 'center' }}>
          <div style={{ fontSize: '28px', fontWeight: '700', color: '#f59e0b' }}>{ehrResult.not_in_ehr}</div>
          <div style={{ fontSize: '11px', color: '#94a3b8' }}>Not in EHR</div>
        </div>
        <div style={{ background: 'rgba(239, 68, 68, 0.2)', padding: '16px', borderRadius: '8px', textAlign: 'center' }}>
          <div style={{ fontSize: '28px', fontWeight: '700', color: '#ef4444' }}>{ehrResult.contradicted}</div>
          <div style={{ fontSize: '11px', color: '#94a3b8' }}>Contradicted</div>
        </div>
        <div style={{ background: 'rgba(139, 92, 246, 0.2)', padding: '16px', borderRadius: '8px', textAlign: 'center' }}>
          <div style={{ fontSize: '28px', fontWeight: '700', color: '#8b5cf6' }}>{ehrResult.ehr_medications_count}</div>
          <div style={{ fontSize: '11px', color: '#94a3b8' }}>Meds on File</div>
        </div>
      </div>
      
      {ehrResult.results && ehrResult.results.length > 0 && (
        <div style={{ marginTop: '16px', maxHeight: '150px', overflow: 'auto' }}>
          {ehrResult.results.map((claim, idx) => (
            <div key={idx} style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              padding: '8px 12px',
              background: 'rgba(255,255,255,0.05)',
              borderRadius: '6px',
              marginBottom: '4px',
              borderLeft: `3px solid ${
                claim.status === 'verified' ? '#10b981' :
                claim.status === 'not_in_ehr' ? '#f59e0b' : '#ef4444'
              }`
            }}>
              <div>
                <div style={{ fontSize: '12px', fontWeight: '500' }}>{claim.claim_text}</div>
                <div style={{ fontSize: '10px', color: '#94a3b8' }}>{claim.explanation}</div>
              </div>
              <span style={{
                padding: '2px 8px',
                borderRadius: '4px',
                fontSize: '10px',
                fontWeight: '600',
                background: claim.status === 'verified' ? '#10b981' :
                           claim.status === 'not_in_ehr' ? '#f59e0b' : '#ef4444',
                color: 'white'
              }}>
                {claim.status.toUpperCase()}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default function GovernanceDashboard() {
  const [activeStage, setActiveStage] = useState(5);
  const [showDetails, setShowDetails] = useState(false);
  const [ehrResult, setEhrResult] = useState(null);
  const [ehrLoading, setEhrLoading] = useState(false);
  const [apiStatus, setApiStatus] = useState('checking');

  // Check API connection on mount
  useEffect(() => {
    const checkApi = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/scribe/health`);
        if (response.ok) {
          setApiStatus('connected');
        } else {
          setApiStatus('error');
        }
      } catch {
        setApiStatus('error');
      }
    };
    checkApi();
  }, []);

  // Fetch EHR verification
  const runEHRVerification = async () => {
    setEhrLoading(true);
    try {
      const noteData = {
        note_id: 'governance-demo',
        patient_id: '12724066',
        patient_name: 'Scott Jackson',
        encounter_type: 'Pre-op',
        sections: {
          medications: 'metoprolol 50mg po bid, lisinopril 10mg po daily, aspirin 81mg po daily, atorvastatin 40mg daily',
          allergies: 'NKDA',
        },
        source_transcript: 'Patient on metoprolol for blood pressure. No known allergies.'
      };
      
      const response = await fetch(`${API_BASE_URL}/scribe/verify-ehr/12724066`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(noteData),
      });
      
      if (response.ok) {
        const data = await response.json();
        setEhrResult(data);
      }
    } catch (err) {
      console.error('EHR verification failed:', err);
    } finally {
      setEhrLoading(false);
    }
  };

  // Calculate summary stats
  const stats = {
    total: sampleNote.extractedClaims.length,
    verified: sampleNote.extractedClaims.filter(c => c.status === 'verified').length,
    flagged: sampleNote.extractedClaims.filter(c => c.status === 'flagged').length,
    hallucinations: sampleNote.extractedClaims.filter(c => c.status === 'hallucination').length,
    inferred: sampleNote.extractedClaims.filter(c => c.status === 'inferred').length,
    unverified: sampleNote.extractedClaims.filter(c => c.status === 'unverified').length,
  };

  const reviewReduction = Math.round((stats.verified / stats.total) * 100);

  return (
    <div style={{ 
      fontFamily: "'Outfit', system-ui, sans-serif", 
      maxWidth: '1400px', 
      margin: '0 auto', 
      padding: '40px 20px',
      backgroundColor: '#f8fafc',
      minHeight: '100vh'
    }}>
      {/* Header */}
      <div style={{ textAlign: 'center', marginBottom: '24px' }}>
        <h1 style={{ 
          fontFamily: "'Sora', sans-serif", 
          fontSize: '28px', 
          color: '#0f172a',
          marginBottom: '8px'
        }}>
          TRUST AI Governance Pipeline
        </h1>
        <p style={{ color: '#64748b', fontSize: '16px', marginBottom: '16px' }}>
          Hybrid Source Verification + Targeted Semantic Entropy + EHR Validation
        </p>
        
        {/* API Status */}
        <div style={{ display: 'flex', justifyContent: 'center', gap: '16px', alignItems: 'center' }}>
          <span style={{
            padding: '4px 12px',
            borderRadius: '20px',
            fontSize: '12px',
            fontWeight: '600',
            background: apiStatus === 'connected' ? '#dcfce7' : apiStatus === 'error' ? '#fee2e2' : '#f3f4f6',
            color: apiStatus === 'connected' ? '#166534' : apiStatus === 'error' ? '#991b1b' : '#4b5563',
          }}>
            {apiStatus === 'connected' ? '🟢 Backend Connected' : 
             apiStatus === 'error' ? '🔴 Backend Offline' : '⏳ Checking...'}
          </span>
          
          <button
            onClick={runEHRVerification}
            disabled={apiStatus !== 'connected' || ehrLoading}
            style={{
              padding: '8px 20px',
              background: apiStatus === 'connected' ? '#0891b2' : '#9ca3af',
              color: 'white',
              border: 'none',
              borderRadius: '8px',
              cursor: apiStatus === 'connected' ? 'pointer' : 'not-allowed',
              fontWeight: '600',
              fontSize: '13px',
            }}
          >
            {ehrLoading ? '⏳ Verifying...' : '🏥 Run Live EHR Verification'}
          </button>
        </div>
      </div>

      {/* EHR Verification Results */}
      <EHRVerificationCard ehrResult={ehrResult} loading={ehrLoading} />

      {/* Architecture Overview */}
      <div style={{ 
        display: 'grid', 
        gridTemplateColumns: '1fr 2fr', 
        gap: '24px',
        marginBottom: '40px'
      }}>
        {/* Left: Pipeline Stages */}
        <div>
          <h2 style={{ fontSize: '18px', fontWeight: '600', marginBottom: '16px', color: '#0f172a' }}>
            Processing Pipeline
          </h2>
          
          <PipelineStage 
            number="1" 
            title="Input Capture" 
            subtitle="Audio/Transcript + AI Output"
            isActive={activeStage >= 1}
          >
            <div style={{ fontSize: '12px', color: '#64748b' }}>
              • Source audio/transcript preserved<br/>
              • AI scribe note captured<br/>
              • Timestamp alignment
            </div>
          </PipelineStage>

          <PipelineStage 
            number="2" 
            title="Claim Extraction" 
            subtitle="NLP parses discrete assertions"
            isActive={activeStage >= 2}
          >
            <div style={{ fontSize: '12px', color: '#64748b' }}>
              • Extract clinical claims<br/>
              • Categorize by type & risk<br/>
              • {stats.total} claims extracted
            </div>
          </PipelineStage>

          <PipelineStage 
            number="3" 
            title="Source Verification" 
            subtitle="Fast first-pass validation"
            isActive={activeStage >= 3}
          >
            <div style={{ fontSize: '12px', color: '#64748b' }}>
              • Match claims to transcript<br/>
              • Direct match → Pass<br/>
              • {stats.verified}/{stats.total} verified ({reviewReduction}%)
            </div>
          </PipelineStage>

          <PipelineStage 
            number="4" 
            title="Semantic Entropy" 
            subtitle="Targeted uncertainty analysis"
            isActive={activeStage >= 4}
          >
            <div style={{ fontSize: '12px', color: '#64748b' }}>
              • Only on unverified claims<br/>
              • 10 regenerations per claim<br/>
              • SE {'>'} 0.5 → Flag for review
            </div>
          </PipelineStage>

          <PipelineStage 
            number="5" 
            title="EHR Verification" 
            subtitle="Real-time Cerner validation"
            isActive={activeStage >= 5}
          >
            <div style={{ fontSize: '12px', color: '#64748b' }}>
              • Compare to actual EHR data<br/>
              • Verify medications & allergies<br/>
              • Catch fabricated claims
            </div>
          </PipelineStage>

          <PipelineStage 
            number="6" 
            title="Review Triage" 
            subtitle="Tiered physician workflow"
            isActive={activeStage >= 6}
          >
            <div style={{ fontSize: '12px', color: '#64748b' }}>
              • Brief: All green<br/>
              • Standard: Some flags<br/>
              • Detailed: Hallucinations detected
            </div>
          </PipelineStage>
        </div>

        {/* Right: Claim Analysis Table */}
        <div>
          <div style={{ 
            display: 'flex', 
            justifyContent: 'space-between', 
            alignItems: 'center',
            marginBottom: '16px'
          }}>
            <h2 style={{ fontSize: '18px', fontWeight: '600', color: '#0f172a' }}>
              Claim-Level Analysis
            </h2>
            <button 
              onClick={() => setShowDetails(!showDetails)}
              style={{
                padding: '6px 12px',
                backgroundColor: '#0891b2',
                color: 'white',
                border: 'none',
                borderRadius: '6px',
                cursor: 'pointer',
                fontSize: '12px',
              }}
            >
              {showDetails ? 'Hide Notes' : 'Show Notes'}
            </button>
          </div>

          {/* Summary Cards */}
          <div style={{ 
            display: 'grid', 
            gridTemplateColumns: 'repeat(5, 1fr)', 
            gap: '12px',
            marginBottom: '16px'
          }}>
            <div style={{ background: '#dcfce7', padding: '12px', borderRadius: '8px', textAlign: 'center' }}>
              <div style={{ fontSize: '24px', fontWeight: '700', color: '#166534' }}>{stats.verified}</div>
              <div style={{ fontSize: '11px', color: '#166534' }}>Verified</div>
            </div>
            <div style={{ background: '#dbeafe', padding: '12px', borderRadius: '8px', textAlign: 'center' }}>
              <div style={{ fontSize: '24px', fontWeight: '700', color: '#1e40af' }}>{stats.inferred}</div>
              <div style={{ fontSize: '11px', color: '#1e40af' }}>Inferred</div>
            </div>
            <div style={{ background: '#f3f4f6', padding: '12px', borderRadius: '8px', textAlign: 'center' }}>
              <div style={{ fontSize: '24px', fontWeight: '700', color: '#4b5563' }}>{stats.unverified}</div>
              <div style={{ fontSize: '11px', color: '#4b5563' }}>Unverified</div>
            </div>
            <div style={{ background: '#fef3c7', padding: '12px', borderRadius: '8px', textAlign: 'center' }}>
              <div style={{ fontSize: '24px', fontWeight: '700', color: '#92400e' }}>{stats.flagged}</div>
              <div style={{ fontSize: '11px', color: '#92400e' }}>Flagged</div>
            </div>
            <div style={{ background: '#fee2e2', padding: '12px', borderRadius: '8px', textAlign: 'center' }}>
              <div style={{ fontSize: '24px', fontWeight: '700', color: '#991b1b' }}>{stats.hallucinations}</div>
              <div style={{ fontSize: '11px', color: '#991b1b' }}>Hallucinations</div>
            </div>
          </div>

          {/* Claims Table */}
          <div style={{ 
            background: 'white', 
            borderRadius: '8px', 
            border: '1px solid #e5e7eb',
            overflow: 'hidden'
          }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '12px' }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  <th style={{ padding: '10px', textAlign: 'left', borderBottom: '1px solid #e5e7eb' }}>Claim</th>
                  <th style={{ padding: '10px', textAlign: 'left', borderBottom: '1px solid #e5e7eb' }}>Type</th>
                  <th style={{ padding: '10px', textAlign: 'center', borderBottom: '1px solid #e5e7eb' }}>In Source?</th>
                  <th style={{ padding: '10px', textAlign: 'left', borderBottom: '1px solid #e5e7eb' }}>SE Score</th>
                  <th style={{ padding: '10px', textAlign: 'left', borderBottom: '1px solid #e5e7eb' }}>Status</th>
                </tr>
              </thead>
              <tbody>
                {sampleNote.extractedClaims.map((claim, idx) => (
                  <React.Fragment key={claim.id}>
                    <tr style={{ 
                      background: claim.status === 'hallucination' ? '#fef2f2' : 
                                  claim.status === 'flagged' ? '#fffbeb' : 
                                  idx % 2 === 0 ? 'white' : '#f9fafb'
                    }}>
                      <td style={{ padding: '8px 10px', borderBottom: '1px solid #f3f4f6' }}>
                        <RiskBadge risk={claim.risk} />
                        {claim.text}
                      </td>
                      <td style={{ padding: '8px 10px', borderBottom: '1px solid #f3f4f6', color: '#64748b' }}>
                        {claim.type}
                      </td>
                      <td style={{ padding: '8px 10px', borderBottom: '1px solid #f3f4f6', textAlign: 'center' }}>
                        {claim.inSource ? '✓' : '✗'}
                      </td>
                      <td style={{ padding: '8px 10px', borderBottom: '1px solid #f3f4f6' }}>
                        <SEIndicator score={claim.seScore} />
                      </td>
                      <td style={{ padding: '8px 10px', borderBottom: '1px solid #f3f4f6' }}>
                        <StatusBadge status={claim.status} />
                      </td>
                    </tr>
                    {showDetails && claim.note && (
                      <tr style={{ background: '#fefce8' }}>
                        <td colSpan="5" style={{ 
                          padding: '6px 10px 6px 30px', 
                          borderBottom: '1px solid #f3f4f6',
                          fontSize: '11px',
                          color: '#92400e',
                          fontStyle: 'italic'
                        }}>
                          💡 {claim.note}
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Review Recommendation */}
      <div style={{
        background: 'linear-gradient(135deg, #0f172a 0%, #1e293b 100%)',
        borderRadius: '12px',
        padding: '24px',
        color: 'white',
        marginBottom: '40px'
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <h3 style={{ fontFamily: "'Sora', sans-serif", fontSize: '18px', marginBottom: '8px' }}>
              TRUST Review Recommendation
            </h3>
            <p style={{ color: '#94a3b8', fontSize: '14px', margin: 0 }}>
              Based on {stats.hallucinations} hallucination(s) and {stats.flagged} flagged claim(s) detected
            </p>
          </div>
          <div style={{ textAlign: 'right' }}>
            <div style={{
              display: 'inline-block',
              padding: '8px 20px',
              background: '#dc2626',
              borderRadius: '8px',
              fontWeight: '600',
              fontSize: '16px',
            }}>
              ⚠️ DETAILED REVIEW
            </div>
            <div style={{ fontSize: '12px', color: '#94a3b8', marginTop: '4px' }}>
              Physician must verify all medication claims
            </div>
          </div>
        </div>

        {/* Key Issues */}
        <div style={{ 
          marginTop: '20px', 
          padding: '16px', 
          background: 'rgba(239, 68, 68, 0.1)', 
          borderRadius: '8px',
          border: '1px solid rgba(239, 68, 68, 0.3)'
        }}>
          <div style={{ fontWeight: '600', marginBottom: '8px', color: '#fca5a5' }}>
            ⚠️ Critical Issues Detected:
          </div>
          <ul style={{ margin: 0, paddingLeft: '20px', color: '#fecaca', fontSize: '13px' }}>
            <li><strong>Lisinopril 10mg PO daily</strong> — Not mentioned in source audio (SE: 1.89)</li>
            <li><strong>Aspirin 81mg PO daily</strong> — Not mentioned in source audio (SE: 1.76)</li>
            <li><strong>Metoprolol 50mg PO BID</strong> — Dosage/frequency not in source (SE: 1.42)</li>
          </ul>
        </div>
      </div>

      {/* Computational Efficiency */}
      <div style={{
        background: 'white',
        borderRadius: '12px',
        padding: '24px',
        border: '1px solid #e5e7eb'
      }}>
        <h3 style={{ fontFamily: "'Sora', sans-serif", fontSize: '18px', marginBottom: '16px', color: '#0f172a' }}>
          Computational Efficiency
        </h3>
        
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '20px' }}>
          <div style={{ padding: '16px', background: '#f0fdf4', borderRadius: '8px' }}>
            <div style={{ fontSize: '12px', color: '#166534', fontWeight: '600', marginBottom: '8px' }}>
              NAIVE APPROACH
            </div>
            <div style={{ fontSize: '24px', fontWeight: '700', color: '#166534' }}>~2,000</div>
            <div style={{ fontSize: '12px', color: '#166534' }}>API calls (full note × 10 regenerations)</div>
            <div style={{ fontSize: '11px', color: '#15803d', marginTop: '4px' }}>Cost: ~$2-5 per note</div>
          </div>
          
          <div style={{ padding: '16px', background: '#f0f9ff', borderRadius: '8px' }}>
            <div style={{ fontSize: '12px', color: '#0369a1', fontWeight: '600', marginBottom: '8px' }}>
              TRUST HYBRID APPROACH
            </div>
            <div style={{ fontSize: '24px', fontWeight: '700', color: '#0369a1' }}>~65</div>
            <div style={{ fontSize: '12px', color: '#0369a1' }}>API calls (13 claims × 5 unverified × 10 regen)</div>
            <div style={{ fontSize: '11px', color: '#0284c7', marginTop: '4px' }}>Cost: ~$0.05-0.15 per note</div>
          </div>
          
          <div style={{ padding: '16px', background: '#fdf4ff', borderRadius: '8px' }}>
            <div style={{ fontSize: '12px', color: '#86198f', fontWeight: '600', marginBottom: '8px' }}>
              EFFICIENCY GAIN
            </div>
            <div style={{ fontSize: '24px', fontWeight: '700', color: '#86198f' }}>97%</div>
            <div style={{ fontSize: '12px', color: '#86198f' }}>Reduction in compute</div>
            <div style={{ fontSize: '11px', color: '#a21caf', marginTop: '4px' }}>Same detection accuracy</div>
          </div>
        </div>

        <div style={{ 
          marginTop: '20px', 
          padding: '16px', 
          background: '#f8fafc', 
          borderRadius: '8px',
          fontSize: '13px',
          color: '#475569'
        }}>
          <strong>Key Insight:</strong> Source verification eliminates {reviewReduction}% of claims from SE analysis. 
          Only high-risk unverified claims (medications, dosages, procedures not found in transcript) 
          require the expensive multi-regeneration semantic entropy check.
        </div>
      </div>
    </div>
  );
}
