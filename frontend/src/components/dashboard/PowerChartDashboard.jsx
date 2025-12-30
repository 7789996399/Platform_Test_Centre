import React, { useState, useEffect } from 'react';

// API Configuration
const API_BASE_URL = 'https://api.trustplatform.ca/api/v1';

// TRUST Color scheme matching brand guidelines + PowerChart
const COLORS = {
  // PowerChart colors (from screenshot)
  headerTeal: '#006272',
  selectedBlue: '#0078d4',
  folderHover: '#e8f4fc',
  statusGreen: '#008000',
  gridBorder: '#d0d0d0',
  toolbarBg: '#f0f0f0',
  
  // TRUST semantic colors
  trustTeal: '#0891b2',
  trustGreen: '#10b981',
  trustAmber: '#f59e0b',
  trustRed: '#ef4444',
  trustPurple: '#8b5cf6',
};

// Sample AI Documents data with noteData for API calls
const initialAiDocuments = [
  {
    id: 1,
    patientName: 'JACKSON, SCOTT',
    patientId: '12724066',
    from: 'AI Scribe',
    subject: 'Pre-Op Assessment',
    author: 'Cerner AI Scribe',
    description: 'Anesthesiology Consult',
    createDate: '27-Dec-2025',
    status: 'Review Required',
    type: 'HIGH_UNCERTAINTY',
    confidence: 72,
    semanticEntropy: 0.84,
    assigned: 'Raubenheimer...',
    reviewLevel: 'Detailed',
    noteData: {
      note_id: 'note-001',
      patient_id: '12724066',
      patient_name: 'Scott Jackson',
      encounter_type: 'Pre-op',
      sections: {
        medications: 'metoprolol 50mg po bid, lisinopril 10mg po daily, aspirin 81mg po daily, atorvastatin 40mg po daily',
        allergies: 'NKDA',
        history: 'Moderate aortic stenosis, AVA 1.1 cm²',
        vitals: 'BP 132/78, HR 68, SpO2 98%',
        exam: 'Regular rhythm, 3/6 systolic murmur at right upper sternal border'
      },
      source_transcript: 'Patient takes metoprolol 50mg twice daily for blood pressure. No known allergies. Echo shows moderate AS with valve area 1.1.'
    }
  },
  {
    id: 2,
    patientName: 'SMARTS, NANCYS II',
    patientId: '12724066',
    from: 'AI Scribe',
    subject: 'PIFP Block Note',
    author: 'Cerner AI Scribe',
    description: 'Regional Anesthesia',
    createDate: '27-Dec-2025',
    status: 'Pending',
    type: 'STANDARD',
    confidence: 94,
    semanticEntropy: 0.21,
    assigned: 'Raubenheimer...',
    reviewLevel: 'Brief',
    noteData: {
      note_id: 'note-002',
      patient_id: '12724066',
      patient_name: 'Nancys Smarts',
      encounter_type: 'Procedure',
      sections: {
        medications: 'acetaminophen 500mg prn',
        allergies: 'Penicillin allergy, Latex allergy',
        procedure: 'PIFP block performed under ultrasound guidance'
      },
      source_transcript: 'Patient takes tylenol as needed. Allergic to penicillin and latex. Block performed successfully.'
    }
  },
  {
    id: 3,
    patientName: 'CHEN, DAVID',
    patientId: '12724066',
    from: 'AI Scribe',
    subject: 'Anesthesia Consult',
    author: 'Cerner AI Scribe',
    description: 'Anesthesiology Consult',
    createDate: '27-Dec-2025',
    status: 'Pending',
    type: 'STANDARD',
    confidence: 91,
    semanticEntropy: 0.28,
    assigned: 'Raubenheimer...',
    reviewLevel: 'Standard',
    noteData: {
      note_id: 'note-003',
      patient_id: '12724066',
      patient_name: 'David Chen',
      encounter_type: 'Consult',
      sections: {
        medications: 'aspirin 81mg daily, atorvastatin 40mg daily',
        allergies: 'NKDA',
      },
      source_transcript: 'Patient on aspirin and statin. No allergies.'
    }
  },
  {
    id: 4,
    patientName: 'MARTINEZ, ANA',
    patientId: '12724066',
    from: 'AI Scribe',
    subject: 'Post-Op Note',
    author: 'Cerner AI Scribe',
    description: 'Cardiac Surgery',
    createDate: '26-Dec-2025',
    status: 'Flagged',
    type: 'HALLUCINATION_DETECTED',
    confidence: 45,
    semanticEntropy: 1.42,
    assigned: 'Raubenheimer...',
    reviewLevel: 'Detailed',
    flagReason: 'Medication dosage inconsistency detected - Warfarin not in EHR',
    noteData: {
      note_id: 'note-004',
      patient_id: '12724066',
      patient_name: 'Ana Martinez',
      encounter_type: 'Post-op',
      sections: {
        medications: 'warfarin 5mg daily, metformin 1000mg bid, carvedilol 25mg bid',
        allergies: 'Sulfa - anaphylaxis',
        vitals: 'BP 145/92, HR 72, SpO2 96%',
        exam: 'Bilateral leg edema, clear lungs'
      },
      source_transcript: 'Patient on metformin and carvedilol. Allergic to penicillin only. No swelling noted.'
    }
  },
  {
    id: 5,
    patientName: 'ZAD, AXTON',
    from: 'Rajamohan, Ra...',
    subject: 'CARD OR TEE',
    author: 'Rajamohan, Ra...',
    description: 'CARD OR TEE',
    createDate: '25-Dec-2025',
    status: 'Pending',
    type: 'REVIEW_DOC',
    confidence: null,
    semanticEntropy: null,
    assigned: 'Raubenheimer...',
    reviewLevel: null,
  },
];

// Confidence badge component
const ConfidenceBadge = ({ confidence, semanticEntropy }) => {
  if (confidence === null) return <span style={{ color: '#666' }}>—</span>;
  
  let bgColor, textColor;
  if (confidence >= 90) {
    bgColor = '#dcfce7';
    textColor = '#166534';
  } else if (confidence >= 75) {
    bgColor = '#fef3c7';
    textColor = '#92400e';
  } else {
    bgColor = '#fee2e2';
    textColor = '#991b1b';
  }
  
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
      <span style={{
        padding: '2px 8px',
        borderRadius: '4px',
        fontSize: '11px',
        fontWeight: '600',
        backgroundColor: bgColor,
        color: textColor,
      }}>
        {confidence}%
      </span>
      {semanticEntropy !== null && semanticEntropy > 0.5 && (
        <span title={`Semantic Entropy: ${semanticEntropy.toFixed(2)}`} style={{ cursor: 'help' }}>
          ⚠️
        </span>
      )}
    </div>
  );
};

// Review level badge
const ReviewBadge = ({ level }) => {
  if (!level) return <span style={{ color: '#666' }}>—</span>;
  
  const styles = {
    Brief: { bg: '#dbeafe', color: '#1e40af' },
    Standard: { bg: '#e0e7ff', color: '#3730a3' },
    Detailed: { bg: '#fae8ff', color: '#86198f' },
  };
  
  const style = styles[level] || styles.Standard;
  
  return (
    <span style={{
      padding: '2px 8px',
      borderRadius: '4px',
      fontSize: '11px',
      fontWeight: '500',
      backgroundColor: style.bg,
      color: style.color,
    }}>
      {level}
    </span>
  );
};

// Status indicator
const StatusBadge = ({ status, type }) => {
  let color = '#008000';
  let fontWeight = 'normal';
  
  if (status === 'Flagged' || type === 'HALLUCINATION_DETECTED') {
    color = '#dc2626';
    fontWeight = 'bold';
  } else if (status === 'Review Required' || type === 'HIGH_UNCERTAINTY') {
    color = '#d97706';
    fontWeight = 'bold';
  }
  
  return (
    <span style={{ color, fontWeight }}>
      {type === 'HALLUCINATION_DETECTED' && '⚠️ '}
      {status}
    </span>
  );
};

// TRUST Mini Logo
const TrustMiniLogo = ({ size = 24 }) => (
  <svg width={size} height={size} viewBox="0 0 90 90" fill="none">
    <circle cx="45" cy="45" r="32" stroke="#e2e8f0" strokeWidth="4" fill="none"/>
    <path d="M45 13 A32 32 0 1 1 17 52" stroke="#0891b2" strokeWidth="4" fill="none" strokeLinecap="round"/>
    <circle cx="17" cy="52" r="7" fill="#0891b2"/>
    <path d="M13 52 L16 55 L22 48" stroke="#ffffff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" fill="none"/>
  </svg>
);
// TRUST ML Analysis Panel Component - Compact & Non-disruptive
const TRUSTAnalysisPanel = ({ analysis, loading, onClose }) => {
  const [expanded, setExpanded] = useState(false);
  
  if (loading) {
    return (
      <div style={{ padding: '8px 16px', background: '#f8fafc', borderTop: '1px solid #e2e8f0', fontSize: '11px', color: '#64748b', display: 'flex', alignItems: 'center', gap: '8px' }}>
        <span>⏳</span> Running TRUST Analysis...
      </div>
    );
  }
  
  if (!analysis) return null;
  
  const getRiskColor = (risk) => {
    switch (risk) {
      case 'HIGH': return { bg: '#fee2e2', text: '#991b1b' };
      case 'MEDIUM': return { bg: '#fef3c7', text: '#92400e' };
      case 'LOW': return { bg: '#dcfce7', text: '#166534' };
      default: return { bg: '#f1f5f9', text: '#475569' };
    }
  };
  
  const riskStyle = getRiskColor(analysis.overall_risk);

  // Compact summary bar (always visible)
  if (!expanded) {
    return (
      <div style={{ 
        padding: '8px 16px', 
        background: riskStyle.bg, 
        borderTop: '2px solid #8b5cf6',
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'space-between',
        fontSize: '11px'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <span style={{ fontWeight: '600', color: '#8b5cf6' }}>TRUST Analysis</span>
          <span style={{ padding: '2px 8px', borderRadius: '4px', background: riskStyle.text, color: 'white', fontWeight: '600', fontSize: '10px' }}>
            {analysis.overall_risk} RISK
          </span>
          <span>{analysis.summary.total_claims} claims | {analysis.summary.verified} verified | {analysis.summary.contradictions} contradictions</span>
          <span style={{ color: '#166534', fontWeight: '600' }}>{analysis.review_burden.time_saved_percent}% time saved</span>
        </div>
        <div style={{ display: 'flex', gap: '8px' }}>
          <button 
            onClick={() => setExpanded(true)}
            style={{ background: 'transparent', border: '1px solid #8b5cf6', color: '#8b5cf6', padding: '2px 8px', borderRadius: '4px', cursor: 'pointer', fontSize: '10px' }}
          >
            Details ▼
          </button>
          <button 
            onClick={onClose}
            style={{ background: 'transparent', border: 'none', color: '#64748b', cursor: 'pointer', fontSize: '14px' }}
          >
            ×
          </button>
        </div>
      </div>
    );
  }

  // Expanded view (only when user clicks Details)
  return (
    <div style={{ borderTop: '2px solid #8b5cf6', background: '#faf5ff' }}>
      <div style={{ padding: '8px 16px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid #e2e8f0' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ fontWeight: '600', color: '#8b5cf6' }}>TRUST ML Analysis</span>
          <span style={{ padding: '2px 8px', borderRadius: '4px', background: riskStyle.text, color: 'white', fontWeight: '600', fontSize: '10px' }}>
            {analysis.overall_risk} RISK
          </span>
        </div>
        <div style={{ display: 'flex', gap: '8px' }}>
          <button 
            onClick={() => setExpanded(false)}
            style={{ background: 'transparent', border: '1px solid #8b5cf6', color: '#8b5cf6', padding: '2px 8px', borderRadius: '4px', cursor: 'pointer', fontSize: '10px' }}
          >
            Collapse ▲
          </button>
          <button 
            onClick={onClose}
            style={{ background: 'transparent', border: 'none', color: '#64748b', cursor: 'pointer', fontSize: '14px' }}
          >
            ×
          </button>
        </div>
      </div>
      
      {/* Summary Stats */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '8px', padding: '12px 16px' }}>
        <div style={{ background: 'white', padding: '8px', borderRadius: '4px', textAlign: 'center', border: '1px solid #e2e8f0' }}>
          <div style={{ fontSize: '18px', fontWeight: '700', color: '#475569' }}>{analysis.summary.total_claims}</div>
          <div style={{ fontSize: '9px', color: '#64748b' }}>Claims</div>
        </div>
        <div style={{ background: '#dcfce7', padding: '8px', borderRadius: '4px', textAlign: 'center' }}>
          <div style={{ fontSize: '18px', fontWeight: '700', color: '#166534' }}>{analysis.summary.verified}</div>
          <div style={{ fontSize: '9px', color: '#166534' }}>Verified</div>
        </div>
        <div style={{ background: '#fef3c7', padding: '8px', borderRadius: '4px', textAlign: 'center' }}>
          <div style={{ fontSize: '18px', fontWeight: '700', color: '#92400e' }}>{analysis.summary.needs_review}</div>
          <div style={{ fontSize: '9px', color: '#92400e' }}>Review</div>
        </div>
        <div style={{ background: '#fee2e2', padding: '8px', borderRadius: '4px', textAlign: 'center' }}>
          <div style={{ fontSize: '18px', fontWeight: '700', color: '#991b1b' }}>{analysis.summary.contradictions}</div>
          <div style={{ fontSize: '9px', color: '#991b1b' }}>Contradict</div>
        </div>
        <div style={{ background: '#dbeafe', padding: '8px', borderRadius: '4px', textAlign: 'center' }}>
          <div style={{ fontSize: '18px', fontWeight: '700', color: '#1e40af' }}>{analysis.review_burden.time_saved_percent}%</div>
          <div style={{ fontSize: '9px', color: '#1e40af' }}>Saved</div>
        </div>
      </div>

      {/* Review Queue - Compact */}
      {analysis.review_queue && analysis.review_queue.length > 0 && (
        <div style={{ maxHeight: '120px', overflow: 'auto', borderTop: '1px solid #e2e8f0' }}>
          {analysis.review_queue.slice(0, 5).map((item, idx) => (
            <div key={idx} style={{ display: 'flex', justifyContent: 'space-between', padding: '4px 16px', fontSize: '11px', borderBottom: '1px solid #f1f5f9' }}>
              <span><strong>#{item.rank}</strong> {item.claim.text}</span>
              <span style={{ color: '#64748b' }}>{item.uncertainty.review_tier.toUpperCase()}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
// EHR Verification Panel Component
const EHRVerificationPanel = ({ ehrResult, loading }) => {
  if (loading) {
    return (
      <div style={{ padding: '20px', textAlign: 'center', color: '#64748b' }}>
        <div style={{ marginBottom: '8px' }}>🔄 Verifying against Cerner EHR...</div>
        <div style={{ fontSize: '11px' }}>Connecting to FHIR server</div>
      </div>
    );
  }
  
  if (!ehrResult) return null;
  
  const getStatusColor = (status) => {
    switch (status) {
      case 'verified': return COLORS.trustGreen;
      case 'not_in_ehr': return COLORS.trustAmber;
      case 'contradicted': return COLORS.trustRed;
      default: return '#64748b';
    }
  };
  
  return (
    <div style={{ borderTop: '2px solid #0891b2', background: '#f8fafc' }}>
      <div style={{ padding: '12px 16px', borderBottom: '1px solid #e2e8f0' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
          <TrustMiniLogo size={18} />
          <span style={{ fontWeight: '600', color: COLORS.trustTeal }}>EHR Verification Results</span>
          <span style={{ fontSize: '11px', color: '#64748b' }}>| Patient: {ehrResult.patient_name}</span>
        </div>
        
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px' }}>
          <div style={{ background: '#dcfce7', padding: '10px', borderRadius: '6px', textAlign: 'center' }}>
            <div style={{ fontSize: '20px', fontWeight: '700', color: '#166534' }}>{ehrResult.verified}</div>
            <div style={{ fontSize: '10px', color: '#166534' }}>Verified in EHR</div>
          </div>
          <div style={{ background: '#fef3c7', padding: '10px', borderRadius: '6px', textAlign: 'center' }}>
            <div style={{ fontSize: '20px', fontWeight: '700', color: '#92400e' }}>{ehrResult.not_in_ehr}</div>
            <div style={{ fontSize: '10px', color: '#92400e' }}>Not in EHR</div>
          </div>
          <div style={{ background: '#fee2e2', padding: '10px', borderRadius: '6px', textAlign: 'center' }}>
            <div style={{ fontSize: '20px', fontWeight: '700', color: '#991b1b' }}>{ehrResult.contradicted}</div>
            <div style={{ fontSize: '10px', color: '#991b1b' }}>Contradicted</div>
          </div>
          <div style={{ background: '#f1f5f9', padding: '10px', borderRadius: '6px', textAlign: 'center' }}>
            <div style={{ fontSize: '20px', fontWeight: '700', color: '#475569' }}>{ehrResult.ehr_medications_count}</div>
            <div style={{ fontSize: '10px', color: '#475569' }}>EHR Meds on File</div>
          </div>
        </div>
      </div>
      
      {/* Claim-by-claim results */}
      <div style={{ maxHeight: '200px', overflow: 'auto' }}>
        {ehrResult.results && ehrResult.results.map((claim, idx) => (
          <div 
            key={idx}
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              padding: '8px 16px',
              borderBottom: '1px solid #f1f5f9',
              background: claim.status === 'contradicted' ? '#fef2f2' : 
                         claim.status === 'not_in_ehr' ? '#fffbeb' : 'white',
              borderLeft: `3px solid ${getStatusColor(claim.status)}`
            }}
          >
            <div>
              <div style={{ fontWeight: '500', fontSize: '12px' }}>{claim.claim_text}</div>
              <div style={{ fontSize: '10px', color: '#64748b' }}>{claim.explanation}</div>
            </div>
            <span style={{
              padding: '2px 8px',
              borderRadius: '4px',
              fontSize: '10px',
              fontWeight: '600',
              background: getStatusColor(claim.status),
              color: 'white'
            }}>
              {claim.status.toUpperCase().replace('_', ' ')}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default function PowerChartDashboard() {
  const [selectedFolder, setSelectedFolder] = useState('AI Documents');
  const [selectedDoc, setSelectedDoc] = useState(null);
  const [aiDocuments] = useState(initialAiDocuments);
  const [ehrResult, setEhrResult] = useState(null);
  const [ehrLoading, setEhrLoading] = useState(false);
  const [expandedSections, setExpandedSections] = useState({
    inboxItems: true,
    aiReview: true,
    workItems: false,
    notifications: false,
  });

  const toggleSection = (section) => {
    setExpandedSections(prev => ({ ...prev, [section]: !prev[section] }));
  };
const [trustAnalysis, setTrustAnalysis] = useState(null);
// Run TRUST Analysis when document is selected
  const handleSelectDoc = async (doc) => {
    setSelectedDoc(doc);
    setEhrResult(null);
    setTrustAnalysis(null);
    
    if (doc.noteData) {
      setEhrLoading(true);
      try {
        // Run TRUST ML Analysis
        const analysisResponse = await fetch(`${API_BASE_URL}/scribe/analyze`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(doc.noteData),
        });
        if (analysisResponse.ok) {
          const analysisData = await analysisResponse.json();
          setTrustAnalysis(analysisData);
        }
        
        // Also run EHR verification
        if (doc.patientId) {
          const ehrResponse = await fetch(`${API_BASE_URL}/scribe/verify-ehr/${doc.patientId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(doc.noteData),
          });
          if (ehrResponse.ok) {
            const ehrData = await ehrResponse.json();
            setEhrResult(ehrData);
          }
        }
      } catch (err) {
        console.error('Analysis failed:', err);
      } finally {
        setEhrLoading(false);
      }
    }
  };

  // Count AI documents by type
  const highUncertainty = aiDocuments.filter(d => d.type === 'HIGH_UNCERTAINTY').length;
  const hallucinations = aiDocuments.filter(d => d.type === 'HALLUCINATION_DETECTED').length;
  const standardReview = aiDocuments.filter(d => d.type === 'STANDARD').length;
  const totalAI = highUncertainty + hallucinations + standardReview;

  return (
    <div style={{ fontFamily: 'Segoe UI, Tahoma, sans-serif', fontSize: '12px', height: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* Title Bar - Lighter teal matching PowerChart */}
      <div style={{ 
        background: 'linear-gradient(180deg, #5ba4b0 0%, #4a939f 100%)', 
        padding: '4px 8px', 
        borderBottom: '1px solid #3d8591',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        color: 'white'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ 
            background: '#2d7a86', 
            padding: '2px 6px', 
            borderRadius: '2px',
            fontWeight: 'bold',
            fontSize: '12px'
          }}>P</span>
        </div>
        <span style={{ fontSize: '11px', fontWeight: '500' }}>PowerChart Organizer for Raubenheimer, Jean Louis, MD</span>
        <div style={{ display: 'flex', gap: '4px' }}>
          <button style={{ width: '20px', height: '20px', background: 'rgba(255,255,255,0.2)', border: '1px solid rgba(255,255,255,0.3)', cursor: 'pointer', color: 'white' }}>−</button>
          <button style={{ width: '20px', height: '20px', background: 'rgba(255,255,255,0.2)', border: '1px solid rgba(255,255,255,0.3)', cursor: 'pointer', color: 'white' }}>□</button>
          <button style={{ width: '20px', height: '20px', background: '#c42b1c', border: '1px solid #a02218', color: 'white', cursor: 'pointer' }}>×</button>
        </div>
      </div>

      {/* Menu Bar */}
      <div style={{ background: '#f8f8f8', padding: '2px 8px', borderBottom: '1px solid #ccc', fontSize: '11px' }}>
        <span style={{ marginRight: '16px', cursor: 'pointer' }}>Task</span>
        <span style={{ marginRight: '16px', cursor: 'pointer' }}>Edit</span>
        <span style={{ marginRight: '16px', cursor: 'pointer' }}>View</span>
        <span style={{ marginRight: '16px', cursor: 'pointer' }}>Patient</span>
        <span style={{ marginRight: '16px', cursor: 'pointer' }}>Chart</span>
        <span style={{ marginRight: '16px', cursor: 'pointer' }}>Links</span>
        <span style={{ marginRight: '16px', cursor: 'pointer' }}>Notifications</span>
        <span style={{ marginRight: '16px', cursor: 'pointer' }}>Inbox</span>
        <span style={{ cursor: 'pointer' }}>Help</span>
      </div>

      {/* Toolbar Row 1 */}
      <div style={{ background: '#f0f0f0', padding: '4px 8px', borderBottom: '1px solid #ccc', display: 'flex', gap: '8px', flexWrap: 'wrap', fontSize: '10px' }}>
        <span style={{ padding: '2px 6px', cursor: 'pointer' }}>📧 Message Centre</span>
        <span style={{ padding: '2px 6px', cursor: 'pointer' }}>👤 Patient Overview</span>
        <span style={{ padding: '2px 6px', cursor: 'pointer' }}>📊 Perioperative Tracking</span>
        <span style={{ padding: '2px 6px', cursor: 'pointer' }}>🏠 Home</span>
        <span style={{ padding: '2px 6px', cursor: 'pointer' }}>📋 Patient List</span>
        <span style={{ padding: '2px 6px', cursor: 'pointer' }}>🔍 Dynamic Worklist</span>
      </div>

      {/* TRUST Integration Notice - Blue to match Message Centre */}
      <div style={{ 
        background: '#005580', 
        padding: '6px 12px', 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'space-between',
        color: 'white',
        fontSize: '11px'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <TrustMiniLogo size={22} />
          <span style={{ fontWeight: '600' }}>TRUST AI Governance</span>
          <span style={{ opacity: 0.8 }}>|</span>
          <span>Live Cerner EHR Verification Active</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <span style={{ 
            background: hallucinations > 0 ? '#ef4444' : '#10b981', 
            padding: '2px 8px', 
            borderRadius: '10px',
            fontSize: '10px',
            fontWeight: '600'
          }}>
            {hallucinations > 0 ? `${hallucinations} Alert${hallucinations > 1 ? 's' : ''}` : '✓ All Clear'}
          </span>
          <span style={{ opacity: 0.8, cursor: 'pointer' }}>⚙️ Settings</span>
        </div>
      </div>

      {/* Main Content Area */}
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
        
        {/* Left Sidebar - Inbox Summary */}
        <div style={{ width: '220px', borderRight: '1px solid #ccc', background: '#fafafa', display: 'flex', flexDirection: 'column' }}>
          
          {/* Message Centre Header - Blue to match PowerChart */}
          <div style={{ 
            background: '#005580', 
            color: 'white', 
            padding: '8px 12px', 
            fontWeight: '600',
            fontSize: '13px'
          }}>
            Message Centre
          </div>

          {/* Inbox Summary Header */}
          <div style={{ 
            background: '#e8e8e8', 
            padding: '6px 12px', 
            borderBottom: '1px solid #ccc',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center'
          }}>
            <span style={{ fontWeight: '600' }}>Inbox Summary</span>
            <span style={{ cursor: 'pointer' }}>📌</span>
          </div>

          {/* Tabs */}
          <div style={{ display: 'flex', borderBottom: '1px solid #ccc' }}>
            <div style={{ 
              flex: 1, 
              padding: '6px', 
              textAlign: 'center', 
              background: 'white',
              borderBottom: '2px solid #0078d4',
              fontWeight: '600',
              cursor: 'pointer'
            }}>
              Inbox
            </div>
            <div style={{ flex: 1, padding: '6px', textAlign: 'center', background: '#f0f0f0', cursor: 'pointer' }}>Proxies</div>
            <div style={{ flex: 1, padding: '6px', textAlign: 'center', background: '#f0f0f0', cursor: 'pointer' }}>Pools</div>
          </div>

          {/* Display Filter */}
          <div style={{ padding: '8px 12px', borderBottom: '1px solid #ddd', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span>Display:</span>
            <select style={{ flex: 1, padding: '2px 4px', fontSize: '11px' }}>
              <option>Last 90 Days</option>
              <option>Last 30 Days</option>
              <option>Last 7 Days</option>
            </select>
            <span style={{ cursor: 'pointer' }}>...</span>
          </div>

          {/* Folder Tree */}
          <div style={{ flex: 1, overflow: 'auto', padding: '4px 0' }}>
            
            {/* AI Review Section - NEW */}
            <div>
              <div 
                onClick={() => toggleSection('aiReview')}
                style={{ 
                  padding: '4px 12px', 
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px',
                  background: '#e8f4f8',
                  borderLeft: '3px solid #0891b2'
                }}
              >
                <span>{expandedSections.aiReview ? '▼' : '▶'}</span>
                <TrustMiniLogo size={14} />
                <span style={{ fontWeight: '600', color: '#0891b2' }}>AI Documents ({totalAI})</span>
              </div>
              {expandedSections.aiReview && (
                <div style={{ marginLeft: '20px' }}>
                  <div 
                    onClick={() => setSelectedFolder('High Uncertainty')}
                    style={{ 
                      padding: '3px 12px', 
                      cursor: 'pointer',
                      background: selectedFolder === 'High Uncertainty' ? COLORS.selectedBlue : 'transparent',
                      color: selectedFolder === 'High Uncertainty' ? 'white' : highUncertainty > 0 ? '#d97706' : 'inherit',
                      fontWeight: highUncertainty > 0 ? '600' : 'normal'
                    }}
                  >
                    ⚠ High Uncertainty ({highUncertainty})
                  </div>
                  <div 
                    onClick={() => setSelectedFolder('Hallucination Flagged')}
                    style={{ 
                      padding: '3px 12px', 
                      cursor: 'pointer',
                      background: selectedFolder === 'Hallucination Flagged' ? COLORS.selectedBlue : 'transparent',
                      color: selectedFolder === 'Hallucination Flagged' ? 'white' : hallucinations > 0 ? '#dc2626' : 'inherit',
                      fontWeight: hallucinations > 0 ? '600' : 'normal'
                    }}
                  >
                    🚨 Hallucination ({hallucinations})
                  </div>
                  <div 
                    onClick={() => setSelectedFolder('Standard Review')}
                    style={{ 
                      padding: '3px 12px', 
                      cursor: 'pointer',
                      background: selectedFolder === 'Standard Review' ? COLORS.selectedBlue : 'transparent',
                      color: selectedFolder === 'Standard Review' ? 'white' : 'inherit'
                    }}
                  >
                    ✓ Standard Review ({standardReview})
                  </div>
                </div>
              )}
            </div>

            {/* Inbox Items - Traditional */}
            <div>
              <div 
                onClick={() => toggleSection('inboxItems')}
                style={{ padding: '4px 12px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '4px' }}
              >
                <span>{expandedSections.inboxItems ? '▼' : '▶'}</span>
                <span>Inbox Items (4)</span>
              </div>
              {expandedSections.inboxItems && (
                <div style={{ marginLeft: '20px' }}>
                  <div style={{ padding: '3px 12px', cursor: 'pointer' }}>Results</div>
                  <div style={{ padding: '3px 12px', cursor: 'pointer' }}>Orders</div>
                  <div 
                    onClick={() => setSelectedFolder('Documents')}
                    style={{ 
                      padding: '3px 12px', 
                      cursor: 'pointer',
                      background: selectedFolder === 'Documents' ? COLORS.selectedBlue : 'transparent',
                      color: selectedFolder === 'Documents' ? 'white' : 'inherit'
                    }}
                  >
                    ▼ Documents (4/4)
                  </div>
                  <div style={{ marginLeft: '16px', padding: '2px 12px', cursor: 'pointer', fontSize: '11px' }}>Sign (3/3)</div>
                  <div style={{ marginLeft: '16px', padding: '2px 12px', cursor: 'pointer', fontSize: '11px' }}>Review (1/1)</div>
                  <div style={{ padding: '3px 12px', cursor: 'pointer' }}>Messages</div>
                  <div style={{ padding: '3px 12px', cursor: 'pointer' }}>Results FYI</div>
                </div>
              )}
            </div>

            {/* Work Items */}
            <div>
              <div 
                onClick={() => toggleSection('workItems')}
                style={{ padding: '4px 12px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '4px' }}
              >
                <span>{expandedSections.workItems ? '▼' : '▶'}</span>
                <span>Work Items (0)</span>
              </div>
              {expandedSections.workItems && (
                <div style={{ marginLeft: '20px' }}>
                  <div style={{ padding: '3px 12px', cursor: 'pointer' }}>Saved Documents</div>
                  <div style={{ padding: '3px 12px', cursor: 'pointer' }}>Reminders</div>
                  <div style={{ padding: '3px 12px', cursor: 'pointer' }}>Deficient Documents</div>
                </div>
              )}
            </div>

            {/* Notifications */}
            <div>
              <div 
                onClick={() => toggleSection('notifications')}
                style={{ padding: '4px 12px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '4px' }}
              >
                <span>{expandedSections.notifications ? '▼' : '▶'}</span>
                <span>Notifications</span>
              </div>
              {expandedSections.notifications && (
                <div style={{ marginLeft: '20px' }}>
                  <div style={{ padding: '3px 12px', cursor: 'pointer' }}>Sent Items</div>
                  <div style={{ padding: '3px 12px', cursor: 'pointer' }}>Trash</div>
                  <div style={{ padding: '3px 12px', cursor: 'pointer' }}>Notify Receipts</div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Main Content Area */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', background: 'white' }}>
          
          {/* Content Header */}
          <div style={{ 
            padding: '6px 12px', 
            borderBottom: '1px solid #ccc',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            background: '#f8f8f8'
          }}>
            <span style={{ fontWeight: '600' }}>
              {selectedFolder === 'AI Documents' ? 'AI Documents' : selectedFolder}
            </span>
            <span style={{ cursor: 'pointer' }}>×</span>
          </div>

          {/* Data Grid */}
          <div style={{ flex: 1, overflow: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '11px' }}>
              <thead>
                <tr style={{ background: '#f0f0f0', position: 'sticky', top: 0 }}>
                  <th style={{ padding: '6px 8px', textAlign: 'left', borderBottom: '1px solid #ccc', fontWeight: '600' }}>Patient Name</th>
                  <th style={{ padding: '6px 8px', textAlign: 'left', borderBottom: '1px solid #ccc', fontWeight: '600' }}>From</th>
                  <th style={{ padding: '6px 8px', textAlign: 'left', borderBottom: '1px solid #ccc', fontWeight: '600' }}>Subject</th>
                  <th style={{ padding: '6px 8px', textAlign: 'left', borderBottom: '1px solid #ccc', fontWeight: '600' }}>Description</th>
                  <th style={{ padding: '6px 8px', textAlign: 'left', borderBottom: '1px solid #ccc', fontWeight: '600' }}>Create Date</th>
                  <th style={{ padding: '6px 8px', textAlign: 'left', borderBottom: '1px solid #ccc', fontWeight: '600' }}>Status</th>
                  <th style={{ padding: '6px 8px', textAlign: 'left', borderBottom: '1px solid #ccc', fontWeight: '600', color: '#0891b2' }}>AI Confidence</th>
                  <th style={{ padding: '6px 8px', textAlign: 'left', borderBottom: '1px solid #ccc', fontWeight: '600', color: '#0891b2' }}>Review Level</th>
                  <th style={{ padding: '6px 8px', textAlign: 'left', borderBottom: '1px solid #ccc', fontWeight: '600' }}>Assigned</th>
                </tr>
              </thead>
              <tbody>
                {aiDocuments.map((doc, index) => (
                  <tr 
                    key={doc.id}
                    onClick={() => handleSelectDoc(doc)}
                    style={{ 
                      background: selectedDoc?.id === doc.id ? '#cce5ff' : index % 2 === 0 ? 'white' : '#fafafa',
                      cursor: 'pointer',
                      borderLeft: doc.type === 'HALLUCINATION_DETECTED' ? '3px solid #ef4444' : 
                                  doc.type === 'HIGH_UNCERTAINTY' ? '3px solid #f59e0b' : 'none'
                    }}
                  >
                    <td style={{ padding: '6px 8px', borderBottom: '1px solid #eee' }}>{doc.patientName}</td>
                    <td style={{ padding: '6px 8px', borderBottom: '1px solid #eee' }}>{doc.from}</td>
                    <td style={{ padding: '6px 8px', borderBottom: '1px solid #eee' }}>{doc.subject}</td>
                    <td style={{ padding: '6px 8px', borderBottom: '1px solid #eee' }}>{doc.description}</td>
                    <td style={{ padding: '6px 8px', borderBottom: '1px solid #eee' }}>{doc.createDate}</td>
                    <td style={{ padding: '6px 8px', borderBottom: '1px solid #eee' }}>
                      <StatusBadge status={doc.status} type={doc.type} />
                    </td>
                    <td style={{ padding: '6px 8px', borderBottom: '1px solid #eee' }}>
                      <ConfidenceBadge confidence={doc.confidence} semanticEntropy={doc.semanticEntropy} />
                    </td>
                    <td style={{ padding: '6px 8px', borderBottom: '1px solid #eee' }}>
                      <ReviewBadge level={doc.reviewLevel} />
                    </td>
                    <td style={{ padding: '6px 8px', borderBottom: '1px solid #eee' }}>{doc.assigned}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Detail Panel - Shows when document selected */}
          {selectedDoc && selectedDoc.confidence !== null && (
            <div style={{ 
              borderTop: '2px solid #0891b2', 
              background: 'linear-gradient(180deg, #f0f9ff 0%, #ffffff 100%)',
            }}>
              <div style={{ padding: '16px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                  <div>
                    <h3 style={{ margin: '0 0 8px 0', color: '#0891b2', display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <TrustMiniLogo size={20} />
                      TRUST AI Analysis
                    </h3>
                    <p style={{ margin: '4px 0', fontSize: '12px' }}>
                      <strong>Patient:</strong> {selectedDoc.patientName} | <strong>Document:</strong> {selectedDoc.subject}
                    </p>
                  </div>
                  <div style={{ textAlign: 'right' }}>
                    <div style={{ 
                      fontSize: '24px', 
                      fontWeight: 'bold', 
                      color: selectedDoc.confidence >= 90 ? '#10b981' : selectedDoc.confidence >= 75 ? '#f59e0b' : '#ef4444'
                    }}>
                      {selectedDoc.confidence}%
                    </div>
                    <div style={{ fontSize: '10px', color: '#666' }}>Confidence Score</div>
                  </div>
                </div>
                
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px', marginTop: '12px' }}>
                  <div style={{ background: 'white', padding: '10px', borderRadius: '6px', border: '1px solid #e2e8f0' }}>
                    <div style={{ fontSize: '10px', color: '#64748b', textTransform: 'uppercase' }}>Semantic Entropy</div>
                    <div style={{ fontSize: '16px', fontWeight: '600', color: selectedDoc.semanticEntropy > 0.5 ? '#f59e0b' : '#10b981' }}>
                      {selectedDoc.semanticEntropy?.toFixed(2) || '—'}
                    </div>
                  </div>
                  <div style={{ background: 'white', padding: '10px', borderRadius: '6px', border: '1px solid #e2e8f0' }}>
                    <div style={{ fontSize: '10px', color: '#64748b', textTransform: 'uppercase' }}>Review Level</div>
                    <div style={{ fontSize: '16px', fontWeight: '600' }}>{selectedDoc.reviewLevel}</div>
                  </div>
                  <div style={{ background: 'white', padding: '10px', borderRadius: '6px', border: '1px solid #e2e8f0' }}>
                    <div style={{ fontSize: '10px', color: '#64748b', textTransform: 'uppercase' }}>Detection Method</div>
                    <div style={{ fontSize: '12px', fontWeight: '500' }}>SE + EHR Verify</div>
                  </div>
                  <div style={{ background: 'white', padding: '10px', borderRadius: '6px', border: '1px solid #e2e8f0' }}>
                    <div style={{ fontSize: '10px', color: '#64748b', textTransform: 'uppercase' }}>Audit Trail</div>
                    <div style={{ fontSize: '12px', fontWeight: '500', color: '#0891b2', cursor: 'pointer' }}>View Log →</div>
                  </div>
                </div>

                {selectedDoc.flagReason && (
                  <div style={{ 
                    marginTop: '12px', 
                    padding: '10px', 
                    background: '#fef2f2', 
                    border: '1px solid #fecaca',
                    borderRadius: '6px',
                    color: '#991b1b'
                  }}>
                    <strong>⚠️ Alert:</strong> {selectedDoc.flagReason}
                  </div>
                )}

                <div style={{ marginTop: '12px', display: 'flex', gap: '8px' }}>
                  <button style={{ 
                    padding: '8px 16px', 
                    background: '#0891b2', 
                    color: 'white', 
                    border: 'none', 
                    borderRadius: '4px',
                    cursor: 'pointer',
                    fontWeight: '600'
                  }}>
                    ✓ Approve & Sign
                  </button>
                  <button style={{ 
                    padding: '8px 16px', 
                    background: 'white', 
                    color: '#0891b2', 
                    border: '1px solid #0891b2', 
                    borderRadius: '4px',
                    cursor: 'pointer'
                  }}>
                    🔍 Open for Detailed Review
                  </button>
                  <button style={{ 
                    padding: '8px 16px', 
                    background: 'white', 
                    color: '#dc2626', 
                    border: '1px solid #dc2626', 
                    borderRadius: '4px',
                    cursor: 'pointer'
                  }}>
                    ✗ Reject with Comment
                  </button>
                </div>
              </div>
              <TRUSTAnalysisPanel analysis={trustAnalysis} loading={ehrLoading} onClose={() => setTrustAnalysis(null)} />
              {/* EHR Verification Panel */}
              <EHRVerificationPanel ehrResult={ehrResult} loading={ehrLoading} />
            </div>
          )}
        </div>
      </div>

      {/* Status Bar */}
      <div style={{ 
        background: '#f0f0f0', 
        padding: '4px 12px', 
        borderTop: '1px solid #ccc',
        display: 'flex',
        justifyContent: 'space-between',
        fontSize: '10px',
        color: '#666'
      }}>
        <span>Clinical link on P0783 JRAUBENHEIMER | TRUST Platform v0.2.0 | Cerner FHIR Connected</span>
        <span>{new Date().toLocaleString()}</span>
      </div>
    </div>
  );
}
