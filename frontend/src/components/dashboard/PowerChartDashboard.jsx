import React, { useState, useEffect } from 'react';
import { analyzeNote, transformApiResponse } from '../../services/api';

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
  trustAmber: '#f59e0b',  // Amber/Orange for warnings
  trustRed: '#ef4444',
  trustPurple: '#8b5cf6',
};

// =============================================================================
// TIME SAVED CALCULATION LOGIC
// =============================================================================

// Average time (in seconds) to manually verify each claim type against EHR
const MANUAL_VERIFICATION_TIME = {
  medication: 45,      // Check medication list, reconcile
  allergy: 30,         // Check allergy list
  vital_sign: 20,      // Check flowsheet
  lab_result: 40,      // Navigate to labs, find result
  diagnosis: 60,       // Review problem list
  procedure: 90,       // Check surgical/procedure history
  demographic: 15,     // Check patient info
  default: 35,         // Average for unspecified claim types
};

// TRUST automated verification time per claim (API call)
const TRUST_VERIFICATION_TIME_PER_CLAIM = 2; // seconds

/**
 * Calculate time saved by TRUST automated verification
 * @param {Array} claims - Array of claim objects with 'type' property
 * @param {number} contradictions - Number of contradictions found (adds review time)
 * @returns {Object} { timeSavedPercent, manualTimeSeconds, trustTimeSeconds }
 */
const calculateTimeSaved = (claims, contradictions = 0) => {
  if (!claims || claims.length === 0) {
    return { timeSavedPercent: 0, manualTimeSeconds: 0, trustTimeSeconds: 0 };
  }
  
  // Calculate total manual verification time
  const manualTimeSeconds = claims.reduce((total, claim) => {
    const claimTime = MANUAL_VERIFICATION_TIME[claim.type] || MANUAL_VERIFICATION_TIME.default;
    return total + claimTime;
  }, 0);
  
  // Calculate TRUST automated time
  let trustTimeSeconds = claims.length * TRUST_VERIFICATION_TIME_PER_CLAIM;
  
  // Add penalty time for each contradiction (requires manual review)
  const CONTRADICTION_REVIEW_TIME = 60; // seconds to investigate each contradiction
  trustTimeSeconds += contradictions * CONTRADICTION_REVIEW_TIME;
  
  // Calculate percentage saved
  const timeSavedPercent = Math.max(0, ((manualTimeSeconds - trustTimeSeconds) / manualTimeSeconds) * 100);
  
  return {
    timeSavedPercent: Math.round(timeSavedPercent * 10) / 10, // Round to 1 decimal
    manualTimeSeconds,
    trustTimeSeconds
  };
};

/**
 * Determine risk level based on semantic entropy, contradictions, and verification status
 */
const calculateRiskLevel = (doc) => {
  if (!doc.ehrVerification) return { level: 'UNKNOWN', color: '#666' };
  
  const { contradictions, unverified } = doc.ehrVerification;
  const se = doc.semanticEntropy || 0;
  
  // HIGH RISK: Any contradiction OR very high SE
  if (contradictions > 0 || se > 1.0) {
    return { level: 'HIGH RISK', color: COLORS.trustRed, bgColor: '#fee2e2' };
  }
  
  // MEDIUM RISK: High SE OR many unverified claims
  if (se > 0.5 || unverified > 2) {
    return { level: 'MEDIUM RISK', color: COLORS.trustAmber, bgColor: '#fef3c7' };
  }
  
  // LOW RISK: Low SE and no contradictions
  return { level: 'LOW RISK', color: COLORS.trustGreen, bgColor: '#dcfce7' };
};


// =============================================================================
// SAMPLE DATA WITH EHR VERIFICATION DETAILS
// =============================================================================

const aiDocuments = [
  {
    id: 1,
    patientName: 'JACKSON, SCOTT',
    from: 'AI Scribe',
    subject: 'Pre-Op Assessment',
    author: 'Cerner AI Scribe',
    description: 'Anesthesiology Consult',
    createDate: '27-Dec-2025',
    status: 'Review Required',
    type: 'HIGH_UNCERTAINTY',
    semanticEntropy: 0.84,
    assigned: 'Raubenheimer...',
    reviewLevel: 'Detailed',
    // EHR Verification data
    ehrVerification: {
      claims: [
        { type: 'medication', text: 'Metoprolol 50mg daily', verified: true },
        { type: 'medication', text: 'Lisinopril 10mg daily', verified: true },
        { type: 'allergy', text: 'NKDA', verified: true },
        { type: 'diagnosis', text: 'Hypertension', verified: true },
        { type: 'vital_sign', text: 'BP 142/88', verified: false }, // Not yet documented
        { type: 'procedure', text: 'Scheduled for hip replacement', verified: true },
      ],
      totalClaims: 6,
      verified: 5,
      contradictions: 0,
      unverified: 1,
    }
  },
  {
    id: 2,
    patientName: 'SMARTS, NANCYS II',
    from: 'AI Scribe',
    subject: 'PIFP Block Note',
    author: 'Cerner AI Scribe',
    description: 'Regional Anesthesia',
    createDate: '27-Dec-2025',
    status: 'Pending',
    type: 'STANDARD',
    semanticEntropy: 0.21,
    assigned: 'Raubenheimer...',
    reviewLevel: 'Brief',
    ehrVerification: {
      claims: [
        { type: 'procedure', text: 'PIFP block performed', verified: true },
        { type: 'medication', text: 'Ropivacaine 0.5% 20mL', verified: true },
        { type: 'vital_sign', text: 'Vitals stable', verified: true },
      ],
      totalClaims: 3,
      verified: 3,
      contradictions: 0,
      unverified: 0,
    }
  },
  {
    id: 3,
    patientName: 'CHEN, DAVID',
    from: 'AI Scribe',
    subject: 'Anesthesia Consult',
    author: 'Cerner AI Scribe',
    description: 'Anesthesiology Consult',
    createDate: '27-Dec-2025',
    status: 'Pending',
    type: 'STANDARD',
    semanticEntropy: 0.28,
    assigned: 'Raubenheimer...',
    reviewLevel: 'Standard',
    ehrVerification: {
      claims: [
        { type: 'medication', text: 'Aspirin 81mg daily', verified: true },
        { type: 'diagnosis', text: 'Type 2 Diabetes', verified: true },
        { type: 'diagnosis', text: 'CAD', verified: true },
        { type: 'lab_result', text: 'HbA1c 7.2%', verified: true },
        { type: 'allergy', text: 'Penicillin - rash', verified: true },
      ],
      totalClaims: 5,
      verified: 5,
      contradictions: 0,
      unverified: 0,
    }
  },
  {
    id: 4,
    patientName: 'MARTINEZ, ANA',
    from: 'AI Scribe',
    subject: 'Post-Op Note',
    author: 'Cerner AI Scribe',
    description: 'Cardiac Surgery',
    createDate: '26-Dec-2025',
    status: 'Flagged',
    type: 'HALLUCINATION_DETECTED',
    semanticEntropy: 1.42,
    assigned: 'Raubenheimer...',
    reviewLevel: 'Detailed',
    flagReason: 'Medication inconsistency - "Warfarin 5mg" mentioned but NOT found in EHR',
    ehrVerification: {
      claims: [
        { type: 'medication', text: 'Warfarin 5mg daily', verified: false, contradiction: true }, // NOT IN EHR!
        { type: 'procedure', text: 'CABG x3 completed', verified: true },
        { type: 'diagnosis', text: 'CAD', verified: true },
        { type: 'vital_sign', text: 'Hemodynamically stable', verified: true },
        { type: 'lab_result', text: 'Hgb 10.2', verified: true },
      ],
      totalClaims: 5,
      verified: 4,
      contradictions: 1,  // Warfarin contradiction!
      unverified: 0,
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
    semanticEntropy: null,
    assigned: 'Raubenheimer...',
    reviewLevel: null,
    ehrVerification: null, // Not an AI document
  },
];


// =============================================================================
// UI COMPONENTS
// =============================================================================

// Status indicator - Review Required now uses amber color
const StatusBadge = ({ status, type }) => {
  let color = '#008000';  // Default green
  let fontWeight = 'normal';
  let prefix = '';
  
  if (status === 'Flagged' || type === 'HALLUCINATION_DETECTED') {
    color = COLORS.trustRed;
    fontWeight = 'bold';
    prefix = '⚠ ';
  } else if (status === 'Review Required' || type === 'HIGH_UNCERTAINTY') {
    color = COLORS.trustAmber;
    fontWeight = 'bold';
  }
  
  return (
    <span style={{ color, fontWeight }}>
      {prefix}{status}
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


// =============================================================================
// MAIN COMPONENT
// =============================================================================

export default function TrustPowerChartDashboard() {
  const [selectedFolder, setSelectedFolder] = useState('AI Documents');
  const [selectedDoc, setSelectedDoc] = useState(null);
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Test function to analyze a note against real Cerner data
  const testAnalyzeNote = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await analyzeNote({
        note_id: `test_${Date.now()}`,
        patient_id: '12724066',
        sections: {
          medications: [
            { name: 'Acetaminophen', dose: '500mg', frequency: 'PRN' },
            { name: 'Amlodipine', dose: '5mg', frequency: 'daily' },
            { name: 'Warfarin', dose: '5mg', frequency: 'daily' }
          ],
          allergies: 'Penicillin, Latex'
        },
        source_transcript: 'Patient confirms taking acetaminophen, amlodipine and warfarin'
      });
      const transformed = transformApiResponse(response);
      transformed.patientName = 'SMARTS, NANCYS II';
      setDocuments(prev => [transformed, ...prev]);
    } catch (err) {
      setError(err.message);
      console.error('Analysis failed:', err);
    }
    setLoading(false);
  };
  const [expandedSections, setExpandedSections] = useState({
    inboxItems: true,
    aiReview: true,
    workItems: false,
    notifications: false,
  });

  const toggleSection = (section) => {
    setExpandedSections(prev => ({ ...prev, [section]: !prev[section] }));
  };

  // Count documents by type
  const allDocs = [...documents, ...aiDocuments];
  const highUncertainty = allDocs.filter(d => d.type === 'HIGH_UNCERTAINTY').length;
  const hallucinations = allDocs.filter(d => d.type === 'HALLUCINATION_DETECTED').length;
  const standardReview = allDocs.filter(d => d.type === 'STANDARD').length;
  const totalAI = highUncertainty + hallucinations + standardReview;

  // Get uncertainty level text based on semantic entropy
  const getUncertaintyLevel = (se) => {
    if (se === null) return { text: '—', color: '#666' };
    if (se <= 0.3) return { text: `Low (${se.toFixed(2)})`, color: COLORS.trustGreen };
    if (se <= 0.6) return { text: `Medium (${se.toFixed(2)})`, color: COLORS.trustAmber };
    return { text: `High (${se.toFixed(2)})`, color: COLORS.trustRed };
  };

  // Calculate stats for selected document
  const getDocStats = (doc) => {
    if (!doc || !doc.ehrVerification) {
      return {
        timeSaved: { timeSavedPercent: 0 },
        risk: { level: 'N/A', color: '#666', bgColor: '#f0f0f0' },
        claims: 0,
        verified: 0,
        contradictions: 0
      };
    }
    
    const { claims, verified, contradictions } = doc.ehrVerification;
    const timeSaved = calculateTimeSaved(claims, contradictions);
    const risk = calculateRiskLevel(doc);
    
    return {
      timeSaved,
      risk,
      claims: doc.ehrVerification.totalClaims,
      verified,
      contradictions
    };
  };

  const selectedDocStats = getDocStats(selectedDoc);

  return (
    <div style={{ 
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif',
      fontSize: '12px',
      height: '100vh',
      display: 'flex',
      flexDirection: 'column',
      background: '#e8e8e8'
    }}>
      {/* PowerChart Header Bar */}
      <div style={{ 
        background: 'linear-gradient(180deg, #006878 0%, #004d5a 100%)', 
        color: 'white', 
        padding: '4px 12px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        fontSize: '12px'
      }}>
        <span style={{ fontWeight: '500' }}>PowerChart Organizer for Raubenheimer, Jean Louis, MD</span>
      </div>

      {/* Menu Bar */}
      <div style={{ 
        background: '#f0f0f0', 
        borderBottom: '1px solid #ccc',
        padding: '2px 8px',
        display: 'flex',
        gap: '16px',
        fontSize: '11px'
      }}>
        {['Task', 'Edit', 'View', 'Patient', 'Chart', 'Links', 'Notifications', 'Inbox', 'Help'].map(item => (
          <span key={item} style={{ cursor: 'pointer', padding: '2px 4px' }}>{item}</span>
        ))}
      </div>

      {/* Toolbar */}
      <div style={{ 
        background: '#f5f5f5', 
        borderBottom: '1px solid #ccc',
        padding: '4px 8px',
        display: 'flex',
        gap: '8px',
        alignItems: 'center',
        fontSize: '11px'
      }}>
        <span style={{ padding: '2px 8px', cursor: 'pointer' }}>📧 Message Centre</span>
        <span style={{ padding: '2px 8px', cursor: 'pointer' }}>👤 Patient Overview</span>
        <span style={{ padding: '2px 8px', cursor: 'pointer' }}>📊 Perioperative Tracking</span>
        <span style={{ padding: '2px 8px', cursor: 'pointer' }}>🏠 Home</span>
        <span style={{ padding: '2px 8px', cursor: 'pointer' }}>📋 Patient List</span>
        <span style={{ padding: '2px 8px', cursor: 'pointer' }}>📑 Dynamic Worklist</span>
      </div>

      {/* TRUST AI Governance Bar */}
      <div style={{ 
        background: 'linear-gradient(90deg, #0891b2 0%, #0e7490 100%)', 
        color: 'white', 
        padding: '6px 12px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        fontSize: '11px'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <TrustMiniLogo size={18} />
          <span style={{ fontWeight: '600' }}>TRUST AI Governance</span>
          <span style={{ opacity: 0.8 }}>|</span>
          <span style={{ opacity: 0.9 }}>Live Cerner EHR Verification Active</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <span style={{ 
            background: COLORS.trustRed, 
            padding: '2px 8px', 
            borderRadius: '10px',
            fontSize: '10px',
            fontWeight: '600'
          }}>
            {hallucinations} Alert{hallucinations !== 1 ? 's' : ''}
          </span>
          <span style={{ cursor: 'pointer' }}>⚙ Settings</span>
        </div>
      </div>

      {/* Main Content Area */}
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
        
        {/* Left Sidebar - Inbox Summary */}
        <div style={{ width: '220px', borderRight: '1px solid #ccc', background: '#fafafa', display: 'flex', flexDirection: 'column' }}>
          
          {/* Message Centre Header */}
          <div style={{ 
            background: '#005580', 
            color: 'white', 
            padding: '8px 12px', 
            fontWeight: '600',
            fontSize: '13px'
          }}>
            Message Centre
          </div>

          {/* Inbox Summary Header - No pin icon */}
          <div style={{ 
            background: '#e8e8e8', 
            padding: '6px 12px', 
            borderBottom: '1px solid #ccc',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center'
          }}>
            <span style={{ fontWeight: '600' }}>Inbox Summary</span>
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
            
            {/* AI Review Section */}
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
                      color: selectedFolder === 'High Uncertainty' ? 'white' : COLORS.trustAmber,
                      fontWeight: '600'
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
                      color: selectedFolder === 'Hallucination Flagged' ? 'white' : COLORS.trustRed,
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
           <span style={{ fontWeight: '600' }}>AI Documents</span>
            <button 
              onClick={testAnalyzeNote}
              disabled={loading}
              style={{
                marginLeft: '16px',
                padding: '4px 12px',
                background: loading ? '#ccc' : '#0891b2',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: loading ? 'wait' : 'pointer',
                fontSize: '11px'
              }}
            >
              {loading ? 'Analyzing...' : '+ Test Real API'}
            </button>
            {error && <span style={{ color: 'red', fontSize: '11px' }}>{error}</span>}
            <span style={{ cursor: 'pointer' }}>×</span> 
          </div>

          {/* Data Grid - Simplified columns */}
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
                  <th style={{ padding: '6px 8px', textAlign: 'left', borderBottom: '1px solid #ccc', fontWeight: '600' }}>Assigned</th>
                </tr>
              </thead>
              <tbody>
                {allDocs.map((doc, index) => (
                  <tr 
                    key={doc.id}
                    onClick={() => setSelectedDoc(doc)}
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
                    <td style={{ padding: '6px 8px', borderBottom: '1px solid #eee' }}>{doc.assigned}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Detail Panel - Shows when document selected */}
          {selectedDoc && selectedDoc.semanticEntropy !== null && (
            <div style={{ 
              borderTop: '2px solid #0891b2', 
              padding: '16px', 
              background: 'linear-gradient(180deg, #f0f9ff 0%, #ffffff 100%)',
              maxHeight: '240px',
              overflow: 'auto'
            }}>
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
                    fontSize: '28px', 
                    fontWeight: 'bold', 
                    color: selectedDoc.semanticEntropy <= 0.3 ? COLORS.trustGreen : 
                           selectedDoc.semanticEntropy <= 0.6 ? COLORS.trustAmber : COLORS.trustRed
                  }}>
                    {selectedDoc.semanticEntropy.toFixed(2)}
                  </div>
                  <div style={{ fontSize: '10px', color: '#666' }}>Semantic Entropy</div>
                </div>
              </div>
              
              {/* Info boxes - 3 columns: Uncertainty, Review Level, EHR Verification */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '12px', marginTop: '12px' }}>
                <div style={{ background: 'white', padding: '12px', borderRadius: '6px', border: '1px solid #e2e8f0' }}>
                  <div style={{ fontSize: '10px', color: '#64748b', textTransform: 'uppercase', marginBottom: '4px' }}>Uncertainty Level</div>
                  <div style={{ 
                    fontSize: '16px', 
                    fontWeight: '600', 
                    color: getUncertaintyLevel(selectedDoc.semanticEntropy).color 
                  }}>
                    {getUncertaintyLevel(selectedDoc.semanticEntropy).text}
                  </div>
                </div>
                <div style={{ background: 'white', padding: '12px', borderRadius: '6px', border: '1px solid #e2e8f0' }}>
                  <div style={{ fontSize: '10px', color: '#64748b', textTransform: 'uppercase', marginBottom: '4px' }}>Review Level</div>
                  <div style={{ fontSize: '16px', fontWeight: '600' }}>{selectedDoc.reviewLevel}</div>
                </div>
                <div style={{ background: 'white', padding: '12px', borderRadius: '6px', border: '1px solid #e2e8f0' }}>
                  <div style={{ fontSize: '10px', color: '#64748b', textTransform: 'uppercase', marginBottom: '4px' }}>EHR Verification</div>
                  <div style={{ 
                    fontSize: '14px', 
                    fontWeight: '600',
                    color: selectedDocStats.contradictions > 0 ? COLORS.trustRed : COLORS.trustGreen
                  }}>
                    {selectedDocStats.contradictions > 0 
                      ? `${selectedDocStats.contradictions} contradiction${selectedDocStats.contradictions > 1 ? 's' : ''} found`
                      : `${selectedDocStats.verified}/${selectedDocStats.claims} verified`
                    }
                  </div>
                </div>
              </div>

              {selectedDoc.flagReason && (
                <div style={{ 
                  marginTop: '12px', 
                  padding: '10px', 
                  background: '#fef2f2', 
                  border: '1px solid #fecaca',
                  borderRadius: '6px',
                  color: '#991b1b',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px'
                }}>
                  <span>⚠</span>
                  <span><strong>Alert:</strong> {selectedDoc.flagReason}</span>
                </div>
              )}

              <div style={{ marginTop: '12px', display: 'flex', gap: '8px' }}>
                <button style={{ 
                  padding: '8px 16px', 
                  background: selectedDocStats.contradictions > 0 ? '#9ca3af' : '#10b981',
                  color: 'white', 
                  border: 'none', 
                  borderRadius: '4px',
                  cursor: selectedDocStats.contradictions > 0 ? 'not-allowed' : 'pointer',
                  fontWeight: '600',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px'
                }}
                disabled={selectedDocStats.contradictions > 0}
                title={selectedDocStats.contradictions > 0 ? 'Cannot approve - contradictions found' : ''}
                >
                  ✓ Approve & Sign
                </button>
                <button style={{ 
                  padding: '8px 16px', 
                  background: 'white', 
                  color: '#0891b2', 
                  border: '1px solid #0891b2', 
                  borderRadius: '4px',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px'
                }}>
                  🔍 Open for Detailed Review
                </button>
                <button style={{ 
                  padding: '8px 16px', 
                  background: 'white', 
                  color: '#dc2626', 
                  border: '1px solid #dc2626', 
                  borderRadius: '4px',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px'
                }}>
                  ✗ Reject with Comment
                </button>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Status Bar - DYNAMIC based on selected document */}
      <div style={{ 
        background: '#f0f0f0', 
        padding: '4px 12px', 
        borderTop: '1px solid #ccc',
        display: 'flex',
        justifyContent: 'space-between',
        fontSize: '10px',
        color: '#666'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <span style={{ 
            background: COLORS.trustTeal, 
            color: 'white', 
            padding: '2px 8px', 
            borderRadius: '4px',
            fontWeight: '600'
          }}>
            TRUST Analysis
          </span>
          
          {selectedDoc && selectedDoc.ehrVerification ? (
            <>
              {/* Dynamic Risk Badge */}
              <span style={{ 
                background: selectedDocStats.risk.bgColor, 
                color: selectedDocStats.risk.color, 
                padding: '2px 8px', 
                borderRadius: '4px',
                fontWeight: '600'
              }}>
                {selectedDocStats.risk.level}
              </span>
              
              {/* Dynamic Claims Stats */}
              <span>
                {selectedDocStats.claims} claims | {selectedDocStats.verified} verified | {selectedDocStats.contradictions} contradiction{selectedDocStats.contradictions !== 1 ? 's' : ''}
              </span>
              
              {/* Dynamic Time Saved */}
              <span style={{ 
                color: selectedDocStats.timeSaved.timeSavedPercent > 50 ? COLORS.trustGreen : COLORS.trustAmber,
                fontWeight: '600' 
              }}>
                {selectedDocStats.timeSaved.timeSavedPercent}% time saved
              </span>
            </>
          ) : (
            <span style={{ color: '#999' }}>Select an AI document to see analysis</span>
          )}
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ color: '#0891b2', cursor: 'pointer' }}>Details ▼</span>
          <span>×</span>
        </div>
      </div>

      {/* Footer */}
      <div style={{ 
        background: '#e0e0e0', 
        padding: '2px 12px', 
        fontSize: '9px',
        color: '#666',
        display: 'flex',
        justifyContent: 'space-between'
      }}>
        <span>TRUST Platform v0.2.0 | Cerner FHIR Connected</span>
        <span>{new Date().toLocaleDateString()} {new Date().toLocaleTimeString()}</span>
      </div>
    </div>
  );
}
