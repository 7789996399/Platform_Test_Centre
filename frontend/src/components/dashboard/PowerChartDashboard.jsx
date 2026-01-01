import React, { useState } from 'react';

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

// Sample AI Documents data
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
    flagReason: 'Medication dosage inconsistency detected - Warfarin not in EHR',
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
  },
];

// Semantic Entropy badge for table (color-coded)
const SemanticEntropyBadge = ({ semanticEntropy }) => {
  if (semanticEntropy === null) return <span style={{ color: '#666' }}>—</span>;
  
  let textColor;
  // Lower SE = better (inverted from confidence)
  if (semanticEntropy <= 0.3) {
    textColor = COLORS.trustGreen;  // Low uncertainty = good
  } else if (semanticEntropy <= 0.6) {
    textColor = COLORS.trustAmber;  // Medium uncertainty
  } else {
    textColor = COLORS.trustRed;    // High uncertainty = concerning
  }
  
  return (
    <span style={{
      fontWeight: '600',
      color: textColor,
    }}>
      {semanticEntropy.toFixed(2)}
    </span>
  );
};

// Status indicator - UPDATED: Review Required now uses amber color
const StatusBadge = ({ status, type }) => {
  let color = '#008000';  // Default green
  let fontWeight = 'normal';
  let prefix = '';
  
  if (status === 'Flagged' || type === 'HALLUCINATION_DETECTED') {
    color = COLORS.trustRed;  // Red for hallucinations
    fontWeight = 'bold';
    prefix = '⚠ ';
  } else if (status === 'Review Required' || type === 'HIGH_UNCERTAINTY') {
    color = COLORS.trustAmber;  // AMBER for high uncertainty (matches border strip)
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

export default function TrustPowerChartDashboard() {
  const [selectedFolder, setSelectedFolder] = useState('AI Documents');
  const [selectedDoc, setSelectedDoc] = useState(null);
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
  const highUncertainty = aiDocuments.filter(d => d.type === 'HIGH_UNCERTAINTY').length;
  const hallucinations = aiDocuments.filter(d => d.type === 'HALLUCINATION_DETECTED').length;
  const standardReview = aiDocuments.filter(d => d.type === 'STANDARD').length;
  const totalAI = highUncertainty + hallucinations + standardReview;

  // Uncertainty level text based on semantic entropy
  const getUncertaintyLevel = (se) => {
    if (se === null) return { text: '—', color: '#666' };
    if (se <= 0.3) return { text: `Low (${se.toFixed(2)})`, color: COLORS.trustGreen };
    if (se <= 0.6) return { text: `Medium (${se.toFixed(2)})`, color: COLORS.trustAmber };
    return { text: `High (${se.toFixed(2)})`, color: COLORS.trustRed };
  };

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
            1 Alert
          </span>
          <span style={{ cursor: 'pointer' }}>⚙ Settings</span>
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

          {/* Inbox Summary Header - PIN REMOVED */}
          <div style={{ 
            background: '#e8e8e8', 
            padding: '6px 12px', 
            borderBottom: '1px solid #ccc',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center'
          }}>
            <span style={{ fontWeight: '600' }}>Inbox Summary</span>
            {/* Pin icon removed per user request */}
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
                      color: selectedFolder === 'High Uncertainty' ? 'white' : COLORS.trustAmber,  // AMBER color
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

          {/* Data Grid - SIMPLIFIED: Removed Semantic Entropy and Review Level columns */}
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
                {aiDocuments.map((doc, index) => (
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
              maxHeight: '220px',
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
              
              {/* Info boxes - Only Uncertainty Level and Review Level */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '16px', marginTop: '12px' }}>
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
                  background: '#10b981', 
                  color: 'white', 
                  border: 'none', 
                  borderRadius: '4px',
                  cursor: 'pointer',
                  fontWeight: '600',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px'
                }}>
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
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <span style={{ 
            background: '#10b981', 
            color: 'white', 
            padding: '2px 8px', 
            borderRadius: '4px',
            fontWeight: '600'
          }}>
            TRUST Analysis
          </span>
          <span style={{ 
            background: '#dcfce7', 
            color: '#166534', 
            padding: '2px 8px', 
            borderRadius: '4px'
          }}>
            LOW RISK
          </span>
          <span>5 claims | 1 verified | 0 contradictions</span>
          <span style={{ color: '#10b981', fontWeight: '600' }}>91.7% time saved</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ color: '#0891b2', cursor: 'pointer' }}>Details ▼</span>
          <span>×</span>
        </div>
      </div>
    </div>
  );
}
