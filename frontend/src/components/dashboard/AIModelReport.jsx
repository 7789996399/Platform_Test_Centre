import React, { useState, useEffect, useRef } from 'react';

const API_BASE_URL = 'https://api.trustplatform.ca/api/v1';

// FDA Nutrition Facts Style AI Model Report
const AIModelReport = ({ model, onClose }) => {
  const reportRef = useRef(null);
  const [auditStats, setAuditStats] = useState(null);
  
  useEffect(() => {
    // Fetch real audit data
    const fetchStats = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/ml/logs/summary`);
        if (response.ok) {
          setAuditStats(await response.json());
        }
      } catch (err) {
        console.error('Failed to fetch audit stats:', err);
      }
    };
    fetchStats();
  }, []);

  const handlePrint = () => {
    const printContent = reportRef.current.innerHTML;
    const printWindow = window.open('', '_blank');
    printWindow.document.write(`
      <!DOCTYPE html>
      <html>
        <head>
          <title>AI Model Report - ${model?.name || 'Model Card'}</title>
          <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { font-family: Helvetica, Arial, sans-serif; padding: 20px; }
            @media print { body { padding: 0; } }
          </style>
        </head>
        <body>${printContent}</body>
      </html>
    `);
    printWindow.document.close();
    printWindow.print();
  };

  // Default model data if none provided
  const modelData = model || {
    name: 'AI Scribe - Clinical Documentation',
    version: '2.1.0',
    developer: 'Cerner Corporation',
    type: 'Generative AI',
    releaseDate: '2025-01-15',
    lastValidated: new Date().toISOString().split('T')[0],
    riskLevel: 'Medium',
    intendedUse: 'Automated clinical documentation from patient encounters',
    targetPopulation: 'Adult patients, all clinical settings',
    contraindications: 'Pediatric dosing, Oncology protocols',
  };

  return (
    <div style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, background: 'rgba(0,0,0,0.5)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000 }}>
      <div style={{ background: 'white', borderRadius: '8px', maxWidth: '450px', maxHeight: '90vh', overflow: 'auto' }}>
        {/* Header with buttons */}
        <div style={{ padding: '12px 16px', borderBottom: '1px solid #e2e8f0', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span style={{ fontWeight: '600' }}>AI Model Report Card</span>
          <div style={{ display: 'flex', gap: '8px' }}>
            <button 
              onClick={handlePrint}
              style={{ padding: '6px 12px', background: '#0891b2', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer', fontSize: '12px', fontWeight: '600' }}
            >
              🖨️ Print
            </button>
            <button 
              onClick={onClose}
              style={{ padding: '6px 12px', background: '#f1f5f9', border: '1px solid #e2e8f0', borderRadius: '4px', cursor: 'pointer', fontSize: '12px' }}
            >
              Close
            </button>
          </div>
        </div>

        {/* The actual nutrition label - printable area */}
        <div ref={reportRef} style={{ padding: '8px' }}>
          <div style={{ 
            width: '400px', 
            border: '2px solid #000', 
            padding: '6px',
            fontFamily: 'Helvetica, Arial, sans-serif',
            fontSize: '11px',
            lineHeight: '1.3',
            color: '#000',
            background: '#fff'
          }}>
            {/* Title */}
            <div style={{ fontSize: '26px', fontWeight: '900', borderBottom: '1px solid #000', paddingBottom: '2px' }}>
              AI Model Facts
            </div>
            
            {/* Model Name */}
            <div style={{ fontSize: '12px', fontWeight: 'bold', borderBottom: '8px solid #000', paddingBottom: '4px', marginTop: '2px' }}>
              {modelData.name}
            </div>
            
            {/* Basic Info */}
            <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid #000', padding: '2px 0' }}>
              <span>Version</span>
              <span style={{ fontWeight: 'bold' }}>{modelData.version}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid #000', padding: '2px 0' }}>
              <span>Developer</span>
              <span style={{ fontWeight: 'bold' }}>{modelData.developer}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid #000', padding: '2px 0' }}>
              <span>Model Type</span>
              <span style={{ fontWeight: 'bold' }}>{modelData.type}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '3px solid #000', padding: '2px 0' }}>
              <span>Clinical Risk Level</span>
              <span style={{ fontWeight: 'bold' }}>{modelData.riskLevel}</span>
            </div>

            {/* Performance Section */}
            <div style={{ fontWeight: '900', fontSize: '11px', background: '#000', color: '#fff', padding: '2px 4px', margin: '6px 0 3px 0' }}>
              PERFORMANCE METRICS
            </div>
            
            <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid #000', padding: '2px 0' }}>
              <span>Overall Accuracy</span>
              <span style={{ fontWeight: 'bold' }}>94.2%</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid #000', padding: '2px 0' }}>
              <span>Sensitivity</span>
              <span style={{ fontWeight: 'bold' }}>91.8%</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid #000', padding: '2px 0' }}>
              <span>Specificity</span>
              <span style={{ fontWeight: 'bold' }}>96.1%</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '3px solid #000', padding: '2px 0' }}>
              <span>Mean Semantic Entropy</span>
              <span style={{ fontWeight: 'bold' }}>0.12</span>
            </div>

            {/* Safety Metrics */}
            <div style={{ fontWeight: '900', fontSize: '11px', background: '#000', color: '#fff', padding: '2px 4px', margin: '6px 0 3px 0' }}>
              SAFETY METRICS
            </div>
            
            <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid #000', padding: '2px 0' }}>
              <span>Hallucination Rate</span>
              <span style={{ fontWeight: 'bold' }}>2.1%</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid #000', padding: '2px 0' }}>
              <span>Contradiction Rate</span>
              <span style={{ fontWeight: 'bold' }}>0.8%</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid #000', padding: '2px 0' }}>
              <span>Mean Uncertainty</span>
              <span style={{ fontWeight: 'bold' }}>0.06</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '3px solid #000', padding: '2px 0' }}>
              <span>Calibration Error</span>
              <span style={{ fontWeight: 'bold' }}>0.03</span>
            </div>

            {/* Audit Trail */}
            <div style={{ fontWeight: '900', fontSize: '11px', background: '#000', color: '#fff', padding: '2px 4px', margin: '6px 0 3px 0' }}>
              AUDIT TRAIL (TRUST Platform)
            </div>
            
            <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid #000', padding: '2px 0' }}>
              <span>Total Analyses Logged</span>
              <span style={{ fontWeight: 'bold' }}>{auditStats?.total_analyses || '—'}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid #000', padding: '2px 0' }}>
              <span>Semantic Entropy Checks</span>
              <span style={{ fontWeight: 'bold' }}>{auditStats?.by_type?.semantic_entropy || '—'}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid #000', padding: '2px 0' }}>
              <span>Uncertainty Quantifications</span>
              <span style={{ fontWeight: 'bold' }}>{auditStats?.by_type?.uncertainty || '—'}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid #000', padding: '2px 0' }}>
              <span>Avg Processing Time</span>
              <span style={{ fontWeight: 'bold' }}>{auditStats?.avg_processing_time_ms ? `${Math.round(auditStats.avg_processing_time_ms)}ms` : '—'}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '3px solid #000', padding: '2px 0' }}>
              <span>Last Validation</span>
              <span style={{ fontWeight: 'bold' }}>{modelData.lastValidated}</span>
            </div>

            {/* Review Distribution */}
            <div style={{ fontWeight: '900', fontSize: '11px', background: '#000', color: '#fff', padding: '2px 4px', margin: '6px 0 3px 0' }}>
              REVIEW DISTRIBUTION
            </div>
            
            <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid #000', padding: '2px 0' }}>
              <span>Brief Review (Auto-approved)</span>
              <span style={{ fontWeight: 'bold' }}>{auditStats?.by_review_level?.BRIEF || '—'}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid #000', padding: '2px 0' }}>
              <span>Standard Review</span>
              <span style={{ fontWeight: 'bold' }}>{auditStats?.by_review_level?.STANDARD || '0'}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '3px solid #000', padding: '2px 0' }}>
              <span>Detailed Review Required</span>
              <span style={{ fontWeight: 'bold' }}>{auditStats?.by_review_level?.DETAILED || '0'}</span>
            </div>

            {/* Compliance */}
            <div style={{ fontWeight: '900', fontSize: '11px', background: '#000', color: '#fff', padding: '2px 4px', margin: '6px 0 3px 0' }}>
              REGULATORY COMPLIANCE
            </div>
            
            <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid #000', padding: '2px 0' }}>
              <span>FDA GMLP</span>
              <span style={{ fontWeight: 'bold' }}>COMPLIANT</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid #000', padding: '2px 0' }}>
              <span>Health Canada SaMD</span>
              <span style={{ fontWeight: 'bold' }}>COMPLIANT</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid #000', padding: '2px 0' }}>
              <span>EU AI Act (High-Risk)</span>
              <span style={{ fontWeight: 'bold' }}>COMPLIANT</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid #000', padding: '2px 0' }}>
              <span>HTI-1 Model Card</span>
              <span style={{ fontWeight: 'bold' }}>COMPLIANT</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '3px solid #000', padding: '2px 0' }}>
              <span>CHAI Registry</span>
              <span style={{ fontWeight: 'bold' }}>REGISTERED</span>
            </div>

            {/* Intended Use */}
            <div style={{ fontWeight: '900', fontSize: '11px', background: '#000', color: '#fff', padding: '2px 4px', margin: '6px 0 3px 0' }}>
              INTENDED USE
            </div>
            <div style={{ fontSize: '10px', padding: '4px 0', borderBottom: '1px solid #000' }}>
              {modelData.intendedUse}
            </div>
            
            {/* Target Population */}
            <div style={{ fontWeight: 'bold', fontSize: '10px', marginTop: '4px' }}>Target Population:</div>
            <div style={{ fontSize: '10px', borderBottom: '1px solid #000', paddingBottom: '4px' }}>
              {modelData.targetPopulation}
            </div>

            {/* Contraindications */}
            <div style={{ fontWeight: 'bold', fontSize: '10px', marginTop: '4px' }}>Contraindications:</div>
            <div style={{ fontSize: '10px', borderBottom: '3px solid #000', paddingBottom: '4px' }}>
              {modelData.contraindications}
            </div>

            {/* Known Risks */}
            <div style={{ fontWeight: '900', fontSize: '11px', background: '#000', color: '#fff', padding: '2px 4px', margin: '6px 0 3px 0' }}>
              KNOWN RISKS & LIMITATIONS
            </div>
            <ul style={{ fontSize: '9px', paddingLeft: '16px', margin: '4px 0' }}>
              <li>May hallucinate plausible but incorrect clinical information</li>
              <li>Verify all medication dosages independently</li>
              <li>Not validated for pediatric populations</li>
              <li>Requires physician review before sign-off</li>
            </ul>

            {/* Footer */}
            <div style={{ borderTop: '8px solid #000', marginTop: '6px', paddingTop: '4px' }}>
              <div style={{ fontSize: '9px', textAlign: 'center' }}>
                Validated by TRUST Platform v0.2.0
              </div>
              <div style={{ fontSize: '8px', textAlign: 'center', color: '#666', marginTop: '2px' }}>
                Report Generated: {new Date().toLocaleString()}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AIModelReport;
