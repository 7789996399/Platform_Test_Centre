import React, { useState, useEffect } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar,
  BarChart, Bar, AreaChart, Area
} from 'recharts';

const API_BASE_URL = 'http://127.0.0.1:8002/api/v1';

// TRUST Brand Colors
const COLORS = {
  teal: '#0891b2',
  green: '#10b981',
  amber: '#f59e0b',
  red: '#ef4444',
  purple: '#8b5cf6',
  slate: {
    50: '#f8fafc',
    100: '#f1f5f9',
    200: '#e2e8f0',
    600: '#475569',
    700: '#334155',
    800: '#1e293b',
    900: '#0f172a',
  }
};

// Sample data
const performanceTrendData = [
  { month: 'Jul', accuracy: 92.1, sensitivity: 89.3, specificity: 94.2 },
  { month: 'Aug', accuracy: 93.4, sensitivity: 90.8, specificity: 94.8 },
  { month: 'Sep', accuracy: 94.2, sensitivity: 91.5, specificity: 95.1 },
  { month: 'Oct', accuracy: 93.8, sensitivity: 92.1, specificity: 94.6 },
  { month: 'Nov', accuracy: 94.9, sensitivity: 93.2, specificity: 95.4 },
  { month: 'Dec', accuracy: 95.2, sensitivity: 94.1, specificity: 95.8 },
];

const complianceRadarData = [
  { framework: 'FDA GMLP', score: 94, fullMark: 100 },
  { framework: 'Health Canada', score: 92, fullMark: 100 },
  { framework: 'EU AI Act', score: 89, fullMark: 100 },
  { framework: 'WHO Guidelines', score: 93, fullMark: 100 },
  { framework: 'HIPAA', score: 100, fullMark: 100 },
  { framework: 'PIPEDA', score: 100, fullMark: 100 },
];

const physicianFeedbackData = [
  { month: 'Jul', approved: 145, modified: 23, rejected: 8 },
  { month: 'Aug', approved: 162, modified: 19, rejected: 5 },
  { month: 'Sep', approved: 178, modified: 15, rejected: 4 },
  { month: 'Oct', approved: 189, modified: 12, rejected: 3 },
  { month: 'Nov', approved: 201, modified: 10, rejected: 2 },
  { month: 'Dec', approved: 215, modified: 8, rejected: 2 },
];

const alertHistoryData = [
  { month: 'Jul', critical: 12, warning: 34, info: 89 },
  { month: 'Aug', critical: 8, warning: 28, info: 76 },
  { month: 'Sep', critical: 5, warning: 22, info: 65 },
  { month: 'Oct', critical: 3, warning: 18, info: 58 },
  { month: 'Nov', critical: 2, warning: 14, info: 52 },
  { month: 'Dec', critical: 1, warning: 11, info: 48 },
];

const driftDetectionData = [
  { week: 'W1', dataIntegrity: 99.2, featureDrift: 0.8, conceptDrift: 0.3 },
  { week: 'W2', dataIntegrity: 99.1, featureDrift: 1.2, conceptDrift: 0.5 },
  { week: 'W3', dataIntegrity: 98.9, featureDrift: 1.8, conceptDrift: 0.7 },
  { week: 'W4', dataIntegrity: 99.3, featureDrift: 1.1, conceptDrift: 0.4 },
];

const complianceHeatmapData = [
  { model: 'Sepsis Early Warning', type: 'Predictive', fda: 98, hc: 96, eu: 94, who: 97, hipaa: 100, pipeda: 100 },
  { model: 'Pneumonia Detection', type: 'Predictive', fda: 96, hc: 94, eu: 91, who: 95, hipaa: 100, pipeda: 100 },
  { model: 'Cardiac Risk Score', type: 'Predictive', fda: 94, hc: 92, eu: 88, who: 93, hipaa: 100, pipeda: 100 },
  { model: 'AI Scribe - Pre-op', type: 'Generative', fda: 95, hc: 93, eu: 90, who: 94, hipaa: 100, pipeda: 100 },
  { model: 'Clinical Note Assistant', type: 'Generative', fda: 97, hc: 95, eu: 92, who: 96, hipaa: 100, pipeda: 100 },
  { model: 'Patient Message Responder', type: 'Generative', fda: 82, hc: 80, eu: 78, who: 84, hipaa: 98, pipeda: 98 },
  { model: 'Chest X-Ray AI', type: 'Radiology', fda: 98, hc: 96, eu: 94, who: 97, hipaa: 100, pipeda: 100 },
  { model: 'CT Head - Hemorrhage', type: 'Radiology', fda: 94, hc: 92, eu: 89, who: 93, hipaa: 100, pipeda: 100 },
  { model: 'Mammography AI', type: 'Radiology', fda: 91, hc: 88, eu: 85, who: 90, hipaa: 100, pipeda: 100 },
];

// TRUST Logo
const TrustLogo = ({ size = 40 }) => {
  const center = size / 2;
  const radius = (size / 2) - 6;
  const circumference = 2 * Math.PI * radius;
  const dashOffset = circumference - (0.85 * circumference);
  const angle = (0.85 * 360) - 90;
  const angleRad = (angle * Math.PI) / 180;
  const checkX = center + radius * Math.cos(angleRad);
  const checkY = center + radius * Math.sin(angleRad);
  const checkRadius = size * 0.09;

  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      <circle cx={center} cy={center} r={radius} stroke={COLORS.slate[200]} strokeWidth="3" fill="none"/>
      <circle cx={center} cy={center} r={radius} stroke={COLORS.teal} strokeWidth="3" fill="none" strokeLinecap="round"
        strokeDasharray={circumference} strokeDashoffset={dashOffset} transform={`rotate(-90 ${center} ${center})`}/>
      <circle cx={checkX} cy={checkY} r={checkRadius} fill={COLORS.teal}/>
      <path d={`M${checkX - checkRadius * 0.4} ${checkY} L${checkX - checkRadius * 0.1} ${checkY + checkRadius * 0.3} L${checkX + checkRadius * 0.5} ${checkY - checkRadius * 0.4}`}
        stroke="#ffffff" strokeWidth={checkRadius * 0.3} strokeLinecap="round" strokeLinejoin="round" fill="none"/>
    </svg>
  );
};

// TRUST Metric Ring - WIDER and THINNER
const MetricRing = ({ value, label, title, color = COLORS.teal, size = 120 }) => {
  const center = size / 2;
  const radius = (size / 2) - 6; // Wider - less padding
  const strokeWidth = 4; // Thinner stroke
  const circumference = 2 * Math.PI * radius;
  const normalizedValue = Math.min(Math.max(value, 0), 100);
  const dashOffset = circumference - (normalizedValue / 100) * circumference;
  
  // Checkmark position
  const angle = (normalizedValue / 100) * 360 - 90;
  const angleRad = (angle * Math.PI) / 180;
  const checkX = center + radius * Math.cos(angleRad);
  const checkY = center + radius * Math.sin(angleRad);
  const checkRadius = 8;
  
  const isWarning = value < 90;

  return (
    <div style={{ textAlign: 'center' }}>
      {title && (
        <div style={{
          fontSize: '10px',
          fontWeight: '600',
          color: COLORS.slate[600],
          textTransform: 'uppercase',
          letterSpacing: '0.5px',
          marginBottom: '6px'
        }}>
          {title}
        </div>
      )}
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <circle cx={center} cy={center} r={radius} stroke={COLORS.slate[200]} strokeWidth={strokeWidth} fill="none"/>
        <circle cx={center} cy={center} r={radius} stroke={color} strokeWidth={strokeWidth} fill="none" strokeLinecap="round"
          strokeDasharray={circumference} strokeDashoffset={dashOffset} transform={`rotate(-90 ${center} ${center})`}
          style={{ transition: 'stroke-dashoffset 1s ease-out' }}/>
        <text x={center} y={center - 2} fontSize="18" fontWeight="700" fill={COLORS.slate[800]} textAnchor="middle" dominantBaseline="middle">
          {typeof value === 'number' ? `${value}%` : value}
        </text>
        <text x={center} y={center + 14} fontSize="8" fontWeight="600" fill={color} textAnchor="middle">
          {label}
        </text>
        {normalizedValue > 5 && (
          <>
            <circle cx={checkX} cy={checkY} r={checkRadius} fill={color} />
            {isWarning ? (
              <text x={checkX} y={checkY + 1} fontSize="10" fontWeight="700" fill="#fff" textAnchor="middle" dominantBaseline="middle">!</text>
            ) : (
              <path d={`M${checkX - 3} ${checkY} L${checkX - 1} ${checkY + 2} L${checkX + 3} ${checkY - 2}`}
                stroke="#ffffff" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" fill="none"/>
            )}
          </>
        )}
      </svg>
    </div>
  );
};

// Compliance Cell
const ComplianceCell = ({ value }) => {
  let bgColor, textColor;
  if (value >= 95) { bgColor = '#dcfce7'; textColor = '#166534'; }
  else if (value >= 85) { bgColor = '#fef3c7'; textColor = '#92400e'; }
  else { bgColor = '#fee2e2'; textColor = '#991b1b'; }
  
  return (
    <td style={{ padding: '8px 12px', textAlign: 'center', background: bgColor, color: textColor, fontWeight: '600', fontSize: '12px', border: '1px solid white' }}>
      {value}%
    </td>
  );
};

export default function AnalyticsDashboard() {
  const [activeTab, setActiveTab] = useState('overview');
  const [apiStatus, setApiStatus] = useState('checking');
  const [ehrResult, setEhrResult] = useState(null);

  useEffect(() => {
    const checkApi = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/scribe/health`);
        setApiStatus(response.ok ? 'connected' : 'error');
      } catch { setApiStatus('error'); }
    };
    checkApi();
  }, []);

  const runEHRDemo = async () => {
    try {
      const noteData = {
        note_id: 'analytics-demo', patient_id: '12724066', patient_name: 'Demo Patient',
        sections: { medications: 'metoprolol 50mg po bid, lisinopril 10mg po daily', allergies: 'NKDA' },
        source_transcript: 'Patient on metoprolol. No allergies.'
      };
      const response = await fetch(`${API_BASE_URL}/scribe/verify-ehr/12724066`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(noteData),
      });
      if (response.ok) setEhrResult(await response.json());
    } catch (err) { console.error('EHR verification failed:', err); }
  };

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'compliance', label: 'Compliance' },
    { id: 'models', label: 'AI Models' },
    { id: 'audit', label: 'Audit Trail' },
  ];

  return (
    <div style={{ fontFamily: "'Inter', system-ui, sans-serif", background: COLORS.slate[100], minHeight: '100vh' }}>
      {/* Header */}
      <header style={{ background: 'white', borderBottom: `1px solid ${COLORS.slate[200]}`, padding: '16px 24px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
            <TrustLogo size={40} />
            <div>
              <h1 style={{ margin: 0, fontSize: '20px', fontWeight: '700', color: COLORS.slate[900] }}>TRUST Analytics</h1>
              <p style={{ margin: 0, fontSize: '12px', color: COLORS.slate[600] }}>Healthcare AI Governance Platform</p>
            </div>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
            <span style={{
              padding: '6px 12px',
              background: apiStatus === 'connected' ? '#dcfce7' : '#fee2e2',
              color: apiStatus === 'connected' ? '#166534' : '#991b1b',
              borderRadius: '4px', fontSize: '11px', fontWeight: '600'
            }}>
              {apiStatus === 'connected' ? 'API Connected' : 'API Offline'}
            </span>
            <span style={{ fontSize: '13px', color: COLORS.slate[700] }}>Raubenheimer, Jean MD</span>
          </div>
        </div>
      </header>

      {/* Tabs */}
      <nav style={{ background: 'white', borderBottom: `1px solid ${COLORS.slate[200]}`, padding: '0 24px', display: 'flex', gap: '4px' }}>
        {tabs.map(tab => (
          <button key={tab.id} onClick={() => setActiveTab(tab.id)} style={{
            padding: '14px 20px', background: 'transparent', border: 'none',
            borderBottom: activeTab === tab.id ? `2px solid ${COLORS.teal}` : '2px solid transparent',
            color: activeTab === tab.id ? COLORS.teal : COLORS.slate[600],
            fontWeight: activeTab === tab.id ? '600' : '400', fontSize: '13px', cursor: 'pointer'
          }}>
            {tab.label}
          </button>
        ))}
      </nav>

      {/* Main Content */}
      <main style={{ padding: '24px' }}>
        {/* Overview Tab */}
        {activeTab === 'overview' && (
          <div>
            {/* Metric Rings */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '20px', marginBottom: '24px' }}>
              <div style={{ background: 'white', borderRadius: '8px', padding: '24px', boxShadow: '0 1px 3px rgba(0,0,0,0.05)', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                <MetricRing value={95.2} label="ACCURACY" title="Model Accuracy" color={COLORS.green} size={100} />
                <div style={{ marginTop: '8px', fontSize: '11px', color: COLORS.slate[600] }}>30-day average</div>
              </div>
              <div style={{ background: 'white', borderRadius: '8px', padding: '24px', boxShadow: '0 1px 3px rgba(0,0,0,0.05)', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                <MetricRing value={97} label="COMPLIANT" title="Compliance Score" color={COLORS.teal} size={100} />
                <div style={{ marginTop: '8px', fontSize: '11px', color: COLORS.slate[600] }}>6 frameworks</div>
              </div>
              <div style={{ background: 'white', borderRadius: '8px', padding: '24px', boxShadow: '0 1px 3px rgba(0,0,0,0.05)', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                <MetricRing value={100} label="EXPLAINABLE" title="SHAP Coverage" color={COLORS.purple} size={100} />
                <div style={{ marginTop: '8px', fontSize: '11px', color: COLORS.slate[600] }}>All models</div>
              </div>
              <div style={{ background: 'white', borderRadius: '8px', padding: '24px', boxShadow: '0 1px 3px rgba(0,0,0,0.05)', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                <MetricRing value={99.1} label="UPTIME" title="System Uptime" color={COLORS.teal} size={100} />
                <div style={{ marginTop: '8px', fontSize: '11px', color: COLORS.slate[600] }}>Last 30 days</div>
              </div>
            </div>

            {/* Charts Row 1 */}
            <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '20px', marginBottom: '24px' }}>
              <div style={{ background: 'white', borderRadius: '8px', padding: '20px', boxShadow: '0 1px 3px rgba(0,0,0,0.05)' }}>
                <h3 style={{ margin: '0 0 16px 0', fontSize: '14px', fontWeight: '600', color: COLORS.slate[800] }}>Model Performance Trends</h3>
                <ResponsiveContainer width="100%" height={260}>
                  <LineChart data={performanceTrendData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis dataKey="month" tick={{ fontSize: 11 }} />
                    <YAxis domain={[85, 100]} tick={{ fontSize: 11 }} />
                    <Tooltip />
                    <Legend wrapperStyle={{ fontSize: '11px' }} />
                    <Line type="monotone" dataKey="accuracy" stroke={COLORS.teal} strokeWidth={2} dot={{ r: 3 }} name="Accuracy" />
                    <Line type="monotone" dataKey="sensitivity" stroke={COLORS.green} strokeWidth={2} dot={{ r: 3 }} name="Sensitivity" />
                    <Line type="monotone" dataKey="specificity" stroke={COLORS.purple} strokeWidth={2} dot={{ r: 3 }} name="Specificity" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <div style={{ background: 'white', borderRadius: '8px', padding: '20px', boxShadow: '0 1px 3px rgba(0,0,0,0.05)' }}>
                <h3 style={{ margin: '0 0 16px 0', fontSize: '14px', fontWeight: '600', color: COLORS.slate[800] }}>Compliance Status</h3>
                <ResponsiveContainer width="100%" height={260}>
                  <RadarChart data={complianceRadarData}>
                    <PolarGrid stroke="#e2e8f0" />
                    <PolarAngleAxis dataKey="framework" tick={{ fontSize: 9 }} />
                    <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fontSize: 9 }} />
                    <Radar name="Compliance" dataKey="score" stroke={COLORS.teal} fill={COLORS.teal} fillOpacity={0.4} />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Charts Row 2 */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginBottom: '24px' }}>
              <div style={{ background: 'white', borderRadius: '8px', padding: '20px', boxShadow: '0 1px 3px rgba(0,0,0,0.05)' }}>
                <h3 style={{ margin: '0 0 16px 0', fontSize: '14px', fontWeight: '600', color: COLORS.slate[800] }}>Physician Feedback Patterns</h3>
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={physicianFeedbackData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis dataKey="month" tick={{ fontSize: 11 }} />
                    <YAxis tick={{ fontSize: 11 }} />
                    <Tooltip />
                    <Legend wrapperStyle={{ fontSize: '11px' }} />
                    <Bar dataKey="approved" stackId="a" fill={COLORS.green} name="Approved" />
                    <Bar dataKey="modified" stackId="a" fill={COLORS.amber} name="Modified" />
                    <Bar dataKey="rejected" stackId="a" fill={COLORS.red} name="Rejected" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <div style={{ background: 'white', borderRadius: '8px', padding: '20px', boxShadow: '0 1px 3px rgba(0,0,0,0.05)' }}>
                <h3 style={{ margin: '0 0 16px 0', fontSize: '14px', fontWeight: '600', color: COLORS.slate[800] }}>Alert History</h3>
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={alertHistoryData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis dataKey="month" tick={{ fontSize: 11 }} />
                    <YAxis tick={{ fontSize: 11 }} />
                    <Tooltip />
                    <Legend wrapperStyle={{ fontSize: '11px' }} />
                    <Bar dataKey="critical" stackId="a" fill={COLORS.red} name="Critical" />
                    <Bar dataKey="warning" stackId="a" fill={COLORS.amber} name="Warning" />
                    <Bar dataKey="info" stackId="a" fill={COLORS.teal} name="Info" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Drift Detection */}
            <div style={{ background: 'white', borderRadius: '8px', padding: '20px', boxShadow: '0 1px 3px rgba(0,0,0,0.05)' }}>
              <h3 style={{ margin: '0 0 16px 0', fontSize: '14px', fontWeight: '600', color: COLORS.slate[800] }}>Drift Detection (Last 4 Weeks)</h3>
              <ResponsiveContainer width="100%" height={180}>
                <AreaChart data={driftDetectionData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="week" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Legend wrapperStyle={{ fontSize: '11px' }} />
                  <Area type="monotone" dataKey="dataIntegrity" stroke={COLORS.green} fill={COLORS.green} fillOpacity={0.3} name="Data Integrity %" />
                  <Area type="monotone" dataKey="featureDrift" stroke={COLORS.amber} fill={COLORS.amber} fillOpacity={0.3} name="Feature Drift %" />
                  <Area type="monotone" dataKey="conceptDrift" stroke={COLORS.red} fill={COLORS.red} fillOpacity={0.3} name="Concept Drift %" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* Compliance Tab */}
        {activeTab === 'compliance' && (
          <div style={{ background: 'white', borderRadius: '8px', padding: '24px', boxShadow: '0 1px 3px rgba(0,0,0,0.05)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
              <h3 style={{ margin: 0, fontSize: '16px', fontWeight: '600', color: COLORS.slate[800] }}>Compliance Heatmap</h3>
              <div style={{ display: 'flex', gap: '16px', fontSize: '11px' }}>
                <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <span style={{ width: '14px', height: '14px', background: '#dcfce7', borderRadius: '2px' }}></span> 95%+
                </span>
                <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <span style={{ width: '14px', height: '14px', background: '#fef3c7', borderRadius: '2px' }}></span> 85-94%
                </span>
                <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <span style={{ width: '14px', height: '14px', background: '#fee2e2', borderRadius: '2px' }}></span> Below 85%
                </span>
              </div>
            </div>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '12px' }}>
              <thead>
                <tr style={{ background: COLORS.slate[800], color: 'white' }}>
                  <th style={{ padding: '12px', textAlign: 'left' }}>AI Model</th>
                  <th style={{ padding: '12px', textAlign: 'center' }}>FDA GMLP</th>
                  <th style={{ padding: '12px', textAlign: 'center' }}>Health Canada</th>
                  <th style={{ padding: '12px', textAlign: 'center' }}>EU AI Act</th>
                  <th style={{ padding: '12px', textAlign: 'center' }}>WHO</th>
                  <th style={{ padding: '12px', textAlign: 'center' }}>HIPAA</th>
                  <th style={{ padding: '12px', textAlign: 'center' }}>PIPEDA</th>
                  <th style={{ padding: '12px', textAlign: 'center' }}>Avg</th>
                </tr>
              </thead>
              <tbody>
                {complianceHeatmapData.map((row, idx) => {
                  const avg = Math.round((row.fda + row.hc + row.eu + row.who + row.hipaa + row.pipeda) / 6);
                  return (
                    <tr key={idx}>
                      <td style={{ padding: '12px', fontWeight: '500' }}>{row.model} <span style={{ fontSize: '10px', color: COLORS.slate[600] }}>({row.type})</span></td>
                      <ComplianceCell value={row.fda} />
                      <ComplianceCell value={row.hc} />
                      <ComplianceCell value={row.eu} />
                      <ComplianceCell value={row.who} />
                      <ComplianceCell value={row.hipaa} />
                      <ComplianceCell value={row.pipeda} />
                      <ComplianceCell value={avg} />
                    </tr>
                  );
                })}
              </tbody>
            </table>
            <div style={{ marginTop: '20px', padding: '16px', background: '#fef2f2', border: '1px solid #fecaca', borderRadius: '6px' }}>
              <div style={{ fontWeight: '600', color: '#991b1b', fontSize: '13px' }}>Action Required</div>
              <div style={{ fontSize: '12px', color: '#991b1b' }}>Patient Message Responder below threshold on EU AI Act (78%)</div>
            </div>
          </div>
        )}

        {/* AI Models Tab - FIXED RINGS */}
        {activeTab === 'models' && (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '20px' }}>
            {[
              { name: 'Sepsis Early Warning', type: 'Predictive', accuracy: 94.2, status: 'Active' },
              { name: 'AI Scribe - Pre-op', type: 'Generative', accuracy: 92.1, status: 'Active' },
              { name: 'Chest X-Ray AI', type: 'Radiology', accuracy: 96.8, status: 'Active' },
              { name: 'Cardiac Risk Score', type: 'Predictive', accuracy: 91.5, status: 'Active' },
              { name: 'Clinical Note Assistant', type: 'Generative', accuracy: 93.4, status: 'Active' },
              { name: 'Mammography AI', type: 'Radiology', accuracy: 89.2, status: 'Review' },
            ].map((model, idx) => (
              <div key={idx} style={{
                background: 'white', borderRadius: '8px', padding: '24px', boxShadow: '0 1px 3px rgba(0,0,0,0.05)',
                border: model.status === 'Review' ? `1px solid ${COLORS.amber}` : `1px solid ${COLORS.slate[200]}`
              }}>
                <div style={{ marginBottom: '16px' }}>
                  <div style={{ fontWeight: '600', color: COLORS.slate[800], fontSize: '14px' }}>{model.name}</div>
                  <div style={{ fontSize: '11px', color: COLORS.slate[600] }}>{model.type}</div>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <MetricRing 
                    value={model.accuracy} 
                    label="ACCURACY" 
                    color={model.accuracy >= 90 ? COLORS.green : COLORS.amber} 
                    size={90}
                  />
                  <span style={{
                    padding: '4px 10px', borderRadius: '4px', fontSize: '10px', fontWeight: '600',
                    background: model.status === 'Active' ? '#dcfce7' : '#fef3c7',
                    color: model.status === 'Active' ? '#166534' : '#92400e'
                  }}>
                    {model.status}
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Audit Tab */}
        {activeTab === 'audit' && (
          <div style={{ background: 'white', borderRadius: '8px', padding: '24px', boxShadow: '0 1px 3px rgba(0,0,0,0.05)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
              <h3 style={{ margin: 0, fontSize: '16px', fontWeight: '600', color: COLORS.slate[800] }}>Audit Trail</h3>
              <button onClick={runEHRDemo} style={{
                padding: '8px 16px', background: COLORS.teal, color: 'white', border: 'none',
                borderRadius: '4px', cursor: 'pointer', fontWeight: '600', fontSize: '12px'
              }}>
                Run Live EHR Verification
              </button>
            </div>
            {ehrResult && (
              <div style={{ marginBottom: '20px', padding: '16px', background: '#f0f9ff', border: `1px solid ${COLORS.teal}`, borderRadius: '6px' }}>
                <div style={{ fontWeight: '600', color: COLORS.teal, marginBottom: '8px', fontSize: '13px' }}>
                  EHR Verification Complete - {ehrResult.patient_name}
                </div>
                <div style={{ display: 'flex', gap: '24px', fontSize: '12px' }}>
                  <span>Verified: <strong>{ehrResult.verified}</strong></span>
                  <span>Not in EHR: <strong>{ehrResult.not_in_ehr}</strong></span>
                  <span>Contradicted: <strong>{ehrResult.contradicted}</strong></span>
                </div>
              </div>
            )}
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '12px' }}>
              <thead>
                <tr style={{ background: COLORS.slate[100] }}>
                  <th style={{ padding: '12px', textAlign: 'left', borderBottom: `1px solid ${COLORS.slate[200]}` }}>Time</th>
                  <th style={{ padding: '12px', textAlign: 'left', borderBottom: `1px solid ${COLORS.slate[200]}` }}>Event</th>
                  <th style={{ padding: '12px', textAlign: 'left', borderBottom: `1px solid ${COLORS.slate[200]}` }}>User</th>
                  <th style={{ padding: '12px', textAlign: 'left', borderBottom: `1px solid ${COLORS.slate[200]}` }}>Model</th>
                  <th style={{ padding: '12px', textAlign: 'left', borderBottom: `1px solid ${COLORS.slate[200]}` }}>Status</th>
                </tr>
              </thead>
              <tbody>
                {[
                  { time: '10:42:15', event: 'EHR Verification', user: 'System', model: 'AI Scribe', status: 'Completed' },
                  { time: '10:38:22', event: 'Note Approved', user: 'Dr. Raubenheimer', model: 'AI Scribe', status: 'Success' },
                  { time: '10:35:01', event: 'Hallucination Detected', user: 'System', model: 'AI Scribe', status: 'Flagged' },
                  { time: '10:30:45', event: 'Model Drift Alert', user: 'System', model: 'Sepsis AI', status: 'Warning' },
                  { time: '10:28:12', event: 'Compliance Check', user: 'System', model: 'All Models', status: 'Passed' },
                ].map((row, idx) => (
                  <tr key={idx} style={{ borderBottom: `1px solid ${COLORS.slate[100]}` }}>
                    <td style={{ padding: '12px', fontFamily: 'monospace', fontSize: '11px' }}>{row.time}</td>
                    <td style={{ padding: '12px' }}>{row.event}</td>
                    <td style={{ padding: '12px', color: COLORS.slate[600] }}>{row.user}</td>
                    <td style={{ padding: '12px' }}>{row.model}</td>
                    <td style={{ padding: '12px' }}>
                      <span style={{
                        padding: '2px 8px', borderRadius: '4px', fontSize: '10px', fontWeight: '600',
                        background: row.status === 'Flagged' ? '#fee2e2' : row.status === 'Warning' ? '#fef3c7' : '#dcfce7',
                        color: row.status === 'Flagged' ? '#991b1b' : row.status === 'Warning' ? '#92400e' : '#166534'
                      }}>
                        {row.status}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </main>

      <footer style={{ textAlign: 'center', padding: '16px', color: COLORS.slate[600], fontSize: '11px', borderTop: `1px solid ${COLORS.slate[200]}`, background: 'white' }}>
        TRUST Platform v0.2.0 | FDA GMLP | Health Canada | EU AI Act | WHO | HIPAA | PIPEDA
      </footer>
    </div>
  );
}
