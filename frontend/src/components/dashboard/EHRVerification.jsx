import React from 'react';

const COLORS = {
  green: '#10b981',
  amber: '#f59e0b', 
  red: '#ef4444',
  teal: '#0891b2',
  slate: { 100: '#f1f5f9', 600: '#475569', 800: '#1e293b' }
};

const getStatusColor = (status) => {
  switch (status) {
    case 'verified': return COLORS.green;
    case 'not_in_ehr': return COLORS.amber;
    case 'contradicted': return COLORS.red;
    default: return COLORS.slate[600];
  }
};

export default function EHRVerification({ ehrResult }) {
  if (!ehrResult) return null;

  return (
    <div style={{ 
      background: 'white', 
      borderRadius: '12px', 
      padding: '20px',
      margin: '20px',
      border: '2px solid ' + COLORS.teal
    }}>
      <h2 style={{ margin: '0 0 16px 0', color: COLORS.teal }}>
        EHR Verification Results
      </h2>
      
      <div style={{ 
        display: 'grid', 
        gridTemplateColumns: 'repeat(4, 1fr)', 
        gap: '16px',
        marginBottom: '20px'
      }}>
        <div style={{ textAlign: 'center', padding: '12px', background: COLORS.slate[100], borderRadius: '8px' }}>
          <div style={{ fontSize: '18px', fontWeight: '700' }}>{ehrResult.patient_name}</div>
          <div style={{ fontSize: '12px', color: COLORS.slate[600] }}>Patient</div>
        </div>
        <div style={{ textAlign: 'center', padding: '12px', background: COLORS.slate[100], borderRadius: '8px' }}>
          <div style={{ fontSize: '24px', fontWeight: '700', color: COLORS.green }}>{ehrResult.verified}</div>
          <div style={{ fontSize: '12px', color: COLORS.slate[600] }}>Verified</div>
        </div>
        <div style={{ textAlign: 'center', padding: '12px', background: COLORS.slate[100], borderRadius: '8px' }}>
          <div style={{ fontSize: '24px', fontWeight: '700', color: COLORS.amber }}>{ehrResult.not_in_ehr}</div>
          <div style={{ fontSize: '12px', color: COLORS.slate[600] }}>Not in EHR</div>
        </div>
        <div style={{ textAlign: 'center', padding: '12px', background: COLORS.slate[100], borderRadius: '8px' }}>
          <div style={{ fontSize: '24px', fontWeight: '700', color: COLORS.red }}>{ehrResult.contradicted}</div>
          <div style={{ fontSize: '12px', color: COLORS.slate[600] }}>Contradicted</div>
        </div>
      </div>

      <h3 style={{ margin: '0 0 12px 0' }}>Claim Results</h3>
      {ehrResult.results.map((r, i) => (
        <div key={i} style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          padding: '12px',
          marginBottom: '8px',
          background: COLORS.slate[100],
          borderRadius: '8px',
          borderLeft: '4px solid ' + getStatusColor(r.status)
        }}>
          <div>
            <div style={{ fontWeight: '600' }}>{r.claim_text}</div>
            <div style={{ fontSize: '12px', color: COLORS.slate[600] }}>{r.explanation}</div>
          </div>
          <span style={{
            background: getStatusColor(r.status),
            color: 'white',
            padding: '4px 12px',
            borderRadius: '4px',
            fontSize: '12px',
            fontWeight: '600'
          }}>
            {r.status.toUpperCase()}
          </span>
        </div>
      ))}
    </div>
  );
}
