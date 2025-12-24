/**
 * TRUST Platform - Main Dashboard
 * ================================
 * Displays AI scribe analysis results with review queue.
 */

import React, { useState } from 'react';

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

// Risk level colors
const getRiskColor = (risk) => {
  switch (risk?.toUpperCase()) {
    case 'HIGH': return COLORS.red;
    case 'MEDIUM': return COLORS.amber;
    case 'LOW': return COLORS.green;
    default: return COLORS.slate[600];
  }
};

// Review tier colors
const getTierColor = (tier) => {
  switch (tier) {
    case 'detailed': return COLORS.red;
    case 'standard': return COLORS.amber;
    case 'brief': return COLORS.green;
    default: return COLORS.slate[600];
  }
};

// Verification status colors
const getStatusColor = (status) => {
  switch (status) {
    case 'verified': return COLORS.green;
    case 'contradicted': return COLORS.red;
    case 'not_found': return COLORS.amber;
    case 'partial': return COLORS.purple;
    default: return COLORS.slate[600];
  }
};

/**
 * Metric Ring Component - matches TRUST brand
 */
function MetricRing({ value, label, color = 'teal', size = 100 }) {
  const colorValue = COLORS[color] || COLORS.teal;
  const radius = (size / 2) - 10;
  const circumference = 2 * Math.PI * radius;
  const progress = (value / 100) * circumference;
  
  return (
    <div style={{ textAlign: 'center' }}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        {/* Background ring */}
        <circle
          cx={size/2}
          cy={size/2}
          r={radius}
          fill="none"
          stroke={COLORS.slate[200]}
          strokeWidth={size * 0.06}
        />
        {/* Progress ring */}
        <circle
          cx={size/2}
          cy={size/2}
          r={radius}
          fill="none"
          stroke={colorValue}
          strokeWidth={size * 0.06}
          strokeDasharray={circumference}
          strokeDashoffset={circumference - progress}
          strokeLinecap="round"
          transform={`rotate(-90 ${size/2} ${size/2})`}
        />
        {/* Center text */}
        <text
          x={size/2}
          y={size/2 - 5}
          textAnchor="middle"
          fontSize={size * 0.22}
          fontWeight="700"
          fill={COLORS.slate[800]}
        >
          {typeof value === 'number' ? `${Math.round(value)}%` : value}
        </text>
        <text
          x={size/2}
          y={size/2 + 15}
          textAnchor="middle"
          fontSize={size * 0.1}
          fontWeight="600"
          fill={COLORS.slate[600]}
        >
          {label}
        </text>
      </svg>
    </div>
  );
}

/**
 * Claim Card Component
 */
function ClaimCard({ claim, rank }) {
  const statusColor = getStatusColor(claim.verification.status);
  const tierColor = getTierColor(claim.uncertainty.review_tier);
  
  return (
    <div style={{
      background: 'white',
      borderRadius: '8px',
      padding: '16px',
      marginBottom: '12px',
      border: `2px solid ${claim.verification.status === 'contradicted' ? COLORS.red : COLORS.slate[200]}`,
      boxShadow: '0 1px 3px rgba(0,0,0,0.1)'
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div style={{ flex: 1 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
            <span style={{
              background: COLORS.slate[800],
              color: 'white',
              borderRadius: '50%',
              width: '24px',
              height: '24px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: '12px',
              fontWeight: '600'
            }}>
              {rank}
            </span>
            <span style={{
              background: statusColor,
              color: 'white',
              padding: '2px 8px',
              borderRadius: '4px',
              fontSize: '11px',
              fontWeight: '600',
              textTransform: 'uppercase'
            }}>
              {claim.verification.status}
            </span>
            <span style={{
              background: tierColor,
              color: 'white',
              padding: '2px 8px',
              borderRadius: '4px',
              fontSize: '11px',
              fontWeight: '600'
            }}>
              {claim.uncertainty.review_tier} review
            </span>
          </div>
          
          <p style={{ 
            margin: '0 0 8px 0', 
            fontWeight: '500',
            color: COLORS.slate[800]
          }}>
            {claim.claim.text}
          </p>
          
          <p style={{ 
            margin: 0, 
            fontSize: '13px',
            color: COLORS.slate[600]
          }}>
            {claim.verification.explanation}
          </p>
          
          {claim.uncertainty.flags.length > 0 && (
            <div style={{ marginTop: '8px', display: 'flex', gap: '4px', flexWrap: 'wrap' }}>
              {claim.uncertainty.flags.map((flag, i) => (
                <span key={i} style={{
                  background: COLORS.amber,
                  color: 'white',
                  padding: '2px 6px',
                  borderRadius: '3px',
                  fontSize: '10px',
                  fontWeight: '600'
                }}>
                  {flag}
                </span>
              ))}
            </div>
          )}
        </div>
        
        <div style={{ 
          textAlign: 'right',
          minWidth: '80px'
        }}>
          <div style={{ 
            fontSize: '24px', 
            fontWeight: '700',
            color: claim.priority_score > 40 ? COLORS.red : COLORS.slate[700]
          }}>
            {claim.priority_score.toFixed(1)}
          </div>
          <div style={{ fontSize: '11px', color: COLORS.slate[500] }}>
            priority
          </div>
        </div>
      </div>
    </div>
  );
}

/**
 * Main Dashboard Component
 */
export default function TrustDashboard({ analysisResult }) {
  if (!analysisResult) {
    return (
      <div style={{ 
        textAlign: 'center', 
        padding: '60px 20px',
        color: COLORS.slate[500]
      }}>
        <h2>No Analysis Results</h2>
        <p>Submit a note for analysis to see results here.</p>
      </div>
    );
  }
  
  const { summary, review_burden, review_queue, overall_risk } = analysisResult;
  
  return (
    <div style={{ padding: '20px', maxWidth: '1200px', margin: '0 auto' }}>
      {/* Header */}
      <div style={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center',
        marginBottom: '24px'
      }}>
        <div>
          <h1 style={{ margin: 0, color: COLORS.slate[800] }}>
            Analysis Results
          </h1>
          <p style={{ margin: '4px 0 0 0', color: COLORS.slate[500] }}>
            Note ID: {analysisResult.note_id} | Patient: {analysisResult.patient_id}
          </p>
        </div>
        <div style={{
          background: getRiskColor(overall_risk),
          color: 'white',
          padding: '8px 20px',
          borderRadius: '8px',
          fontWeight: '700',
          fontSize: '18px'
        }}>
          {overall_risk} RISK
        </div>
      </div>
      
      {/* Metrics Row */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(4, 1fr)',
        gap: '20px',
        marginBottom: '30px'
      }}>
        <div style={{ background: 'white', borderRadius: '12px', padding: '20px', textAlign: 'center' }}>
          <MetricRing 
            value={summary.verified / summary.total_claims * 100} 
            label="VERIFIED" 
            color="green"
          />
          <p style={{ margin: '10px 0 0 0', color: COLORS.slate[600] }}>
            {summary.verified} of {summary.total_claims} claims
          </p>
        </div>
        
        <div style={{ background: 'white', borderRadius: '12px', padding: '20px', textAlign: 'center' }}>
          <MetricRing 
            value={review_burden.time_saved_percent} 
            label="TIME SAVED" 
            color="teal"
          />
          <p style={{ margin: '10px 0 0 0', color: COLORS.slate[600] }}>
            vs. manual review
          </p>
        </div>
        
        <div style={{ background: 'white', borderRadius: '12px', padding: '20px', textAlign: 'center' }}>
          <MetricRing 
            value={review_burden.brief_percent || (review_burden.brief_review / review_burden.total_claims * 100)} 
            label="BRIEF" 
            color="green"
          />
          <p style={{ margin: '10px 0 0 0', color: COLORS.slate[600] }}>
            {review_burden.brief_review} quick reviews
          </p>
        </div>
        
        <div style={{ background: 'white', borderRadius: '12px', padding: '20px', textAlign: 'center' }}>
          <div style={{ fontSize: '48px', fontWeight: '700', color: summary.contradictions > 0 ? COLORS.red : COLORS.green }}>
            {summary.contradictions}
          </div>
          <div style={{ fontSize: '14px', fontWeight: '600', color: COLORS.slate[600] }}>
            CONTRADICTIONS
          </div>
          <p style={{ margin: '10px 0 0 0', color: COLORS.slate[600] }}>
            {summary.high_priority_count} high priority
          </p>
        </div>
      </div>
      
      {/* Review Queue */}
      <div style={{ background: 'white', borderRadius: '12px', padding: '20px' }}>
        <h2 style={{ margin: '0 0 16px 0', color: COLORS.slate[800] }}>
          Review Queue
          <span style={{ 
            fontSize: '14px', 
            fontWeight: 'normal',
            color: COLORS.slate[500],
            marginLeft: '12px'
          }}>
            Sorted by priority (highest first)
          </span>
        </h2>
        
        {review_queue.map((claim, index) => (
          <ClaimCard key={index} claim={claim} rank={index + 1} />
        ))}
      </div>
    </div>
  );
}
