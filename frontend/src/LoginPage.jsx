/**
 * TRUST Platform - Login / Landing Page
 * Clean, minimal design
 */

import React, { useState } from 'react';
import { useAuth } from './AuthProvider';
import { isAzureConfigured } from './authConfig';

// Brand colors
const COLORS = {
  teal500: '#0891b2',
  slate100: '#f1f5f9',
  slate400: '#94a3b8',
  slate500: '#64748b',
  slate700: '#334155',
  slate800: '#1e293b',
  slate900: '#0f172a',
  warning: '#f59e0b',
  error: '#ef4444',
};

export function LoginPage() {
  const { login, isLoading, error } = useAuth();
  const [isSigningIn, setIsSigningIn] = useState(false);
  const [localError, setLocalError] = useState(null);

  const handleSignIn = async () => {
    setIsSigningIn(true);
    setLocalError(null);
    
    try {
      await login();
    } catch (err) {
      setLocalError(err.message || 'Sign in failed. Please try again.');
    } finally {
      setIsSigningIn(false);
    }
  };

  const azureConfigured = isAzureConfigured();

  return (
    <div style={styles.container}>
      <div style={styles.content}>
        
        {/* TRUST Logo */}
        <div style={styles.logoContainer}>
          <svg width="140" height="140" viewBox="0 0 140 140" fill="none">
            {/* Background ring */}
            <circle 
              cx="70" 
              cy="70" 
              r="50" 
              stroke={COLORS.slate700} 
              strokeWidth="6" 
              fill="none"
            />
            
            {/* Progress ring */}
            <path 
              d="M70 20 A50 50 0 1 1 25 85" 
              stroke={COLORS.teal500} 
              strokeWidth="6" 
              fill="none" 
              strokeLinecap="round"
              style={{ filter: `drop-shadow(0 0 10px ${COLORS.teal500}40)` }}
            />
            
            {/* Center text - TRUST only */}
            <text 
              x="70" 
              y="75" 
              fontFamily="'Sora', sans-serif" 
              fontSize="22" 
              fontWeight="700" 
              fill={COLORS.slate100} 
              textAnchor="middle"
            >
              TRUST
            </text>
            
            {/* Checkmark circle */}
            <circle cx="25" cy="85" r="12" fill={COLORS.teal500}/>
            <path 
              d="M19 85 L23 89 L31 80" 
              stroke="#ffffff" 
              strokeWidth="3" 
              strokeLinecap="round" 
              strokeLinejoin="round" 
              fill="none"
            />
          </svg>
          
          {/* PLATFORM text below the ring */}
          <p style={styles.platformText}>PLATFORM</p>
        </div>
        
        {/* Tagline */}
        <h1 style={styles.headline}>AI Governance for Healthcare</h1>
        <p style={styles.tagline}>Auditing AI. Protecting Patients. Empowering Physicians.</p>
        
        {/* Sign In Section */}
        <div style={styles.signInSection}>
          
          {!azureConfigured && (
            <div style={styles.demoNotice}>
              <span style={styles.demoBadge}>DEMO MODE</span>
              <p style={styles.demoText}>Azure AD not configured</p>
            </div>
          )}
          
          <button 
            onClick={handleSignIn}
            disabled={isSigningIn || isLoading}
            style={{
              ...styles.signInButton,
              opacity: (isSigningIn || isLoading) ? 0.7 : 1,
            }}
          >
            {(isSigningIn || isLoading) ? (
              <span>Signing in...</span>
            ) : (
              <>
                <MicrosoftIcon />
                <span>Sign in with Microsoft</span>
              </>
            )}
          </button>
          
          {(error || localError) && (
            <p style={styles.errorText}>⚠️ {error || localError}</p>
          )}
          
          <p style={styles.privacyNote}>
            Protected under PIPEDA compliance
          </p>
        </div>
        
        {/* Footer */}
        <footer style={styles.footer}>
          <p>© 2025 TRUST Platform • Providence Health Care</p>
        </footer>
      </div>
    </div>
  );
}

function MicrosoftIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 21 21" style={{ marginRight: '10px' }}>
      <rect x="1" y="1" width="9" height="9" fill="#f25022"/>
      <rect x="11" y="1" width="9" height="9" fill="#7fba00"/>
      <rect x="1" y="11" width="9" height="9" fill="#00a4ef"/>
      <rect x="11" y="11" width="9" height="9" fill="#ffb900"/>
    </svg>
  );
}

const styles = {
  container: {
    minHeight: '100vh',
    background: `linear-gradient(135deg, ${COLORS.slate900} 0%, ${COLORS.slate800} 100%)`,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  
  content: {
    textAlign: 'center',
    padding: '40px 20px',
  },
  
  logoContainer: {
    marginBottom: '40px',
  },
  
  platformText: {
    fontFamily: "'Outfit', sans-serif",
    fontSize: '14px',
    fontWeight: 500,
    color: COLORS.slate400,
    letterSpacing: '4px',
    marginTop: '16px',
    marginBottom: 0,
  },
  
  headline: {
    fontFamily: "'Sora', sans-serif",
    fontSize: '2rem',
    fontWeight: 700,
    color: COLORS.slate100,
    marginBottom: '12px',
    letterSpacing: '-0.02em',
  },
  
  tagline: {
    fontFamily: "'Outfit', sans-serif",
    fontSize: '1rem',
    color: COLORS.slate400,
    marginBottom: '48px',
  },
  
  signInSection: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: '16px',
  },
  
  demoNotice: {
    background: `${COLORS.warning}15`,
    border: `1px solid ${COLORS.warning}40`,
    borderRadius: '8px',
    padding: '10px 20px',
    marginBottom: '8px',
  },
  
  demoBadge: {
    background: COLORS.warning,
    color: COLORS.slate900,
    padding: '2px 8px',
    borderRadius: '4px',
    fontSize: '11px',
    fontWeight: 700,
  },
  
  demoText: {
    color: COLORS.slate400,
    fontSize: '12px',
    marginTop: '6px',
    marginBottom: 0,
  },
  
  signInButton: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    background: '#ffffff',
    color: COLORS.slate800,
    border: 'none',
    borderRadius: '8px',
    padding: '14px 32px',
    fontSize: '15px',
    fontWeight: 600,
    fontFamily: "'Outfit', sans-serif",
    cursor: 'pointer',
    boxShadow: '0 4px 14px rgba(0,0,0,0.25)',
    minWidth: '280px',
  },
  
  errorText: {
    color: COLORS.error,
    fontSize: '14px',
  },
  
  privacyNote: {
    color: COLORS.slate500,
    fontSize: '12px',
    marginTop: '8px',
  },
  
  footer: {
    marginTop: '80px',
    color: COLORS.slate500,
    fontSize: '12px',
    fontFamily: "'Outfit', sans-serif",
  },
};

export default LoginPage;
