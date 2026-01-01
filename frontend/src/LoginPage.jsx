/**
 * TRUST Platform - Login / Landing Page
 * Clean, minimal design with animated verification ring
 */

import React, { useState, useEffect } from 'react';
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

// CSS for the ring animation
const animationStyles = `
  @keyframes drawRing {
    0% {
      stroke-dashoffset: 283;
    }
    100% {
      stroke-dashoffset: 47;
    }
  }
  
  @keyframes fadeInCheckmark {
    0% {
      opacity: 0;
      transform: scale(0.5);
    }
    100% {
      opacity: 1;
      transform: scale(1);
    }
  }
  
  @keyframes pulseGlow {
    0%, 100% {
      filter: drop-shadow(0 0 8px rgba(8, 145, 178, 0.4));
    }
    50% {
      filter: drop-shadow(0 0 16px rgba(8, 145, 178, 0.6));
    }
  }
  
  .trust-ring-animated {
    stroke-dasharray: 283;
    stroke-dashoffset: 283;
    animation: drawRing 1.8s ease-out forwards;
  }
  
  .trust-checkmark {
    opacity: 0;
    transform-origin: center;
    animation: fadeInCheckmark 0.4s ease-out 1.6s forwards;
  }
  
  .trust-ring-glow {
    animation: pulseGlow 3s ease-in-out infinite;
    animation-delay: 2s;
  }
`;

export function LoginPage() {
  const { login, isLoading, error } = useAuth();
  const [isSigningIn, setIsSigningIn] = useState(false);
  const [localError, setLocalError] = useState(null);

  // Inject animation styles
  useEffect(() => {
    const styleElement = document.createElement('style');
    styleElement.textContent = animationStyles;
    document.head.appendChild(styleElement);
    return () => {
      document.head.removeChild(styleElement);
    };
  }, []);

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
        
        {/* TRUST Logo with Animated Ring */}
        <div style={styles.logoContainer}>
          <svg 
            width="140" 
            height="140" 
            viewBox="0 0 140 140" 
            fill="none"
            className="trust-ring-glow"
          >
            {/* Background ring */}
            <circle 
              cx="70" 
              cy="70" 
              r="45" 
              stroke={COLORS.slate700} 
              strokeWidth="6" 
              fill="none"
            />
            
            {/* Animated progress ring */}
            <circle
              cx="70"
              cy="70"
              r="45"
              stroke={COLORS.teal500}
              strokeWidth="6"
              fill="none"
              strokeLinecap="round"
              className="trust-ring-animated"
              style={{
                transform: 'rotate(-90deg)',
                transformOrigin: 'center',
              }}
            />
            
            {/* Center text - TRUST */}
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
            
            {/* Checkmark circle - animated fade in */}
            <g className="trust-checkmark">
              <circle cx="25" cy="85" r="12" fill={COLORS.teal500}/>
              <path 
                d="M19 85 L23 89 L31 80" 
                stroke="#ffffff" 
                strokeWidth="3" 
                strokeLinecap="round" 
                strokeLinejoin="round" 
                fill="none"
              />
            </g>
          </svg>
          
          {/* PLATFORM text below the ring */}
          <p style={styles.platformText}>PLATFORM</p>
        </div>
        
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
          
          
        </div>
      </div>
      
      {/* Footer - positioned at bottom */}
      <footer style={styles.footer}>
        <p style={styles.privacyNote}>Protected under PIPEDA</p>
        <p style={styles.footerText}>© 2025 TRUST Platform • Providence Health Care</p>
      </footer>
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
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    position: 'relative',
  },
  
  content: {
    textAlign: 'center',
    padding: '40px 20px',
  },
  
  logoContainer: {
    marginBottom: '48px',
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
    marginBottom: '4px',
  },
  
  footer: {
    position: 'absolute',
    bottom: '24px',
    left: 0,
    right: 0,
    textAlign: 'center',
  },
  
  footerText: {
    color: COLORS.slate500,
    fontSize: '12px',
    fontFamily: "'Outfit', sans-serif",
    margin: 0,
  },
};

export default LoginPage;
