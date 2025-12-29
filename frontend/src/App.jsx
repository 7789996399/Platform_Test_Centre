import React from 'react';
import { AuthProvider, ProtectedRoute, useAuth } from './AuthProvider';
import PowerChartDashboard from './components/dashboard/PowerChartDashboard';
import AnalyticsDashboard from './components/dashboard/AnalyticsDashboard';

// TRUST Brand Colors
const COLORS = {
  teal: '#0891b2',
  green: '#10b981',
  slate: {
    50: '#f8fafc',
    100: '#f1f5f9',
    200: '#e2e8f0',
    400: '#94a3b8',
    600: '#475569',
    700: '#334155',
    800: '#1e293b',
    900: '#0f172a',
  }
};

// TRUST Logo Component
const TrustLogo = ({ size = 80 }) => {
  const center = size / 2;
  const radius = (size / 2) - 8;
  const strokeWidth = size * 0.05;
  const circumference = 2 * Math.PI * radius;
  const dashOffset = circumference - (0.85 * circumference);
  const angle = (0.85 * 360) - 90;
  const angleRad = (angle * Math.PI) / 180;
  const checkX = center + radius * Math.cos(angleRad);
  const checkY = center + radius * Math.sin(angleRad);
  const checkRadius = size * 0.09;

  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      <circle cx={center} cy={center} r={radius} stroke={COLORS.slate[200]} strokeWidth={strokeWidth} fill="none"/>
      <circle cx={center} cy={center} r={radius} stroke={COLORS.teal} strokeWidth={strokeWidth} fill="none" strokeLinecap="round"
        strokeDasharray={circumference} strokeDashoffset={dashOffset} transform={`rotate(-90 ${center} ${center})`}/>
      <circle cx={checkX} cy={checkY} r={checkRadius} fill={COLORS.teal}/>
      <path d={`M${checkX - checkRadius * 0.4} ${checkY} L${checkX - checkRadius * 0.1} ${checkY + checkRadius * 0.3} L${checkX + checkRadius * 0.5} ${checkY - checkRadius * 0.4}`}
        stroke="#ffffff" strokeWidth={checkRadius * 0.3} strokeLinecap="round" strokeLinejoin="round" fill="none"/>
    </svg>
  );
};

// Landing Page - With User Info and Logout
const LandingPage = () => {
  const { user, logout } = useAuth();
  
  return (
    <div style={{
      minHeight: '100vh',
      background: COLORS.slate[50],
      fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, sans-serif",
      display: 'flex',
      flexDirection: 'column'
    }}>
      {/* Header */}
      <header style={{
        background: 'white',
        borderBottom: `1px solid ${COLORS.slate[200]}`,
        padding: '16px 40px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <TrustLogo size={36} />
          <span style={{ fontSize: '18px', fontWeight: '700', color: COLORS.slate[900] }}>TRUST</span>
        </div>
        
        {/* User Info & Logout */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <div style={{ textAlign: 'right' }}>
            <div style={{ fontSize: '14px', fontWeight: '500', color: COLORS.slate[800] }}>
              {user?.name || 'User'}
            </div>
            <div style={{ fontSize: '11px', color: COLORS.slate[600] }}>
              {user?.email || 'Providence Health Care'}
            </div>
          </div>
          <button
            onClick={logout}
            style={{
              background: 'transparent',
              border: `1px solid ${COLORS.slate[200]}`,
              borderRadius: '6px',
              padding: '8px 12px',
              fontSize: '13px',
              color: COLORS.slate[600],
              cursor: 'pointer',
            }}
          >
            Sign Out
          </button>
        </div>
      </header>

      {/* Main Content */}
      <main style={{
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '40px'
      }}>
        <TrustLogo size={80} />
        <h1 style={{
          fontSize: '32px',
          fontWeight: '700',
          color: COLORS.slate[900],
          margin: '20px 0 8px 0'
        }}>
          TRUST Platform
        </h1>
        <p style={{
          fontSize: '13px',
          color: COLORS.slate[600],
          letterSpacing: '1.5px',
          textTransform: 'uppercase',
          marginBottom: '40px'
        }}>
          Healthcare AI Governance
        </p>

        {/* Dashboard Links */}
        <div style={{ display: 'flex', gap: '24px' }}>
          <a
            href="/physician"
            style={{
              display: 'block',
              width: '280px',
              background: 'white',
              border: `1px solid ${COLORS.slate[200]}`,
              borderRadius: '8px',
              padding: '28px',
              textDecoration: 'none',
              color: 'inherit',
              transition: 'all 0.2s ease',
            }}
            onMouseOver={(e) => {
              e.currentTarget.style.borderColor = COLORS.teal;
              e.currentTarget.style.boxShadow = '0 4px 12px rgba(8, 145, 178, 0.12)';
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.borderColor = COLORS.slate[200];
              e.currentTarget.style.boxShadow = 'none';
            }}
          >
            <h2 style={{ fontSize: '16px', fontWeight: '600', color: COLORS.slate[900], margin: '0 0 6px 0' }}>
              Physician Dashboard
            </h2>
            <p style={{ fontSize: '13px', color: COLORS.slate[600], margin: '0 0 16px 0' }}>
              PowerChart AI Document Review
            </p>
            <span style={{ fontSize: '13px', fontWeight: '600', color: COLORS.teal }}>
              Open →
            </span>
          </a>

          <a
            href="/analytics"
            style={{
              display: 'block',
              width: '280px',
              background: 'white',
              border: `1px solid ${COLORS.slate[200]}`,
              borderRadius: '8px',
              padding: '28px',
              textDecoration: 'none',
              color: 'inherit',
              transition: 'all 0.2s ease',
            }}
            onMouseOver={(e) => {
              e.currentTarget.style.borderColor = COLORS.teal;
              e.currentTarget.style.boxShadow = '0 4px 12px rgba(8, 145, 178, 0.12)';
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.borderColor = COLORS.slate[200];
              e.currentTarget.style.boxShadow = 'none';
            }}
          >
            <h2 style={{ fontSize: '16px', fontWeight: '600', color: COLORS.slate[900], margin: '0 0 6px 0' }}>
              Analytics Dashboard
            </h2>
            <p style={{ fontSize: '13px', color: COLORS.slate[600], margin: '0 0 16px 0' }}>
              Governance & Compliance
            </p>
            <span style={{ fontSize: '13px', fontWeight: '600', color: COLORS.teal }}>
              Open →
            </span>
          </a>
        </div>
      </main>

      {/* Footer */}
      <footer style={{
        padding: '16px 40px',
        textAlign: 'center',
        fontSize: '11px',
        color: COLORS.slate[600],
        borderTop: `1px solid ${COLORS.slate[200]}`,
        background: 'white'
      }}>
        v0.3.0 | Cerner EHR Connected | Azure AD Authenticated
      </footer>
    </div>
  );
};

// Main App Router (inside ProtectedRoute)
const AppRouter = () => {
  const path = window.location.pathname;
  
  if (path === '/physician') return <PowerChartDashboard />;
  if (path === '/analytics') return <AnalyticsDashboard />;
  return <LandingPage />;
};

// App with Auth Wrapper
function App() {
  return (
    <AuthProvider>
      <ProtectedRoute>
        <AppRouter />
      </ProtectedRoute>
    </AuthProvider>
  );
}

export default App;
