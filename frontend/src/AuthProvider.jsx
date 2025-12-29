/**
 * ============================================================
 * TRUST Platform - Authentication Provider
 * ============================================================
 * 
 * WHAT THIS FILE DOES:
 * This is like a "security guard" that wraps your entire app.
 * It keeps track of:
 *   - Is anyone logged in?
 *   - Who is logged in?
 *   - How to log in / log out
 * 
 * HOW TO USE:
 * 1. Wrap your app with <AuthProvider>
 * 2. Use the useAuth() hook anywhere to check login status
 * 
 * ============================================================
 */

import React, { createContext, useContext, useState, useEffect } from 'react';
import { PublicClientApplication, InteractionStatus } from '@azure/msal-browser';
import { MsalProvider, useMsal, useIsAuthenticated } from '@azure/msal-react';
import { msalConfig, loginRequest, graphConfig, isAzureConfigured } from './authConfig';

// ===========================================
// STEP 1: Create the MSAL Instance
// ===========================================

/**
 * This creates the "connection" to Microsoft's login service.
 * Think of it as opening a phone line to Microsoft.
 */
let msalInstance = null;

if (isAzureConfigured()) {
  msalInstance = new PublicClientApplication(msalConfig);
}

// ===========================================
// STEP 2: Create Auth Context
// ===========================================

/**
 * Context = a way to share data across all components
 * Think of it like a "bulletin board" that any component can read
 */
const AuthContext = createContext(null);

// ===========================================
// STEP 3: Auth Provider Component
// ===========================================

/**
 * AuthProvider - Wraps your app and provides login functionality
 * 
 * Usage:
 *   <AuthProvider>
 *     <YourApp />
 *   </AuthProvider>
 */
export function AuthProvider({ children }) {
  // If Azure isn't configured, use demo mode
  if (!isAzureConfigured()) {
    return (
      <DemoAuthProvider>
        {children}
      </DemoAuthProvider>
    );
  }

  // Use real Azure authentication
  return (
    <MsalProvider instance={msalInstance}>
      <MsalAuthProvider>
        {children}
      </MsalAuthProvider>
    </MsalProvider>
  );
}

// ===========================================
// STEP 4: Real Azure Auth Provider
// ===========================================

function MsalAuthProvider({ children }) {
  const { instance, accounts, inProgress } = useMsal();
  const isAuthenticated = useIsAuthenticated();
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Get user info when logged in
  useEffect(() => {
    const fetchUserData = async () => {
      if (isAuthenticated && accounts.length > 0) {
        try {
          // Get access token
          const response = await instance.acquireTokenSilent({
            ...loginRequest,
            account: accounts[0],
          });

          // Fetch user profile from Microsoft Graph
          const graphResponse = await fetch(graphConfig.graphMeEndpoint, {
            headers: {
              Authorization: `Bearer ${response.accessToken}`,
            },
          });

          if (graphResponse.ok) {
            const userData = await graphResponse.json();
            setUser({
              id: userData.id,
              name: userData.displayName,
              email: userData.mail || userData.userPrincipalName,
              firstName: userData.givenName,
              lastName: userData.surname,
              jobTitle: userData.jobTitle,
              department: userData.department,
              // Add a photo URL if available
              photoUrl: `https://graph.microsoft.com/v1.0/me/photo/$value`,
            });
          }
        } catch (err) {
          console.error('Error fetching user data:', err);
          // Still set basic user info from the account
          setUser({
            name: accounts[0].name,
            email: accounts[0].username,
          });
        }
      } else {
        setUser(null);
      }
      setLoading(false);
    };

    if (inProgress === InteractionStatus.None) {
      fetchUserData();
    }
  }, [isAuthenticated, accounts, instance, inProgress]);

  // Login function
  const login = async () => {
    try {
      setError(null);
      await instance.loginPopup(loginRequest);
    } catch (err) {
      console.error('Login error:', err);
      setError(err.message);
      throw err;
    }
  };

  // Logout function
  const logout = async () => {
    try {
      await instance.logoutPopup({
        postLogoutRedirectUri: msalConfig.auth.postLogoutRedirectUri,
      });
      setUser(null);
    } catch (err) {
      console.error('Logout error:', err);
    }
  };

  // Get access token (for API calls)
  const getAccessToken = async () => {
    if (!isAuthenticated || accounts.length === 0) {
      return null;
    }

    try {
      const response = await instance.acquireTokenSilent({
        ...loginRequest,
        account: accounts[0],
      });
      return response.accessToken;
    } catch (err) {
      // If silent token acquisition fails, try popup
      try {
        const response = await instance.acquireTokenPopup(loginRequest);
        return response.accessToken;
      } catch (popupErr) {
        console.error('Error acquiring token:', popupErr);
        return null;
      }
    }
  };

  // The value we share with all components
  const value = {
    user,
    isAuthenticated,
    isLoading: loading || inProgress !== InteractionStatus.None,
    error,
    login,
    logout,
    getAccessToken,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
}

// ===========================================
// STEP 5: Demo Auth Provider (for testing)
// ===========================================

/**
 * This is used when Azure isn't configured yet.
 * Allows you to test the UI with a fake login.
 */
function DemoAuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  // Check for saved demo session
  useEffect(() => {
    const savedUser = sessionStorage.getItem('trust_demo_user');
    if (savedUser) {
      setUser(JSON.parse(savedUser));
      setIsAuthenticated(true);
    }
  }, []);

  const login = async () => {
    setIsLoading(true);
    // Simulate login delay
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    const demoUser = {
      id: 'demo-user-001',
      name: 'Demo Physician',
      email: 'demo@trustplatform.ca',
      firstName: 'Demo',
      lastName: 'Physician',
      jobTitle: 'Cardiac Anesthesiologist',
      department: 'Anesthesiology',
    };
    
    setUser(demoUser);
    setIsAuthenticated(true);
    sessionStorage.setItem('trust_demo_user', JSON.stringify(demoUser));
    setIsLoading(false);
  };

  const logout = async () => {
    setUser(null);
    setIsAuthenticated(false);
    sessionStorage.removeItem('trust_demo_user');
  };

  const getAccessToken = async () => {
    return 'demo-token-not-real';
  };

  const value = {
    user,
    isAuthenticated,
    isLoading,
    error: null,
    login,
    logout,
    getAccessToken,
    isDemo: true, // Flag to show demo mode indicator
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
}

// ===========================================
// STEP 6: useAuth Hook
// ===========================================

/**
 * useAuth - Hook to access auth state from any component
 * 
 * Usage:
 *   function MyComponent() {
 *     const { user, isAuthenticated, login, logout } = useAuth();
 *     
 *     if (!isAuthenticated) {
 *       return <button onClick={login}>Sign In</button>;
 *     }
 *     
 *     return <p>Welcome, {user.name}!</p>;
 *   }
 */
export function useAuth() {
  const context = useContext(AuthContext);
  
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  
  return context;
}

// ===========================================
// STEP 7: ProtectedRoute Component
// ===========================================

/**
 * ProtectedRoute - Only shows content if user is logged in
 * 
 * Usage:
 *   <ProtectedRoute>
 *     <Dashboard />  {/* Only visible when logged in *\/}
 *   </ProtectedRoute>
 */
export function ProtectedRoute({ children, fallback }) {
  const { isAuthenticated, isLoading } = useAuth();

  // Show loading spinner while checking auth status
  if (isLoading) {
    return fallback || <LoadingSpinner />;
  }

  // Show login page if not authenticated
  if (!isAuthenticated) {
    return <LoginPage />;
  }

  // User is authenticated, show the protected content
  return children;
}

// ===========================================
// STEP 8: Loading Spinner Component
// ===========================================

function LoadingSpinner() {
  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      height: '100vh',
      background: 'linear-gradient(135deg, #0f172a 0%, #1e293b 100%)',
      fontFamily: "'Outfit', sans-serif",
    }}>
      {/* Animated TRUST Ring */}
      <svg width="80" height="80" viewBox="0 0 80 80" style={{ animation: 'spin 2s linear infinite' }}>
        <circle cx="40" cy="40" r="30" stroke="#334155" strokeWidth="4" fill="none"/>
        <path 
          d="M40 10 A30 30 0 0 1 70 40" 
          stroke="#0891b2" 
          strokeWidth="4" 
          fill="none" 
          strokeLinecap="round"
        />
      </svg>
      <p style={{ color: '#94a3b8', marginTop: '16px', fontSize: '14px' }}>
        Verifying credentials...
      </p>
      <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}

// ===========================================
// STEP 9: Login Page Component (imported)
// ===========================================

// We'll import this from a separate file
import { LoginPage } from './LoginPage';

// ===========================================
// EXPORTS
// ===========================================

export { msalInstance };
export default AuthProvider;
