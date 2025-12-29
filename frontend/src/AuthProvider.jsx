/**
 * TRUST Platform - Authentication Provider
 * With Audit Logging
 */

import React, { createContext, useContext, useState, useEffect } from 'react';
import { PublicClientApplication, InteractionStatus } from '@azure/msal-browser';
import { MsalProvider, useMsal, useIsAuthenticated } from '@azure/msal-react';
import { msalConfig, loginRequest, graphConfig, isAzureConfigured } from './authConfig';
import { LoginPage } from './LoginPage';
import { logAuditEvent } from './services/api';

let msalInstance = null;

if (isAzureConfigured()) {
  msalInstance = new PublicClientApplication(msalConfig);
}

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  if (!isAzureConfigured()) {
    return (
      <DemoAuthProvider>
        {children}
      </DemoAuthProvider>
    );
  }

  return (
    <MsalProvider instance={msalInstance}>
      <MsalAuthProvider>
        {children}
      </MsalAuthProvider>
    </MsalProvider>
  );
}

function MsalAuthProvider({ children }) {
  const { instance, accounts, inProgress } = useMsal();
  const isAuthenticated = useIsAuthenticated();
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchUserData = async () => {
      if (isAuthenticated && accounts.length > 0) {
        try {
          const response = await instance.acquireTokenSilent({
            ...loginRequest,
            account: accounts[0],
          });

          const graphResponse = await fetch(graphConfig.graphMeEndpoint, {
            headers: {
              Authorization: `Bearer ${response.accessToken}`,
            },
          });

          if (graphResponse.ok) {
            const userData = await graphResponse.json();
            const userObj = {
              id: userData.id,
              name: userData.displayName,
              email: userData.mail || userData.userPrincipalName,
              firstName: userData.givenName,
              lastName: userData.surname,
              jobTitle: userData.jobTitle,
              department: userData.department,
            };
            setUser(userObj);
            
            // Log successful login
            logAuditEvent({
              user_id: userObj.id,
              user_email: userObj.email,
              user_name: userObj.name,
              action: 'LOGIN',
              resource_type: 'auth',
              details: { method: 'azure_ad' }
            });
          }
        } catch (err) {
          console.error('Error fetching user data:', err);
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

  const logout = async () => {
    // Log logout before actually logging out
    if (user) {
      await logAuditEvent({
        user_id: user.id,
        user_email: user.email,
        user_name: user.name,
        action: 'LOGOUT',
        resource_type: 'auth',
        details: { method: 'user_initiated' }
      });
    }
    
    try {
      await instance.logoutPopup({
        postLogoutRedirectUri: msalConfig.auth.postLogoutRedirectUri,
      });
      setUser(null);
    } catch (err) {
      console.error('Logout error:', err);
    }
  };

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
      try {
        const response = await instance.acquireTokenPopup(loginRequest);
        return response.accessToken;
      } catch (popupErr) {
        console.error('Error acquiring token:', popupErr);
        return null;
      }
    }
  };

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

function DemoAuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    const savedUser = sessionStorage.getItem('trust_demo_user');
    if (savedUser) {
      setUser(JSON.parse(savedUser));
      setIsAuthenticated(true);
    }
  }, []);

  const login = async () => {
    setIsLoading(true);
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
    
    // Log demo login
    logAuditEvent({
      user_id: demoUser.id,
      user_email: demoUser.email,
      user_name: demoUser.name,
      action: 'LOGIN',
      resource_type: 'auth',
      details: { method: 'demo_mode' }
    });
  };

  const logout = async () => {
    // Log logout
    if (user) {
      await logAuditEvent({
        user_id: user.id,
        user_email: user.email,
        user_name: user.name,
        action: 'LOGOUT',
        resource_type: 'auth',
        details: { method: 'demo_mode' }
      });
    }
    
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
    isDemo: true,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  
  return context;
}

export function ProtectedRoute({ children, fallback }) {
  const { isAuthenticated, isLoading } = useAuth();

  if (isLoading) {
    return fallback || <LoadingSpinner />;
  }

  if (!isAuthenticated) {
    return <LoginPage />;
  }

  return children;
}

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

export { msalInstance };
export default AuthProvider;
