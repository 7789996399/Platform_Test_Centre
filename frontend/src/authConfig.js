/**
 * ============================================================
 * TRUST Platform - Azure AD Configuration
 * ============================================================
 * 
 * WHAT THIS FILE DOES:
 * Think of this as your "address book" for Azure. It tells 
 * the app WHERE to find Microsoft's login service and 
 * WHO you are (your app's ID).
 * 
 * BEFORE USING:
 * Replace the placeholder values below with your real 
 * Azure App Registration values (from the setup instructions).
 * 
 * ============================================================
 */

// ===========================================
// STEP 1: YOUR AZURE CREDENTIALS
// ===========================================
// 🔴 REPLACE THESE with your values from Azure Portal!

const AZURE_CLIENT_ID = "5a4755ae-d766-49fc-9fd5-ff68d2cba8de";  // From Azure App Registration
const AZURE_TENANT_ID = "aa89db9d-ee60-4244-b01b-764f6dc5cd2a";               // Use "common" for multi-tenant, or your specific tenant ID

// ===========================================
// STEP 2: WHERE USERS GO AFTER LOGIN
// ===========================================
// This should match what you entered in Azure Portal

const REDIRECT_URI = window.location.hostname === "localhost" 
  ? "http://localhost:3000"           // For local development
  : "https://www.trustplatform.ca";   // For production

// ===========================================
// MSAL CONFIGURATION (Don't change this part)
// ===========================================

/**
 * MSAL = Microsoft Authentication Library
 * This is the "engine" that handles login/logout
 */
export const msalConfig = {
  auth: {
    // Your app's unique ID (like a social security number for your app)
    clientId: AZURE_CLIENT_ID,
    
    // The "address" of Microsoft's login service
    // "common" = accepts any Microsoft account
    // Your tenant ID = only accepts accounts from your organization
    authority: `https://login.microsoftonline.com/${AZURE_TENANT_ID}`,
    
    // Where to send users after they log in
    redirectUri: REDIRECT_URI,
    
    // Where to send users after they log out
    postLogoutRedirectUri: REDIRECT_URI,
  },
  
  cache: {
    // Where to store the login info in the browser
    // "sessionStorage" = cleared when browser closes (more secure)
    // "localStorage" = stays until manually cleared (more convenient)
    cacheLocation: "sessionStorage",
    
    // Prevents certain security attacks in older browsers
    storeAuthStateInCookie: false,
  },
};

// ===========================================
// WHAT PERMISSIONS WE'RE ASKING FOR
// ===========================================

/**
 * "Scopes" = the things we're allowed to see/do
 * 
 * Think of it like hospital badge permissions:
 * - "User.Read" = can see your own profile
 * - "openid" = can verify your identity
 * - "profile" = can see your name
 * - "email" = can see your email
 */
export const loginRequest = {
  scopes: ["User.Read", "openid", "profile", "email"]
};

// ===========================================
// GRAPH API SETTINGS (for getting user info)
// ===========================================

/**
 * Microsoft Graph = Microsoft's API for user data
 * We use this to get the logged-in user's name, email, etc.
 */
export const graphConfig = {
  graphMeEndpoint: "https://graph.microsoft.com/v1.0/me"
};

// ===========================================
// HELPER: Check if Azure is configured
// ===========================================

export const isAzureConfigured = () => {
  return AZURE_CLIENT_ID !== "YOUR-CLIENT-ID-HERE";
};

// ===========================================
// EXPORT EVERYTHING
// ===========================================

export default {
  msalConfig,
  loginRequest,
  graphConfig,
  isAzureConfigured,
};
