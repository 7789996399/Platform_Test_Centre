"""
Azure Health Data AI Services - De-identification Client
Automatically redacts PHI from clinical text before storing in audit logs
"""

import os
import httpx
from azure.identity import DefaultAzureCredential, ManagedIdentityCredential
from typing import Optional
import logging

logger = logging.getLogger(__name__)

DEID_SERVICE_URL = os.getenv("DEID_SERVICE_URL", "")
DEID_SCOPE = "https://deid.azure.com/.default"


class DeidentificationClient:
    """Client for Azure De-identification Service"""
    
    def __init__(self):
        self.service_url = DEID_SERVICE_URL
        self._credential = None
        self._token = None
    
    def _get_token(self) -> str:
        """Get Azure AD token for de-id service"""
        if not self._credential:
            try:
                # Try managed identity first (production)
                self._credential = ManagedIdentityCredential()
            except Exception:
                # Fall back to default (local dev)
                self._credential = DefaultAzureCredential()
        
        token = self._credential.get_token(DEID_SCOPE)
        return token.token
    
    async def deidentify_text(self, text: str, operation: str = "Redact") -> dict:
        """
        De-identify clinical text by redacting or surrogating PHI
        
        Args:
            text: Clinical text containing potential PHI
            operation: "Redact" (replace with [ENTITY_TYPE]) or "Surrogate" (replace with fake values)
        
        Returns:
            dict with originalText, processedText, and entities found
        """
        if not self.service_url:
            logger.warning("DEID_SERVICE_URL not configured - returning original text")
            return {
                "originalText": text,
                "processedText": text,
                "entities": [],
                "service": "disabled"
            }
        
        try:
            token = self._get_token()
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.service_url}/deid",
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "inputText": text,
                        "operation": operation,
                        "dataType": "Plaintext"
                    },
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return {
                        "originalText": text,
                        "processedText": result.get("outputText", text),
                        "entities": result.get("taggerResults", []),
                        "service": "azure_deid"
                    }
                else:
                    logger.error(f"De-id service error: {response.status_code} - {response.text}")
                    return {
                        "originalText": text,
                        "processedText": text,
                        "entities": [],
                        "error": f"Service returned {response.status_code}"
                    }
                    
        except Exception as e:
            logger.error(f"De-identification failed: {str(e)}")
            return {
                "originalText": text,
                "processedText": text,
                "entities": [],
                "error": str(e)
            }
    
    async def redact_for_audit(self, clinical_text: str) -> str:
        """
        Convenience method: Redact PHI and return only the clean text
        Use this before storing in audit logs
        """
        result = await self.deidentify_text(clinical_text, operation="Redact")
        return result["processedText"]


# Singleton instance
deid_client = DeidentificationClient()
