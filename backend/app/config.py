"""
TRUST Platform Configuration
============================
Centralized configuration management for all environments.
"""
from pydantic_settings import BaseSettings
from typing import Optional
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Environment
    environment: str = "development"
    debug: bool = True
    
    # API
    api_v1_prefix: str = "/api/v1"
    project_name: str = "TRUST Platform"
    version: str = "0.2.0"
    
    # Database
    database_url: Optional[str] = None
    
    # Cerner FHIR
    cerner_fhir_base_url: str = "https://fhir-open.cerner.com/r4/ec2458f2-1e24-41c8-b71b-0e701af7583d"
    cerner_client_id: Optional[str] = None
    cerner_client_secret: Optional[str] = None
    
    # AI Services
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    
    # Azure
    azure_storage_connection_string: Optional[str] = None
    azure_key_vault_url: Optional[str] = None
    
    # Security
    secret_key: str = "dev-secret-key-change-in-production"
    api_key: Optional[str] = None
    
    # CORS
    cors_origins: list = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "https://www.trustplatform.ca",
        "https://trustplatform.ca",
    ]
    
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Cached settings instance."""
    return Settings()


# Convenience export
settings = get_settings()
