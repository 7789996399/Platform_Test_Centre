# Claude Code Instructions for Platform_Test_Centre

## CRITICAL: This is a TEST repository

DO NOT:
- Deploy anything to Azure
- Use any Azure CLI commands (az webapp, az staticwebapp, etc.)
- Modify any .env files with production credentials
- Push to any Azure-connected GitHub Actions
- Access any external APIs (OpenAI, Anthropic, Cerner) using real credentials

DO:
- Create new Python modules and test scripts
- Run tests locally with mock data
- Build new architecture components
- Use placeholder/mock credentials only

## Environment
All testing should use:
- SQLite (not PostgreSQL)
- Mock API responses (not real LLM calls)
- Local file storage (not Azure Blob)

This repo is for architecture experimentation ONLY.
