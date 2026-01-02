# TRUST Platform - Comprehensive Developer README

> **Transparent, Responsible, Unbiased, Safe, Traceable**  
> AI Governance Layer for Healthcare

**Version:** 3.0  
**Last Updated:** January 2, 2026  
**Primary Developer:** Jean Raubenheimer (Cardiac Anesthesiologist, St. Paul's Hospital)  
**Technical Partner:** CL (Cerner/Oracle EHR Implementation Expert)

---

## Table of Contents

1. [What is TRUST?](#what-is-trust)
2. [Repository Structure](#repository-structure)
3. [The Three AI Types](#the-three-ai-types)
4. [Architecture Overview](#architecture-overview)
5. [Azure Resources](#azure-resources)
6. [Deployment Pipeline](#deployment-pipeline)
7. [Troubleshooting Deployments](#troubleshooting-deployments)
8. [Local Development](#local-development)
9. [Current Status](#current-status)
10. [Research Papers](#research-papers)
11. [Roadmap](#roadmap)

---

## What is TRUST?

TRUST is an **independent governance layer** that sits between AI systems and clinical workflows. It does NOT replace AI scribes, radiology AI, or predictive models — instead, it **monitors, validates, and audits** them.

### The Core Problem

AI scribes can **hallucinate** medications, dosages, and diagnoses. Standard validation catches "uncertain wrong" but misses **"confident hallucinators"** — AI that is confidently WRONG. These are the most dangerous errors because physicians trust high-confidence outputs.

### TRUST's Solution

| Layer | What It Catches | Method |
|-------|-----------------|--------|
| **Semantic Entropy** | AI "doesn't know" | Multiple generations → cluster → entropy |
| **EHR Verification** | Claims contradict record | FHIR cross-reference |
| **Hallucination Detection** | Confident but wrong | SE + Ground Truth |

### Key Insight: "Physician in the Loop"

Unlike generic "human in the loop", TRUST implements **tiered physician review**:

| Review Level | When Triggered | Physician Effort |
|--------------|----------------|------------------|
| **Brief** | High confidence, verified | 15 seconds |
| **Standard** | Moderate uncertainty | 2-3 minutes |
| **Detailed** | High uncertainty / flagged | 5+ minutes |

**Result:** 87% reduction in review burden while maintaining 100% physician oversight.

---

## Repository Structure

```
TRUST-platform/
│
├── .github/
│   └── workflows/              # GitHub Actions CI/CD
│       └── azure-deploy.yml    # Auto-deploy to Azure on push
│
├── backend/                    # FastAPI Python Backend
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py             # FastAPI entry point
│   │   ├── models.py           # Pydantic models
│   │   ├── database.py         # PostgreSQL connection
│   │   ├── routers/            # API route handlers
│   │   └── services/           # Business logic
│   ├── tests/                  # pytest test suite
│   ├── requirements.txt        # Python dependencies
│   ├── startup.sh              # Azure App Service startup
│   ├── .env.example            # Environment template
│   └── test_*.py               # Integration tests
│
├── frontend/                   # React Frontend
│   ├── src/
│   │   ├── App.jsx             # Main React component
│   │   ├── components/         # UI components
│   │   └── services/           # API client
│   ├── public/
│   ├── build/                  # Production build (generated)
│   ├── package.json
│   └── .env                    # Frontend environment
│
├── ml-service/                 # ML Microservice (Semantic Entropy)
│   ├── app/
│   │   ├── main.py             # FastAPI ML endpoints
│   │   ├── semantic_entropy.py
│   │   └── hallucination_detector.py
│   ├── ml_client.py            # Client for backend integration
│   ├── requirements.txt
│   └── startup.sh
│
├── cerner_sandbox/             # Cerner Integration Testing
│   ├── fhir_profiles/          # FHIR resource definitions
│   ├── smart_apps/             # SMART on FHIR app configs
│   └── cds_hooks/              # CDS Hooks definitions
│
├── docs/                       # Documentation
│   ├── architecture/           # System diagrams
│   ├── brand/                  # Brand guidelines, logos
│   ├── papers/                 # Research paper drafts
│   ├── TRUST-Verification-Methods-Explained.md
│   ├── TRUST-Uncertainty-Quantification-Explained.md
│   ├── TRUST-Competitive-Differentiation-vs-Vendor-Self-Audit.md
│   └── Research-Note-Semantic-Entropy-Radiology-AI.md
│
├── infrastructure/             # Infrastructure as Code
│   └── terraform/              # Azure resource definitions
│
├── mock_data/                  # Test Data
│   ├── scribe_notes/           # Sample AI-generated notes
│   ├── transcripts/            # Sample audio transcripts
│   └── README.md
│
├── validation/                 # Research Validation Code
│   ├── paper1_hallucination/   # MedHallu benchmark
│   ├── paper2_uncertainty/     # PubMedQA validation
│   └── shared/                 # Common utilities
│
├── scripts/                    # Utility scripts
│
├── .env                        # Root environment (gitignored)
├── .env.example                # Environment template
├── .gitignore
└── README.md                   # This file
```

---

## The Three AI Types

TRUST uses **different validation methods** for different AI types. Each runs as a separate microservice.

### 1. Generative AI (AI Scribes) — Papers 1 & 2

**Validates:** Dragon Medical, Nuance DAX, Cerner AI Scribe

**Methods:**
- Semantic Entropy (SE) — Multiple generations, cluster by meaning
- EHR Cross-Reference — FHIR verification against patient data
- Tiered routing based on uncertainty

**Key Formula:**
```
IF (SE < 0.3) AND (contradicts EHR):
    → CONFIDENT HALLUCINATOR (highest risk)
ELIF (SE > 0.6):
    → HIGH UNCERTAINTY (flag for review)
ELSE:
    → Standard routing
```

### 2. Radiology AI — Paper 4

**Validates:** Chest X-ray AI, CT analysis, Vision-Language Models

**Methods:**
- Semantic Entropy (adapted for images)
- Confident Hallucinator Detection (TRUST innovation)
- IoU/Dice localization metrics
- Attention validation
- ECE/MCE calibration

### 3. Predictive AI — Paper 3

**Validates:** Sepsis prediction, mortality risk, readmission models

**Methods:**
- Calibration monitoring
- SHAP explainability
- Bias detection
- Drift alerts

**Note:** No semantic entropy (deterministic outputs)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        INTERNET                                  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              AZURE FRONT DOOR / CUSTOM DOMAIN                    │
│         trustplatform.ca  │  api.trustplatform.ca               │
└───────────────────────────┼─────────────────────────────────────┘
                            │
         ┌──────────────────┼──────────────────┐
         │                  │                  │
         ▼                  ▼                  ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  Static Web App │ │  App Service    │ │  Key Vault      │
│  (React)        │ │  (FastAPI)      │ │  (Secrets)      │
│  Frontend       │ │  Backend API    │ │  trust-prod-kv  │
└─────────────────┘ └────────┬────────┘ └─────────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  PostgreSQL     │ │  ML Service     │ │  External APIs  │
│  (Audit Logs)   │ │  (SE Pipeline)  │ │  (OpenAI, etc)  │
└─────────────────┘ └─────────────────┘ └─────────────────┘
                             │
                             ▼
                   ┌─────────────────┐
                   │  Cerner EHR     │
                   │  (FHIR R4)      │
                   └─────────────────┘
```

### Tech Stack

| Layer | Technology |
|-------|------------|
| **Frontend** | React 18, Tailwind CSS, Recharts |
| **Backend** | FastAPI (Python 3.11), Pydantic |
| **Database** | PostgreSQL 16 (Azure Flexible Server) |
| **ML Service** | FastAPI, sentence-transformers |
| **Hosting** | Azure Static Web App + App Service |
| **Secrets** | Azure Key Vault |
| **CI/CD** | GitHub Actions |
| **Domain** | trustplatform.ca (Cloudflare DNS) |

---

## Azure Resources

### Resource Group: `trust-dev-rg`

**Subscription ID:** `830d7e6e-7b72-408d-82bd-da02ad220d91`  
**Region:** Canada Central

| Resource Name | Type | Purpose |
|---------------|------|---------|
| `trust-prod-kv` | Key Vault | Secrets (API keys, DB credentials) |
| `trust-dev-postgres` | PostgreSQL Flexible Server | Audit database |
| `trust-dashboard-swa` | Static Web App | React frontend |
| `trust-backend-app` | App Service | FastAPI backend |

### Key Vault Secrets

| Secret Name | Purpose |
|-------------|---------|
| `OPENAI-API-KEY` | OpenAI API access |
| `ANTHROPIC-API-KEY` | Claude API access |
| `DATABASE-URL` | PostgreSQL connection string |
| `CERNER-CLIENT-ID` | Cerner sandbox OAuth |
| `CERNER-CLIENT-SECRET` | Cerner sandbox OAuth |

### URLs

| Environment | Frontend | Backend API |
|-------------|----------|-------------|
| **Production** | https://trustplatform.ca | https://api.trustplatform.ca |
| **Azure Direct** | https://trust-dashboard-swa.azurestaticapps.net | https://trust-backend-app.azurewebsites.net |

---

## Deployment Pipeline

### How It Works

1. **Push to `main` branch** → Triggers GitHub Actions
2. **GitHub Actions** runs `.github/workflows/azure-deploy.yml`
3. **Backend deploys** to Azure App Service via ZIP deploy
4. **Frontend deploys** to Azure Static Web App

### GitHub Actions Workflow Location

```
.github/workflows/azure-deploy.yml
```

### Required GitHub Secrets

| Secret Name | Where to Get It |
|-------------|-----------------|
| `AZURE_WEBAPP_PUBLISH_PROFILE` | Azure Portal → App Service → Deployment Center → Manage publish profile |
| `AZURE_STATIC_WEB_APPS_API_TOKEN` | Azure Portal → Static Web App → Manage deployment token |

---

## Troubleshooting Deployments

### Step 1: Check GitHub Actions

1. Go to: `https://github.com/YOUR_USERNAME/TRUST-platform/actions`
2. Click on the failed workflow run
3. Expand the failed step to see error logs

### Common Failures

| Error | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError` | Missing dependency | Add to `requirements.txt` |
| `401 Unauthorized` | Bad publish profile | Regenerate in Azure Portal |
| `Application Error` | Startup crash | Check Azure logs (see below) |
| `Build failed` | Frontend syntax error | Run `npm run build` locally first |

### Step 2: Check Azure Logs

**Via Azure Portal:**
1. Go to App Service → "Log stream" (left sidebar)
2. Watch real-time logs during startup

**Via Azure CLI:**
```bash
# Stream live logs
az webapp log tail --name trust-backend-app --resource-group trust-dev-rg

# Download recent logs
az webapp log download --name trust-backend-app --resource-group trust-dev-rg
```

**Via Kudu Console:**
1. Go to: `https://trust-backend-app.scm.azurewebsites.net`
2. Navigate to: Debug Console → CMD
3. Check: `LogFiles/` directory

### Step 3: Test Backend Startup Locally

```bash
cd backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

If this fails locally, the Azure deployment will also fail.

### Step 4: Check startup.sh

The `backend/startup.sh` file tells Azure how to start your app:

```bash
#!/bin/bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Make sure:
- File has execute permissions (`chmod +x startup.sh`)
- Path to `app.main:app` matches your actual file structure
- No Windows line endings (use `dos2unix startup.sh` if needed)

---

## Local Development

### Prerequisites

- Python 3.11+
- Node.js 18+
- PostgreSQL (local or Azure)
- Azure CLI (for secrets)

### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file (copy from .env.example)
cp .env.example .env
# Edit .env with your values

# Run locally
uvicorn app.main:app --reload --port 8000
```

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Create .env file
echo "REACT_APP_API_URL=http://localhost:8000" > .env

# Run locally
npm start
```

### Environment Variables

**Backend `.env`:**
```env
DATABASE_URL=postgresql://user:pass@localhost:5432/trust_dev
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
AZURE_KEY_VAULT_URL=https://trust-prod-kv.vault.azure.net/
ENVIRONMENT=development
```

**Frontend `.env`:**
```env
REACT_APP_API_URL=http://localhost:8000
```

---

## Current Status

### ✅ Completed

- [x] Azure infrastructure deployed (App Service, PostgreSQL, Key Vault)
- [x] Custom domain configured (trustplatform.ca)
- [x] GitHub Actions CI/CD pipeline created
- [x] FastAPI backend scaffold
- [x] React frontend scaffold  
- [x] Cerner FHIR sandbox connection tested
- [x] Semantic entropy pipeline (research validation)
- [x] Paper 1 validation (95.5% accuracy on MedHallu)
- [x] Paper 2 draft (uncertainty quantification)
- [x] Documentation in `/docs`

### 🔄 In Progress

- [ ] Debug GitHub → Azure deployment failures
- [ ] ML service integration
- [ ] Frontend dashboard components
- [ ] EHR verification endpoint

### 📋 Next Steps (Priority Order)

1. **Fix deployment** — Check GitHub Actions logs, verify startup.sh
2. **Test /health endpoint** — Ensure backend responds
3. **Add ML endpoints** — Integrate semantic entropy
4. **Build dashboard** — Physician review interface

---

## Research Papers

| Paper | Title | Status | Target Venue |
|-------|-------|--------|--------------|
| **Paper 1** | Hallucination Detection via Semantic Entropy | Draft complete | NEJM AI |
| **Paper 2** | Uncertainty Quantification for Review Routing | Draft complete | npj Digital Medicine |
| **Paper 2.1** | Evidence-Calibrated Uncertainty | Planning | - |
| **Paper 3** | Predictive AI Governance (Calibration/SHAP) | Planning | - |
| **Paper 4** | Radiology AI + Confident Hallucinator | Outlined | - |

### Key Results

- **Paper 1:** 95.5% hallucination detection using semantic entropy
- **Paper 2:** 87% reduction in review burden with tiered routing
- **Innovation:** "Confident Hallucinator" detection (low SE + wrong)

---

## Roadmap

### Phase 1: Monitoring (Current — Q1 2026)
- Semantic entropy pipeline
- EHR verification  
- Audit logging
- Tiered review routing
- Providence pilot preparation

### Phase 2: Evaluation (Q2-Q3 2026)
- Randomized deployment testing
- Federated learning (cross-site learning without data sharing)
- AI Behavioral Phenotyping research

### Phase 3: Intervention (Q4 2026+)
- Auto-retraining recommendations
- Advanced governance automation
- Multi-site deployment across Providence network

---

## Quick Reference Commands

### Azure CLI

```bash
# Login to Azure
az login

# Stream backend logs
az webapp log tail --name trust-backend-app --resource-group trust-dev-rg

# Get a secret from Key Vault
az keyvault secret show --vault-name trust-prod-kv --name OPENAI-API-KEY --query value -o tsv

# Restart App Service
az webapp restart --name trust-backend-app --resource-group trust-dev-rg

# Check deployment status
az webapp show --name trust-backend-app --resource-group trust-dev-rg --query state -o tsv
```

### Git Commands

```bash
# Standard workflow
git add .
git commit -m "Description of changes"
git push origin main  # Triggers deployment

# Check remote
git remote -v

# Pull latest
git pull origin main
```

### Local Testing

```bash
# Test backend
cd backend && uvicorn app.main:app --reload

# Test frontend  
cd frontend && npm start

# Run backend tests
cd backend && pytest
```

---

## Key URLs

| Resource | URL |
|----------|-----|
| **Production Site** | https://trustplatform.ca |
| **API Endpoint** | https://api.trustplatform.ca |
| **GitHub Repo** | https://github.com/[username]/TRUST-platform |
| **GitHub Actions** | https://github.com/[username]/TRUST-platform/actions |
| **Azure Portal** | https://portal.azure.com |
| **Kudu Console** | https://trust-backend-app.scm.azurewebsites.net |

---

## Support

**Jean Raubenheimer**  
Cardiac Anesthesiologist, St. Paul's Hospital  
Providence Health Care, Vancouver

**CL** — Technical Partner (Cerner/Oracle EHR)

---

*TRUST Medical AI Governance Platform*  
*Transparent • Responsible • Unbiased • Safe • Traceable*
