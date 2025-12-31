# TRUST Verification Methods: How It Works
## Semantic Entropy + EHR Cross-Reference Explained

**Document Purpose:** Technical explanation of how TRUST verifies AI-generated clinical notes using semantic entropy and EHR cross-reference.

---

## Overview: The Two-Layer Verification System

TRUST uses two complementary methods to verify AI scribe outputs:

| Layer | Method | Question It Answers | Ground Truth Source |
|-------|--------|---------------------|---------------------|
| **Layer 1** | Semantic Entropy | "Is the AI confident?" | AI's own consistency |
| **Layer 2** | EHR Cross-Reference | "Is this factually correct?" | Cerner patient data |

These catch **different failure modes**—that's why we need both.

---

## Layer 1: Semantic Entropy

### The Core Idea

> **Ask the AI the same question multiple times. If it gives different answers, it's uncertain.**

Semantic entropy measures the AI's internal consistency. High entropy means the AI is uncertain; low entropy means the AI is confident.

**Important:** Low entropy does NOT mean correct—it means consistent. An AI can be consistently wrong (the "confident hallucinator" problem).

### Step-by-Step Implementation

#### Step 1: Extract Claims from the AI-Generated Note

The AI scribe produces a clinical note. We break it into individual verifiable claims:

```
AI Scribe Note:
─────────────────────────────────────────────────────────────
"72-year-old female with history of hypertension and diabetes 
presents with chest pain. Currently taking metformin 1000mg 
and lisinopril 20mg. Denies shortness of breath."
─────────────────────────────────────────────────────────────

Extracted Claims:
├── Claim 1: Patient is 72 years old, female
├── Claim 2: History of hypertension
├── Claim 3: History of diabetes  
├── Claim 4: Presenting complaint is chest pain
├── Claim 5: Takes metformin 1000mg
├── Claim 6: Takes lisinopril 20mg
└── Claim 7: Denies shortness of breath
```

#### Step 2: Probe Each Claim Multiple Times

For each claim, we ask the AI to regenerate or confirm it multiple times (typically 5-10 times) with temperature > 0 to allow variation:

```python
# Example: Probing Claim 5 - "Takes metformin 1000mg"

prompt = """
Based on this clinical encounter, what diabetes medication 
is the patient taking and at what dose?
"""

# Ask 5 times with temperature > 0
responses = [
    "Metformin 1000mg daily",
    "Metformin 1000mg twice daily", 
    "Metformin 500mg twice daily",
    "Metformin 1000mg",
    "Metformin 1000mg once daily"
]
```

#### Step 3: Cluster Responses by Semantic Meaning

Group responses that mean the same thing (semantic equivalence):

```
Cluster A: "1000mg daily/once daily" → 3 responses (60%)
Cluster B: "1000mg twice daily"      → 1 response  (20%)
Cluster C: "500mg twice daily"       → 1 response  (20%)
```

#### Step 4: Calculate Entropy

```
Entropy = -Σ p(cluster) × log(p(cluster))
        = -(0.6 × log(0.6) + 0.2 × log(0.2) + 0.2 × log(0.2))
        = 0.95
```

**Interpretation:**
- Entropy near 0 = AI gives same answer every time (confident)
- Entropy near 1+ = AI gives different answers (uncertain)

#### Step 5: Flag High-Entropy Claims for Review

```
Claim Analysis Results:
─────────────────────────────────────────────────────────────
├── Claim 1: Age/gender      → Entropy: 0.12 ✅ Low (confident)
├── Claim 2: Hypertension    → Entropy: 0.08 ✅ Low (confident)
├── Claim 3: Diabetes        → Entropy: 0.15 ✅ Low (confident)
├── Claim 4: Chest pain      → Entropy: 0.22 ✅ Low (confident)
├── Claim 5: Metformin dose  → Entropy: 0.95 ⚠️ HIGH (uncertain!)
├── Claim 6: Lisinopril dose → Entropy: 0.31 ✅ Low (confident)
└── Claim 7: Denies SOB      → Entropy: 0.18 ✅ Low (confident)
─────────────────────────────────────────────────────────────

RESULT: Flag Claim 5 for detailed physician review
        AI is uncertain about metformin dosing
```

### What Semantic Entropy Catches

| Scenario | Entropy | Meaning |
|----------|---------|---------|
| AI gives same answer 5/5 times | Low (~0.1) | AI is confident |
| AI gives 3 similar, 2 different | Medium (~0.5) | AI is somewhat uncertain |
| AI gives 5 different answers | High (~1.5+) | AI is very uncertain |

### What Semantic Entropy Does NOT Catch

- Factual errors where AI is confident
- Contradictions with patient record
- Hallucinations stated with certainty

**That's why we need Layer 2.**

---

## Layer 2: EHR Cross-Reference

### The Core Idea

> **Compare the AI's claims against the patient's actual medical record.**

The EHR (Electronic Health Record) serves as external ground truth for factual verification.

### Step-by-Step Implementation

#### Step 1: Query Cerner FHIR for Patient Data

```python
# Fetch patient context from Cerner
patient_id = "12742400"

conditions = get_conditions(patient_id)
# Returns: ["Hypertension", "Type 2 Diabetes", "Hyperlipidemia"]

medications = get_medications(patient_id)
# Returns: ["Metformin 1000mg", "Lisinopril 20mg", "Atorvastatin 40mg"]

allergies = get_allergies(patient_id)
# Returns: ["Penicillin", "Sulfa drugs"]
```

#### Step 2: Compare Each Claim Against EHR

```python
def verify_claim(claim: str, patient_context: dict) -> dict:
    """
    Verify a claim from AI scribe against EHR data.
    """
    
    # Example: Claim = "Patient has diabetes"
    
    if "diabetes" in claim.lower():
        if any("diabetes" in c.lower() for c in patient_context["conditions"]):
            return {"status": "verified", "source": "EHR Conditions"}
        else:
            return {"status": "contradiction", "alert": "No diabetes in EHR"}
    
    # Example: Claim = "Takes metformin 1000mg"
    
    if "metformin" in claim.lower():
        ehr_meds = patient_context["medications"]
        if any("metformin" in m.lower() for m in ehr_meds):
            # Check dose matches
            if "1000mg" in claim and any("1000mg" in m for m in ehr_meds):
                return {"status": "verified", "source": "EHR Medications"}
            else:
                return {"status": "dose_mismatch", "alert": "Check metformin dose"}
        else:
            return {"status": "contradiction", "alert": "Metformin not in EHR"}
    
    return {"status": "unverified", "note": "No EHR data to compare"}
```

#### Step 3: Generate Verification Report

```
EHR Cross-Reference Results:
─────────────────────────────────────────────────────────────
├── Claim 1: "72yo female"    → ✅ VERIFIED (EHR: DOB, Gender)
├── Claim 2: "Hypertension"   → ✅ VERIFIED (EHR: Conditions)
├── Claim 3: "Diabetes"       → ✅ VERIFIED (EHR: Conditions)
├── Claim 4: "Chest pain"     → ⚪ UNVERIFIED (Chief complaint not in EHR)
├── Claim 5: "Metformin 1000mg" → ⚠️ DOSE MISMATCH (EHR: 500mg BID)
├── Claim 6: "Lisinopril 20mg"  → ✅ VERIFIED (EHR: Medications)
└── Claim 7: "Denies SOB"     → ⚪ UNVERIFIED (Symptoms not in EHR)
─────────────────────────────────────────────────────────────

ALERTS:
- Claim 5: Metformin dose in note (1000mg) differs from EHR (500mg BID)
  → Flag for physician verification
```

### What EHR Cross-Reference Catches

| Scenario | Result |
|----------|--------|
| Note says "diabetes" but EHR has no diabetes | Contradiction detected |
| Note says "Metformin 1000mg" but EHR says 500mg | Dose mismatch detected |
| Note mentions allergy patient doesn't have | Contradiction detected |
| Note says "no allergies" but EHR lists allergies | Contradiction detected |

### What EHR Cross-Reference Does NOT Catch

- New information from current encounter (not yet in EHR)
- AI uncertainty (confident but wrong)
- Subtle meaning distortions
- Information the EHR doesn't contain

**That's why we need BOTH layers.**

---

## The Combined System

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    AI SCRIBE NOTE                           │
│         "Patient has diabetes, takes metformin"             │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 1: SEMANTIC ENTROPY                                  │
│  ─────────────────────────                                  │
│  Question: "Is the AI certain about this claim?"            │
│                                                             │
│  Method:                                                    │
│  → Extract claims from note                                 │
│  → Ask AI same question 5x per claim                        │
│  → Cluster responses by semantic meaning                    │
│  → Calculate entropy for each claim                         │
│                                                             │
│  Output: Entropy score per claim                            │
│  Example: Claim 5 (metformin dose) = 0.95 (HIGH)            │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 2: EHR CROSS-REFERENCE                               │
│  ────────────────────────────                               │
│  Question: "Does this match the patient's record?"          │
│                                                             │
│  Method:                                                    │
│  → Query Cerner FHIR for patient data                       │
│  → Compare each claim against conditions, meds, allergies   │
│  → Flag contradictions and mismatches                       │
│                                                             │
│  Output: Verification status per claim                      │
│  Example: Claim 5 = DOSE MISMATCH (EHR says 500mg BID)      │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  COMBINED ASSESSMENT                                        │
│  ───────────────────                                        │
│                                                             │
│  Claim 5: Metformin 1000mg                                  │
│  ├── Semantic Entropy: 0.95 (HIGH - AI uncertain)           │
│  └── EHR Cross-Reference: DOSE MISMATCH                     │
│                                                             │
│  Risk Level: HIGH                                           │
│  Review Level: DETAILED                                     │
│  Physician Alert: "Verify metformin dose with patient"      │
└─────────────────────────────────────────────────────────────┘
```

### Decision Matrix

| Entropy | EHR Match | Interpretation | Review Level |
|---------|-----------|----------------|--------------|
| Low | Verified | ✅ Confident and correct | Brief |
| Low | Contradiction | 🚨 Confident but WRONG | Detailed |
| Low | Unverified | ⚠️ Confident, can't verify | Standard |
| High | Verified | ⚠️ Uncertain but correct | Standard |
| High | Contradiction | 🚨 Uncertain AND wrong | Detailed |
| High | Unverified | ⚠️ Uncertain, can't verify | Detailed |

### The Confident Hallucinator Problem

This is why the two-layer system is essential:

```
Scenario: AI confidently states "Patient takes Warfarin 5mg"
          but patient actually takes Metformin (no Warfarin)

Semantic Entropy Check:
→ AI gives same answer 5/5 times
→ Entropy = 0.08 (LOW)
→ AI is CONFIDENT ✓

EHR Cross-Reference Check:
→ Query Cerner medications
→ Warfarin NOT in patient's medication list
→ Status = CONTRADICTION ✗

Combined Result:
→ Low entropy + EHR contradiction
→ CONFIDENT HALLUCINATION DETECTED
→ Flag for DETAILED review
```

Without EHR cross-reference, this hallucination would pass undetected because the AI is confident.

---

## Frequently Asked Questions

### Q: Do we ask the AI a question from the generated text?

**Yes.** We extract claims from the note, then probe each claim by asking the AI to regenerate or confirm it multiple times. The questions are targeted at specific claims, not random.

### Q: Do we ask random questions?

**No.** Each question is designed to probe a specific claim from the AI-generated note. The goal is to measure the AI's confidence in that particular claim.

### Q: Where is the ground truth?

| For What | Ground Truth Source |
|----------|---------------------|
| Semantic Entropy | None needed—measures self-consistency |
| EHR Cross-Reference | Cerner patient data (conditions, meds, allergies) |

### Q: What if information isn't in the EHR?

Some claims can't be verified against the EHR (e.g., symptoms from current encounter). These are marked as "unverified" and rely on semantic entropy alone. High-entropy unverified claims get escalated to detailed review.

### Q: How many times do we query the AI?

Typically 5-10 times per claim, with temperature > 0 to allow natural variation. More queries = more accurate entropy estimate, but higher API cost.

### Q: What's a "high" entropy threshold?

Based on Paper 1 validation:
- < 0.3: Low entropy (confident)
- 0.3 - 0.7: Medium entropy (somewhat uncertain)
- \> 0.7: High entropy (uncertain, flag for review)

Thresholds may be adjusted based on clinical context and risk tolerance.

---

## Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| Claim extraction | ✅ Built | Uses NLP to parse clinical text |
| Semantic entropy calculation | ✅ Built | Validated in Paper 1 (95.5% accuracy) |
| Semantic clustering | ✅ Built | Uses NLI for meaning comparison |
| Cerner FHIR integration | ✅ Built | Fetches conditions, meds, allergies |
| EHR cross-reference logic | ✅ Built | Compares claims to patient data |
| Combined scoring | ✅ Built | Assigns review levels |
| Physician dashboard | ✅ Built | Displays flagged claims |

---

## References

1. Farquhar, S., et al. (2024). "Detecting hallucinations in large language models using semantic entropy." *Nature*.

2. Manakul, P., et al. (2023). "SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models." *EMNLP*.

3. TRUST Paper 1 (in preparation). "Multi-Layer Hallucination Detection in Clinical AI Scribes Using Semantic Entropy."

---

*TRUST Medical AI Governance Platform*
*Transparent • Responsible • Unbiased • Safe • Traceable*
*Auditing AI. Protecting Patients. Empowering Physicians.*
