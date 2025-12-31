# TRUST Uncertainty Quantification: How It Works
## Knowing When AI Doesn't Know

**Document Purpose:** Technical explanation of how TRUST quantifies and communicates AI uncertainty to physicians, enabling appropriate trust calibration.

---

## The Core Problem

> **AI systems rarely say "I don't know."**

When a physician asks a colleague for an opinion, the colleague might say:
- "I'm confident this is pneumonia"
- "I think it might be pneumonia, but consider TB"
- "I'm not sure—get a pulmonology consult"

AI systems typically output a single answer with no indication of confidence. This is dangerous in healthcare.

**TRUST's role:** Force AI to express uncertainty, then communicate it to physicians in actionable terms.

---

## Two Types of Uncertainty

| Type | Name | Meaning | Example |
|------|------|---------|---------|
| **Type 1** | Aleatoric | Inherent randomness in the data | Noisy lab values, ambiguous symptoms |
| **Type 2** | Epistemic | AI's lack of knowledge | Rare disease AI wasn't trained on |

### Why This Matters

```
Aleatoric Uncertainty (Data Noise):
───────────────────────────────────
"The chest X-ray is blurry—I can't tell if there's a nodule"
→ Solution: Get a better image
→ More AI training won't help

Epistemic Uncertainty (Knowledge Gap):
──────────────────────────────────────
"I've never seen this presentation before"
→ Solution: Get expert consultation
→ More training data might help
```

**TRUST distinguishes these** so physicians know the appropriate response.

---

## Uncertainty Quantification vs. Hallucination Detection

These are related but address different failure modes:

| Method | Question | Detects |
|--------|----------|---------|
| **Semantic Entropy** (Paper 1) | "Is the AI consistent?" | Hallucinations, confusion |
| **Uncertainty Quantification** (Paper 2) | "How confident is the AI?" | Knowledge gaps, edge cases |

### The Key Insight

> **Low uncertainty ≠ Correct**
> **High uncertainty ≠ Incorrect**

```
Scenario Matrix:
─────────────────────────────────────────────────────────────
                        AI is Correct    AI is Wrong
                        ─────────────    ───────────
Low Uncertainty         ✅ Ideal         🚨 Dangerous!
(AI is confident)       (trust it)       (confident hallucination)

High Uncertainty        ⚠️ Verify        ✅ Appropriate
(AI admits doubt)       (AI got lucky)   (AI knows its limits)
─────────────────────────────────────────────────────────────
```

**The dangerous quadrant:** Low uncertainty + Wrong answer = Confident hallucination

**TRUST catches this** by combining uncertainty quantification with EHR cross-reference.

---

## How TRUST Quantifies Uncertainty

### Method 1: Response Variance (For Generative AI)

Same as semantic entropy—measure consistency across multiple generations:

```python
def calculate_response_variance(prompt: str, n_samples: int = 10) -> dict:
    """
    Generate multiple responses and measure variance.
    """
    responses = []
    for _ in range(n_samples):
        response = ai_model.generate(prompt, temperature=0.7)
        responses.append(response)
    
    # Cluster by semantic similarity
    clusters = cluster_by_meaning(responses)
    
    # Calculate entropy
    entropy = calculate_entropy(clusters)
    
    # Map to uncertainty level
    if entropy < 0.3:
        uncertainty = "low"
    elif entropy < 0.7:
        uncertainty = "medium"
    else:
        uncertainty = "high"
    
    return {
        "entropy": entropy,
        "uncertainty_level": uncertainty,
        "n_clusters": len(clusters),
        "dominant_answer_frequency": max(len(c) for c in clusters) / n_samples
    }
```

### Method 2: Calibration Analysis (For Predictive AI)

For models that output probabilities (sepsis risk, mortality risk, readmission risk):

```
What is Calibration?
────────────────────
When a model says "70% probability of sepsis":
- Good calibration: 70% of those patients actually have sepsis
- Poor calibration: Only 40% actually have sepsis (overconfident!)

┌─────────────────────────────────────────────────────────────┐
│                  CALIBRATION CURVE                          │
│                                                             │
│  100% ┤                                        ✓ Perfect    │
│       │                                      ⋰              │
│   80% ┤                                   ⋰                 │
│       │                               ⋰                     │
│   60% ┤                           ⋰ ← Overconfident         │
│       │                       ⋰      (predicts 80%,         │
│   40% ┤                   ⋰           actual 60%)           │
│       │               ⋰                                     │
│   20% ┤           ⋰                                         │
│       │       ⋰                                             │
│    0% ┼───┬───┬───┬───┬───┬───┬───┬───┬───┬───             │
│       0%  20% 40% 60% 80% 100%                              │
│              Predicted Probability                          │
└─────────────────────────────────────────────────────────────┘
```

**TRUST Implementation:**

```python
def assess_calibration(predictions: list, outcomes: list) -> dict:
    """
    Measure how well-calibrated a predictive model is.
    
    predictions: list of predicted probabilities (0-1)
    outcomes: list of actual outcomes (0 or 1)
    """
    # Bin predictions into deciles
    bins = np.linspace(0, 1, 11)
    bin_indices = np.digitize(predictions, bins)
    
    calibration_data = []
    for i in range(1, 11):
        mask = bin_indices == i
        if mask.sum() > 0:
            mean_predicted = np.mean(predictions[mask])
            mean_actual = np.mean(outcomes[mask])
            calibration_data.append({
                "bin": i,
                "predicted": mean_predicted,
                "actual": mean_actual,
                "gap": abs(mean_predicted - mean_actual),
                "n_samples": mask.sum()
            })
    
    # Calculate Expected Calibration Error (ECE)
    ece = sum(d["gap"] * d["n_samples"] for d in calibration_data) / len(predictions)
    
    # Interpret
    if ece < 0.05:
        calibration_quality = "excellent"
    elif ece < 0.10:
        calibration_quality = "good"
    elif ece < 0.15:
        calibration_quality = "fair"
    else:
        calibration_quality = "poor"
    
    return {
        "expected_calibration_error": ece,
        "calibration_quality": calibration_quality,
        "calibration_curve": calibration_data,
        "recommendation": get_calibration_recommendation(ece)
    }

def get_calibration_recommendation(ece: float) -> str:
    if ece < 0.05:
        return "Model probabilities are reliable for clinical decision-making"
    elif ece < 0.10:
        return "Model probabilities are generally reliable; verify high-stakes predictions"
    elif ece < 0.15:
        return "Model tends to be overconfident; apply clinical judgment"
    else:
        return "Model is poorly calibrated; do not rely on probability estimates"
```

### Method 3: Confidence Intervals (For All AI Types)

Instead of point estimates, require AI to provide ranges:

```
Without Uncertainty Quantification:
───────────────────────────────────
AI: "Patient's sepsis risk is 73%"
Physician: Treats as precise number

With Uncertainty Quantification:
────────────────────────────────
AI: "Patient's sepsis risk is 73% (95% CI: 58-85%)"
Physician: Understands the range of possible values
```

**TRUST Implementation:**

```python
def generate_with_confidence_interval(
    model, 
    input_data, 
    n_bootstrap: int = 100
) -> dict:
    """
    Generate prediction with confidence interval via bootstrapping.
    """
    predictions = []
    
    for _ in range(n_bootstrap):
        # Resample or use dropout for uncertainty
        pred = model.predict_with_uncertainty(input_data)
        predictions.append(pred)
    
    mean_pred = np.mean(predictions)
    ci_lower = np.percentile(predictions, 2.5)
    ci_upper = np.percentile(predictions, 97.5)
    std_dev = np.std(predictions)
    
    # Uncertainty level based on CI width
    ci_width = ci_upper - ci_lower
    if ci_width < 0.1:
        uncertainty = "low"
    elif ci_width < 0.25:
        uncertainty = "medium"
    else:
        uncertainty = "high"
    
    return {
        "prediction": mean_pred,
        "confidence_interval": [ci_lower, ci_upper],
        "std_dev": std_dev,
        "uncertainty_level": uncertainty,
        "interpretation": interpret_ci(mean_pred, ci_lower, ci_upper)
    }
```

### Method 4: Evidence-Based Uncertainty (Paper 2.1)

Calibrate AI confidence against strength of supporting evidence:

```
Evidence Calibration Framework:
───────────────────────────────────────────────────────────────

Evidence Level          AI Confidence Should Be
──────────────          ───────────────────────
Strong RCT evidence     High confidence appropriate
Observational only      Medium confidence max
Expert opinion only     Low confidence appropriate
No evidence             Express high uncertainty

TRUST Check:
"AI is 95% confident, but recommendation is based on 
single case report → MISMATCH → Flag for review"
```

**TRUST Implementation:**

```python
def assess_evidence_calibration(
    ai_confidence: float,
    evidence_sources: list
) -> dict:
    """
    Check if AI confidence matches evidence strength.
    """
    # Score evidence quality
    evidence_score = calculate_evidence_score(evidence_sources)
    
    # Expected confidence range based on evidence
    if evidence_score > 0.8:  # Strong evidence
        expected_confidence_range = (0.7, 1.0)
    elif evidence_score > 0.5:  # Moderate evidence
        expected_confidence_range = (0.4, 0.8)
    else:  # Weak evidence
        expected_confidence_range = (0.0, 0.5)
    
    # Check for mismatch
    ci_low, ci_high = expected_confidence_range
    
    if ai_confidence > ci_high:
        calibration_status = "overconfident"
        alert = f"AI confidence ({ai_confidence:.0%}) exceeds evidence strength"
    elif ai_confidence < ci_low:
        calibration_status = "underconfident"
        alert = f"AI confidence ({ai_confidence:.0%}) is lower than evidence supports"
    else:
        calibration_status = "appropriate"
        alert = None
    
    return {
        "ai_confidence": ai_confidence,
        "evidence_score": evidence_score,
        "expected_range": expected_confidence_range,
        "calibration_status": calibration_status,
        "alert": alert
    }
```

---

## Communicating Uncertainty to Physicians

### The Translation Problem

AI outputs numbers. Physicians need actionable guidance.

```
❌ Bad: "Entropy = 0.847, ECE = 0.12, σ = 0.23"

✅ Good: "AI is uncertain about this diagnosis. 
         Consider alternative diagnoses or specialist consult."
```

### TRUST's Uncertainty Communication Framework

#### Level 1: Visual Indicators

```
┌─────────────────────────────────────────────────────────────┐
│  AI SCRIBE NOTE - Uncertainty Summary                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Overall Confidence: ████████░░ 78%                         │
│                                                             │
│  Claims Breakdown:                                          │
│  ├── Demographics      ██████████ High confidence           │
│  ├── Medical history   ████████░░ Good confidence           │
│  ├── Current meds      ██████░░░░ Moderate - verify         │
│  ├── Assessment        ████░░░░░░ Low - review carefully    │
│  └── Plan              ████████░░ Good confidence           │
│                                                             │
│  ⚠️ 2 claims flagged for detailed review                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### Level 2: Natural Language Explanations

```python
def generate_uncertainty_explanation(uncertainty_data: dict) -> str:
    """
    Convert uncertainty metrics to physician-friendly language.
    """
    level = uncertainty_data["uncertainty_level"]
    claim = uncertainty_data["claim"]
    
    explanations = {
        "low": f"The AI is confident about '{claim}'. "
               f"Verified against EHR where applicable.",
        
        "medium": f"The AI shows some uncertainty about '{claim}'. "
                  f"Please verify this information with the patient.",
        
        "high": f"The AI is uncertain about '{claim}'. "
                f"When asked multiple times, the AI gave different answers. "
                f"This requires your direct verification."
    }
    
    return explanations[level]
```

#### Level 3: Actionable Recommendations

| Uncertainty Level | Physician Action | Time Estimate |
|-------------------|------------------|---------------|
| Low | Quick scan, sign if appropriate | ~15 seconds |
| Medium | Verify flagged items with patient/chart | ~45 seconds |
| High | Detailed review, may need to rewrite | ~90+ seconds |

---

## The Complete Uncertainty Pipeline

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    AI OUTPUT                                │
│    (Note, Prediction, or Diagnosis)                         │
└─────────────────────────┬───────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
          ▼               ▼               ▼
┌─────────────────┐ ┌───────────┐ ┌─────────────────┐
│ GENERATIVE AI   │ │PREDICTIVE │ │ DIAGNOSTIC AI   │
│ (Scribes)       │ │ AI        │ │ (Radiology)     │
├─────────────────┤ ├───────────┤ ├─────────────────┤
│                 │ │           │ │                 │
│ • Semantic      │ │• Calibra- │ │ • Confidence    │
│   Entropy       │ │  tion     │ │   Scores        │
│                 │ │  Analysis │ │                 │
│ • Response      │ │           │ │ • Attention     │
│   Variance      │ │• Confid-  │ │   Validation    │
│                 │ │  ence     │ │                 │
│ • Claim-level   │ │  Intervals│ │ • Second-read   │
│   Uncertainty   │ │           │ │   Consistency   │
│                 │ │• Drift    │ │                 │
│                 │ │  Detection│ │                 │
└────────┬────────┘ └─────┬─────┘ └────────┬────────┘
         │                │                │
         └────────────────┼────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              UNCERTAINTY AGGREGATION                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Inputs:                                                    │
│  • Per-claim uncertainty scores                             │
│  • Model calibration metrics                                │
│  • Evidence strength assessment                             │
│  • EHR verification results                                 │
│                                                             │
│  Processing:                                                │
│  • Weight uncertainties by clinical importance              │
│  • Identify highest-risk claims                             │
│  • Generate overall confidence score                        │
│                                                             │
│  Output:                                                    │
│  • Document-level uncertainty (0-100%)                      │
│  • Claim-level flags                                        │
│  • Review level recommendation                              │
│                                                             │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              PHYSICIAN DASHBOARD                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Visual: Confidence bars, color coding                      │
│  Text: Plain-language uncertainty explanations              │
│  Action: Clear review level (Brief/Standard/Detailed)       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Decision Flow

```
START
  │
  ▼
┌─────────────────────────────────┐
│ Calculate Uncertainty Metrics   │
│ (entropy, calibration, CI)      │
└───────────────┬─────────────────┘
                │
                ▼
        ┌───────────────┐
        │ Uncertainty   │
        │ Level?        │
        └───────┬───────┘
                │
    ┌───────────┼───────────┐
    │           │           │
    ▼           ▼           ▼
┌───────┐   ┌───────┐   ┌───────┐
│ LOW   │   │MEDIUM │   │ HIGH  │
└───┬───┘   └───┬───┘   └───┬───┘
    │           │           │
    ▼           ▼           ▼
┌───────┐   ┌───────┐   ┌───────┐
│Check  │   │Check  │   │Always │
│EHR    │   │EHR    │   │flag   │
│match? │   │match? │   │       │
└───┬───┘   └───┬───┘   └───────┘
    │           │           │
   YES         YES          │
    │           │           │
    ▼           ▼           ▼
┌───────┐   ┌───────┐   ┌────────┐
│BRIEF  │   │STANDARD│  │DETAILED│
│REVIEW │   │REVIEW │   │REVIEW  │
│(15s)  │   │(45s)  │   │(90s+)  │
└───────┘   └───────┘   └────────┘
```

---

## Uncertainty Quantification by AI Type

### Generative AI (Scribes)

| Method | What It Measures | Implementation |
|--------|------------------|----------------|
| Semantic Entropy | Consistency across regenerations | Ask same question 5-10x |
| Token Probability | Model's confidence per word | Extract logits from model |
| Claim Decomposition | Per-statement uncertainty | Parse and probe each claim |

### Predictive AI (Sepsis, Mortality, Readmission)

| Method | What It Measures | Implementation |
|--------|------------------|----------------|
| Calibration (ECE) | Probability accuracy | Compare predictions to outcomes |
| Confidence Intervals | Range of possible values | Bootstrap or MC Dropout |
| Prediction Variance | Stability of predictions | Perturb inputs slightly |
| Drift Detection | Model degradation over time | Monitor calibration longitudinally |

### Diagnostic AI (Radiology, Pathology)

| Method | What It Measures | Implementation |
|--------|------------------|----------------|
| Softmax Entropy | Classification uncertainty | Entropy of output probabilities |
| Attention Consistency | Where model "looks" | Compare attention across runs |
| Out-of-Distribution | Unfamiliar inputs | Detect inputs unlike training data |

---

## Why Uncertainty Quantification Matters Clinically

### Case Study 1: The Overconfident Sepsis Model

```
Scenario:
─────────
Patient: 45yo male, routine post-op day 1
AI Prediction: "Sepsis risk: 85%"
Actual: Patient was fine (false positive)

Without Uncertainty Quantification:
───────────────────────────────────
Physician orders unnecessary workup, antibiotics
Patient experiences side effects, delayed discharge
Cost: $5,000+ unnecessary care

With TRUST Uncertainty Quantification:
──────────────────────────────────────
TRUST reports: "Model confidence: 85%, but calibration 
analysis shows this model overestimates by ~20% in 
post-surgical patients. Adjusted estimate: 65% (CI: 45-78%)"

Physician: Reviews more carefully, decides to monitor
Patient: Avoids unnecessary intervention
```

### Case Study 2: The Uncertain AI Scribe

```
Scenario:
─────────
AI Scribe Note: "Patient takes lisinopril 40mg daily"
Reality: Patient said "lisinopril, I think 40... or maybe 20?"

Without Uncertainty Quantification:
───────────────────────────────────
Note signed as-is
Wrong dose documented
Potential medication error downstream

With TRUST Uncertainty Quantification:
──────────────────────────────────────
TRUST flags: "High uncertainty on medication dose claim.
AI gave different doses in 4/10 regenerations.
Confidence: 45%. Please verify with patient."

Physician: Confirms correct dose (20mg)
Correct documentation preserved
```

### Case Study 3: The Appropriately Uncertain Diagnosis

```
Scenario:
─────────
Radiology AI: "Possible early pneumonia vs. atelectasis"
AI Confidence: 55% pneumonia

Without Uncertainty Communication:
──────────────────────────────────
Report just says: "Findings consistent with pneumonia"
Uncertainty hidden from clinician

With TRUST Uncertainty Quantification:
──────────────────────────────────────
TRUST displays: "AI confidence: 55% (CI: 40-70%)
Differential: Pneumonia vs. atelectasis
Recommendation: Consider clinical correlation, 
follow-up imaging if symptoms persist"

Physician: Appreciates the nuance, makes informed decision
```

---

## Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| Semantic entropy calculation | ✅ Built | Validated in Paper 1 |
| Response variance analysis | ✅ Built | Part of entropy pipeline |
| Calibration analysis | 🔄 In Progress | Paper 2 focus |
| Confidence interval generation | 🔄 In Progress | Paper 2 focus |
| Evidence-based calibration | 📋 Planned | Paper 2.1 focus |
| Physician-friendly translation | ✅ Built | Dashboard displays |
| Per-claim uncertainty | ✅ Built | Claim extraction pipeline |
| Longitudinal drift detection | 📋 Planned | Phase 2 feature |

---

## Relationship to TRUST Papers

| Paper | Focus | Uncertainty Method |
|-------|-------|-------------------|
| **Paper 1** | Hallucination Detection | Semantic entropy as uncertainty signal |
| **Paper 2** | Uncertainty Quantification | Calibration, confidence intervals |
| **Paper 2.1** | Evidence-Calibrated Uncertainty | Matching confidence to evidence strength |
| **Paper 3** | Predictive AI Governance | Calibration monitoring for risk models |
| **Paper 4** | Radiology AI Governance | Diagnostic uncertainty quantification |

---

## Key Takeaways

1. **Uncertainty ≠ Error**
   - High uncertainty means "AI doesn't know"—this is valuable information
   - Low uncertainty doesn't guarantee correctness

2. **Different AI types need different methods**
   - Generative: Semantic entropy, response variance
   - Predictive: Calibration analysis, confidence intervals
   - Diagnostic: Softmax entropy, attention consistency

3. **Communication matters as much as measurement**
   - Physicians need actionable guidance, not statistical jargon
   - Visual indicators + plain language + clear recommendations

4. **Uncertainty quantification enables appropriate trust**
   - When AI is confident AND correct: efficiency gain
   - When AI is uncertain: physician expertise applied where needed
   - When AI is confident but wrong: cross-reference catches it

---

## References

1. Guo, C., et al. (2017). "On Calibration of Modern Neural Networks." *ICML*.

2. Kompa, B., et al. (2021). "Second opinion needed: communicating uncertainty in medical machine learning." *npj Digital Medicine*.

3. Farquhar, S., et al. (2024). "Detecting hallucinations in large language models using semantic entropy." *Nature*.

4. Naeini, M.P., et al. (2015). "Obtaining Well Calibrated Probabilities Using Bayesian Binning." *AAAI*.

5. TRUST Paper 2 (in preparation). "Uncertainty Quantification for Healthcare AI: A Clinical Framework."

---

*TRUST Medical AI Governance Platform*
*Transparent • Responsible • Unbiased • Safe • Traceable*
*Auditing AI. Protecting Patients. Empowering Physicians.*
