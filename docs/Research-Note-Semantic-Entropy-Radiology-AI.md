# Research Note: Semantic Entropy in Radiology AI
## Literature Review and TRUST Positioning

**Date:** December 31, 2024
**Purpose:** Document current research on semantic entropy for radiology AI hallucination detection and position TRUST's differentiation.

---

## Key Finding: Independent Validation

Semantic entropy for radiology AI hallucination detection has been independently validated by researchers at TruhnLab (RWTH Aachen University Hospital). This strengthens TRUST's approach and validates our Paper 1 methodology.

---

## Primary Paper

### Hallucination Filtering in Radiology Vision-Language Models Using Discrete Semantic Entropy

| Field | Details |
|-------|---------|
| **Authors** | Patrick Wienholt et al. |
| **arXiv** | 2510.09256 |
| **Date** | October 10, 2025 |
| **Code** | https://github.com/TruhnLab/VisionSemanticEntropy |
| **Institution** | TruhnLab, RWTH Aachen University Hospital |

### Abstract Summary

The study investigated Discrete Semantic Entropy (DSE) as a method for identifying and filtering unreliable Vision-Language Model (VLM) outputs in radiologic image interpretation tasks. DSE extends semantic entropy from text-only applications to the multimodal domain of radiology.

### Methodology

```
1. Input: Radiology image + clinical question
2. Generate 15 responses at temperature=1.0
3. Cluster semantically equivalent responses using bidirectional entailment
4. Calculate DSE from cluster distribution
5. Filter out questions with DSE > threshold (0.3 or 0.6)
6. Evaluate accuracy on retained questions
```

### Datasets

| Dataset | Description |
|---------|-------------|
| VQA-Med 2019 | 500 images with clinical questions and short-text answers |
| Diagnostic Radiology | 206 cases: 60 CT, 60 MRI, 60 radiographs, 26 angiograms |

### Key Results

| Model | Baseline Accuracy | After DSE Filtering (>0.3) | Questions Retained |
|-------|-------------------|---------------------------|-------------------|
| **GPT-4o** | 51.7% | **76.3%** | 334/706 (47%) |
| **GPT-4.1** | 54.8% | **63.8%** | 499/706 (71%) |

**Key Finding:** Filtering out high-entropy responses improved GPT-4o accuracy from 51.7% to 76.3%—a 48% relative improvement.

### Strengths Noted

1. **Black-box compatible:** No access to model internals required
2. **Robust to paraphrasing:** Semantic clustering handles linguistic variation
3. **Lower latency:** Compared to RadFlag or vision-amplified methods
4. **Question-level screening:** Can withhold unstable outputs before reaching clinicians

### Limitations Acknowledged

> "DSE focuses on semantic consistency alone—efficient yet still limited when hallucinations are repeated with high confidence."

This is the "confident hallucinator" problem we address in TRUST Paper 1.

### Clinical Integration Vision

> "Integrating DSE into PACS or reporting systems can streamline workflows by screening queries at the question level to either withhold unstable outputs or attach an interpretable uncertainty score derived from semantic dispersion across sampled responses."

---

## Secondary Paper: RadFlag

### RadFlag: A Black-Box Hallucination Detection Method for Medical Vision Language Models

| Field | Details |
|-------|---------|
| **arXiv** | 2411.00299 |
| **Date** | November 16, 2024 |
| **Focus** | Report-level hallucination detection |

### Methodology Difference

RadFlag compares a candidate radiology report against a corpus of additional reports sampled at higher temperature, using GPT-4 to count how many high-temperature reports support each sentence.

### Comparison to DSE

| Aspect | DSE (Wienholt) | RadFlag |
|--------|----------------|---------|
| Level | Question/claim | Sentence/report |
| Method | Entropy of response clusters | Entailment scoring against corpus |
| Compute | Lower (15 samples) | Higher (multiple reports + GPT-4 calls) |
| Black-box | Yes | Yes |

### Key Quote on Entropy Limitations

> "Despite requiring access to per-token probabilities and therefore requiring more technical expertise to implement, we found entropy baselines were not successful in identifying hallucinatory outputs. This is likely because there are many valid ways to express the same radiological idea, thus low probability does not equate to low confidence in the overall claim."

**Note:** This refers to token-level entropy, not semantic entropy. Semantic entropy (clustering by meaning) addresses this limitation.

---

## Additional Related Work

### Confidence Calibration for Medical Image Segmentation (Mehrtash et al., 2020)

| Field | Details |
|-------|---------|
| **Journal** | IEEE TMI / PMC7704933 |
| **Institution** | Harvard/UBC |
| **Focus** | Uncertainty estimation in FCNs for segmentation |

**Key Contribution:** Proposed model ensembling for confidence calibration and average entropy as a predictive metric for segmentation quality.

**Relevance:** Different application (segmentation vs. reports) but establishes entropy-based uncertainty in radiology AI.

### On the Interpretability of AI in Radiology (Reyes et al., 2020)

| Field | Details |
|-------|---------|
| **Journal** | Radiology: Artificial Intelligence |
| **Focus** | Interpretability methods including uncertainty |

**Key Quote:**
> "Uncertainty estimates can in fact act as a proxy to enhance trust in a system, as a radiologist can verify whether the generated confidence levels of a computer-generated result match with their own assessment."

---

## TRUST Differentiation

### What Existing Research Does

| Paper | Approach | Limitation |
|-------|----------|------------|
| Wienholt 2025 | Filter high-entropy responses | Binary reject/accept |
| RadFlag 2024 | Flag sentences with low support | Computationally expensive |
| Mehrtash 2020 | Ensemble calibration for segmentation | Segmentation only, not reports |

### What TRUST Does Differently

| TRUST Feature | Differentiation |
|---------------|-----------------|
| **Tiered Review** | Not binary filter—Brief/Standard/Detailed based on uncertainty |
| **Claim-Level Uncertainty** | Per-claim entropy, not just document-level |
| **Multi-Layer Verification** | Entropy + EHR cross-reference catches confident hallucinators |
| **Governance Framework** | Audit trail, compliance reporting, not just detection |
| **Physician Communication** | Actionable guidance, not statistical scores |
| **Vendor Agnostic** | Works with any AI system, not model-specific |

### Positioning Statement for Papers

> "While recent work demonstrates that discrete semantic entropy can identify unreliable radiology VLM outputs (Wienholt et al., 2025), clinical workflows require more than binary filtering. Healthcare AI governance demands claim-level uncertainty quantification, integration with EHR verification, tiered physician review, and comprehensive audit trails. TRUST extends semantic entropy from a filtering mechanism to a complete governance framework—enabling physician-in-the-loop oversight rather than automated gatekeeping."

---

## Implications for TRUST Papers

### Paper 1 (Hallucination Detection)

- **Validation:** Independent research confirms semantic entropy works for medical AI
- **Citation:** Reference Wienholt et al. as concurrent/supporting work
- **Differentiation:** Our multi-layer approach (entropy + NLI + EHR) catches confident hallucinators that DSE alone misses

### Paper 4 (Radiology AI Governance)

- **Foundation:** Build on validated DSE methodology
- **Extension:** Add attention validation, concept-based explanations
- **Cross-reference:** Show TRUST provides comprehensive coverage (Paper 1 + Paper 4)

### Publication Strategy

| Journal | Angle |
|---------|-------|
| **Radiology: AI** | Governance framework extending DSE |
| **npj Digital Medicine** | Clinical implementation focus |
| **JAMA Network Open** | Physician workflow integration |

---

## Code Resources

### TruhnLab Implementation
- **Repository:** https://github.com/TruhnLab/VisionSemanticEntropy
- **Potential Use:** Reference implementation, benchmark comparison

### TRUST Implementation
- **Our Approach:** Already built in Paper 1 codebase
- **Differentiation:** Multi-layer (entropy + NLI + EHR), governance-focused

---

## Key Quotes for Reference

### On Clinical Utility (Wienholt 2025)
> "Integrating DSE into PACS or reporting systems can streamline workflows by screening queries at the question level to either withhold unstable outputs or attach an interpretable uncertainty score derived from semantic dispersion across sampled responses. In practice, this identifies questions that are more likely to elicit hallucinations and, when answers are returned, provides a quantitative signal that radiologists can reference when judging the safety of acting on a given question–answer pair."

### On Limitations (Wienholt 2025)
> "DSE focuses on semantic consistency alone—efficient yet still limited when hallucinations are repeated with high confidence."

### On Trust Building (Reyes 2020)
> "Uncertainty estimates can in fact act as a proxy to enhance trust in a system, as a radiologist can verify whether the generated confidence levels of a computer-generated result match with their own assessment."

---

## Action Items

- [ ] Add Wienholt et al. 2025 to Paper 1 literature review
- [ ] Reference in Paper 4 as foundation for radiology extension
- [ ] Review TruhnLab GitHub for implementation insights
- [ ] Position TRUST as governance layer, not just filter
- [ ] Emphasize multi-layer approach addressing confident hallucinator problem

---

## Summary

**Semantic entropy for radiology AI hallucination detection is independently validated (October 2025).** This strengthens TRUST's technical foundation while creating clear differentiation:

| They Do | TRUST Does |
|---------|------------|
| Filter unreliable outputs | Govern AI with physician-in-the-loop |
| Binary accept/reject | Tiered review levels |
| Single method (DSE) | Multi-layer (Entropy + EHR + Audit) |
| Detection only | Detection + Communication + Compliance |

**TRUST's value proposition:** We don't just detect problems—we enable appropriate physician oversight at scale.

---

*Research note prepared for TRUST Medical AI Governance Platform*
*Last updated: December 31, 2024*
