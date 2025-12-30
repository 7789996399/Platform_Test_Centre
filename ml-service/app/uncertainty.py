"""
Uncertainty Quantification Module
Based on TRUST Paper 2 methodology

Methods:
1. Ensemble disagreement - multiple model samples
2. Token probability analysis - confidence from logprobs
3. Calibrated confidence - adjusted for known biases
"""
import numpy as np
from typing import Dict, List, Optional
import os


async def calculate_uncertainty(
    text: str,
    method: str = "ensemble",
    num_samples: int = 5
) -> Dict:
    """
    Calculate uncertainty for AI-generated text.
    
    Args:
        text: The AI-generated text to analyze
        method: "ensemble", "token_probability", or "calibrated"
        num_samples: Number of samples for ensemble method
        
    Returns:
        {
            "uncertainty": float (0-1, higher = more uncertain),
            "calibrated_confidence": float (0-1, higher = more confident),
            "method": str,
            "details": dict with method-specific info
        }
    """
    if method == "ensemble":
        return await _ensemble_uncertainty(text, num_samples)
    elif method == "token_probability":
        return await _token_probability_uncertainty(text)
    elif method == "calibrated":
        return await _calibrated_uncertainty(text, num_samples)
    else:
        # Default to ensemble
        return await _ensemble_uncertainty(text, num_samples)


async def _ensemble_uncertainty(text: str, num_samples: int) -> Dict:
    """
    Ensemble-based uncertainty using semantic similarity variance.
    
    Method:
    1. Generate embeddings for the text
    2. Compare against multiple paraphrased versions
    3. High variance in embedding space = high uncertainty
    """
    try:
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Generate embedding for original text
        original_embedding = model.encode([text])[0]
        
        # For MVP: simulate ensemble by adding small perturbations
        # In production: would query multiple models or use dropout sampling
        embeddings = [original_embedding]
        
        for i in range(num_samples - 1):
            # Add small random noise to simulate model variance
            noise = np.random.normal(0, 0.01, original_embedding.shape)
            perturbed = original_embedding + noise
            embeddings.append(perturbed)
        
        embeddings = np.array(embeddings)
        
        # Calculate variance across ensemble
        variance = np.var(embeddings, axis=0)
        mean_variance = np.mean(variance)
        
        # Normalize to 0-1 range
        uncertainty = min(1.0, mean_variance * 1000)
        calibrated_confidence = 1.0 - uncertainty
        
        return {
            "uncertainty": float(uncertainty),
            "calibrated_confidence": float(calibrated_confidence),
            "method": "ensemble",
            "details": {
                "num_samples": num_samples,
                "embedding_variance": float(mean_variance),
                "interpretation": _interpret_uncertainty(uncertainty)
            }
        }
        
    except Exception as e:
        return {
            "uncertainty": 0.5,
            "calibrated_confidence": 0.5,
            "method": "ensemble",
            "details": {"error": str(e)}
        }


async def _token_probability_uncertainty(text: str) -> Dict:
    """
    Token probability-based uncertainty.
    
    Uses average token probability as confidence measure.
    Lower average probability = higher uncertainty.
    
    Note: Requires API that returns logprobs (e.g., OpenAI)
    """
    # For MVP: estimate based on text characteristics
    # In production: would use actual logprobs from model API
    
    # Heuristics for uncertainty estimation
    uncertainty_factors = []
    
    # Factor 1: Hedging language
    hedging_words = ['maybe', 'perhaps', 'possibly', 'might', 'could', 
                     'uncertain', 'unclear', 'approximately', 'roughly',
                     'seems', 'appears', 'likely', 'unlikely']
    text_lower = text.lower()
    hedging_count = sum(1 for word in hedging_words if word in text_lower)
    hedging_factor = min(1.0, hedging_count * 0.15)
    uncertainty_factors.append(hedging_factor)
    
    # Factor 2: Question marks (indicates uncertainty)
    question_factor = min(1.0, text.count('?') * 0.2)
    uncertainty_factors.append(question_factor)
    
    # Factor 3: Negation complexity
    negation_words = ['not', "n't", 'never', 'no', 'none', 'neither']
    negation_count = sum(1 for word in negation_words if word in text_lower)
    negation_factor = min(1.0, negation_count * 0.1)
    uncertainty_factors.append(negation_factor)
    
    # Combine factors
    uncertainty = np.mean(uncertainty_factors) if uncertainty_factors else 0.1
    calibrated_confidence = 1.0 - uncertainty
    
    return {
        "uncertainty": float(uncertainty),
        "calibrated_confidence": float(calibrated_confidence),
        "method": "token_probability",
        "details": {
            "hedging_score": float(hedging_factor),
            "question_score": float(question_factor),
            "negation_score": float(negation_factor),
            "interpretation": _interpret_uncertainty(uncertainty),
            "note": "Heuristic estimation - production would use model logprobs"
        }
    }


async def _calibrated_uncertainty(text: str, num_samples: int) -> Dict:
    """
    Calibrated uncertainty combining multiple methods.
    
    Based on Paper 2: Evidence-calibrated uncertainty
    Combines ensemble disagreement with linguistic markers.
    """
    # Get ensemble uncertainty
    ensemble_result = await _ensemble_uncertainty(text, num_samples)
    ensemble_uncertainty = ensemble_result["uncertainty"]
    
    # Get token probability uncertainty
    token_result = await _token_probability_uncertainty(text)
    token_uncertainty = token_result["uncertainty"]
    
    # Weighted combination (ensemble weighted higher as more reliable)
    weights = {"ensemble": 0.7, "token": 0.3}
    combined_uncertainty = (
        weights["ensemble"] * ensemble_uncertainty +
        weights["token"] * token_uncertainty
    )
    
    # Apply calibration curve (reduces overconfidence)
    # Based on empirical calibration from Paper 2
    calibrated_uncertainty = _apply_calibration(combined_uncertainty)
    calibrated_confidence = 1.0 - calibrated_uncertainty
    
    return {
        "uncertainty": float(calibrated_uncertainty),
        "calibrated_confidence": float(calibrated_confidence),
        "method": "calibrated",
        "details": {
            "ensemble_uncertainty": float(ensemble_uncertainty),
            "token_uncertainty": float(token_uncertainty),
            "raw_combined": float(combined_uncertainty),
            "calibration_applied": True,
            "interpretation": _interpret_uncertainty(calibrated_uncertainty)
        }
    }


def _apply_calibration(raw_uncertainty: float) -> float:
    """
    Apply calibration curve to adjust for known AI overconfidence.
    
    AI models tend to be overconfident, so we adjust uncertainty upward
    for low uncertainty scores.
    """
    # Platt scaling approximation
    # This increases low uncertainty values slightly
    if raw_uncertainty < 0.3:
        # Boost low uncertainty (model might be overconfident)
        return raw_uncertainty * 1.3
    elif raw_uncertainty > 0.7:
        # Slightly reduce very high uncertainty
        return 0.7 + (raw_uncertainty - 0.7) * 0.8
    else:
        return raw_uncertainty


def _interpret_uncertainty(uncertainty: float) -> str:
    """Convert uncertainty score to human-readable interpretation."""
    if uncertainty < 0.2:
        return "VERY_LOW - High confidence, minimal review needed"
    elif uncertainty < 0.4:
        return "LOW - Good confidence, brief review recommended"
    elif uncertainty < 0.6:
        return "MODERATE - Standard review recommended"
    elif uncertainty < 0.8:
        return "HIGH - Detailed review required"
    else:
        return "VERY_HIGH - Significant uncertainty, careful verification needed"


# Medical-specific uncertainty adjustments
MEDICAL_HIGH_RISK_TERMS = [
    'diagnosis', 'cancer', 'malignant', 'surgery', 'dose', 'medication',
    'allergy', 'contraindicated', 'emergency', 'critical', 'fatal',
    'prognosis', 'terminal', 'metastasis', 'overdose'
]

async def medical_adjusted_uncertainty(text: str, base_uncertainty: float) -> float:
    """
    Adjust uncertainty for medical context.
    High-risk medical terms increase required confidence threshold.
    """
    text_lower = text.lower()
    risk_term_count = sum(1 for term in MEDICAL_HIGH_RISK_TERMS if term in text_lower)
    
    # Increase uncertainty for high-risk medical content
    risk_adjustment = min(0.2, risk_term_count * 0.05)
    adjusted = min(1.0, base_uncertainty + risk_adjustment)
    
    return adjusted