"""
Hallucination Detection
Uses NLI (Natural Language Inference) to detect unsupported claims
"""
from typing import Dict

async def detect(claim: str, context: str) -> Dict:
    """
    Detect if a claim is supported by the context.
    
    Uses entailment classification:
    - ENTAILMENT: claim is supported by context
    - CONTRADICTION: claim contradicts context  
    - NEUTRAL: claim is not supported (potential hallucination)
    """
    try:
        from transformers import pipeline
        
        # Load NLI model
        classifier = pipeline(
            "text-classification",
            model="facebook/bart-large-mnli",
            device=-1  # CPU
        )
        
        # Format for NLI
        premise = context
        hypothesis = claim
        
        result = classifier(
            f"{premise}</s></s>{hypothesis}",
            candidate_labels=["entailment", "contradiction", "neutral"]
        )
        
        # Interpret results
        label = result[0]["label"].lower()
        score = result[0]["score"]
        
        is_hallucination = label in ["neutral", "contradiction"]
        
        reasoning = {
            "entailment": "Claim is supported by the context",
            "contradiction": "Claim contradicts the context",
            "neutral": "Claim is not supported by available context"
        }.get(label, "Unknown")
        
        return {
            "is_hallucination": is_hallucination,
            "confidence": score,
            "reasoning": reasoning
        }
        
    except Exception as e:
        # Fallback: assume uncertain
        return {
            "is_hallucination": False,
            "confidence": 0.5,
            "reasoning": f"Could not analyze: {str(e)}"
        }