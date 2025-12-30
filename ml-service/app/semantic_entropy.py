"""
Semantic Entropy Calculation
Based on TRUST Paper 1 methodology
"""
import numpy as np
from typing import List, Dict
import os

async def calculate_entropy(
    text: str,
    num_samples: int = 5,
    model: str = "local"
) -> Dict:
    """
    Calculate semantic entropy using multiple model samples.
    
    Method:
    1. Generate multiple paraphrases/responses for the same input
    2. Calculate embeddings for each
    3. Measure variance in embedding space
    4. Higher variance = higher entropy = less certain
    """
    
    samples = []
    
    if model == "local":
        # Use local sentence transformers for paraphrase generation
        samples = await _generate_local_samples(text, num_samples)
    elif model == "openai":
        samples = await _generate_openai_samples(text, num_samples)
    else:
        # Default to simple variation
        samples = [text] * num_samples
    
    # Calculate entropy from samples
    entropy, confidence = await _calculate_entropy_from_samples(samples)
    
    return {
        "entropy": entropy,
        "confidence": confidence,
        "samples": samples
    }


async def _generate_local_samples(text: str, num_samples: int) -> List[str]:
    """Generate sample variations using local models"""
    # For MVP: return the same text (entropy will be 0 = very confident)
    # TODO: Integrate with local paraphrase model
    return [text] * num_samples


async def _generate_openai_samples(text: str, num_samples: int) -> List[str]:
    """Generate sample variations using OpenAI"""
    import openai
    
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    samples = []
    for i in range(num_samples):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Paraphrase the following text while keeping the same meaning. Be concise."},
                {"role": "user", "content": text}
            ],
            temperature=0.7 + (i * 0.1)  # Vary temperature for diversity
        )
        samples.append(response.choices[0].message.content)
    
    return samples


async def _calculate_entropy_from_samples(samples: List[str]) -> tuple:
    """
    Calculate semantic entropy from text samples.
    Uses embedding variance as proxy for semantic uncertainty.
    """
    if len(samples) < 2:
        return 0.0, 1.0  # Single sample = no entropy, full confidence
    
    try:
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(samples)
        
        # Calculate variance across embedding dimensions
        variance = np.var(embeddings, axis=0)
        mean_variance = np.mean(variance)
        
        # Normalize to 0-1 range (empirically tuned)
        entropy = min(1.0, mean_variance * 100)
        
        # Confidence is inverse of entropy
        confidence = 1.0 - entropy
        
        return float(entropy), float(confidence)
        
    except Exception as e:
        print(f"Error calculating entropy: {e}")
        return 0.5, 0.5  # Default to uncertain


# Bidirectional entailment clustering (from Paper 1)
async def cluster_by_entailment(samples: List[str]) -> List[List[str]]:
    """
    Cluster samples by semantic equivalence using bidirectional entailment.
    Samples that entail each other belong to same cluster.
    More clusters = higher entropy.
    """
    # TODO: Implement NLI-based clustering
    # For MVP: each sample is its own cluster
    return [[s] for s in samples]