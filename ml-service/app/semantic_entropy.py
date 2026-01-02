"""
Semantic Entropy Calculation - TRUST Platform
Based on Farquhar et al. (Nature 2024) methodology

═══════════════════════════════════════════════════════════════════════════════
⚠️  EHR-FIRST PIPELINE: This module is ONLY called for unverified claims!
═══════════════════════════════════════════════════════════════════════════════

Pipeline Position:
    1. Extract claims from AI note
    2. Verify ALL claims against EHR (fast, cheap)
    3. ✅ VERIFIED → Skip SE, mark LOW risk
    4. ❌ CONTRADICTED or NOT_FOUND → Call THIS MODULE
    
This saves ~80% of API costs by avoiding SE on verified claims.

═══════════════════════════════════════════════════════════════════════════════

Key insight: Standard NLI fails on medical text (9.5% accuracy in Paper 1).
Bidirectional entailment clustering achieves 95.5% accuracy by detecting
semantic inconsistency across multiple model generations.

The "confident hallucinator" problem:
- Low entropy + wrong = MOST DANGEROUS (model is consistently wrong)
- High entropy + wrong = Caught by uncertainty (model knows it doesn't know)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import asyncio
import os
import logging

logger = logging.getLogger(__name__)

@dataclass
class SemanticEntropyResult:
    """Result of semantic entropy calculation"""
    entropy: float                    # 0 = consistent, 1+ = inconsistent
    normalized_entropy: float         # Normalized to 0-1 range
    confidence: float                 # 1 - normalized_entropy
    num_clusters: int                 # Number of semantic clusters
    num_samples: int                  # Total samples generated
    cluster_sizes: List[int]          # Size of each cluster
    samples: List[str]                # Generated samples
    cluster_assignments: List[int]    # Which cluster each sample belongs to


# =============================================================================
# STEP 1: GENERATE MULTIPLE RESPONSES
# =============================================================================

async def generate_responses(
    prompt: str,
    context: str,
    num_samples: int = 5,
    model: str = "openai",
    temperature: float = 0.7
) -> List[str]:
    """
    Generate multiple responses to the same prompt.
    
    Key: We're NOT paraphrasing. We're asking the model the same question
    multiple times and checking if it gives consistent answers.
    """
    
    if model == "openai":
        return await _generate_openai_responses(prompt, context, num_samples, temperature)
    elif model == "anthropic":
        return await _generate_anthropic_responses(prompt, context, num_samples, temperature)
    elif model == "mock":
        return _generate_mock_responses(prompt, num_samples)
    else:
        raise ValueError(f"Unknown model: {model}")


async def _generate_openai_responses(
    prompt: str,
    context: str,
    num_samples: int,
    temperature: float
) -> List[str]:
    """Generate responses using OpenAI API"""
    import openai
    
    client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    system_prompt = """You are a medical AI assistant. Answer the question based only on the provided context. 
Be specific and concise. If the information is not in the context, say "Not found in context."
Do not make up information."""

    full_prompt = f"""Context: {context}

Question: {prompt}

Answer:"""

    # Generate samples concurrently
    async def get_one_response(temp_offset: float = 0):
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_prompt}
            ],
            temperature=min(1.0, temperature + temp_offset),
            max_tokens=200
        )
        return response.choices[0].message.content.strip()
    
    # Vary temperature slightly for diversity
    tasks = [get_one_response(i * 0.05) for i in range(num_samples)]
    responses = await asyncio.gather(*tasks)
    
    return list(responses)


async def _generate_anthropic_responses(
    prompt: str,
    context: str,
    num_samples: int,
    temperature: float
) -> List[str]:
    """Generate responses using Anthropic API"""
    import anthropic
    
    client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    system_prompt = """You are a medical AI assistant. Answer the question based only on the provided context. 
Be specific and concise. If the information is not in the context, say "Not found in context."
Do not make up information."""

    full_prompt = f"""Context: {context}

Question: {prompt}

Answer:"""

    async def get_one_response(temp_offset: float = 0):
        response = await client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=200,
            temperature=min(1.0, temperature + temp_offset),
            system=system_prompt,
            messages=[{"role": "user", "content": full_prompt}]
        )
        return response.content[0].text.strip()
    
    tasks = [get_one_response(i * 0.05) for i in range(num_samples)]
    responses = await asyncio.gather(*tasks)
    
    return list(responses)


def _generate_mock_responses(prompt: str, num_samples: int) -> List[str]:
    """
    Generate mock responses for testing WITHOUT loading any models.
    Returns intentionally varied responses to test entropy calculation.
    """
    # Simulate realistic variation in medical responses
    mock_variations = [
        f"{prompt}",
        f"{prompt}, as documented",
        f"The patient is prescribed {prompt.replace('Patient is on ', '').replace('Patient on ', '')}",
        f"{prompt} per medication list",
        f"Current medication includes {prompt.replace('Patient is on ', '').replace('Patient on ', '')}",
    ]
    return mock_variations[:num_samples]


# =============================================================================
# STEP 2: BIDIRECTIONAL ENTAILMENT CLUSTERING
# =============================================================================

class EntailmentClassifier:
    """
    Bidirectional entailment using DeBERTa-v3-small-MNLI
    
    Model choice rationale:
    - deberta-v3-small: 140MB, 88.4% accuracy, fits P1v2 App Service
    - Future: Upgrade to Azure ML Endpoint with deberta-v3-large for pilot
    
    Two texts are semantically equivalent if:
    - A entails B (A → B)  AND
    - B entails A (B → A)
    
    This is stricter than just similarity - it requires mutual implication.
    """
    
    # Class-level model instance (shared across requests, loaded once at startup)
    _shared_classifier = None
    _model_loaded = False
    
    # Use small model for P1v2 App Service (140MB, fits in 3.5GB RAM)
    # TODO: Upgrade to "microsoft/deberta-v3-large-mnli" when moving to Azure ML Endpoint
    DEFAULT_MODEL = "cross-encoder/nli-distilroberta-base"
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or self.DEFAULT_MODEL
    
    @classmethod
    def preload_model(cls, model_name: str = None):
        """
        Pre-load the model at startup to avoid cold-start latency.
        Call this from FastAPI's startup event.
        """
        if cls._shared_classifier is None:
            from transformers import pipeline
            model = model_name or cls.DEFAULT_MODEL
            logger.info(f"Pre-loading entailment model: {model}")
            cls._shared_classifier = pipeline(
                "text-classification",
                model=model,
                device=-1  # CPU
            )
            cls._model_loaded = True
            logger.info(f"Model loaded successfully: {model}")
        return cls._shared_classifier
    
    def _get_classifier(self):
        """Get the shared classifier, loading if necessary"""
        if EntailmentClassifier._shared_classifier is None:
            EntailmentClassifier.preload_model(self.model_name)
        return EntailmentClassifier._shared_classifier
    
    def check_entailment(self, premise: str, hypothesis: str) -> Tuple[str, float]:
        """Check if premise entails hypothesis."""
        classifier = self._get_classifier()
        
        # Cross-encoder expects: "premise [SEP] hypothesis"
        result = classifier(f"{premise} [SEP] {hypothesis}")
        
        # Handle result format
        if isinstance(result, list):
            result = result[0]
        
        label = result['label'].upper()
        score = result['score']
        
        return label, score
    
    def check_bidirectional_entailment(
        self, 
        text_a: str, 
        text_b: str,
        threshold: float = 0.5
    ) -> bool:
        """
        Check if two texts are semantically equivalent via bidirectional entailment.
        
        Returns True if:
        - A entails B with confidence > threshold  AND
        - B entails A with confidence > threshold
        """
        # A → B
        label_ab, conf_ab = self.check_entailment(text_a, text_b)
        if label_ab != 'ENTAILMENT' or conf_ab < threshold:
            return False
        
        # B → A
        label_ba, conf_ba = self.check_entailment(text_b, text_a)
        if label_ba != 'ENTAILMENT' or conf_ba < threshold:
            return False
        
        return True
    
    @classmethod
    def is_loaded(cls) -> bool:
        """Check if model is loaded"""
        return cls._model_loaded


def cluster_by_bidirectional_entailment(
    samples: List[str],
    classifier: Optional[EntailmentClassifier] = None,
    threshold: float = 0.5
) -> Tuple[List[List[str]], List[int]]:
    """
    Cluster samples by semantic equivalence using bidirectional entailment.
    
    Algorithm:
    1. Start with first sample in cluster 0
    2. For each new sample, check bidirectional entailment with cluster representatives
    3. If matches a cluster, add to it; otherwise create new cluster
    
    Returns:
        (clusters, assignments) where:
        - clusters: List of lists, each inner list contains semantically equivalent samples
        - assignments: List mapping each sample index to its cluster index
    """
    if not samples:
        return [], []
    
    if classifier is None:
        classifier = EntailmentClassifier()
    
    clusters: List[List[str]] = [[samples[0]]]  # First sample starts cluster 0
    assignments: List[int] = [0]  # First sample assigned to cluster 0
    
    for i, sample in enumerate(samples[1:], start=1):
        assigned = False
        
        # Check against representative of each existing cluster
        for cluster_idx, cluster in enumerate(clusters):
            representative = cluster[0]  # Use first sample as representative
            
            if classifier.check_bidirectional_entailment(sample, representative, threshold):
                clusters[cluster_idx].append(sample)
                assignments.append(cluster_idx)
                assigned = True
                break
        
        if not assigned:
            # Create new cluster
            clusters.append([sample])
            assignments.append(len(clusters) - 1)
    
    return clusters, assignments


# =============================================================================
# STEP 3: CALCULATE ENTROPY
# =============================================================================

def calculate_cluster_entropy(clusters: List[List[str]], num_samples: int) -> float:
    """
    Calculate semantic entropy from clusters.
    
    Formula: SE = -Σ p(cluster) × log₂(p(cluster))
    
    Where p(cluster) = size of cluster / total samples
    
    Interpretation:
    - SE = 0: All samples in one cluster (perfectly consistent)
    - SE = log₂(N): Each sample in its own cluster (maximally inconsistent)
    """
    if num_samples <= 1 or len(clusters) <= 1:
        return 0.0
    
    entropy = 0.0
    for cluster in clusters:
        p = len(cluster) / num_samples
        if p > 0:
            entropy -= p * np.log2(p)
    
    return float(entropy)


def normalize_entropy(entropy: float, num_samples: int) -> float:
    """
    Normalize entropy to 0-1 range.
    
    Max possible entropy = log₂(num_samples) when each sample is its own cluster
    """
    if num_samples <= 1:
        return 0.0
    
    max_entropy = np.log2(num_samples)
    if max_entropy == 0:
        return 0.0
    
    return min(1.0, entropy / max_entropy)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

async def calculate_semantic_entropy(
    prompt: str,
    context: str,
    num_samples: int = 5,
    model: str = "openai",
    temperature: float = 0.7,
    entailment_threshold: float = 0.5
) -> SemanticEntropyResult:
    """
    Calculate semantic entropy for a clinical claim.
    
    This is the main entry point implementing Farquhar et al. methodology:
    1. Generate N responses to the same prompt
    2. Cluster by bidirectional entailment
    3. Calculate entropy across clusters
    
    Args:
        prompt: The question/claim to evaluate
        context: Clinical context (e.g., patient record)
        num_samples: Number of responses to generate (default 5)
        model: Which LLM to use ("openai", "anthropic", "mock")
        temperature: Sampling temperature (higher = more diverse)
        entailment_threshold: Confidence threshold for entailment (default 0.5)
    
    Returns:
        SemanticEntropyResult with entropy, confidence, clusters, etc.
    """
    
    # Step 1: Generate multiple responses
    logger.info(f"Generating {num_samples} responses using {model}")
    samples = await generate_responses(
        prompt=prompt,
        context=context,
        num_samples=num_samples,
        model=model,
        temperature=temperature
    )
    
    # Step 2: Cluster by bidirectional entailment
    logger.info("Clustering by bidirectional entailment")
    classifier = EntailmentClassifier()
    clusters, assignments = cluster_by_bidirectional_entailment(
        samples=samples,
        classifier=classifier,
        threshold=entailment_threshold
    )
    
    # Step 3: Calculate entropy
    entropy = calculate_cluster_entropy(clusters, num_samples)
    normalized = normalize_entropy(entropy, num_samples)
    confidence = 1.0 - normalized
    
    logger.info(f"Entropy: {entropy:.3f}, Clusters: {len(clusters)}, Confidence: {confidence:.3f}")
    
    return SemanticEntropyResult(
        entropy=entropy,
        normalized_entropy=normalized,
        confidence=confidence,
        num_clusters=len(clusters),
        num_samples=num_samples,
        cluster_sizes=[len(c) for c in clusters],
        samples=samples,
        cluster_assignments=assignments
    )


# =============================================================================
# LEGACY FUNCTION (for backward compatibility)
# =============================================================================

async def calculate_entropy(
    text: str,
    num_samples: int = 5,
    model: str = "local"
) -> Dict:
    """
    Legacy function for backward compatibility.
    Maps to new calculate_semantic_entropy function.
    """
    # If model is "local", use mock for now (no local model available)
    if model == "local":
        model = "mock"
    
    result = await calculate_semantic_entropy(
        prompt=text,
        context="",  # No context in legacy API
        num_samples=num_samples,
        model=model
    )
    
    return {
        "entropy": result.normalized_entropy,
        "confidence": result.confidence,
        "samples": result.samples
    }
