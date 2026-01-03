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
import httpx

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
# STEP 2: BIDIRECTIONAL ENTAILMENT CLUSTERING (via Hugging Face API)
# =============================================================================

class EntailmentClassifier:
    """
    Bidirectional entailment using Hugging Face Inference API
    
    Uses DeBERTa-large-MNLI via API (91.3% accuracy) - no local model loading.
    This avoids all the memory/tokenizer issues with local deployment.
    
    Two texts are semantically equivalent if:
    - A entails B (A → B)  AND
    - B entails A (B → A)
    
    This is stricter than just similarity - it requires mutual implication.
    """
    
    _model_loaded = False
    
    # Best model for NLI - runs on HF infrastructure
    DEFAULT_MODEL = "microsoft/deberta-large-mnli"
    API_URL = "https://api-inference.huggingface.co/models/"
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or self.DEFAULT_MODEL
        self.api_token = os.getenv("HUGGINGFACE_API_TOKEN")
        if not self.api_token:
            logger.warning("HUGGINGFACE_API_TOKEN not set - will use mock responses")
    
    @classmethod
    def preload_model(cls, model_name: str = None):
        """No preloading needed for API - just mark as ready"""
        cls._model_loaded = True
        logger.info(f"Hugging Face Inference API ready for: {model_name or cls.DEFAULT_MODEL}")
        return None
    
    @classmethod
    def is_loaded(cls) -> bool:
        """Check if ready (always true for API)"""
        return True
    
    async def check_entailment_async(self, premise: str, hypothesis: str) -> Tuple[str, float]:
        """
        Check if premise entails hypothesis using HF Inference API (async version).
        
        Returns:
            (label, confidence) where label is 'ENTAILMENT', 'CONTRADICTION', or 'NEUTRAL'
        """
        if not self.api_token:
            # Mock response for testing without API key
            return 'ENTAILMENT', 0.85
        
        url = f"{self.API_URL}{self.model_name}"
        headers = {"Authorization": f"Bearer {self.api_token}"}
        
        # DeBERTa-MNLI expects premise and hypothesis as a pair
        payload = {
            "inputs": f"{premise} [SEP] {hypothesis}",
            "parameters": {"candidate_labels": ["entailment", "contradiction", "neutral"]}
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                result = response.json()
                
                # Handle different response formats
                if isinstance(result, list) and len(result) > 0:
                    result = result[0]
                
                # Parse the response
                if 'label' in result:
                    label = result['label'].upper()
                    score = result.get('score', 0.5)
                elif 'labels' in result:
                    # Zero-shot classification format
                    labels = result['labels']
                    scores = result['scores']
                    max_idx = scores.index(max(scores))
                    label = labels[max_idx].upper()
                    score = scores[max_idx]
                else:
                    logger.warning(f"Unexpected API response format: {result}")
                    return 'NEUTRAL', 0.5
                
                return label, float(score)
                
        except httpx.HTTPStatusError as e:
            logger.error(f"HF API error: {e.response.status_code} - {e.response.text}")
            # Model might be loading - return neutral
            if e.response.status_code == 503:
                logger.info("Model is loading on HF, retrying...")
                await asyncio.sleep(2)
                return await self.check_entailment_async(premise, hypothesis)
            return 'NEUTRAL', 0.5
        except Exception as e:
            logger.error(f"Entailment check failed: {e}")
            return 'NEUTRAL', 0.5
    
    def check_entailment(self, premise: str, hypothesis: str) -> Tuple[str, float]:
        """Synchronous wrapper for check_entailment_async"""
        return asyncio.get_event_loop().run_until_complete(
            self.check_entailment_async(premise, hypothesis)
        )
    
    async def check_bidirectional_entailment_async(
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
        label_ab, conf_ab = await self.check_entailment_async(text_a, text_b)
        if label_ab != 'ENTAILMENT' or conf_ab < threshold:
            return False
        
        # B → A
        label_ba, conf_ba = await self.check_entailment_async(text_b, text_a)
        if label_ba != 'ENTAILMENT' or conf_ba < threshold:
            return False
        
        return True
    
    def check_bidirectional_entailment(
        self, 
        text_a: str, 
        text_b: str,
        threshold: float = 0.5
    ) -> bool:
        """Synchronous wrapper"""
        return asyncio.get_event_loop().run_until_complete(
            self.check_bidirectional_entailment_async(text_a, text_b, threshold)
        )


async def cluster_by_bidirectional_entailment_async(
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
        - clusters: List of lists, each containing equivalent samples
        - assignments: List mapping each sample to its cluster index
    """
    if classifier is None:
        classifier = EntailmentClassifier()
    
    if not samples:
        return [], []
    
    # First sample starts cluster 0
    clusters = [[samples[0]]]
    assignments = [0]
    
    # Process remaining samples
    for sample in samples[1:]:
        found_cluster = False
        
        for cluster_idx, cluster in enumerate(clusters):
            # Check against first sample in cluster (representative)
            representative = cluster[0]
            
            is_equivalent = await classifier.check_bidirectional_entailment_async(
                sample, 
                representative, 
                threshold
            )
            
            if is_equivalent:
                cluster.append(sample)
                assignments.append(cluster_idx)
                found_cluster = True
                break
        
        if not found_cluster:
            # Create new cluster
            clusters.append([sample])
            assignments.append(len(clusters) - 1)
    
    return clusters, assignments


def cluster_by_bidirectional_entailment(
    samples: List[str],
    classifier: Optional[EntailmentClassifier] = None,
    threshold: float = 0.5
) -> Tuple[List[List[str]], List[int]]:
    """Synchronous wrapper"""
    return asyncio.get_event_loop().run_until_complete(
        cluster_by_bidirectional_entailment_async(samples, classifier, threshold)
    )


# =============================================================================
# STEP 3: CALCULATE SEMANTIC ENTROPY
# =============================================================================

def calculate_cluster_entropy(cluster_sizes: List[int]) -> float:
    """
    Calculate Shannon entropy across clusters.
    
    SE = -Σ p(cluster) × log₂(p(cluster))
    
    Where p(cluster) = cluster_size / total_samples
    """
    total = sum(cluster_sizes)
    if total == 0:
        return 0.0
    
    entropy = 0.0
    for size in cluster_sizes:
        if size > 0:
            p = size / total
            entropy -= p * np.log2(p)
    
    return entropy


def normalize_entropy(entropy: float, num_samples: int) -> float:
    """
    Normalize entropy to 0-1 range.
    
    Max entropy occurs when each sample is in its own cluster:
    max_entropy = log₂(num_samples)
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
        num_samples: Number of responses to generate
        model: Which LLM to use ("openai", "anthropic", "mock")
        temperature: Sampling temperature (higher = more diverse)
        entailment_threshold: Confidence threshold for entailment
    
    Returns:
        SemanticEntropyResult with entropy metrics
    """
    # Step 1: Generate multiple responses
    logger.info(f"Generating {num_samples} responses using {model}")
    samples = await generate_responses(prompt, context, num_samples, model, temperature)
    
    # Step 2: Cluster by bidirectional entailment
    logger.info("Clustering responses by bidirectional entailment")
    classifier = EntailmentClassifier()
    clusters, assignments = await cluster_by_bidirectional_entailment_async(
        samples, 
        classifier, 
        entailment_threshold
    )
    
    cluster_sizes = [len(c) for c in clusters]
    
    # Step 3: Calculate entropy
    entropy = calculate_cluster_entropy(cluster_sizes)
    normalized = normalize_entropy(entropy, num_samples)
    confidence = 1.0 - normalized
    
    logger.info(f"SE={entropy:.3f}, Normalized={normalized:.3f}, Clusters={len(clusters)}")
    
    return SemanticEntropyResult(
        entropy=entropy,
        normalized_entropy=normalized,
        confidence=confidence,
        num_clusters=len(clusters),
        num_samples=num_samples,
        cluster_sizes=cluster_sizes,
        samples=samples,
        cluster_assignments=assignments
    )
