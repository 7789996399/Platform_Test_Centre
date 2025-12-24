"""
TRUST Platform - Semantic Entropy for Hallucination Detection
=============================================================
Implements Farquhar et al. (Nature, 2024) methodology.

Paper 1: Detecting hallucinations using semantic entropy.
- Generate multiple responses
- Cluster by bidirectional entailment
- High entropy = hallucination risk
"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class EntropyResult:
    """Result of semantic entropy calculation."""
    entropy: float
    n_clusters: int
    n_responses: int
    cluster_sizes: List[int]
    risk_level: str  # LOW, MEDIUM, HIGH
    

def calculate_entropy(cluster_sizes: List[int], total_responses: int) -> float:
    """
    Calculate Shannon entropy over semantic clusters.
    
    Args:
        cluster_sizes: Number of responses in each cluster
        total_responses: Total number of generated responses
        
    Returns:
        Entropy value (higher = more uncertainty = hallucination risk)
    
    Example:
        - All 5 responses in 1 cluster → entropy = 0 (consistent, trustworthy)
        - 5 responses in 5 clusters → entropy = 1.6 (inconsistent, risky)
    """
    if total_responses == 0:
        return 0.0
    
    probabilities = [size / total_responses for size in cluster_sizes]
    
    # Shannon entropy: -Σ p(x) * log(p(x))
    entropy = 0.0
    for p in probabilities:
        if p > 0:
            entropy -= p * np.log2(p)
    
    return entropy


def classify_risk(entropy: float, n_clusters: int) -> str:
    """
    Classify hallucination risk based on entropy.
    
    Thresholds based on Paper 1 validation:
    - LOW: entropy < 0.5 (consistent responses)
    - MEDIUM: 0.5 <= entropy < 1.0 (some variation)
    - HIGH: entropy >= 1.0 (inconsistent, likely hallucination)
    """
    if entropy < 0.5:
        return "LOW"
    elif entropy < 1.0:
        return "MEDIUM"
    else:
        return "HIGH"


def analyze_claim(
    claim: str,
    responses: List[str],
    clusters: List[List[int]]
) -> EntropyResult:
    """
    Analyze a single claim using semantic entropy.
    
    Args:
        claim: The clinical claim to verify
        responses: Multiple LLM responses about the claim
        clusters: Indices grouped by semantic equivalence
        
    Returns:
        EntropyResult with risk assessment
    """
    cluster_sizes = [len(c) for c in clusters]
    n_responses = len(responses)
    n_clusters = len(clusters)
    
    entropy = calculate_entropy(cluster_sizes, n_responses)
    risk_level = classify_risk(entropy, n_clusters)
    
    return EntropyResult(
        entropy=entropy,
        n_clusters=n_clusters,
        n_responses=n_responses,
        cluster_sizes=cluster_sizes,
        risk_level=risk_level
    )


# =============================================================
# ENTAILMENT CHECKING (requires transformers - used in production)
# =============================================================

def check_bidirectional_entailment(text_a: str, text_b: str, nli_model) -> bool:
    """
    Check if two texts are semantically equivalent.
    
    Bidirectional entailment: A entails B AND B entails A
    This is the key insight from Farquhar et al.
    
    Args:
        text_a: First text
        text_b: Second text
        nli_model: HuggingFace NLI pipeline (DeBERTa-large-MNLI)
        
    Returns:
        True if texts are semantically equivalent
    """
    # A → B
    result_ab = nli_model(f"{text_a} [SEP] {text_b}", truncation=True)
    a_entails_b = result_ab[0]['label'] == 'ENTAILMENT'
    
    # B → A
    result_ba = nli_model(f"{text_b} [SEP] {text_a}", truncation=True)
    b_entails_a = result_ba[0]['label'] == 'ENTAILMENT'
    
    return a_entails_b and b_entails_a


def cluster_by_entailment(responses: List[str], nli_model) -> List[List[int]]:
    """
    Cluster responses using bidirectional entailment.
    
    Greedy algorithm from Farquhar et al.:
    1. Start with first response as cluster 0
    2. For each new response, check entailment with existing cluster representatives
    3. If entails, add to that cluster; otherwise create new cluster
    
    Args:
        responses: List of LLM responses
        nli_model: HuggingFace NLI pipeline
        
    Returns:
        List of clusters (each cluster is list of response indices)
    """
    if not responses:
        return []
    
    clusters = [[0]]  # First response starts first cluster
    
    for i in range(1, len(responses)):
        found_cluster = False
        
        for cluster in clusters:
            # Compare with first response in cluster (representative)
            representative_idx = cluster[0]
            if check_bidirectional_entailment(
                responses[representative_idx],
                responses[i],
                nli_model
            ):
                cluster.append(i)
                found_cluster = True
                break
        
        if not found_cluster:
            clusters.append([i])
    
    return clusters


# =============================================================
# SIMPLE MODE (for testing without heavy ML models)
# =============================================================

def cluster_by_exact_match(responses: List[str]) -> List[List[int]]:
    """
    Simple clustering by exact string match (for testing).
    
    Use this when you don't have the NLI model loaded.
    Less accurate but fast.
    """
    clusters = []
    seen = {}
    
    for i, response in enumerate(responses):
        normalized = response.strip().lower()
        if normalized in seen:
            clusters[seen[normalized]].append(i)
        else:
            seen[normalized] = len(clusters)
            clusters.append([i])
    
    return clusters
