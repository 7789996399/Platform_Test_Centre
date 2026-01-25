"""
Generative Claim Extractor
===========================

Domain-agnostic claim extraction from generative AI outputs.

This module provides utilities for extracting verifiable claims from
free-form text. The actual claim patterns are domain-specific and
come from adapters; this module provides the extraction framework.

Extraction Strategy:
    1. Pattern-based extraction (fast, high precision)
    2. LLM-based extraction (slower, higher recall) - optional
    3. Deduplication and normalization
    4. Confidence scoring

Example:
    >>> extractor = GenerativeClaimExtractor()
    >>> claims = extractor.extract(
    ...     text="Patient takes Lisinopril 10mg daily",
    ...     patterns=MEDICATION_PATTERNS
    ... )
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Pattern, Tuple
import re


__all__ = [
    'GenerativeClaimExtractor',
    'ExtractedClaim',
    'ExtractionConfig',
    'ExtractionMethod',
]


# =============================================================================
# ENUMS & DATA CLASSES
# =============================================================================

class ExtractionMethod(str, Enum):
    """Method used to extract a claim."""
    PATTERN = "pattern"       # Regex pattern matching
    LLM = "llm"              # LLM-based extraction
    HYBRID = "hybrid"        # Combination of methods
    MANUAL = "manual"        # Manually specified


@dataclass
class ExtractedClaim:
    """
    A claim extracted from generative AI output.

    Contains the claim text, its location in the source, extraction
    method, and confidence score.
    """
    text: str
    claim_type: str
    start_pos: int
    end_pos: int
    extraction_method: ExtractionMethod
    confidence: float
    context: str  # Surrounding text for context
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def span(self) -> Tuple[int, int]:
        """Return the character span of the claim."""
        return (self.start_pos, self.end_pos)

    def overlaps(self, other: 'ExtractedClaim') -> bool:
        """Check if this claim overlaps with another."""
        return not (self.end_pos <= other.start_pos or other.end_pos <= self.start_pos)


@dataclass
class ExtractionConfig:
    """Configuration for claim extraction."""
    context_window: int = 50  # Characters of context to include
    min_confidence: float = 0.5  # Minimum confidence to keep claim
    deduplicate: bool = True  # Remove duplicate claims
    normalize: bool = True  # Normalize claim text
    max_claims: int = 100  # Maximum claims to extract


# =============================================================================
# MAIN EXTRACTOR CLASS
# =============================================================================

class GenerativeClaimExtractor:
    """
    Extracts verifiable claims from generative AI text output.

    The extractor is domain-agnostic; domain-specific patterns are
    provided by adapters. This class handles the mechanics of extraction,
    deduplication, and confidence scoring.

    Example:
        >>> extractor = GenerativeClaimExtractor()
        >>>
        >>> # Define patterns for your domain
        >>> patterns = {
        ...     "medication": re.compile(r'takes?\s+(\w+)\s+(\d+\s*mg)', re.I),
        ...     "vital": re.compile(r'(blood pressure|BP)\s+(?:is|was|of)\s+(\d+/\d+)', re.I),
        ... }
        >>>
        >>> claims = extractor.extract(ai_output, patterns)
    """

    def __init__(self, config: Optional[ExtractionConfig] = None):
        """
        Initialize extractor.

        Args:
            config: Extraction configuration (uses defaults if not provided)
        """
        self.config = config or ExtractionConfig()

    def extract(
        self,
        text: str,
        patterns: Dict[str, Pattern],
        claim_types: Optional[Dict[str, str]] = None,
    ) -> List[ExtractedClaim]:
        """
        Extract claims using regex patterns.

        Args:
            text: The text to extract claims from
            patterns: Dict mapping pattern names to compiled regex patterns
            claim_types: Optional mapping from pattern names to claim types

        Returns:
            List of extracted claims
        """
        if not text or not patterns:
            return []

        claims: List[ExtractedClaim] = []
        claim_types = claim_types or {}

        for pattern_name, pattern in patterns.items():
            claim_type = claim_types.get(pattern_name, pattern_name)
            matches = pattern.finditer(text)

            for match in matches:
                claim_text = match.group(0).strip()
                start_pos = match.start()
                end_pos = match.end()

                # Get context
                context_start = max(0, start_pos - self.config.context_window)
                context_end = min(len(text), end_pos + self.config.context_window)
                context = text[context_start:context_end]

                # Compute confidence based on match quality
                confidence = self._compute_confidence(match, pattern_name)

                if confidence >= self.config.min_confidence:
                    claims.append(ExtractedClaim(
                        text=claim_text,
                        claim_type=claim_type,
                        start_pos=start_pos,
                        end_pos=end_pos,
                        extraction_method=ExtractionMethod.PATTERN,
                        confidence=confidence,
                        context=context,
                        metadata={
                            "pattern_name": pattern_name,
                            "groups": match.groups(),
                        }
                    ))

        # Post-process
        if self.config.deduplicate:
            claims = self._deduplicate(claims)

        if self.config.normalize:
            claims = self._normalize(claims)

        # Sort by position
        claims.sort(key=lambda c: c.start_pos)

        # Limit count
        if len(claims) > self.config.max_claims:
            claims = claims[:self.config.max_claims]

        return claims

    def extract_with_llm(
        self,
        text: str,
        claim_types: List[str],
        llm_extractor: Optional[Callable[[str, List[str]], List[Dict]]] = None,
    ) -> List[ExtractedClaim]:
        """
        Extract claims using LLM (mock implementation for testing).

        In production, this would call an LLM to identify claims.

        Args:
            text: Text to extract from
            claim_types: Types of claims to look for
            llm_extractor: Optional custom LLM extractor function

        Returns:
            List of extracted claims
        """
        # MOCK: Return empty list for testing
        # In production, this would call an LLM
        return []

    def extract_hybrid(
        self,
        text: str,
        patterns: Dict[str, Pattern],
        claim_types: List[str],
        llm_extractor: Optional[Callable] = None,
    ) -> List[ExtractedClaim]:
        """
        Extract claims using both patterns and LLM.

        Pattern extraction runs first (fast), then LLM fills gaps.

        Args:
            text: Text to extract from
            patterns: Regex patterns for pattern extraction
            claim_types: Claim types for LLM extraction
            llm_extractor: Optional custom LLM extractor

        Returns:
            Combined list of extracted claims
        """
        # Pattern extraction first
        pattern_claims = self.extract(text, patterns)

        # LLM extraction for additional claims
        llm_claims = self.extract_with_llm(text, claim_types, llm_extractor)

        # Combine and deduplicate
        all_claims = pattern_claims + llm_claims

        # Mark hybrid extraction
        for claim in llm_claims:
            claim.extraction_method = ExtractionMethod.HYBRID

        return self._deduplicate(all_claims)

    def _compute_confidence(
        self,
        match: re.Match,
        pattern_name: str,
    ) -> float:
        """
        Compute confidence score for an extracted claim.

        Factors:
        - Match length (longer = more specific)
        - Number of captured groups (more = more structured)
        - Pattern type (some patterns are more reliable)
        """
        base_confidence = 0.7

        # Bonus for longer matches (more specific)
        match_len = match.end() - match.start()
        length_bonus = min(0.1, match_len / 100)

        # Bonus for captured groups
        groups = match.groups()
        group_bonus = min(0.1, len([g for g in groups if g]) * 0.02)

        # Pattern-specific adjustments could go here
        pattern_bonus = 0.0

        return min(1.0, base_confidence + length_bonus + group_bonus + pattern_bonus)

    def _deduplicate(self, claims: List[ExtractedClaim]) -> List[ExtractedClaim]:
        """
        Remove duplicate and overlapping claims.

        Keeps the claim with highest confidence when duplicates found.
        """
        if not claims:
            return []

        # Sort by confidence (descending)
        sorted_claims = sorted(claims, key=lambda c: c.confidence, reverse=True)

        kept: List[ExtractedClaim] = []
        for claim in sorted_claims:
            # Check if overlaps with any kept claim
            overlaps = any(claim.overlaps(kept_claim) for kept_claim in kept)

            # Also check for text duplicates
            text_dup = any(
                claim.text.lower() == kept_claim.text.lower()
                for kept_claim in kept
            )

            if not overlaps and not text_dup:
                kept.append(claim)

        return kept

    def _normalize(self, claims: List[ExtractedClaim]) -> List[ExtractedClaim]:
        """
        Normalize claim text.

        - Collapse whitespace
        - Strip leading/trailing whitespace
        - Standardize case for comparison
        """
        for claim in claims:
            # Collapse whitespace
            claim.text = ' '.join(claim.text.split())
            claim.context = ' '.join(claim.context.split())

        return claims

    def find_claim_boundaries(
        self,
        text: str,
        start_hint: int,
        end_hint: int,
    ) -> Tuple[int, int]:
        """
        Expand claim boundaries to natural sentence/phrase boundaries.

        Useful when pattern captures partial claim.

        Args:
            text: Full text
            start_hint: Approximate start position
            end_hint: Approximate end position

        Returns:
            Tuple of (start, end) positions at natural boundaries
        """
        # Find sentence start
        start = start_hint
        while start > 0 and text[start - 1] not in '.!?\n':
            start -= 1
        if start > 0:
            start += 1  # Skip the punctuation

        # Find sentence end
        end = end_hint
        while end < len(text) and text[end] not in '.!?\n':
            end += 1
        if end < len(text):
            end += 1  # Include the punctuation

        return (start, end)
