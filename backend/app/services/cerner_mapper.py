"""
TRUST Platform - Cerner Data Mapper
====================================
Converts FHIR DocumentReference to TRUST format.
DETECTS AI-GENERATED CONTENT by checking author references.

HOW AI DETECTION WORKS:
-----------------------
Cerner uses FHIR references to identify authors:
- "Device/..." = AI/automated system (AI SCRIBE!)
- "Practitioner/..." = Human physician/nurse

We detect this automatically and flag for governance review.
"""

import base64
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# =============================================================================
# LOINC CODE MAPPINGS
# =============================================================================

LOINC_TO_NOTE_TYPE: Dict[str, str] = {
    "11488-4": "consult_note",
    "18842-5": "discharge_summary",
    "34117-2": "history_and_physical",
    "28570-0": "procedure_note",
    "34746-8": "anesthesia_note",  # Your specialty!
    "11506-3": "progress_note",
    "11504-8": "surgical_note",
    "34874-8": "pre_operative_note",
    "34878-9": "post_operative_note",
    "18748-4": "radiology_report",
    "59258-4": "emergency_note",
}

# Keywords that indicate AI-generated content
AI_INDICATORS: List[str] = [
    "device",       # FHIR Device reference = automated
    "ai",
    "scribe", 
    "dragon",       # Nuance Dragon/DAX
    "dax",
    "nuance",
    "ambient",
    "automated",
    "bot",
    "agent",
]


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TRUSTDocument:
    """A clinical document in TRUST's internal format."""
    
    # Identifiers
    document_id: str
    patient_id: str
    encounter_id: Optional[str] = None
    
    # Content
    note_text: str = ""
    note_type: str = "unknown"
    
    # Author info (CRITICAL for AI detection!)
    author: str = "Unknown"
    author_type: str = "unknown"  # ai_scribe, physician, nurse, unknown
    author_reference: str = ""
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    # Source tracking
    cerner_resource_id: str = ""
    
    # Governance fields
    review_level: str = "standard"  # brief, standard, detailed
    
    def is_ai_generated(self) -> bool:
        """Check if this document was created by an AI scribe."""
        return self.author_type == "ai_scribe"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "document_id": self.document_id,
            "patient_id": self.patient_id,
            "encounter_id": self.encounter_id,
            "note_text": self.note_text,
            "note_type": self.note_type,
            "author": self.author,
            "author_type": self.author_type,
            "author_reference": self.author_reference,
            "created_at": self.created_at.isoformat(),
            "cerner_resource_id": self.cerner_resource_id,
            "review_level": self.review_level,
            "is_ai_generated": self.is_ai_generated(),
        }


@dataclass
class MappingResult:
    """Result of mapping a FHIR document."""
    success: bool
    document: Optional[TRUSTDocument] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# =============================================================================
# MAPPER CLASS
# =============================================================================

class CernerDocumentMapper:
    """
    Maps Cerner FHIR DocumentReference to TRUST format.
    
    USAGE:
    ------
    mapper = CernerDocumentMapper()
    result = mapper.map(fhir_document)
    
    if result.success:
        doc = result.document
        if doc.is_ai_generated():
            print("AI content detected - needs review!")
    """
    
    def __init__(self):
        self.documents_processed = 0
        self.ai_documents_found = 0
    
    def map(self, fhir_doc: Dict) -> MappingResult:
        """
        Map a FHIR DocumentReference to TRUSTDocument.
        
        Parameters:
        -----------
        fhir_doc : dict
            Raw FHIR DocumentReference from Cerner
            
        Returns:
        --------
        MappingResult with success status and mapped document
        """
        errors = []
        warnings = []
        
        self.documents_processed += 1
        
        # Validate
        if not isinstance(fhir_doc, dict):
            return MappingResult(success=False, errors=["Input must be a dict"])
        
        if fhir_doc.get("resourceType") != "DocumentReference":
            return MappingResult(
                success=False, 
                errors=[f"Expected DocumentReference, got {fhir_doc.get('resourceType')}"]
            )
        
        # Extract fields
        doc_id = f"cerner_{fhir_doc.get('id', 'unknown')}"
        
        patient_id, patient_warn = self._extract_patient_id(fhir_doc)
        if patient_warn:
            warnings.append(patient_warn)
        
        note_text, text_warn = self._extract_note_text(fhir_doc)
        if text_warn:
            warnings.append(text_warn)
        
        note_type, type_warn = self._extract_note_type(fhir_doc)
        if type_warn:
            warnings.append(type_warn)
        
        author, author_type, author_ref = self._extract_author(fhir_doc)
        
        created_at = self._extract_date(fhir_doc)
        
        encounter_id = self._extract_encounter(fhir_doc)
        
        # Determine review level based on author type
        if author_type == "ai_scribe":
            review_level = "detailed"
            self.ai_documents_found += 1
            logger.info(f"AI scribe detected: {author}")
        elif author_type == "unknown":
            review_level = "standard"
        else:
            review_level = "brief"
        
        # Create document
        trust_doc = TRUSTDocument(
            document_id=doc_id,
            patient_id=patient_id,
            encounter_id=encounter_id,
            note_text=note_text,
            note_type=note_type,
            author=author,
            author_type=author_type,
            author_reference=author_ref,
            created_at=created_at,
            cerner_resource_id=fhir_doc.get("id", ""),
            review_level=review_level,
        )
        
        return MappingResult(
            success=True,
            document=trust_doc,
            warnings=warnings
        )
    
    def _extract_patient_id(self, fhir_doc: Dict) -> Tuple[str, Optional[str]]:
        """Extract patient ID from subject reference."""
        subject = fhir_doc.get("subject", {})
        reference = subject.get("reference", "")
        
        if reference and "/" in reference:
            return reference.split("/")[-1], None
        
        display = subject.get("display")
        if display:
            return f"name:{display}", "Patient ID from display name"
        
        return "unknown", "Could not extract patient ID"
    
    def _extract_note_text(self, fhir_doc: Dict) -> Tuple[str, Optional[str]]:
        """Extract and decode note text from content attachment."""
        content_list = fhir_doc.get("content", [])
        
        if not content_list:
            return "", "No content found"
        
        for content in content_list:
            attachment = content.get("attachment", {})
            
            # Try base64 data
            if "data" in attachment:
                try:
                    decoded = base64.b64decode(attachment["data"])
                    return decoded.decode("utf-8"), None
                except Exception as e:
                    logger.warning(f"Base64 decode failed: {e}")
                    continue
            
            # Try URL
            if "url" in attachment:
                return f"[Content at: {attachment['url']}]", "External URL"
        
        return "", "Could not extract note text"
    
    def _extract_note_type(self, fhir_doc: Dict) -> Tuple[str, Optional[str]]:
        """Extract note type from LOINC coding."""
        type_info = fhir_doc.get("type", {})
        codings = type_info.get("coding", [])
        
        for coding in codings:
            code = coding.get("code", "")
            if code in LOINC_TO_NOTE_TYPE:
                return LOINC_TO_NOTE_TYPE[code], None
        
        # Fallback to text
        text = type_info.get("text", "")
        if text:
            return text.lower().replace(" ", "_"), "Type from text field"
        
        return "unknown", "Could not determine note type"
    
    def _extract_author(self, fhir_doc: Dict) -> Tuple[str, str, str]:
        """
        Extract author and classify as AI or human.
        
        THIS IS THE KEY FUNCTION FOR AI DETECTION!
        
        Returns:
        --------
        Tuple of (author_name, author_type, author_reference)
        """
        authors = fhir_doc.get("author", [])
        
        if not authors:
            return "Unknown", "unknown", ""
        
        first_author = authors[0]
        author_name = first_author.get("display", "Unknown")
        author_ref = first_author.get("reference", "")
        
        # CLASSIFY AUTHOR TYPE
        ref_lower = author_ref.lower()
        name_lower = author_name.lower()
        combined = f"{ref_lower} {name_lower}"
        
        # Check for AI indicators
        # PRIMARY: Device reference = automated/AI system
        if "device/" in ref_lower:
            return author_name, "ai_scribe", author_ref
        
        # SECONDARY: AI keywords in name or reference
        for indicator in AI_INDICATORS:
            if indicator in combined:
                return author_name, "ai_scribe", author_ref
        
        # Check for human indicators
        if "practitioner/" in ref_lower:
            if any(t in name_lower for t in ["dr", "md", "physician", "doctor"]):
                return author_name, "physician", author_ref
            if any(t in name_lower for t in ["rn", "nurse", "np"]):
                return author_name, "nurse", author_ref
            return author_name, "clinical_staff", author_ref
        
        return author_name, "unknown", author_ref
    
    def _extract_date(self, fhir_doc: Dict) -> datetime:
        """Extract document creation date."""
        date_str = fhir_doc.get("date")
        
        if date_str:
            try:
                # Handle timezone
                date_str = date_str.replace("Z", "+00:00")
                if "+" in date_str:
                    date_str = date_str.split("+")[0]
                return datetime.fromisoformat(date_str)
            except:
                pass
        
        return datetime.utcnow()
    
    def _extract_encounter(self, fhir_doc: Dict) -> Optional[str]:
        """Extract encounter ID."""
        context = fhir_doc.get("context", {})
        encounters = context.get("encounter", [])
        
        if encounters:
            ref = encounters[0].get("reference", "")
            if "/" in ref:
                return ref.split("/")[-1]
        
        return None
    
    def get_stats(self) -> Dict:
        """Get processing statistics."""
        return {
            "documents_processed": self.documents_processed,
            "ai_documents_found": self.ai_documents_found,
            "ai_detection_rate": (
                self.ai_documents_found / max(1, self.documents_processed)
            ),
        }