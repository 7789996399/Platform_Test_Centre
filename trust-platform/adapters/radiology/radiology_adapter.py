"""
Radiology Adapter
=================

Domain-specific adapter for radiology AI governance.

This adapter provides:
- Standard radiology finding classifications
- Anatomical region definitions for attention validation
- Prior imaging comparison logic
- Mock PACS/DICOM integration for testing

Supported Modalities:
- Chest X-ray (CXR)
- CT scan
- MRI (basic support)

Example:
    >>> adapter = RadiologyAdapter(modality=ModalityType.CHEST_XRAY)
    >>> findings = adapter.get_class_names()
    >>> is_valid, concerns = adapter.validate_attention(attention_map, "pneumonia")
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from diagnostic_governance.attention_validator import AttentionRegion


__all__ = [
    'RadiologyAdapter',
    'RadiologyFinding',
    'AnatomicalRegion',
    'ModalityType',
    'CHEST_XRAY_FINDINGS',
    'CHEST_ANATOMICAL_REGIONS',
]


# =============================================================================
# ENUMS
# =============================================================================

class ModalityType(str, Enum):
    """Imaging modality type."""
    CHEST_XRAY = "chest_xray"
    CT_CHEST = "ct_chest"
    CT_ABDOMEN = "ct_abdomen"
    MRI_BRAIN = "mri_brain"
    MAMMOGRAPHY = "mammography"


class RadiologyFinding(str, Enum):
    """Standard radiology findings for chest X-ray."""
    NORMAL = "normal"
    PNEUMONIA = "pneumonia"
    ATELECTASIS = "atelectasis"
    CARDIOMEGALY = "cardiomegaly"
    EFFUSION = "effusion"
    INFILTRATION = "infiltration"
    MASS = "mass"
    NODULE = "nodule"
    PNEUMOTHORAX = "pneumothorax"
    CONSOLIDATION = "consolidation"
    EDEMA = "edema"
    EMPHYSEMA = "emphysema"
    FIBROSIS = "fibrosis"
    PLEURAL_THICKENING = "pleural_thickening"
    HERNIA = "hernia"


class AnatomicalRegion(str, Enum):
    """Anatomical regions for chest imaging."""
    RIGHT_LUNG = "right_lung"
    LEFT_LUNG = "left_lung"
    RIGHT_UPPER_LOBE = "right_upper_lobe"
    RIGHT_MIDDLE_LOBE = "right_middle_lobe"
    RIGHT_LOWER_LOBE = "right_lower_lobe"
    LEFT_UPPER_LOBE = "left_upper_lobe"
    LEFT_LOWER_LOBE = "left_lower_lobe"
    HEART = "heart"
    MEDIASTINUM = "mediastinum"
    DIAPHRAGM = "diaphragm"
    COSTOPHRENIC_ANGLES = "costophrenic_angles"


# =============================================================================
# STANDARD DEFINITIONS
# =============================================================================

# Chest X-ray findings list
CHEST_XRAY_FINDINGS = [f.value for f in RadiologyFinding]

# Anatomical regions with expected attention areas
# Coordinates are normalized (0-1) for a standard PA chest X-ray
CHEST_ANATOMICAL_REGIONS = [
    AttentionRegion(
        name="right_lung",
        x_min=0.05, y_min=0.15, x_max=0.45, y_max=0.75,
        expected_for=["pneumonia", "atelectasis", "infiltration", "nodule", "mass"]
    ),
    AttentionRegion(
        name="left_lung",
        x_min=0.55, y_min=0.15, x_max=0.95, y_max=0.75,
        expected_for=["pneumonia", "atelectasis", "infiltration", "nodule", "mass"]
    ),
    AttentionRegion(
        name="heart",
        x_min=0.35, y_min=0.30, x_max=0.65, y_max=0.70,
        expected_for=["cardiomegaly", "edema"]
    ),
    AttentionRegion(
        name="costophrenic_angles",
        x_min=0.0, y_min=0.65, x_max=1.0, y_max=0.85,
        expected_for=["effusion", "consolidation"]
    ),
    AttentionRegion(
        name="mediastinum",
        x_min=0.35, y_min=0.10, x_max=0.65, y_max=0.50,
        expected_for=["mass", "lymphadenopathy"]
    ),
]

# Finding to expected region mapping
FINDING_REGION_MAP = {
    "pneumonia": ["right_lung", "left_lung"],
    "atelectasis": ["right_lung", "left_lung", "costophrenic_angles"],
    "cardiomegaly": ["heart"],
    "effusion": ["costophrenic_angles"],
    "infiltration": ["right_lung", "left_lung"],
    "mass": ["right_lung", "left_lung", "mediastinum"],
    "nodule": ["right_lung", "left_lung"],
    "pneumothorax": ["right_lung", "left_lung"],
    "consolidation": ["right_lung", "left_lung", "costophrenic_angles"],
    "edema": ["heart", "right_lung", "left_lung"],
    "emphysema": ["right_lung", "left_lung"],
    "fibrosis": ["right_lung", "left_lung"],
    "pleural_thickening": ["right_lung", "left_lung"],
    "normal": [],  # No specific region expected
}


# =============================================================================
# RADIOLOGY ADAPTER
# =============================================================================

class RadiologyAdapter:
    """
    Radiology-specific adapter for diagnostic governance.

    Provides:
    - Standard finding classifications
    - Attention validation against anatomical regions
    - Prior imaging comparison
    - Mock PACS integration for testing

    Example:
        >>> adapter = RadiologyAdapter()
        >>> classes = adapter.get_class_names()
        >>> is_valid, concerns = adapter.validate_attention(attention, "pneumonia")
    """

    def __init__(
        self,
        modality: ModalityType = ModalityType.CHEST_XRAY,
        anatomical_regions: Optional[List[AttentionRegion]] = None,
    ):
        """
        Initialize radiology adapter.

        Args:
            modality: Imaging modality type
            anatomical_regions: Custom anatomical regions (uses defaults if not provided)
        """
        self.modality = modality
        self.anatomical_regions = anatomical_regions or CHEST_ANATOMICAL_REGIONS

    def get_class_names(self) -> List[str]:
        """
        Get list of possible diagnostic classes.

        Returns:
            List of finding class names
        """
        if self.modality == ModalityType.CHEST_XRAY:
            return CHEST_XRAY_FINDINGS
        else:
            # Default to chest X-ray findings
            return CHEST_XRAY_FINDINGS

    def validate_attention(
        self,
        attention_map: Any,
        predicted_class: str,
    ) -> Tuple[bool, List[str]]:
        """
        Validate attention map for predicted class.

        Checks that model attention is in expected anatomical regions.

        Args:
            attention_map: 2D attention/saliency map (or mock data)
            predicted_class: The predicted finding class

        Returns:
            Tuple of (is_valid, list of concerns)
        """
        concerns: List[str] = []

        # Get expected regions for this finding
        expected_regions = FINDING_REGION_MAP.get(predicted_class.lower(), [])

        if not expected_regions:
            # No specific region expected (e.g., "normal")
            return True, []

        # MOCK: Simulate attention validation
        # In production, this would analyze actual attention maps
        mock_result = self._mock_attention_validation(predicted_class)

        if not mock_result["in_expected_region"]:
            concerns.append(
                f"Attention not focused on expected regions for {predicted_class}: "
                f"{', '.join(expected_regions)}"
            )

        if mock_result["edge_attention"] > 0.2:
            concerns.append("Significant attention on image edges (potential artifact)")

        if mock_result["outside_body"] > 0.1:
            concerns.append("Attention detected outside body region")

        is_valid = len(concerns) == 0
        return is_valid, concerns

    def compare_with_prior(
        self,
        current_prediction: str,
        prior_predictions: List[str],
    ) -> str:
        """
        Compare current finding with prior studies.

        Args:
            current_prediction: Current study's predicted finding
            prior_predictions: Findings from prior studies

        Returns:
            Comparison summary string
        """
        current_lower = current_prediction.lower()

        if not prior_predictions:
            return f"No prior studies available. Current finding: {current_prediction}."

        # Check if finding was present in any prior
        was_present = any(
            current_lower in p.lower() or p.lower() in current_lower
            for p in prior_predictions
        )

        if was_present:
            return f"{current_prediction} - STABLE compared to prior studies."
        else:
            # Check if prior was normal
            prior_normal = any("normal" in p.lower() for p in prior_predictions)
            if prior_normal:
                return f"{current_prediction} - NEW FINDING (prior was normal)."
            else:
                return f"{current_prediction} - different from prior findings ({', '.join(prior_predictions[:3])})."

    def get_mock_pacs_data(
        self,
        patient_id: str,
        study_type: str = "XR",
    ) -> Dict[str, Any]:
        """
        Get mock PACS data for testing.

        Args:
            patient_id: Patient identifier
            study_type: Study type code (XR, CT, MRI)

        Returns:
            Mock PACS response
        """
        # MOCK: Return simulated PACS data
        return {
            "patient_id": patient_id,
            "study_type": study_type,
            "prior_studies": [
                {
                    "study_id": "XR-2023-001",
                    "date": "2023-06-15",
                    "findings": ["normal"],
                    "impression": "Normal chest radiograph",
                },
                {
                    "study_id": "XR-2022-045",
                    "date": "2022-12-01",
                    "findings": ["normal"],
                    "impression": "No acute cardiopulmonary abnormality",
                },
            ],
            "clinical_history": "Cough for 3 days",
            "indication": "Rule out pneumonia",
        }

    def get_dicom_metadata(
        self,
        study_id: str,
    ) -> Dict[str, Any]:
        """
        Get mock DICOM metadata for testing.

        Args:
            study_id: Study identifier

        Returns:
            Mock DICOM metadata
        """
        # MOCK: Return simulated DICOM metadata
        return {
            "study_id": study_id,
            "modality": "CR",
            "body_part": "CHEST",
            "view_position": "PA",
            "image_size": [2048, 2048],
            "bits_stored": 12,
            "photometric_interpretation": "MONOCHROME2",
            "manufacturer": "Mock Medical Systems",
            "institution": "Test Hospital",
        }

    def _mock_attention_validation(self, predicted_class: str) -> Dict[str, Any]:
        """
        Generate mock attention validation results.

        In production, this would analyze actual attention maps.
        """
        # Simulate different validation results based on finding
        easy_findings = {"normal", "cardiomegaly"}
        hard_findings = {"subtle_nodule", "early_pneumonia"}

        if predicted_class.lower() in easy_findings:
            return {
                "in_expected_region": True,
                "expected_region_coverage": 0.7,
                "edge_attention": 0.05,
                "outside_body": 0.02,
            }
        elif predicted_class.lower() in hard_findings:
            return {
                "in_expected_region": False,
                "expected_region_coverage": 0.3,
                "edge_attention": 0.25,
                "outside_body": 0.15,
            }
        else:
            # Default reasonable results
            return {
                "in_expected_region": True,
                "expected_region_coverage": 0.5,
                "edge_attention": 0.1,
                "outside_body": 0.05,
            }

    def get_finding_description(self, finding: str) -> str:
        """Get clinical description of a finding."""
        descriptions = {
            "normal": "No acute cardiopulmonary abnormality",
            "pneumonia": "Infectious consolidation in lung parenchyma",
            "atelectasis": "Partial or complete collapse of lung or lobe",
            "cardiomegaly": "Enlarged cardiac silhouette",
            "effusion": "Fluid accumulation in pleural space",
            "infiltration": "Increased opacity suggesting inflammation or infection",
            "mass": "Discrete opacity > 3cm in diameter",
            "nodule": "Discrete opacity < 3cm in diameter",
            "pneumothorax": "Air in pleural space",
            "consolidation": "Airspace filled with fluid, pus, or cells",
            "edema": "Pulmonary vascular congestion",
            "emphysema": "Chronic destruction of alveoli",
            "fibrosis": "Scarring of lung tissue",
            "pleural_thickening": "Thickened pleural membrane",
        }
        return descriptions.get(finding.lower(), f"Finding: {finding}")
