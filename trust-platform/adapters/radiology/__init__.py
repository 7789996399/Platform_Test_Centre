"""
TRUST Radiology Adapter

Domain-specific adapter for radiology AI governance including:
- Chest X-ray findings classification
- CT scan lesion detection
- Attention map validation for anatomical regions
- Prior imaging comparison
- PACS/DICOM integration (mock)
"""

from adapters.radiology.radiology_adapter import (
    RadiologyAdapter,
    RadiologyFinding,
    AnatomicalRegion,
    ModalityType,
    CHEST_XRAY_FINDINGS,
    CHEST_ANATOMICAL_REGIONS,
)

__all__ = [
    'RadiologyAdapter',
    'RadiologyFinding',
    'AnatomicalRegion',
    'ModalityType',
    'CHEST_XRAY_FINDINGS',
    'CHEST_ANATOMICAL_REGIONS',
]
