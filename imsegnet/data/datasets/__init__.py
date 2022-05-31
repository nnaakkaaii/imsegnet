from .coco import COCODataset
from .pascal import PascalDataset


datasets = {
    'coco': COCODataset,
    'pascal': PascalDataset,
}

__all__ = [
    'COCODataset',
    'PascalDataset',
]
