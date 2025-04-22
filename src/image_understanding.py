from typing import Dict, List

import numpy as np
import torch
from lang_sam import LangSAM
from PIL import Image, ImageOps
from PIL import Image


class ImageUnderstanding:
    """Handles food object detection and masking"""

    def __init__(self, sam_type: str = "sam2.1_hiera_large"):
        self.model = LangSAM(sam_type)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def predict_masks(self, image: Image.Image, prompts: List[str]) -> List[Dict]:
        """Generates masks for given text prompts"""
        results = []
        for prompt in prompts:
            result = self.model.predict([image], [prompt])
            if not isinstance(result[0]["masks"], list):
                results.extend(result)
        return results

    def create_composite_mask(self, results: List[Dict]) -> Image.Image:
        """Creates a composite mask from multiple detection results"""
        combined_mask = (
            sum((result["masks"].sum(axis=0) > 0 for result in results)) == 0
        )
        return Image.fromarray(combined_mask.astype(np.uint8) * 255)

    def cleanup(self):
        """Releases model resources"""
        del self.model
        torch.cuda.empty_cache()
