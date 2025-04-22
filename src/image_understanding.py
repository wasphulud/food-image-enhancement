from typing import Dict, List

import numpy as np
import torch
from lang_sam import LangSAM
from PIL import Image, ImageOps


class ImageUnderstanding:
    """Handles food object detection and masking using a specified model."""

    def __init__(self, sam_type: str = "sam2.1_hiera_large") -> None:
        """Initializes the ImageUnderstanding class with a specified model type.

        Args:
            sam_type (str): The type of the LangSAM model to use.
        """
        self.model = LangSAM(sam_type)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def predict_masks(
        self, image: Image.Image, prompts: List[str]
    ) -> List[Dict[str, np.ndarray]]:
        """Generates masks for given text prompts.

        Args:
            image (Image.Image): The input image for which masks are to be generated.
            prompts (List[str]): A list of text prompts for mask generation.

        Returns:
            List[Dict[str, np.ndarray]]: A list of dictionaries containing mask data for each prompt.
        """
        results = []
        for prompt in prompts:
            result = self.model.predict([image], [prompt])
            if not isinstance(result[0]["masks"], list):
                results.extend(result)
        return results

    def create_composite_mask(
        self, results: List[Dict[str, np.ndarray]]
    ) -> Image.Image:
        """Creates a composite mask from multiple detection results.

        Args:
            results (List[Dict[str, np.ndarray]]): A list of detection results containing masks.

        Returns:
            Image.Image: A composite mask image created from the input results.
        """
        combined_mask = (
            sum((result["masks"].sum(axis=0) > 0 for result in results)) == 0
        )
        return Image.fromarray(combined_mask.astype(np.uint8) * 255)

    def cleanup(self) -> None:
        """Releases model resources."""
        del self.model
        torch.cuda.empty_cache()
