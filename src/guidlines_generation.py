from typing import Dict, List
from PIL import Image

import numpy as np
import torch
from lang_sam import LangSAM
from PIL import Image, ImageOps


class CompositionValidator:
    """Maintains 5:4 aspect ratio through padding and applies composition rules"""

    def __init__(
        self,
        max_width: int = 1000,
        max_coverage: float = 0.6,
        rule_threshold: float = 0.15,
    ):
        self.max_width = max_width
        self.max_coverage = max_coverage
        self.rule_threshold = rule_threshold

    def _get_food_center(self, mask: Image.Image) -> tuple:
        """Get center coordinates of food area from mask"""
        if not isinstance(mask, Image.Image):
            raise ValueError("Mask must be a PIL Image object")

        np_mask = np.array(mask.convert("L"))  # Ensure grayscale
        ys, xs = np.where(np_mask > 128)  # Threshold at midpoint
        if len(xs) == 0:
            return (
                mask.width // 2,
                mask.height // 2,
            )  # Default to center if no food found
        return (int(np.mean(xs)), int(np.mean(ys)))

    def _calculate_padding(self, orig_size: tuple, food_center: tuple) -> tuple:
        """Calculate padding to achieve 5:4 ratio and rule of thirds positioning"""
        orig_width, orig_height = orig_size

        # Calculate target dimensions
        target_width = min(orig_width, self.max_width)
        target_height = int(target_width * 4 / 5)

        # Determine padding needed
        if orig_height < target_height:
            # Vertical padding with rule of thirds
            pad_total = target_height - orig_height
            optimal_y = (
                target_height // 3
                if food_center[1] < orig_height / 2
                else 2 * target_height // 3
            )
            pad_top = max(0, min(pad_total, optimal_y - food_center[1]))
            pad_bottom = pad_total - pad_top
            return (0, pad_top, 0, pad_bottom)
        else:
            # Horizontal padding with rule of thirds
            target_width = int(orig_height * 5 / 4)
            pad_total = min(target_width, self.max_width) - orig_width
            optimal_x = (
                target_width // 3
                if food_center[0] < orig_width / 2
                else 2 * target_width // 3
            )
            pad_left = max(0, min(pad_total, optimal_x - food_center[0]))
            pad_right = pad_total - pad_left
            return (pad_left, 0, pad_right, 0)

    def apply_composition(self, image: Image.Image, mask: Image.Image) -> tuple:
        """Main composition method"""
        # Type validation
        if not isinstance(image, Image.Image) or not isinstance(mask, Image.Image):
            raise ValueError(
                f"Both image and mask must be PIL Image objects, {isinstance(image, Image.Image) , isinstance(mask, Image.Image)}"
            )

        # Ensure mask is binary (0 or 255)
        mask = mask.convert("L").point(lambda x: 255 if x > 128 else 0)

        # Get initial food position
        food_center = self._get_food_center(mask)

        # Calculate and apply padding
        padding = self._calculate_padding(image.size, food_center)
        processed_img = ImageOps.expand(image, padding, fill="white")
        processed_mask = ImageOps.expand(mask, padding, fill=255)

        # Verify food coverage
        coverage = np.mean(np.array(processed_mask) > 128) / 255
        if coverage > self.max_coverage:
            print(
                f"Warning: Food coverage {coverage:.0%} exceeds limit {self.max_coverage:.0%}"
            )

        return processed_img, processed_mask