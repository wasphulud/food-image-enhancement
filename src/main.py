import argparse
import gc
import os

import torch
from PIL import Image

from guidlines_generation import CompositionValidator
from image_understanding import ImageUnderstanding
from inpainting import get_inpainting_strategy


def process_image(image_path: str, prompt: str) -> Image.Image:
    """Processes a single image by applying image understanding, composition validation, and inpainting.

    Args:
        image_path (str): The path to the input image.
        prompt (str): The prompt for inpainting.

    Returns:
        Image.Image: The processed image after inpainting.
    """
    # Initialize all variables to None to avoid NameError
    orig_image = None
    masks = None
    composite_mask = None
    final_img = None
    final_mask = None
    understanding = None
    composer = None
    strategy = None
    result = None

    try:
        # --- IMAGE UNDERSTANDING ---
        orig_image = Image.open(image_path).convert("RGB")
        understanding = ImageUnderstanding()
        masks = understanding.predict_masks(orig_image, ["food", "snack", "vegetables"])
        composite_mask = understanding.create_composite_mask(masks)

        # --- COMPOSITION VALIDATION ---
        composer = CompositionValidator()
        final_img, final_mask = composer.apply_composition(orig_image, composite_mask)

        # --- INPAINTING ---
        inpainting_model = "stabilityai/stable-diffusion-2-inpainting"
        strategy = get_inpainting_strategy(inpainting_model)
        result = strategy.inpaint(prompt=prompt, image=final_img, mask=final_mask)

        return result

    finally:
        # Explicitly delete each variable by name
        variables_to_delete = [
            "orig_image",
            "masks",
            "composite_mask",
            "final_img",
            "final_mask",
            "understanding",
            "composer",
            "strategy",
            "result",
        ]
        for var_name in variables_to_delete:
            try:
                exec(f"del {var_name}", globals(), locals())
            except NameError:
                pass

        # Force cleanup if objects have specific methods (e.g., for GPU release)
        if understanding is not None:
            understanding.cleanup()
        if strategy is not None:
            strategy.cleanup()  # Hypothetical method

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def process_images(
    input_dir: str, output_base_dir: str, experiment_name: str, prompt: str
) -> None:
    """Processes all images in a directory and saves results to a new subfolder, skipping already processed files.

    Args:
        input_dir (str): The directory containing input images.
        output_base_dir (str): The base directory for output results.
        experiment_name (str): The name of the experiment for the output subfolder.
        prompt (str): The prompt for inpainting.
    """
    output_dir = os.path.join(output_base_dir, experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    for filename in sorted(os.listdir(input_dir)):
        name, ext = os.path.splitext(filename)
        output_filename = f"{name}_processed{ext}"
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, output_filename)

        # Skip if already processed
        if os.path.exists(output_path):
            print(f"Skipping {filename}: already processed → {output_path}")
            continue

        try:
            processed_img = process_image(input_path, prompt=prompt)
            if processed_img is None:
                print(f"No result for {filename}, skipping.")
                continue

            processed_img.save(output_path)
            print(f"Processed: {filename} → {output_path}")

        except (IOError, OSError, ValueError) as e:
            print(f"Skipping {filename}: {e}")

        # Extra safety—clear caches again between files
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process images and save results with experiment naming."
    )
    parser.add_argument(
        "--input", required=True, help="Input directory containing images"
    )
    parser.add_argument("--output", required=True, help="Base output directory")
    parser.add_argument(
        "--experiment", required=True, help="Experiment name for subfolder"
    )
    args = parser.parse_args()

    prompt = (
        "the dish is placed on a circular white plate. "
        "the plate is placed on a marble table. Clean table"
    )
    process_images(
        input_dir=args.input,
        output_base_dir=args.output,
        experiment_name=args.experiment,
        prompt=prompt,
    )
