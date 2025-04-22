from PIL import Image
from image_understanding import ImageUnderstanding
from guidlines_generation import CompositionValidator
from inpainting import InpaintingPipeline, YahooInpaintingPipeline
import argparse
import os
import torch
import gc

def process_image(image_path: str, prompt: str):
    # Load original image
    orig_image = Image.open(image_path).convert("RGB")
    result = None

    try:
        # --- IMAGE UNDERSTANDING ---
        understanding = ImageUnderstanding()
        masks = understanding.predict_masks(orig_image, ["food", "snack", "vegetables", "plate"])
        composite_mask = understanding.create_composite_mask(masks)
        understanding.cleanup()
        
        # --- COMPOSITION VALIDATION ---
        composer = CompositionValidator()
        final_img, final_mask = composer.apply_composition(orig_image, composite_mask)
        
        # --- INPAINTING ---
        inpainting_model = "stabilityai/stable-diffusion-2-inpainting"
        inpainting_model = "yah"
        if inpainting_model[:2] =="st":
            inpainter = InpaintingPipeline(inpainting_model)
        else:
            inpainter = YahooInpaintingPipeline()
        result = inpainter(prompt=prompt, image=final_img, mask=final_mask)
        inpainter.cleanup()
        return result

    finally:
        # Delete all large objects and free GPU
        for obj in [
            orig_image, masks, composite_mask,
            final_img, final_mask,
            understanding, composer, inpainter,
            result
        ]:
            try:
                del obj
            except NameError:
                pass

        # force Python GC
        gc.collect()
        # force CUDA to free everything
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def process_images(input_dir: str, output_base_dir: str, experiment_name: str, prompt: str):
    """
    Process all images in a directory and save results to a new subfolder,
    skipping already processed files.
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
    parser.add_argument("--input", required=True, help="Input directory containing images")
    parser.add_argument("--output", required=True, help="Base output directory")
    parser.add_argument("--experiment", required=True, help="Experiment name for subfolder")
    args = parser.parse_args()

    prompt = (
        "the dish is placed on a circular white plate. "
        "the plate is placed on a wooden table. Clean table"
    )
    process_images(
        input_dir=args.input,
        output_base_dir=args.output,
        experiment_name=args.experiment,
        prompt=prompt,
    )
