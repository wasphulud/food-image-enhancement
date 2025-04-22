import torch
from diffusers import DiffusionPipeline, StableDiffusionInpaintPipeline
from PIL import Image
from diffusers import DiffusionPipeline

class InpaintingPipeline:
    """Handles image inpainting using ControlNet"""

    def __init__(self, model_name: str = "yahoo-inc/photo-background-generation"):
        self.pipe = self._initialize_pipeline(model_name)
        self._configure_pipeline()

    def _initialize_pipeline(self, model_name: str):
        return StableDiffusionInpaintPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        ).to("cuda" if torch.cuda.is_available() else "cpu")

    def _configure_pipeline(self):
        self.pipe.set_progress_bar_config(disable=True)

    def __call__(
        self, prompt: str, image: Image.Image, mask: Image.Image, **kwargs
    ) -> Image.Image:
        """Executes inpainting with automatic mixed precision"""
        with torch.autocast(self.pipe.device.type):
            return self.pipe(
                prompt=prompt,
                negative_prompt="no plastic container, no text, no background people, no restaurent background",
                image=image,
                mask_image=mask,
                num_images_per_prompt=1,
                num_inference_steps=20,
                **kwargs
            ).images[0]

    def cleanup(self):
        """Releases pipeline resources"""
        del self.pipe
        torch.cuda.empty_cache()

class YahooInpaintingPipeline:
    """Handles image inpainting using ControlNet"""

    def __init__(self, model_name: str = "yahoo-inc/photo-background-generation"):
        self.pipe = self._initialize_pipeline(model_name)
        self._configure_pipeline()

    def _initialize_pipeline(self, model_name: str):
        return DiffusionPipeline.from_pretrained(model_name , custom_pipeline=model_name,
            torch_dtype=torch.float16,
        ).to("cuda" if torch.cuda.is_available() else "cpu")

    def _configure_pipeline(self):
        self.pipe.set_progress_bar_config(disable=True)

    def __call__(self, prompt, image, mask, progress_bar=True, *args, **kwargs):
        """Allows the pipeline to be called like a function."""
        with torch.autocast("cuda"):
            if not progress_bar:
                self.pipe.set_progress_bar_config(disable=True)
            return self.pipe(prompt=prompt, image=image, mask_image=mask, control_image=mask, num_images_per_prompt=1, num_inference_steps=20, guess_mode=False, controlnet_conditioning_scale=1.0).images[0]


    def cleanup(self):
        """Releases pipeline resources"""
        del self.pipe
        torch.cuda.empty_cache()
