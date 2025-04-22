import torch
from diffusers import DiffusionPipeline, StableDiffusionInpaintPipeline
from PIL import Image


class InpaintingPipeline:
    """Handles image inpainting using ControlNet."""

    def __init__(
        self, model_name: str = "yahoo-inc/photo-background-generation"
    ) -> None:
        """Initializes the InpaintingPipeline with a specified model name.

        Args:
            model_name (str): The name of the model to use for inpainting.
        """
        self.pipe = self._initialize_pipeline(model_name)
        self._configure_pipeline()

    def _initialize_pipeline(self, model_name: str) -> StableDiffusionInpaintPipeline:
        """Initializes the inpainting pipeline with the specified model.

        Args:
            model_name (str): The name of the model to load.

        Returns:
            StableDiffusionInpaintPipeline: The initialized inpainting pipeline.
        """
        return StableDiffusionInpaintPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        ).to("cuda" if torch.cuda.is_available() else "cpu")

    def _configure_pipeline(self) -> None:
        """Configures the pipeline settings."""
        self.pipe.set_progress_bar_config(disable=True)

    def __call__(
        self, prompt: str, image: Image.Image, mask: Image.Image, **kwargs
    ) -> Image.Image:
        """Executes inpainting with automatic mixed precision.

        Args:
            prompt (str): The prompt for inpainting.
            image (Image.Image): The input image to be inpainted.
            mask (Image.Image): The mask image indicating areas to inpaint.
            **kwargs: Additional arguments for the pipeline.

        Returns:
            Image.Image: The inpainted image.
        """
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

    def cleanup(self) -> None:
        """Releases pipeline resources."""
        del self.pipe
        torch.cuda.empty_cache()


class YahooInpaintingPipeline:
    """Handles image inpainting using ControlNet."""

    def __init__(
        self, model_name: str = "yahoo-inc/photo-background-generation"
    ) -> None:
        """Initializes the YahooInpaintingPipeline with a specified model name.

        Args:
            model_name (str): The name of the model to use for inpainting.
        """
        self.pipe = self._initialize_pipeline(model_name)
        self._configure_pipeline()

    def _initialize_pipeline(self, model_name: str) -> DiffusionPipeline:
        """Initializes the Yahoo inpainting pipeline with the specified model.

        Args:
            model_name (str): The name of the model to load.

        Returns:
            DiffusionPipeline: The initialized Yahoo inpainting pipeline.
        """
        return DiffusionPipeline.from_pretrained(
            model_name,
            custom_pipeline=model_name,
            torch_dtype=torch.float16,
        ).to("cuda" if torch.cuda.is_available() else "cpu")

    def _configure_pipeline(self) -> None:
        """Configures the pipeline settings."""
        self.pipe.set_progress_bar_config(disable=True)

    def __call__(
        self,
        prompt: str,
        image: Image.Image,
        mask: Image.Image,
        progress_bar: bool = True,
        *args,
        **kwargs
    ) -> Image.Image:
        """Allows the pipeline to be called like a function.

        Args:
            prompt (str): The prompt for inpainting.
            image (Image.Image): The input image to be inpainted.
            mask (Image.Image): The mask image indicating areas to inpaint.
            progress_bar (bool): Whether to show a progress bar.
            *args: Additional positional arguments for the pipeline.
            **kwargs: Additional keyword arguments for the pipeline.

        Returns:
            Image.Image: The inpainted image.
        """
        with torch.autocast("cuda"):
            if not progress_bar:
                self.pipe.set_progress_bar_config(disable=True)
            return self.pipe(
                prompt=prompt,
                image=image,
                mask_image=mask,
                control_image=mask,
                num_images_per_prompt=1,
                num_inference_steps=20,
                guess_mode=False,
                controlnet_conditioning_scale=1.0,
            ).images[0]

    def cleanup(self) -> None:
        """Releases pipeline resources."""
        del self.pipe
        torch.cuda.empty_cache()
