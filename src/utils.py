from PIL import Image


def resize_to_aspect_ratio(
    image: Image.Image,
    aspect_ratio: tuple = (5, 4),
    resample: Image.Resampling = Image.Resampling.LANCZOS,
) -> Image.Image:
    """Resizes image to maintain original width while forcing 5:4 aspect ratio."""
    original_width = image.width
    target_height = int(original_width * aspect_ratio[1] / aspect_ratio[0])
    return image.resize((original_width, target_height), resample=resample)

