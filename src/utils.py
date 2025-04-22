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

import os

def create_svf_gallery_md():
    svg_dir = "../comparisons_svg"
    output_file = "results/svg_gallery.md"
    svg_files = sorted(f for f in os.listdir(svg_dir) if f.endswith(".svg"))

    with open(output_file, "w") as f:
        f.write("# SVG Gallery\n\n")
        f.write('<div style="display: flex; flex-wrap: wrap; gap: 10px;">\n')
        for svg in svg_files:
            f.write(f'  <img src="{svg_dir}/{svg}" width="100"/>\n')
        f.write('</div>\n')
