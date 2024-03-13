from PIL import Image
from pillow_heif import register_heif_opener
import os

import config


def resize_and_convert(filename, size=config.IMAGE_SIZE):
    if filename.lower().endswith(".heic"):
        register_heif_opener()

    image = Image.open(filename)
    image = image.convert("RGB")
    image = image.resize((size, size), Image.BILINEAR)

    return image


def resize_and_save(filename, output_dir, out_filename, size=config.IMAGE_SIZE):
    image = resize_and_convert(filename, size)
    image.save(os.path.join(output_dir, f"{out_filename}.jpg"))
