from io import BytesIO
from PIL import Image, UnidentifiedImageError
from PIL.Image import DecompressionBombError

from src.config import (
    ALLOWED_IMAGE_FORMATS,
    MAX_FILE_SIZE_BYTES,
    MAX_IMAGE_PIXELS
)

def validate_image(image_bytes: bytes) -> Image.Image:
    if not image_bytes:
        raise ValueError("The uploaded file is empty.")

    if len(image_bytes) > MAX_FILE_SIZE_BYTES:
        raise ValueError("The uploaded file is too big.")

    try:
        with Image.open(BytesIO(image_bytes)) as image:

            if image.format not in ALLOWED_IMAGE_FORMATS:
                raise ValueError("Unsupported image format.")

            width, height = image.size

            if width*height > MAX_IMAGE_PIXELS:
                raise ValueError("Image resolution is too big.")

            image.verify() #Check if image is not corrupted

        with Image.open(BytesIO(image_bytes)) as image:
            image.load()
            return image

    except(
        UnidentifiedImageError,
        OSError,
        DecompressionBombError
    ) as error:
        raise ValueError(
            "Invalid or corrupted image"
        ) from error