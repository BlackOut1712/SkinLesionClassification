from io import BytesIO
import pytest
from PIL import Image

from src.validation import validate_image


def test_valid_image():
    buffer = BytesIO()

    Image.new("RGB", (100, 100)).save(
        buffer,
        format="JPEG"
    )
    result = validate_image(buffer.getvalue())
    
    assert isinstance(result, Image.Image)


def test_empty_file():
    with pytest.raises(ValueError):
        validate_image(b"")


def test_invalid_image():
    with pytest.raises(ValueError):
        validate_image(b"This is not an image.")