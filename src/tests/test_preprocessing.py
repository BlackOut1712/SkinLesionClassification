import tensorflow as tf
from PIL import Image
from src.preprocessing import preprocess_image
from src.config import *

def test_output_shape():
    image = Image.new("RGB", (1000, 500))
    result = preprocess_image(image)
    assert result.shape==(1, *IMAGE_SIZE, NUM_CHANNELS)

def test_output_dtype():
    image = Image.new("RGB", (1000, 500))
    result = preprocess_image(image)
    assert result.dtype == tf.float32

def test_convertion():
    image = Image.new("L", IMAGE_SIZE)
    result = preprocess_image(image)
    assert result.shape == (1, *IMAGE_SIZE, NUM_CHANNELS)

def test_pixel_range():
    image = Image.new("RGB", IMAGE_SIZE, color=(0, 128, 255))
    result = preprocess_image(image)
    assert float(tf.reduce_min(result)) >=0
    assert float(tf.reduce_max(result)) <=255

    tf.debugging.assert_near(
        result,
        tf.ones_like(result) * 255
    )
