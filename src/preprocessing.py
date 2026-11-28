import numpy as np
import tensorflow as tf

from PIL import Image

from src.config import (
    IMAGE_SIZE
)

def preprocess_image(image: Image.Image) -> tf.Tensor:
    image = image.convert("RGB")

    image_tensor = tf.convert_to_tensor(
        np.asanyarray(image),
        dtype=tf.float32
    )

    image_tensor = tf.image.resize(
        image_tensor,
        IMAGE_SIZE,
    )

    image_tensor = tf.expand_dims(image_tensor, axis=0)

    return image_tensor