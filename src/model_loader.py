import json
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import Model

from src.config import (
    MODEL_PATH,
    CLASS_NAMES_PATH,
    IMAGE_SIZE,
    NUM_CHANNELS,
    NUM_CLASSES
)

def load_model(model_path: Path=MODEL_PATH) -> tf.keras.Model:
    return tf.keras.models.load_model(model_path, compile=False)

def load_class_names(class_names_path: Path=CLASS_NAMES_PATH) -> list[str]:
    with open(class_names_path, encoding='utf-8') as file:
        return json.load(file)

def validate_model(model: tf.keras.Model,
                   class_names: list[str]) -> None:
    expected_input_shape = (*IMAGE_SIZE, NUM_CHANNELS)

    if tuple(model.input_shape[1:]) != expected_input_shape:
        raise ValueError("[Model validation failed!] Unexpected model input shape.")

    if model.output_shape[-1] != NUM_CLASSES:
        raise ValueError(
            "[Model validation failed!] Unexpected number of model outputs."
        )

    if len(class_names) != NUM_CLASSES:
        raise ValueError(
            "[Model validation failed!] Incorrect number of class names."
        )

    if len(set(class_names)) != len(class_names):
        raise ValueError(
            "[Model validation failed!] Duplicate class names."
        )
