import tensorflow as tf
import numpy as np
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.scores import CategoricalScore
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear

def build_gradcam(model: tf.keras.Model) -> Gradcam:

    flat_model = flatten_model(model)

    return Gradcam(
        flat_model,
        model_modifier=ReplaceToLinear(),
        clone=True
    )


def generate_heatmap(
    gradcam,
    input_tensor,
    class_idx: int
) -> np.ndarray:

    cam = gradcam(
        CategoricalScore(class_idx),
        input_tensor,
        penultimate_layer="top_activation",
        seek_penultimate_conv_layer=False,
        expand_cam=False
    )

    return np.asarray(cam[0], dtype=np.float32)


def flatten_model(model: tf.keras.Model) -> tf.keras.Model:

    inner_model = model.get_layer("functional")
    backbone = inner_model.get_layer("efficientnetb0")

    x = backbone.output

    for layer in inner_model.layers:
        if isinstance(layer, (
            tf.keras.layers.InputLayer,
            tf.keras.Model
        )):
            continue

        x = layer(x)

    return tf.keras.Model(
        inputs=backbone.input,
        outputs=x,
        name="flattened_model"
    )