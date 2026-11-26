import tensorflow as tf

def build_gradcam_model(
    model: tf.keras.Model,
    target_layer_name: str
) -> tf.keras.Model:
    ...