import tensorflow as tf


def predict(
    model: tf.keras.Model,
    image_tensor: tf.Tensor,
    class_names: list[str]
) -> dict[str, float]:

    probabilities = model(
        image_tensor,
        training=False
    ).numpy()[0]

    return {
        name: float(probability) for name, probability in zip(class_names, probabilities, strict=True)
    }
