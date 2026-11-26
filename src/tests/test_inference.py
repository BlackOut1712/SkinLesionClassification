import numpy as np
import tensorflow as tf

from src.model_loader import load_model, load_class_names
from src.inference import predict


def test_prediction():
    model = load_model()
    class_names = load_class_names()

    image_tensor = tf.zeros(
        (1, 224, 224, 3),
        dtype=tf.float32
    )

    result = predict(
        model,
        image_tensor,
        class_names
    )

    probabilities = list(result.values())

    assert len(probabilities) == 7

    assert np.all(np.isfinite(probabilities))
    
    assert all(0 <= p <= 1 for p in probabilities )

    assert np.isclose(sum(probabilities), 1.0, atol=1e-5)