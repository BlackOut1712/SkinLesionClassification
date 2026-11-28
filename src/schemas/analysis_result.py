from dataclasses import dataclass

import tensorflow as tf
from PIL import Image


@dataclass
class AnalysisResult:

    image: Image.Image
    input_tensor: tf.Tensor
    probabilities: dict[str, float]

    @property
    def predicted_class(self) -> str:
        return max(
            self.probabilities,
            key=self.probabilities.get
        )

    @property
    def best_score(self) -> float:
        return self.probabilities[self.predicted_class]