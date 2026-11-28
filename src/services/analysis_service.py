import tensorflow as tf
import numpy as np
from PIL import Image

from src.gradcam import generate_heatmap
from src.validation import validate_image
from src.preprocessing import preprocess_image
from src.inference import predict
from src.schemas.analysis_result import AnalysisResult
from src.visualization import colorize_heatmap, overlay_heatmap



def analyze_image(
    image_bytes: bytes,
    model: tf.keras.Model,
    class_names: list[str]
) -> AnalysisResult:

    image = validate_image(image_bytes)

    image_tensor = preprocess_image(image)

    probabilities = predict(
        model,
        image_tensor,
        class_names
    )

    return AnalysisResult(
        image=image,
        input_tensor=image_tensor,
        probabilities=probabilities
    )

def explain_prediction(
    result: AnalysisResult,
    gradcam,
    target_class: str,
    class_names: list[str]
) -> np.ndarray:

    class_idx = class_names.index(target_class)

    return generate_heatmap(
        gradcam=gradcam,
        input_tensor=result.input_tensor,
        class_idx=class_idx
    )

def prepare_gradcam_visualization(
    image: Image.Image,
    heatmap: np.ndarray,
    alpha: float,
    colormap: int
) -> tuple[np.ndarray, np.ndarray]:

    colored_heatmap = colorize_heatmap(
        heatmap=heatmap,
        target_size=image.size,
        colormap=colormap
    )

    overlay = overlay_heatmap(
        image=image,
        colored_heatmap=colored_heatmap,
        alpha=alpha
    )

    return colored_heatmap, overlay