import cv2
import numpy as np
from PIL import Image


def colorize_heatmap(
    heatmap: np.ndarray,
    target_size: tuple[int, int],
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:

    resized_heatmap = cv2.resize(
        heatmap,
        target_size,
        interpolation=cv2.INTER_LINEAR
    )

    heatmap_uint8 = np.uint8(
        np.clip(resized_heatmap, 0, 1) * 255
    )

    colored_heatmap = cv2.applyColorMap(
        heatmap_uint8,
        colormap
    )

    return cv2.cvtColor(
        colored_heatmap,
        cv2.COLOR_BGR2RGB
    )

def overlay_heatmap(
    image: Image.Image,
    colored_heatmap: np.ndarray,
    alpha: float = 0.2
) -> np.ndarray:

    if not 0 <= alpha <= 1:
        raise ValueError("Alpha must be between 0 and 1.")

    image_array = np.asarray(
        image.convert("RGB")
    )

    return cv2.addWeighted(
        image_array,
        1 - alpha,
        colored_heatmap,
        alpha,
        0
    )