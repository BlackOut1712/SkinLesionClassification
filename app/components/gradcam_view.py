import cv2
import streamlit as st
import numpy as np

from PIL import Image


COLORMAPS = {
    "JET": cv2.COLORMAP_JET,
    "TURBO": cv2.COLORMAP_TURBO,
    "VIRIDIS": cv2.COLORMAP_VIRIDIS,
    "INFERNO": cv2.COLORMAP_INFERNO
}


def render_gradcam_controls(
    class_names: list[str],
    predicted_class: str
) -> tuple[str, float, int]:

    target_class = st.selectbox(
        "Target class",
        options=class_names,
        index=class_names.index(predicted_class),
        key="gradcam_target_class"
    )

    col1, col2 = st.columns(2)

    with col1:
        alpha = st.slider(
            "Heatmap opacity",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.05
        )

    with col2:
        colormap = st.selectbox(
            "Colormap",
            options=list(COLORMAPS.keys())
        )

    return target_class, alpha, COLORMAPS[colormap]


def render_gradcam_results(
    image: Image.Image,
    heatmap: np.ndarray,
    overlay: np.ndarray,
    target_class: str
) -> None:

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(
            image,
            caption="Original image"
        )

    with col2:
        st.image(
            heatmap,
            caption="Activation heatmap"
        )

    with col3:
        st.image(
            overlay,
            caption=f"Grad-CAM — {target_class.upper()}"
        )

    st.caption(
        "Grad-CAM highlights image regions contributing "
        "to the selected class score. It does not provide "
        "lesion segmentation or a medical diagnosis."
    )