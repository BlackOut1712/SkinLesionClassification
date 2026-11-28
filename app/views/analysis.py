import streamlit as st
import numpy as np

from app.components.image_uploader import render_image_upload
from app.components.prediction_results import render_prediction_results
from app.components.gradcam_view import render_gradcam_controls, render_gradcam_results
from app.resources import get_class_names, get_model, get_gradcam
from app.state import get_heatmap, save_heatmap, save_analysis, initialize_state, reset_analysis, get_analysis

from src.services.analysis_service import analyze_image, explain_prediction, prepare_gradcam_visualization, validate_image
from src.schemas.analysis_result import AnalysisResult

def get_or_generate_heatmap(
    result: AnalysisResult,
    target_class: str,
    class_names: list[str]
) -> np.ndarray:

    heatmap = get_heatmap(target_class)

    if heatmap is None:

        with st.spinner("Generating Grad-CAM..."):

            heatmap = explain_prediction(
                result=result,
                gradcam=get_gradcam(),
                target_class=target_class,
                class_names=class_names
            )

        save_heatmap(target_class, heatmap)

    return heatmap


def render_analysis_page() -> None:

    initialize_state()

    st.title("Skin lesion analysis")

    image_bytes = render_image_upload()

    class_names = get_class_names()

    if st.button(
        "Analyse image",
        disabled=image_bytes is None
    ):

        reset_analysis()

        with st.spinner("Analysing image..."):

            result = analyze_image(
                image_bytes=image_bytes,
                model=get_model(),
                class_names=class_names
            )

            save_analysis(result)

        st.session_state.gradcam_target_class = (
            result.predicted_class
        )

    result = get_analysis()

    if result is not None and image_bytes is not None:

        st.image(result.image)

        render_prediction_results(
            result.probabilities
        )

        render_gradcam_section(
            result,
            class_names
        )

def render_gradcam_section(
    result: AnalysisResult,
    class_names: list[str]
) -> None:

    st.subheader("Model explanation — Grad-CAM")

    target_class, alpha, colormap = (
        render_gradcam_controls(
            class_names,
            result.predicted_class
        )
    )

    heatmap = get_or_generate_heatmap(
        result,
        target_class,
        class_names
    )

    colored_heatmap, overlay = (
        prepare_gradcam_visualization(
            image=result.image,
            heatmap=heatmap,
            alpha=alpha,
            colormap=colormap
        )
    )

    render_gradcam_results(
        image=result.image,
        heatmap=colored_heatmap,
        overlay=overlay,
        target_class=target_class
    )
