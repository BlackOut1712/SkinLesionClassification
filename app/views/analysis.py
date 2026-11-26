import streamlit as st

from app.components.image_uploader import render_image_upload
from app.resources import get_class_names, get_model
from app.components.prediction_results import render_prediction_results
from src.preprocessing import preprocess_image
from src.inference import predict

def render_analysis_page():
    st.title("Skin lesion analysis")

    st.write(
        "System wspomagania klasyfikacji zmian skórnych "
        "z wykorzystaniem metod uczenia głębokiego."
    )

    image = render_image_upload()

    if st.button("Analyze",
        disabled=image is None
    ):
        with st.spinner("Analysis in progress..."):
            model = get_model()
            class_names = get_class_names()

            input_tensor = preprocess_image(image)
            probabilities = predict(model, image_tensor=input_tensor, class_names=class_names)

        st.write(probabilities)
        render_prediction_results(probabilities)
        