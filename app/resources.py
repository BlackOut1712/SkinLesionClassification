import streamlit as st

from src.model_loader import load_model, load_class_names, load_gradcam


@st.cache_data
def get_class_names():
    return load_class_names()

@st.cache_resource
def get_model():
    return load_model()

@st.cache_resource
def get_gradcam():
    return load_gradcam()