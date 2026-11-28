import streamlit as st

from views.analysis import render_analysis_page
from src.config import MODEL_PATH

def main():
    st.set_page_config(
        page_title="Analiza zmian skórnych",
        page_icon="🔬",
        layout="wide"
    )

    render_analysis_page()

if __name__ == "__main__":
    main()