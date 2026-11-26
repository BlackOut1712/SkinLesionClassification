import streamlit as st

from views.analysis import render_analysis_page
from components.image_uploader import render_image_upload


def main():
    st.set_page_config(
        page_title="Analiza zmian skórnych",
        page_icon="🔬",
        layout="wide"
    )

    render_analysis_page()

if __name__ == "__main__":
    main()