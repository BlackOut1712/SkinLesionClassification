import streamlit as st

from app.state import reset_analysis
from src.config import MAX_FILE_SIZE_MB,  ALLOWED_IMAGE_FORMATS

def render_image_upload() -> bytes | None:
    data = st.file_uploader(
        "Upload a lesion image.",
        type= ALLOWED_IMAGE_FORMATS,
        max_upload_size= MAX_FILE_SIZE_MB,
        accept_multiple_files= False,
        key="image_uploader",
        on_change=reset_analysis,
        width="stretch"
    )

    if data is None:
        return None

    return data.getvalue()