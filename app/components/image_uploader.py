import streamlit as st

from PIL import Image
from io import BytesIO

from src.config import MAX_FILE_SIZE_MB,  ALLOWED_IMAGE_FORMATS
from src.validation import validate_image

def render_image_upload():
    data = st.file_uploader(
        "Upload a lesion image.",
        type= ALLOWED_IMAGE_FORMATS,
        max_upload_size= MAX_FILE_SIZE_MB,
        accept_multiple_files= False,
        key="image_uploader",
        on_change="rerun",   #????????
        width="stretch"
    )

    if data is None:
        return None

    try:
        validate_image(data.getvalue())
    except ValueError as e:
        st.error(str(e))
        return None


    with Image.open(BytesIO(data.getvalue())) as decoded_image:
        image = decoded_image.copy()

    with st.container():
        st.caption(f"The image has been uploaded: {data.name}.", text_alignment="center"),
        st.image(data)

    return image