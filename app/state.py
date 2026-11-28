import streamlit as st
import numpy as np

from src.schemas.analysis_result import AnalysisResult


def initialize_state() -> None:
    defaults = {
        "analysis_result": None,
        "heatmaps": {}
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def save_analysis(result: AnalysisResult) -> None:
    st.session_state.analysis_result = result
    st.session_state.heatmaps = {}


def get_analysis() -> AnalysisResult | None:
    return st.session_state.analysis_result


def reset_analysis() -> None:
    st.session_state.analysis_result = None
    st.session_state.heatmaps = {}


def get_heatmap(class_name: str) -> np.ndarray | None:
    return st.session_state.heatmaps.get(class_name)


def save_heatmap(
    class_name: str,
    heatmap: np.ndarray
) -> None:

    st.session_state.heatmaps[class_name] = heatmap