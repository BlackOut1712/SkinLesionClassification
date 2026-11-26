import streamlit as st
import pandas as pd


CLASS_FULL_NAMES = {
    "akiec": "Actinic keratoses and intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions"
}


def render_prediction_results(
    probabilities: dict[str, float]
) -> None:

    predicted_class = max(
        probabilities,
        key=probabilities.get
    )

    best_score = probabilities[predicted_class]

    with st.container(border=True):

        st.subheader("Classification results")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Predicted class",
                predicted_class.upper()
            )

            st.caption(CLASS_FULL_NAMES[predicted_class])

        with col2:
            st.metric(
                "Softmax score",
                f"{best_score:.1%}"
            )

        st.divider()

        results_df = pd.DataFrame(
            probabilities.items(),
            columns=["Class", "Score"]
        )

        results_df["Score"] *= 100

        results_df = results_df.sort_values(
            by="Score",
            ascending=False
        )

        for _, row in results_df.iterrows():

            st.progress(
                float(row["Score"] / 100),
                text=f"{row['Class']} — {row['Score']:.1f}%"
            )
