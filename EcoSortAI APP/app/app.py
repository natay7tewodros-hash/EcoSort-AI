from pathlib import Path

import streamlit as st

from predictor import EcoSortPredictor


st.set_page_config(
    page_title="EcoSort AI",
    page_icon="♻",
    layout="centered",
)

CONFIDENCE_THRESHOLD = 0.60


@st.cache_resource
def load_predictor() -> EcoSortPredictor:
    return EcoSortPredictor()


st.title("EcoSort AI")
st.write("Upload one waste item to receive an AI classification.")
st.info(
    "Version 1 recognizes six TrashNet labels: cardboard, glass, metal, paper, plastic, and trash. "
    "The prediction is experimental and is not official recycling or hazardous-waste guidance."
)

uploaded_file = st.file_uploader(
    "Choose one image",
    type=["jpg", "jpeg", "png"],
    help="For best results, photograph one item with good lighting and a clear view.",
)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded image", use_container_width=True)

    try:
        predictor = load_predictor()
        predictions = predictor.predict(uploaded_file, top_k=3)
    except Exception as error:
        st.error(f"Prediction could not be completed: {error}")
        st.stop()

    best = predictions[0]
    st.subheader("Prediction")
    st.metric("Most likely category", best.label.title(), f"{best.confidence:.1%} confidence")

    if best.confidence < CONFIDENCE_THRESHOLD:
        st.warning(
            "The model is unsure. Try another photo with one object, better lighting, "
            "and less background clutter. Do not let the machine sort this item automatically."
        )
    else:
        st.success("The confidence is above the current demonstration threshold.")

    st.subheader("Top predictions")
    for prediction in predictions:
        st.write(f"**{prediction.label.title()}** — {prediction.confidence:.1%}")
        st.progress(prediction.confidence)

st.divider()
st.caption(
    "EcoSort AI is a learning and prototype project. TrashNet images are controlled and "
    "may not represent real household waste."
)
