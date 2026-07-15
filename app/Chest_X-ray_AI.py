import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

import streamlit as st
from src.inference import predict
from PIL import Image
from pathlib import Path


# ---------------------------------
# Page Configuration
# ---------------------------------

st.set_page_config(
    page_title="Chest X-ray AI",
    page_icon="🩺",
    layout="wide"
)



# ---------------------------------
# Title
# ---------------------------------

st.title("🩺 AI-Based Chest X-ray Disease Classification")

st.write(
    """
    Deep learning based multi-label classification
    using DenseNet121 trained on NIH ChestXray14 dataset.
    This application uses deep learning to analyze chest X-ray
    images and predict the presence of multiple thoracic diseases.
    The model is built using DenseNet121 with transfer learning and trained
    on the NIH ChestX-ray14 dataset. It provides disease predictions along
    with confidence scores for each category.
    """
)

st.divider()


# ---------------------------------
# Demo Images
# ---------------------------------

DEMO_DIR = Path("demo_images")

demo_images = {
    "Demo X-ray 1": "sample_1.png",
    "Demo X-ray 2": "sample_2.png",
    "Demo X-ray 3": "sample_3.png",
    "Demo X-ray 4": "sample_4.png",
    "Demo X-ray 5": "sample_5.png"
}


# ---------------------------------
# Image Selection
# ---------------------------------

st.subheader("Choose X-ray Image")


input_type = st.radio(
    "Select input method:",
    [
        "Upload your own image",
        "Use a demo image"
    ]
)


image_source = None


if input_type == "Upload your own image":

    uploaded_file = st.file_uploader(
        "Upload a Chest X-ray image",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:

        image_source = uploaded_file

        image = Image.open(uploaded_file)

        st.subheader("Uploaded X-ray")

        st.image(
            image,
            width=400
        )


else:

    selected_demo = st.selectbox(
        "Select demo X-ray",
        list(demo_images.keys())
    )


    demo_path = DEMO_DIR / demo_images[selected_demo]


    if demo_path.exists():

        image_source = demo_path

        image = Image.open(demo_path)

        st.subheader("Demo X-ray")

        st.image(
            image,
            caption=selected_demo,
            width=400
        )

    else:

        st.error(
            "Demo image not found. Check your demo_images folder."
        )


# ---------------------------------
# Prediction
# ---------------------------------

if image_source is not None:

    st.divider()


    if st.button("🔍 Predict"):

        with st.spinner("Analyzing X-ray..."):

            result = predict(image_source)


        # -----------------------------
        # Detected diseases
        # -----------------------------

        st.subheader("Prediction Result")


        detected = result["detected_diseases"]


        if len(detected) == 0:

            st.success(
                "No disease detected"
            )

        else:

            for disease in detected:

                st.error(
                    f"Detected: {disease}"
                )


        st.divider()


        # -----------------------------
        # Probabilities
        # -----------------------------

        st.subheader(
            "Disease Probabilities"
        )


        probabilities = result["probabilities"]


        for disease, probability in probabilities.items():

            st.write(
                f"{disease}: {probability:.2%}"
            )

            st.progress(
                float(probability)
            )