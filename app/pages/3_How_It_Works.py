import streamlit as st


st.title("How The Model Works")


st.markdown(
"""
## Deep Learning Pipeline

This application uses a deep learning pipeline to analyze chest X-ray images
and estimate the probability of different thoracic diseases.

The model performs **multi-label classification**, meaning a single X-ray can
contain multiple possible findings at the same time.
"""
)


st.subheader("1. Input Chest X-ray")


st.markdown(
"""
The user uploads a chest X-ray image which becomes the input for the model.

Before the image can be processed, it must be converted into a format that
the neural network can understand.
"""
)


st.subheader("2. Image Preprocessing")


st.markdown(
"""
The image undergoes preprocessing steps:

- Resizing to the input size expected by DenseNet121
- Converting grayscale X-rays into a 3-channel image format
- Converting pixel values into tensors
- Normalizing image intensity values

These steps ensure that all images have a consistent representation before
being passed into the neural network.
"""
)


st.subheader("3. Feature Extraction Using DenseNet121")


st.markdown(
"""
The core of the system is **DenseNet121**, a convolutional neural network.

Instead of manually identifying features such as edges, textures, or abnormal
patterns, the network automatically learns visual representations from the
training data.

DenseNet121 uses dense connections between layers, allowing information to
flow efficiently through the network and improving feature reuse.
"""
)


st.subheader("4. Disease Probability Prediction")


st.markdown(
"""
The final layer of the network produces a probability score for each of the
14 disease categories.

Because this is a multi-label problem, each disease is predicted independently.

Example:
Pneumonia → 0.82
Effusion → 0.67
Cardiomegaly → 0.12
Pneumothorax → 0.05


These values represent the model's confidence, not a medical diagnosis.
"""
)


st.subheader("5. Sigmoid Activation")


st.markdown(
"""
The model uses a sigmoid activation function because each disease category can
be independently present or absent.

Unlike softmax classification, where only one class can be selected, sigmoid
allows multiple diseases to have high probabilities simultaneously.
"""
)


st.subheader("6. Threshold Optimization")


st.markdown(
"""
Raw probabilities are converted into final predictions using thresholds.

For example:
Probability > threshold → Positive prediction
Probability < threshold → Negative prediction


The threshold is adjusted during validation to achieve a better balance between
precision and recall.

This is important because medical datasets are often highly imbalanced.
"""
)


st.subheader("Complete Workflow")

st.code(
"""
Chest X-ray Image

        ↓

Preprocessing
(Resize, Normalize, Tensor Conversion)

        ↓

DenseNet121 Feature Extraction

        ↓

Sigmoid Layer

        ↓

Disease Probabilities

        ↓

Optimized Thresholds

        ↓

Final Disease Predictions
"""
)


st.warning(
"""
Important:
This model is developed for educational and research purposes only.
It does not replace professional radiological interpretation.
"""
)