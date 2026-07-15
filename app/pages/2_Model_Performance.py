import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
)

import streamlit as st
from src.utils import disease_names


st.title("Model Performance")

st.markdown(
"""
## Evaluation Overview

The model is evaluated as a **multi-label classification system** where each
chest X-ray can contain multiple disease findings.

Because medical datasets are often highly imbalanced, accuracy alone is not
a reliable measure. This project focuses on metrics commonly used for medical
machine learning evaluation.
"""
)


# Overall Metrics

st.subheader("Overall Metrics")


col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Mean AUROC",
        "0.736"
    )

with col2:
    st.metric(
        "F1 Score",
        "0.280"
    )

with col3:
    st.metric(
        "Recall",
        "0.482"
    )


st.info(
"""
**Interpretation**

- AUROC measures how well the model separates positive and negative cases
  across different classification thresholds.
- F1-score balances precision and recall and is important when disease
  classes are imbalanced.
- Recall measures the ability to identify actual disease cases.
"""
)


# ROC Curve Section

st.subheader("Disease-wise ROC Curves")


st.markdown(
"""
ROC curves show the performance of the model for each disease category.
A curve closer to the top-left corner indicates better discrimination ability.
"""
)



disease = st.selectbox(
    "Select disease for ROC curve: ",
    disease_names,
    key="roc_select",
    width="stretch"
)


st.image( 
         f"outputs/plots/roc_curves/{disease}.png",
         caption=f"{disease} ROC Curve" )


# Confusion Matrix
st.subheader("Confusion Matrix")

st.markdown(
"""
### What is a Confusion Matrix?

A confusion matrix provides a detailed view of how the model's predictions
compare with the actual labels for a specific disease.

Since this project uses **multi-label classification**, a separate confusion
matrix is generated for each disease category.

The matrix contains four outcomes:

| | Predicted Negative | Predicted Positive |
|---|---|---|
| **Actual Negative** | True Negative (TN) | False Positive (FP) |
| **Actual Positive** | False Negative (FN) | True Positive (TP) |

**True Positive (TP):**
The model correctly identifies an image containing the disease.

**True Negative (TN):**
The model correctly identifies an image without the disease.

**False Positive (FP):**
The model predicts the disease when it is not present. This represents a
false alarm.

**False Negative (FN):**
The model fails to detect a disease that is actually present. In medical
applications, reducing false negatives is especially important because missed
cases can be critical.

The confusion matrix helps understand the balance between sensitivity
(recall) and precision, showing not only how often the model is correct but
also what types of mistakes it makes.
"""
)
illness = st.selectbox(
    "Select disease for confusion matrix: ",
    disease_names,
    key="cm_select"
)

st.image( 
         f"outputs/plots/confusion_matrix/{illness}.png",
         caption=f"Confusion Matrix of {illness} " )


# Disease performance

st.subheader("Disease-wise Performance")


st.markdown(
"""
Different diseases have different difficulty levels because of:

- Number of available examples
- Visual similarity with other conditions
- Label uncertainty in the dataset
"""
)


# Limitations

st.subheader("Limitations")


st.warning(
"""
This model is designed for research and educational purposes.

Important limitations:

- NIH ChestX-ray14 labels contain uncertainty.
- Disease categories are highly imbalanced.
- Predictions should not replace expert radiologist interpretation.
- Performance may vary on images from different hospitals or equipment.
"""
)