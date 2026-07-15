from src.model import create_model
from src.dataset import transform
import torch
from PIL import Image
import numpy as np
import json

# -----------------------------
# Device
# -----------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# -----------------------------
# Load disease names
# -----------------------------

from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parent.parent

DISEASE_MAPPING_PATH = BASE_DIR / "models" / "disease_mapping.json"

with open(DISEASE_MAPPING_PATH) as f:
    disease_mapping = json.load(f)

disease_names = list(disease_mapping.values())


# -----------------------------
# Load model
# -----------------------------

MODEL_PATH = BASE_DIR / "models" / "best_model.pth"

model = create_model()

model.load_state_dict(
torch.load(
    MODEL_PATH,
    map_location=device
)
)

model.to(device)
print("Model created")

model.eval()


print("Model ready")
# -----------------------------
# Load thresholds
# -----------------------------


THRESHOLD_PATH = BASE_DIR / "models" / "best_threshold.json"

with open(THRESHOLD_PATH) as f:
    data = json.load(f)

best_thresholds = np.array(
    list(data.values())
)

# -----------------------------
# Prediction function
# -----------------------------

def predict(image_source):

    """
    Predict diseases from a chest X-ray.

    image_source:
        - Streamlit uploaded file
        - image path
    """


    # Load image

    image = Image.open(image_source)


    # Apply same preprocessing as training

    image = transform(image)


    # Add batch dimension

    image = image.unsqueeze(0)


    # Move to GPU/CPU

    image = image.to(device)



    # Model prediction

    with torch.inference_mode():

        outputs = model(image)

        probabilities = torch.sigmoid(outputs)

        probabilities = (
            probabilities
            .squeeze()
            .cpu()
            .numpy()
        )


    # Apply thresholds

    predictions = (
        probabilities >= best_thresholds
    )



    # Store results

    detected = []

    probability_results = {}



    for disease, probability, prediction in zip(
        disease_names,
        probabilities,
        predictions
    ):

        probability_results[disease] = float(probability)


        if prediction:
            detected.append(disease)



    return {
        "detected_diseases": detected,
        "probabilities": probability_results
    }