from src.model import create_model
from src.dataset import XrayDataset
from src.dataset import transform
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import sklearn
from pathlib import Path
import json
import ast

from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parent.parent

DISEASE_MAPPING_PATH = BASE_DIR / "models" / "disease_mapping.json"

print(DISEASE_MAPPING_PATH)
print(DISEASE_MAPPING_PATH.exists())

from sklearn.metrics import(
    roc_auc_score,
    f1_score,
    precision_score,
    accuracy_score,
    recall_score
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = create_model()

MODEL_PATH = BASE_DIR / "models" / "best_model.pth"

model = create_model()

model.load_state_dict(
torch.load(
    MODEL_PATH,
    map_location=device
)
)

model.to(device)
model.eval()


TEST_LIST_PATH = BASE_DIR / "Data" / "original" / "test_list.txt"
RESULT_PATH = BASE_DIR / "Data" / "processed" / "result.csv"


test_list = pd.read_csv(
    TEST_LIST_PATH,
    header=None,
    names=["Image Index"]
)

df3 = pd.read_csv(
    RESULT_PATH
)



df3["Labels_encoded"] = df3["Labels_encoded"].apply(ast.literal_eval)

test_df = df3[
    df3["Image Index"].isin(test_list["Image Index"])
]

from torch.utils.data import DataLoader

test_dataset = XrayDataset(test_df, transform = transform)

test_dataloader = DataLoader(test_dataset, batch_size = 32, shuffle = True, num_workers = 0, pin_memory=True )

all_labels = []
all_outputs = []

model.eval()

with torch.inference_mode():
    
    for images,batch_labels in test_dataloader:
        
        images = images.to(device)
        batch_labels = batch_labels.to(device)
        
        outputs = model(images)
        
        all_outputs.append(outputs.cpu())
        all_labels.append(batch_labels.cpu())
        
        
    
    
all_labels = torch.cat(all_labels)
all_outputs = torch.cat(all_outputs)
# Convert logits to probabilities
probabilities = torch.sigmoid(all_outputs)

# Convert to numpy
probabilities = probabilities.cpu().numpy()
labels = all_labels.numpy()



import json
THRESHOLD_PATH = BASE_DIR / "models" / "best_threshold.json"

with open(THRESHOLD_PATH) as f:
    data = json.load(f)


threshold = data["threshold"]

best_thresholds = np.array(threshold)

y_pred = (probabilities > best_thresholds).astype(int)

y_true = all_labels.numpy()
y_prob = probabilities

f1 = f1_score(
    y_true,
    y_pred,
    average = "macro"
)

precision = precision_score(
    y_true,
    y_pred,
    average = "macro"
)

recall = recall_score(
    y_true,
    y_pred,
    average = "macro"
    
)

auroc = roc_auc_score(
    y_true,
    y_prob,
    average = "macro"
)

accuracy = accuracy_score(
    y_true,
    y_pred,
)

print("F1:", f1)
print("Precision:", precision)
print("Recall:", recall)
print("AUROC:", auroc)
print("Accuracy:", accuracy)


with open(DISEASE_MAPPING_PATH) as f:
    disease_mapping = json.load(f)

disease_names = list(disease_mapping.values())

from sklearn.metrics import roc_auc_score


for i, disease in enumerate(disease_names):

    auc = roc_auc_score(
        all_labels[:, i],
        y_prob[:, i]
    )

    print(f"{disease}: AUROC = {auc:.4f}")
    

# individual threshold

from sklearn.metrics import f1_score, precision_score, recall_score


for i, disease in enumerate(disease_names):

    f1 = f1_score(
        all_labels[:, i],
        y_pred[:, i]
    )

    precision = precision_score(
        all_labels[:, i],
        y_pred[:, i]
    )

    recall = recall_score(
        all_labels[:, i],
        y_pred[:, i]
    )

    print(
        f"{disease}: "
        f"F1={f1:.4f}, "
        f"Precision={precision:.4f},\n "
        f"Recall={recall:.4f}"
    ) 