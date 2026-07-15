import os
import ast
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from src.model import create_model
from src.dataset import XrayDataset, transform

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

TRAIN_CSV = BASE_DIR / "Data" / "processed" / "train.csv"
VAL_CSV = BASE_DIR / "Data" / "processed" / "validation.csv"

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)

train_df["Labels_encoded"] = train_df["Labels_encoded"].apply(ast.literal_eval)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = create_model(num_classes=14)

model = model.to(device)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr = 1e-4
)

import os

num_epochs=15

from torch.utils.data import DataLoader

train_dataset = XrayDataset(train_df, transform = transform)

train_dataloader = DataLoader(train_dataset, batch_size = 32, shuffle = True, num_workers = 0, pin_memory=True )


for epoch in range(num_epochs):
    model.train()
    
    epoch_loss = 0
    
    for images,labels in train_dataloader:
        
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        
        loss = criterion(outputs, labels)
        
        loss.backward()
        
        optimizer.step()
        
        torch.cuda.synchronize()
        
        epoch_loss += loss.item()
        
    avg_loss = epoch_loss / len(train_dataloader)
    
    print(f"Epoch {epoch+1} completed")
    
    print(f"Epoch {epoch+1}/{num_epochs} Loss: {avg_loss:.4f}")
          
    

    torch.save(
        model.state_dict(),
    MODEL_DIR / f"densenet121_epoch{epoch+1}.pth"
    )        



