import pandas as pd
import numpy as np
import torch
from torch import nn

from PIL import Image
from torchvision import transforms

from torch.utils.data import Dataset

import ast


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])


class XrayDataset(Dataset):
    def __init__(self,df,transform=None):
        self.df = df.reset_index(drop = True)
        self.transform = transform 
        
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx):
        row = self.df.iloc[idx]
        
        img_path = row["Image Path"]
        label = row["Labels_encoded"]
        
        image = Image.open(img_path)
        
        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(label, dtype=torch.float32)
            
        return image, label