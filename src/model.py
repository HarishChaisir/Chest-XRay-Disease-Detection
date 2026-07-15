import torch
import torch.nn as nn
from torchvision.models import densenet121


def create_model(num_classes=14):

    model = densenet121(weights="DEFAULT")

    model.classifier = nn.Linear(
        model.classifier.in_features,
        num_classes
    )

    return model