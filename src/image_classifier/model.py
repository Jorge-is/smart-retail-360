import torch
import torch.nn as nn
from torchvision import models
from src.utils.config import IMAGE_CLASSES


def build_model(num_classes: int = len(IMAGE_CLASSES), freeze_backbone: bool = True) -> nn.Module:
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


def load_model(path, num_classes: int = len(IMAGE_CLASSES), device: str = "cpu") -> nn.Module:
    model = build_model(num_classes=num_classes, freeze_backbone=False)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model
