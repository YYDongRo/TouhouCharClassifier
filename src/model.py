import torch
import torch.nn as nn
import torchvision.models as models


def create_model(num_classes: int) -> nn.Module:
    """
    Create a simple ResNet18-based classifier.
    This is just a placeholder; we will improve it later.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
