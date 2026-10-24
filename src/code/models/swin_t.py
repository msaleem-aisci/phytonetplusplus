import torch.nn as nn
from torchvision import models

def build_swin_t(num_classes: int):
    model = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
    num_ftrs = model.head.in_features
    model.head = nn.Linear(num_ftrs, num_classes)
    return model