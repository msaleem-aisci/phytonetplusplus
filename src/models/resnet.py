import os
import sys
import torch.nn as nn
from torchvision import models

# Safely resolve project root so this file can be run standalone
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

def build_resnet(num_classes: int):
    """
    Initializes a ResNet18 model with ImageNet weights and 
    modifies the final fully connected layer for the specified number of classes.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Modify final fully connected layer for PhytoNet++ classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model