import os
import sys
import torch.nn as nn

# Safely resolve project root so this file can be run standalone
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

def get_loss_function():
    """
    Returns the standard Cross Entropy Loss criterion.
    """
    return nn.CrossEntropyLoss()