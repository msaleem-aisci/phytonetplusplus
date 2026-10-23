# PhytoNet++

## Project Structure

The repository is organized into the following structure:

```text
sugarcane-disease-project/
│
├── README.md                  # Project documentation, setup instructions, and execution commands
├── requirements.txt           # List of Python dependencies (torch, torchvision, pyyaml, etc.)
├── config.yml                 # Central configuration for paths and hyperparameters
├── .gitignore                 # Specifies files Git should ignore (datasets, checkpoints, caches)
├── main.py                    # The entry point that ties everything together via CLI
│
└── src/                       # Source folder containing the core modules
    ├── __init__.py            # Empty file that makes 'src' a recognizable Python package
    ├── data.py                # Virtual dataset building, augmentations, and DataLoaders
    ├── model.py               # Model definition (Swin-T initialization and modifications)
    ├── trainer.py             # The Trainer class handling the training and validation loops
    └── utils.py               # Helper classes and functions (e.g., SquarePad, checkpointing)
```

## File Descriptions

| File or Directory | Description |
|---|---|
| `README.md` | Project documentation, setup instructions, and execution commands. |
| `requirements.txt` | Lists the Python dependencies required to run the project. |
| `config.yml` | Stores configuration settings, including dataset paths and model hyperparameters. |
| `.gitignore` | Defines files and directories Git should ignore, such as datasets, checkpoints, and caches. |
| `main.py` | Main entry point for executing the training and evaluation pipeline through the command-line interface (CLI). |
| `src/__init__.py` | Marks `src` as a Python package. |
| `src/data.py` | Handles virtual dataset construction, data augmentation, and PyTorch DataLoaders. |
| `src/model.py` | Defines the model architecture, including Swin Transformer Tiny (Swin-T) initialization and modifications. |
| `src/trainer.py` | Implements the `Trainer` class for model training and validation. |
| `src/utils.py` | Provides utility functions and helper classes, including `SquarePad` and checkpoint management. |

## Overview

PhytoNet++ is ABC...
