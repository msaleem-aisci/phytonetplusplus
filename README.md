# PhytoNet++

## Project Structure

```text
phytonetplusplus/
│
├── README.md                  # Project documentation, setup instructions, and usage
├── requirements.txt           # Python dependencies required to run the project
├── .gitignore                 # Specifies untracked files to ignore (data, weights, logs)
├── config.yml                 # Central configuration for paths and hyperparameters
│
├── data/                      # Raw test images, training datasets, and reference data
├── output/                    # Generated plots, Grad-CAM overlays, and evaluation metrics
├── notebooks/                 # Jupyter notebooks for EDA and standalone Kaggle experiments
│
├── checkpoints/
│   ├── logs/                  # TensorBoard and CSV training logs
│   └── models/                # Saved model weights (.pth files)
│
└── src/                       # Source code for the project
    ├── __init__.py            # Initializes the src package
    ├── main.py                # Main entry point for training and execution
    ├── inference.py           # Model inference and prediction pipeline
    ├── config_parser.py       # Dataclass-based configuration and path management
    ├── dataset.py             # Dataset loading and preprocessing
    ├── trainer.py             # Training and validation engine
    ├── loss.py                # Loss function implementations
    ├── grad.py                # Grad-CAM implementations for ViT and CNN models
    ├── utils.py               # Utility functions for padding and checkpoint management
    │
    └── model/
        ├── __init__.py        # Initializes the model package
        ├── swin_t.py          # Swin Transformer Tiny model
        ├── resnet.py          # ResNet model implementation
        ├── vggnet.py          # VGGNet model implementation
        └── phytonet.py        # PhytoNet++ architecture

```

## File and Directory Descriptions

| File or Directory | Description |
| --- | --- |
| `README.md` | Project overview, documentation, setup instructions, and usage guidelines. |
| `requirements.txt` | Lists the Python packages required to run the project. |
| `.gitignore` | Prevents high-resolution images, `.pth` files, and local logs from being committed to version control. |
| `config.yml` | Stores configuration settings, file paths, and model hyperparameters. |
| `data/` | Stores raw test images or reference data used for training, inference, and evaluation. |
| `output/` | Stores generated visualizations, Grad-CAM overlays, and evaluation metrics. |
| `notebooks/` | Directory for exploratory data analysis (EDA) and rapid Kaggle notebook experiments. |
| `checkpoints/logs/` | Stores TensorBoard events and CSV training logs. |
| `checkpoints/models/` | Stores trained model weights in `.pth` format. |
| `src/main.py` | Entry point for running the training pipeline. |
| `src/inference.py` | Handles model inference and predictions on input images. |
| `src/config_parser.py` | Implements dataclass-based configuration parsing and path management. |
| `src/dataset.py` | Handles dataset loading and preprocessing. |
| `src/trainer.py` | Implements the training and validation workflow. |
| `src/loss.py` | Contains loss functions used during model training. |
| `src/grad.py` | Implements Grad-CAM visualizations for Vision Transformer (ViT) and CNN architectures. |
| `src/utils.py` | Provides helper functions for padding and checkpoint management. |
| `src/model/swin_t.py` | Defines the Swin Transformer Tiny architecture. |
| `src/model/resnet.py` | Defines the ResNet architecture. |
| `src/model/vggnet.py` | Defines the VGGNet architecture. |
| `src/model/phytonet.py` | Implements the proposed PhytoNet++ architecture. |

## Project Overview

PhytoNet++ is organized into modular components to support model development, training, inference, evaluation, and visual explanation. The project includes multiple backbone architectures, a dedicated training engine, configurable experiment settings, and Grad-CAM visualization utilities.
