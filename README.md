Notice that the README draft you pasted reverted to the previous file structure, listing `loss.py` and `grad.py` instead of the `losses/` and `grad_cam/` directories we just established.

To maintain consistency with your actual repository layout, I have corrected the structure and file descriptions in this finalized README. I also added the necessary setup, training, and inference instructions to make this a complete, repository-ready document.

Here is the complete text for your `README.md`:

```markdown
# PhytoNet++

## Project Overview

PhytoNet++ is a modular deep learning pipeline designed for high-resolution crop disease classification and visual interpretability. The project supports multiple backbone architectures—including Swin Transformers, ResNet, VGGNet, and the custom multi-branch PhytoNet++ architecture—along with a dedicated training engine, configurable experiment settings, and class-activation mapping (Grad-CAM) utilities. 

By separating logic (source code) from state (data, model weights, and logs), this structure promotes code reusability, maintainability, and reproducibility across standard academic and engineering environments.

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
    ├── __init__.py            
    ├── main.py                # Main entry point for training and execution
    ├── inference.py           # Model inference and prediction pipeline
    ├── config_parser.py       # Dataclass-based configuration and path management
    ├── dataset.py             # Dataset loading and preprocessing
    ├── trainer.py             # Training and validation engine
    ├── utils.py               # Utility functions for padding and checkpoint management
    │
    ├── grad_cam/              # Explainable AI visualization utilities
    │   ├── __init__.py        
    │   ├── cnn_cam.py         # Grad-CAM for CNN backbones (ResNet, VGGNet)
    │   └── swin_cam.py        # Grad-CAM for Vision Transformers (Swin)
    │
    ├── losses/                # Custom and standard loss function implementations
    │   ├── __init__.py        
    │   ├── cross_entropy.py   # Standard Cross Entropy implementation
    │   └── focal_loss.py      # Focal loss for handling imbalanced datasets
    │
    └── model/
        ├── __init__.py        
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
| `src/utils.py` | Provides helper functions for padding and checkpoint management. |
| `src/grad_cam/` | Directory containing specific Grad-CAM implementations for CNN and ViT architectures. |
| `src/losses/` | Directory containing modular loss functions to address class imbalances. |
| `src/model/swin_t.py` | Defines the Swin Transformer Tiny architecture. |
| `src/model/resnet.py` | Defines the ResNet architecture. |
| `src/model/vggnet.py` | Defines the VGGNet architecture. |
| `src/model/phytonet.py` | Implements the proposed PhytoNet++ architecture. |

## Setup and Installation

1. **Clone the repository:**
```bash
git clone [https://github.com/yourusername/phytonetplusplus.git](https://github.com/yourusername/phytonetplusplus.git)
cd phytonetplusplus

```


2. **Install dependencies:**
Ensure you have a PyTorch environment configured with CUDA support, then install the remaining requirements:
```bash
pip install -r requirements.txt

```


3. **Configure your paths:**
Update `config.yml` to point to your local or Kaggle data directories (`data_dir`) and adjust hyperparameters (batch size, learning rate) according to your available VRAM.

## Usage

**Training:**
To initiate the training pipeline using the parameters defined in `config.yml`:

```bash
python src/main.py

```

**Inference & Grad-CAM Extraction:**
To run predictions on a single image and generate a Grad-CAM overlay heatmap:

```bash
python src/inference.py

```

Outputs will be saved directly to the `output/` directory as specified in your configuration.

```

```