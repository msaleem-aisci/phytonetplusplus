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