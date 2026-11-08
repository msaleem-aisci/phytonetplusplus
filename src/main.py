

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))  # Only one level up
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
    


import argparse
import torch
import torch.optim as optim

from src.config_parser import DatasetConfig, TrainingConfig, CheckpointConfig
from src.utils import test_fun
from src.config_parser import load_config

def parse_args():
    parser = argparse.ArgumentParser(description="PhytoNet++ Training Pipeline")
    
    # Set defaults to None so YAML takes precedence. CLI overrides only if explicitly passed.
    # Dataset Arguments
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--num_classes', type=int, default=None)
    parser.add_argument('--val_split', type=float, default=None)
    
    # Training Arguments
    parser.add_argument('--total_epochs', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--patience', type=int, default=None)
    parser.add_argument('--start_epoch', type=int, default=None)
    
    # Memory Management
    parser.add_argument('--physical_batch', type=int, default=None)
    parser.add_argument('--accumulation_steps', type=int, default=None)
    
    # Checkpoint & CSV Paths
    parser.add_argument('--input_checkpoint_path', type=str, default=None)
    parser.add_argument('--input_csv_path', type=str, default=None)
    parser.add_argument('--working_checkpoint_path', type=str, default=None)
    parser.add_argument('--best_model_path', type=str, default=None)
    parser.add_argument('--working_csv_path', type=str, default=None)

    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Load the base configuration from the YAML file
    config = load_config('config.yml')
    
    # 2. Override YAML defaults with explicitly provided CLI arguments
    if args.data_dir is not None:
        config.dataset.data_dir = args.data_dir
    if args.val_split is not None:
        config.dataset.val_split = args.val_split
        
    if args.total_epochs is not None:
        config.training.epochs = args.total_epochs
    if args.learning_rate is not None:
        config.training.lr = args.learning_rate
    if args.patience is not None:
        config.training.patience = args.patience
    if args.start_epoch is not None: 
        config.training.start_epoch = args.start_epoch
        
    if args.physical_batch is not None:
        config.dataset.batch_size = args.physical_batch
    if args.accumulation_steps is not None:
        config.training.accum_steps = args.accumulation_steps

    if args.input_checkpoint_path is not None:
        config.checkpoint.input_ckpt = "" if args.input_checkpoint_path.lower() == 'none' else args.input_checkpoint_path
    if args.input_csv_path is not None:
        config.checkpoint.input_csv = "" if args.input_csv_path.lower() == 'none' else args.input_csv_path
        
    if args.working_checkpoint_path is not None:
        config.checkpoint.working_ckpt = args.working_checkpoint_path
    if args.best_model_path is not None:
        config.checkpoint.best_model = args.best_model_path
    if args.working_csv_path is not None:
        config.checkpoint.working_csv = args.working_csv_path
    


    os.makedirs(os.path.dirname(config.checkpoint.working_ckpt), exist_ok=True)
    if os.path.dirname(config.checkpoint.working_csv):
        os.makedirs(os.path.dirname(config.checkpoint.working_csv), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Target Epochs: {config.training.epochs} | Effective Batch: {config.dataset.batch_size * config.training.accum_steps}")

    # 3. Initialize components using the fully merged config object
    dataloaders, num_classes_ds = get_dataloaders(config.dataset)
  

if __name__ == '__main__':
    main()

def maina():
    args = parse_args()
    
    # Dynamically build the dataclasses so trainer.py does not need to change
    dataset_cfg = DatasetConfig(
        data_dir=args.data_dir,
        image_size=1500, 
        batch_size=args.physical_batch,
        num_workers=4,
        val_split=args.val_split
    )
    
    training_cfg = TrainingConfig(
        epochs=args.total_epochs,
        lr=args.learning_rate,
        weight_decay=0.01, 
        accum_steps=args.accumulation_steps,
        patience=args.patience
    )
    
    # Handle the 'none' string from CLI
    input_ckpt = "" if args.input_checkpoint_path.lower() == 'none' else args.input_checkpoint_path
    input_csv = "" if args.input_csv_path.lower() == 'none' else args.input_csv_path

    checkpoint_cfg = CheckpointConfig(
        input_ckpt=input_ckpt,
        working_ckpt=args.working_checkpoint_path,
        best_model=args.best_model_path,
        input_csv=input_csv,
        working_csv=args.working_csv_path
    )

    # Ensure output directories exist based on provided Kaggle paths
    os.makedirs(os.path.dirname(args.working_checkpoint_path), exist_ok=True)
    if os.path.dirname(args.working_csv_path):
        os.makedirs(os.path.dirname(args.working_csv_path), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Target Epochs: {training_cfg.epochs} | Effective Batch Size: {dataset_cfg.batch_size * training_cfg.accum_steps}")

    # Initialize components
    dataloaders, num_classes_ds = get_dataloaders(dataset_cfg)
    
    # Enforce class count match
    assert args.num_classes == num_classes_ds, f"CLI classes ({args.num_classes}) != Dataset classes ({num_classes_ds})"
    
    model = build_resnet(args.num_classes)
    criterion = get_loss_function()
    optimizer = optim.AdamW(model.parameters(), lr=training_cfg.lr, weight_decay=training_cfg.weight_decay)

    # Run Engine
    trainer = Trainer(
        model=model, 
        dataloaders=dataloaders, 
        criterion=criterion, 
        optimizer=optimizer, 
        device=device, 
        train_cfg=training_cfg, 
        ckpt_cfg=checkpoint_cfg
    )
    trainer.train()



