

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

    # 3. Print Final Merged Configuration
    print("-" * 50)
    print("Final Configuration (YAML with CLI Overrides):")
    print("-" * 50)
    
    print("\n[Base Paths]")
    print(f"  working_dir: {config.base_paths.working_dir}")
    print(f"  checkpoint_dir: {config.base_paths.checkpoint_dir}")
    print(f"  output_dir: {config.base_paths.output_dir}")
    
    print("\n[Dataset]")
    print(f"  data_dir: {config.dataset.data_dir}")
    print(f"  image_size: {config.dataset.image_size}")
    print(f"  batch_size: {config.dataset.batch_size}")
    print(f"  num_workers: {config.dataset.num_workers}")
    print(f"  val_split: {config.dataset.val_split}")
    
    print("\n[Training]")
    print(f"  epochs: {config.training.epochs}")
    print(f"  learning_rate: {config.training.lr}")
    print(f"  weight_decay: {config.training.weight_decay}")
    print(f"  accumulation_steps: {config.training.accum_steps}")
    print(f"  patience: {config.training.patience}")
    
    print("\n[Checkpoints]")
    print(f"  input_ckpt: {config.checkpoint.input_ckpt}")
    print(f"  input_csv: {config.checkpoint.input_csv}")
    print(f"  working_ckpt: {config.checkpoint.working_ckpt}")
    print(f"  best_model: {config.checkpoint.best_model}")
    print(f"  working_csv: {config.checkpoint.working_csv}")
    
    print("\n[Inference]")
    print(f"  image_path: {config.inference.image_path}")
    print(f"  model_path: {config.inference.model_path}")
    print(f"  output_dir: {config.inference.output_dir}")
    print("-" * 50)
    
    print("\nExecution stopped after printing Final arguments as requested.")

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



