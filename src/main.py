

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
    
    # Dataset Arguments
    parser.add_argument('--data_dir', type=str, default='/kaggle/working/combined_sugarcane_dataset')
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--val_split', type=float, default=0.2)
    
    # Training Arguments
    parser.add_argument('--total_epochs', type=int, default=18)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=5)
    
    # Memory Management
    parser.add_argument('--physical_batch', type=int, default=16)
    parser.add_argument('--accumulation_steps', type=int, default=2)
    
    # Checkpoint & CSV Paths
    parser.add_argument('--input_checkpoint_path', type=str, default='none')
    parser.add_argument('--input_csv_path', type=str, default='none')
    parser.add_argument('--working_checkpoint_path', type=str, default='/kaggle/working/resnet18_checkpoint.pth')
    parser.add_argument('--best_model_path', type=str, default='/kaggle/working/resnet18_best.pth')
    parser.add_argument('--working_csv_path', type=str, default='/kaggle/working/restnet_training_log.csv')

    return parser.parse_args()

def main():
    # You can still parse CLI args if you want to override YAML later
    args = parse_args()
    
    # Load the configuration from the YAML file
    config = load_config('config.yml')
    
    print("-" * 50)
    print("Loaded Configuration from config.yml:")
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
    
    print("\n[Checkpoints - Dynamically Assembled Paths]")
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
    
    print("\nExecution stopped after printing YAML arguments as requested.")

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



