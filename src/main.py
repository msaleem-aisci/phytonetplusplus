

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
    args = parse_args()

    print(test_fun())

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



