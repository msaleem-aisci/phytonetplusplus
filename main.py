import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim

# Absolute imports from the src package
from src.data import build_virtual_dataset, get_dataloaders
from src.model import build_swin_model
from src.trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description="Train Swin-T on Sugarcane Dataset")
    parser.add_argument('--config', type=str, default='config.yml', help='Path to the config file')
    
    parser.add_argument('--epochs', type=int, default=None, help='Override epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Override batch size')
    parser.add_argument('--lr', type=float, default=None, help='Override learning rate')
    parser.add_argument('--image_size', type=int, default=None, help='Override image resolution')
    
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def merge_args_and_config(args, config):
    class ConfigArgs:
        pass
    merged_args = ConfigArgs()
    
    merged_args.ds1_path = config['dataset']['ds1_path']
    merged_args.ds2_path = config['dataset']['ds2_path']
    merged_args.virtual_dir = config['dataset']['virtual_dir']
    merged_args.val_split = config['dataset']['val_split']
    merged_args.num_workers = config['dataset']['num_workers']
    
    merged_args.weight_decay = config['training']['weight_decay']
    merged_args.accum_steps = config['training']['accum_steps']
    merged_args.patience = config['training']['patience']
    
    merged_args.input_ckpt = config['checkpoint']['input_ckpt']
    merged_args.working_ckpt = config['checkpoint']['working_ckpt']
    merged_args.best_model = config['checkpoint']['best_model']
    
    merged_args.epochs = args.epochs if args.epochs is not None else config['training']['epochs']
    merged_args.batch_size = args.batch_size if args.batch_size is not None else config['dataset']['batch_size']
    merged_args.lr = args.lr if args.lr is not None else config['training']['lr']
    merged_args.image_size = args.image_size if args.image_size is not None else config['dataset']['image_size']

    return merged_args

def main():
    args = parse_args()
    config = load_config(args.config)
    final_args = merge_args_and_config(args, config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Training for {final_args.epochs} epochs with image size {final_args.image_size}")

    build_virtual_dataset(final_args.ds1_path, final_args.ds2_path, final_args.virtual_dir)
    
    dataloaders, num_classes = get_dataloaders(
        data_dir=final_args.virtual_dir, 
        batch_size=final_args.batch_size, 
        image_size=final_args.image_size,
        val_split=final_args.val_split,
        num_workers=final_args.num_workers
    )

    model = build_swin_model(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=final_args.lr, weight_decay=final_args.weight_decay)

    trainer = Trainer(model, dataloaders, criterion, optimizer, device, final_args)
    trainer.train()

if __name__ == '__main__':
    main()