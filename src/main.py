import os
import torch
import torch.optim as optim

from core.config_parser import load_config
from core.dataset import get_dataloaders
from core.model.swin_t import build_swin_t
from core.loss import get_loss_function
from core.trainer import Trainer

def main():
    # Setup directories
    os.makedirs('checkpoints/models', exist_ok=True)
    os.makedirs('checkpoints/logs', exist_ok=True)
    
    # Load typed config
    config = load_config('config.yml')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Epochs: {config.training.epochs}")

    # Initialize components
    dataloaders, num_classes = get_dataloaders(config.dataset)
    model = build_swin_t(num_classes)
    criterion = get_loss_function()
    optimizer = optim.AdamW(model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay)

    # Run Engine
    trainer = Trainer(
        model=model, 
        dataloaders=dataloaders, 
        criterion=criterion, 
        optimizer=optimizer, 
        device=device, 
        train_cfg=config.training, 
        ckpt_cfg=config.checkpoint
    )
    trainer.train()

if __name__ == '__main__':
    main()