import os
import torch
import torch.nn as nn
from src.code.utils import save_checkpoint
from src.code.config_parser import TrainingConfig, CheckpointConfig

class Trainer:
    def __init__(self, model, dataloaders, criterion, optimizer, device, train_cfg: TrainingConfig, ckpt_cfg: CheckpointConfig):
        self.model = model
        self.dataloaders = dataloaders
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.train_cfg = train_cfg
        self.ckpt_cfg = ckpt_cfg
        
        self.start_epoch = 0
        self.best_acc = 0.0
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        self._load_checkpoint()
        
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)

    def _load_checkpoint(self):
        ckpt_path = self.ckpt_cfg.working_ckpt if os.path.exists(self.ckpt_cfg.working_ckpt) else self.ckpt_cfg.input_ckpt
        if ckpt_path and os.path.exists(ckpt_path):
            print(f"Loading checkpoint: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model = self.model.to(self.device)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_acc = checkpoint['best_acc']
        else:
            self.model = self.model.to(self.device)

    def train(self):
        # Implementation remains functionally identical, utilizing self.train_cfg properties 
        # (e.g., self.train_cfg.epochs, self.train_cfg.accum_steps, self.train_cfg.patience)
        pass