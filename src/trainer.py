import os
import sys
import csv
import shutil
import torch
import torch.nn as nn

# Safely resolve project root so this file can be run standalone
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.config_parser import TrainingConfig, CheckpointConfig

class Trainer:
    def __init__(self, model, dataloaders, criterion, optimizer, device, train_cfg: TrainingConfig, ckpt_cfg: CheckpointConfig):
        self.model = model
        self.dataloaders = dataloaders
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.train_cfg = train_cfg
        self.ckpt_cfg = ckpt_cfg
        
        # Initialize defaults
        self.start_epoch = self.train_cfg.start_epoch
        self.best_acc = 0.0
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # Load states if resuming
        self._load_checkpoint()
        
        # Distribute across GPUs
        if torch.cuda.device_count() > 1:
            print(f"Utilizing {torch.cuda.device_count()} GPUs via DataParallel.")
            self.model = nn.DataParallel(self.model)

    def _load_checkpoint(self):
        checkpoint_path = None
        csv_exists = False

        # 6a. Find Checkpoint & CSV
        if os.path.exists(self.ckpt_cfg.working_ckpt) and os.path.exists(self.ckpt_cfg.working_csv):
            checkpoint_path = self.ckpt_cfg.working_ckpt
            csv_exists = True
        elif self.ckpt_cfg.input_ckpt and os.path.exists(self.ckpt_cfg.input_ckpt) and self.ckpt_cfg.input_csv and os.path.exists(self.ckpt_cfg.input_csv):
            checkpoint_path = self.ckpt_cfg.input_ckpt
            # Copy read-only CSV from Kaggle input to working directory to append to it
            shutil.copy(self.ckpt_cfg.input_csv, self.ckpt_cfg.working_csv)
            csv_exists = True

        # 6b. Load States
        if checkpoint_path and csv_exists:
            print(f"Loading checkpoint from: {checkpoint_path} and resuming CSV log.")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model = self.model.to(self.device)
            
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Safe memory mapping for optimizer
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
                        
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_acc = checkpoint['best_acc']
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            self.patience_counter = checkpoint.get('patience_counter', 0)
            print(f"Resuming at Epoch {self.start_epoch + 1}. Current Best Acc: {self.best_acc:.4f}")
        else:
            print("No checkpoint or CSV found. Starting training from scratch.")
            self.model = self.model.to(self.device)
            # Initialize new CSV with headers
            with open(self.ckpt_cfg.working_csv, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Epoch', 'Train Loss', 'Train Acc', 'Val Loss', 'Val Acc'])

    def _train_one_epoch(self, epoch):
        """Part 1: Training Loop"""
        self.model.train()
        running_loss = 0.0
        running_corrects = 0
        self.optimizer.zero_grad()
        
        dataloader = self.dataloaders['train']
        total_batches = len(dataloader)
        
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            with torch.set_grad_enabled(True):
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                
                loss = self.criterion(outputs, labels)
                (loss / self.train_cfg.accum_steps).backward()
                
                if (i + 1) % self.train_cfg.accum_steps == 0 or (i + 1) == total_batches:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            print(f"\rTrain | Batch {i+1}/{total_batches}", end="")
            
        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = running_corrects.double() / len(dataloader.dataset)
        
        print(f"\rTrain Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}" + " " * 20)
        return epoch_loss, epoch_acc.item()

    def _validate_one_epoch(self, epoch, train_loss, train_acc):
        """Part 2: Validation, CSV Logging, and Early Stopping"""
        self.model.eval()
        running_loss = 0.0
        running_corrects = 0
        
        dataloader = self.dataloaders['val']
        total_batches = len(dataloader)
        
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)
                
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            print(f"\rVal | Batch {i+1}/{total_batches}", end="")
            
        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = running_corrects.double() / len(dataloader.dataset)
        
        print(f"\rVal Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}" + " " * 20)
        
        # 1. Append to CSV
        with open(self.ckpt_cfg.working_csv, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1, 
                f"{train_loss:.4f}", 
                f"{train_acc:.4f}", 
                f"{epoch_loss:.4f}", 
                f"{epoch_acc.item():.4f}"
            ])
        
        # 2. Early Stopping Logic
        if epoch_loss < self.best_val_loss:
            self.best_val_loss = epoch_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            print(f"Early Stopping Counter: {self.patience_counter}/{self.train_cfg.patience}")
        
        state_dict_to_save = self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict()
        
        # 3. Save Best Model
        if epoch_acc > self.best_acc:
            self.best_acc = epoch_acc
            torch.save(state_dict_to_save, self.ckpt_cfg.best_model)
            print(f"*** New Best Accuracy. Model saved to {self.ckpt_cfg.best_model} ***")
        
        # 4. Save Epoch Checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': state_dict_to_save,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_acc': self.best_acc,
            'best_val_loss': self.best_val_loss,
            'patience_counter': self.patience_counter
        }, self.ckpt_cfg.working_ckpt)
        
        # 5. Trigger Early Stop Check
        if self.patience_counter >= self.train_cfg.patience:
            print("\n>>> Early stopping triggered! Validation loss stopped improving. <<<")
            return True
            
        return False

    def train(self):
        """Main orchestrator for the training and validation loops."""
        early_stop_triggered = False

        for epoch in range(self.start_epoch, self.train_cfg.epochs):
            if early_stop_triggered:
                break
                
            print(f"\nEpoch {epoch+1}/{self.train_cfg.epochs}")
            print("-" * 30)
            
            # Execute Phase 1: Train
            train_loss, train_acc = self._train_one_epoch(epoch)
            
            # Execute Phase 2: Validate
            early_stop_triggered = self._validate_one_epoch(epoch, train_loss, train_acc)

        print("-" * 30)
        print(f"Training complete. Overall Best Validation Accuracy: {self.best_acc:.4f}")