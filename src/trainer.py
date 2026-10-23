import os
import torch
import torch.nn as nn
from src.utils import save_checkpoint

class Trainer:
    def __init__(self, model, dataloaders, criterion, optimizer, device, args):
        self.model = model
        self.dataloaders = dataloaders
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.args = args
        
        self.start_epoch = 0
        self.best_acc = 0.0
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        self._load_checkpoint()
        
        if torch.cuda.device_count() > 1:
            print(f"Utilizing {torch.cuda.device_count()} GPUs via DataParallel.")
            self.model = nn.DataParallel(self.model)

    def _load_checkpoint(self):
        checkpoint_path = None
        if os.path.exists(self.args.working_ckpt):
            checkpoint_path = self.args.working_ckpt
        elif self.args.input_ckpt and os.path.exists(self.args.input_ckpt):
            checkpoint_path = self.args.input_ckpt

        if checkpoint_path:
            print(f"Loading checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model = self.model.to(self.device)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
                        
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_acc = checkpoint['best_acc']
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            self.patience_counter = checkpoint.get('patience_counter', 0)
            print(f"Resuming at Epoch {self.start_epoch}. Current Best Acc: {self.best_acc:.4f}")
        else:
            print("No checkpoint found. Starting training from scratch.")
            self.model = self.model.to(self.device)

    def train(self):
        early_stop_triggered = False

        for epoch in range(self.start_epoch, self.args.epochs):
            if early_stop_triggered:
                break
                
            print(f"\nEpoch {epoch+1}/{self.args.epochs}")
            print("-" * 30)
            
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()
                    
                running_loss = 0.0
                running_corrects = 0
                self.optimizer.zero_grad()
                total_batches = len(self.dataloaders[phase])
                
                for i, (inputs, labels) in enumerate(self.dataloaders[phase]):
                    inputs = inputs.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                    
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)
                        
                        if phase == 'train':
                            (loss / self.args.accum_steps).backward()
                            if (i + 1) % self.args.accum_steps == 0 or (i + 1) == total_batches:
                                self.optimizer.step()
                                self.optimizer.zero_grad()
                                
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    print(f"\r{phase.capitalize()} | Batch {i+1}/{total_batches}", end="")
                    
                epoch_loss = running_loss / len(self.dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(self.dataloaders[phase].dataset)
                print(f"\r{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}" + " " * 20)
                
                if phase == 'val':
                    is_best = False
                    if epoch_acc > self.best_acc:
                        self.best_acc = epoch_acc
                        is_best = True
                        print(f"*** New Best Accuracy. Model saved to {self.args.best_model} ***")

                    if epoch_loss < self.best_val_loss:
                        self.best_val_loss = epoch_loss
                        self.patience_counter = 0
                    else:
                        self.patience_counter += 1
                        print(f"Early Stopping Counter: {self.patience_counter}/{self.args.patience}")
                    
                    state_dict = self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict()
                    
                    save_checkpoint({
                        'epoch': epoch,
                        'model_state_dict': state_dict,
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'best_acc': self.best_acc,
                        'best_val_loss': self.best_val_loss,
                        'patience_counter': self.patience_counter
                    }, is_best, self.args.working_ckpt, self.args.best_model)
                    
                    if self.patience_counter >= self.args.patience:
                        print("\n>>> Early stopping triggered! <<<")
                        early_stop_triggered = True
                        break

        print("-" * 30)
        print(f"Training complete. Overall Best Validation Accuracy: {self.best_acc:.4f}")