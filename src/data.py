import os
from pathlib import Path
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset
from src.utils import SquarePad

class DatasetWrapper(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

def build_virtual_dataset(ds1_path, ds2_path, virtual_dir):
    virtual_dir_path = Path(virtual_dir)
    sources = [('ds1', Path(ds1_path)), ('ds2', Path(ds2_path))]
    virtual_dir_path.mkdir(parents=True, exist_ok=True)

    for prefix, source_path in sources:
        if not source_path.exists():
            print(f"Warning: Source path does not exist: {source_path}")
            continue
            
        for class_folder in source_path.iterdir():
            if class_folder.is_dir():
                class_name = class_folder.name
                if class_name == 'BacterialBlights':
                    continue
                
                target_class_folder = virtual_dir_path / class_name
                target_class_folder.mkdir(exist_ok=True)
                
                for img_file in class_folder.iterdir():
                    if img_file.is_file():
                        symlink_name = f"{prefix}_{img_file.name}"
                        symlink_path = target_class_folder / symlink_name
                        if not symlink_path.exists():
                            os.symlink(img_file, symlink_path)
                            
    print(f"Virtual dataset successfully built at: {virtual_dir}")
    return virtual_dir_path

def get_dataloaders(data_dir, batch_size, image_size, val_split=0.2, num_workers=4):
    train_transforms = transforms.Compose([
        SquarePad(),
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        SquarePad(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(root=data_dir)
    class_names = full_dataset.classes
    print(f"Mapping classes: {class_names}")

    dataset_size = len(full_dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size

    train_subset, val_subset = random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_dataset = DatasetWrapper(train_subset, transform=train_transforms)
    val_dataset = DatasetWrapper(val_subset, transform=val_transforms)

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=num_workers, pin_memory=True, 
                            prefetch_factor=2, persistent_workers=True),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                          num_workers=num_workers, pin_memory=True, 
                          prefetch_factor=2, persistent_workers=True)
    }
    return dataloaders, len(class_names)