import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset
from src.code.utils import SquarePad
from src.code.config_parser import DatasetConfig

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

def get_dataloaders(config: DatasetConfig):
    base_transforms = transforms.Compose([
        SquarePad(),
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_transforms = transforms.Compose([
        SquarePad(),
        transforms.Resize((config.image_size, config.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(root=config.data_dir)
    class_names = full_dataset.classes
    print(f"Dataset classes mapping: {class_names}")

    dataset_size = len(full_dataset)
    val_size = int(config.val_split * dataset_size)
    train_size = dataset_size - val_size

    train_subset, val_subset = random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_dataset = DatasetWrapper(train_subset, transform=train_transforms)
    val_dataset = DatasetWrapper(val_subset, transform=base_transforms)

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, 
                            num_workers=config.num_workers, pin_memory=True, persistent_workers=True),
        'val': DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, 
                          num_workers=config.num_workers, pin_memory=True, persistent_workers=True)
    }
    return dataloaders, len(class_names)