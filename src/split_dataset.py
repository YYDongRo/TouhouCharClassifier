from torch.utils.data import random_split
from src.dataset import TouhouImageDataset

def get_datasets(root="data", val_ratio=0.2):
    full = TouhouImageDataset(root)

    val_size = int(len(full) * val_ratio)
    train_size = len(full) - val_size

    train_ds, val_ds = random_split(full, [train_size, val_size])
    return train_ds, val_ds
