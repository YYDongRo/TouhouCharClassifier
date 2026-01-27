import torch
from torch.utils.data import Subset

from src.dataset import TouhouImageDataset


def get_datasets(root="data", val_ratio=0.2, train_transform=None, val_transform=None, seed=42):
    base = TouhouImageDataset(root)
    total = len(base)

    gen = torch.Generator()
    gen.manual_seed(seed)
    indices = torch.randperm(total, generator=gen).tolist()

    split = int(total * (1 - val_ratio))
    train_indices = indices[:split]
    val_indices = indices[split:]

    train_ds = Subset(TouhouImageDataset(root, transform=train_transform), train_indices)
    val_ds = Subset(TouhouImageDataset(root, transform=val_transform), val_indices)
    return train_ds, val_ds
