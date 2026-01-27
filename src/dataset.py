from pathlib import Path
from typing import Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class TouhouImageDataset(Dataset):
    """
    Expects data like:

    data/
      reimu/
      marisa/
      sakuya/

    You can also pass multiple roots to merge datasets
    (e.g., ["data", "memory"]).
    """

    def __init__(self, root_dir: str | list[str] | tuple[str, ...], transform=None):
        if isinstance(root_dir, (list, tuple)):
            self.root_dirs = [Path(p) for p in root_dir]
        else:
            self.root_dirs = [Path(root_dir)]
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        self._build_index()

    def _build_index(self):
        classes_set = set()
        for root in self.root_dirs:
            if not root.exists():
                continue
            for d in root.iterdir():
                if d.is_dir():
                    classes_set.add(d.name)

        classes = sorted(classes_set)
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        self.idx_to_class = {i: cls_name for cls_name, i in self.class_to_idx.items()}

        for root in self.root_dirs:
            if not root.exists():
                continue
            for cls_name in classes:
                cls_dir = root / cls_name
                if not cls_dir.exists():
                    continue
                for img_path in cls_dir.glob("*.*"):
                    if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                        self.image_paths.append(img_path)
                        self.labels.append(self.class_to_idx[cls_name])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, label
