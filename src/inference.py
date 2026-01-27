import json
import torch
from PIL import Image
from torchvision import transforms

from src.model import create_model
from src.dataset import TouhouImageDataset


def load_model():
    ds = TouhouImageDataset("data")
    num_classes = len(ds.class_to_idx)

    model = create_model(num_classes)
    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    model.eval()

    idx_to_class = ds.idx_to_class
    try:
        with open("class_map.json", "r", encoding="utf-8") as f:
            class_to_idx = json.load(f)
            idx_to_class = {int(v): k for k, v in class_to_idx.items()}
    except FileNotFoundError:
        pass

    return model, idx_to_class


imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])


def predict(image_path):
    model, idx_to_class = load_model()

    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0)  # add batch dim

    with torch.no_grad():
        outputs = model(img)
        pred = torch.argmax(outputs, dim=1).item()

    return idx_to_class[pred]
