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

    return model, ds.idx_to_class


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def predict(image_path):
    model, idx_to_class = load_model()

    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0)  # add batch dim

    with torch.no_grad():
        outputs = model(img)
        pred = torch.argmax(outputs, dim=1).item()

    return idx_to_class[pred]
