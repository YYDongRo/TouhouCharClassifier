import json
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.model import create_model
from src.dataset import TouhouImageDataset


def load_model():
    try:
        with open("class_map.json", "r", encoding="utf-8") as f:
            class_to_idx = json.load(f)
        class_to_idx = {k: int(v) for k, v in class_to_idx.items()}
    except FileNotFoundError:
        ds = TouhouImageDataset("data")
        class_to_idx = ds.class_to_idx

    idx_to_class = {v: k for k, v in class_to_idx.items()}

    model = create_model(len(class_to_idx))
    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    model.eval()
    return model, idx_to_class, class_to_idx


def get_gradcam(image_path, target_class=None, top_k=5):
    model, idx_to_class, class_to_idx = load_model()
    model.eval()

    # For ResNet, target the last convolutional layer (layer4)
    target_layers = [model.layer4[-1]]

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    img = Image.open(image_path).convert("RGB")
    img_resized = img.resize((224, 224))
    rgb_img = np.array(img_resized) / 255.0  # Normalized RGB for overlay
    
    x = transform(img).unsqueeze(0)

    # Get prediction
    with torch.no_grad():
        out = model(x)
        probs = F.softmax(out, dim=1).squeeze(0)
    
    if target_class is None:
        pred = probs.argmax().item()
    elif isinstance(target_class, str):
        pred = class_to_idx[target_class]
    else:
        pred = int(target_class)

    pred_label = idx_to_class[pred]

    # Use pytorch_grad_cam library
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(pred)]
    
    grayscale_cam = cam(input_tensor=x, targets=targets)
    grayscale_cam = grayscale_cam[0, :]  # Get the CAM for the first (only) image
    
    # Resize CAM to original image size
    orig_w, orig_h = img.size
    cam_resized = np.array(Image.fromarray((grayscale_cam * 255).astype(np.uint8)).resize((orig_w, orig_h))) / 255.0

    top_probs = probs.detach().cpu().tolist()
    probs_by_class = {
        idx_to_class[i]: top_probs[i]
        for i in range(len(top_probs))
    }
    probs_sorted = sorted(probs_by_class.items(), key=lambda x: x[1], reverse=True)
    probs_sorted = probs_sorted[:top_k]

    return cam_resized, img, pred_label, probs_sorted


def generate_cam_overlay(image_path, target_class=None, top_k=5):
    cam, img, label, probs_sorted = get_gradcam(
        image_path,
        target_class=target_class,
        top_k=top_k,
    )
    return cam, img, label, probs_sorted
