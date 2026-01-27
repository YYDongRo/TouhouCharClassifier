import json
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

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

    target_layer = model.layer4[-1]
    activations = None
    gradients = None

    def fwd_hook(m, i, o):
        nonlocal activations
        activations = o

    def bwd_hook(m, gin, gout):
        nonlocal gradients
        gradients = gout[0]

    # Better than register_backward_hook (deprecated-ish behavior)
    fwd_handle = target_layer.register_forward_hook(fwd_hook)
    bwd_handle = target_layer.register_full_backward_hook(bwd_hook)

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0)

    out = model(x)
    probs = F.softmax(out, dim=1).squeeze(0)

    if target_class is None:
        pred = probs.argmax().item()
    elif isinstance(target_class, str):
        pred = class_to_idx[target_class]
    else:
        pred = int(target_class)

    pred_label = idx_to_class[pred]

    model.zero_grad(set_to_none=True)
    out[0, pred].backward()

    act = activations.squeeze(0)      # [C,H,W]
    grad = gradients.squeeze(0)       # [C,H,W]
    weights = grad.mean(dim=(1, 2))   # [C]

    cam = (weights[:, None, None] * act).sum(dim=0)
    cam = F.relu(cam)

    # Normalize safely (avoid divide-by-zero)
    cam_min, cam_max = cam.min(), cam.max()
    cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

    # cam currently corresponds to the 224x224 input.
    # Resize cam to ORIGINAL image size for overlay.
    orig_w, orig_h = img.size  # PIL is (W,H)
    cam = cam.unsqueeze(0).unsqueeze(0)  # [1,1,H,W] (H,W here is layer spatial size)
    cam = F.interpolate(cam, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
    cam = cam.squeeze().detach().cpu().numpy()  # [orig_h, orig_w]

    fwd_handle.remove()
    bwd_handle.remove()

    top_probs = probs.detach().cpu().tolist()
    probs_by_class = {
        idx_to_class[i]: top_probs[i]
        for i in range(len(top_probs))
    }
    probs_sorted = sorted(probs_by_class.items(), key=lambda x: x[1], reverse=True)
    probs_sorted = probs_sorted[:top_k]

    return cam, img, pred_label, probs_sorted



def generate_cam_overlay(image_path, target_class=None, top_k=5):
    cam, img, label, probs_sorted = get_gradcam(
        image_path,
        target_class=target_class,
        top_k=top_k,
    )
    return cam, img, label, probs_sorted
