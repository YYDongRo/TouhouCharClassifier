import json
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from src.model import create_model
from src.dataset import TouhouImageDataset


def load_model():
    import json

    with open("class_map.json", "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    model = create_model(len(class_to_idx))
    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    model.eval()
    return model, idx_to_class


def get_gradcam(image_path):
    model, idx_to_class = load_model()
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
    target_layer.register_forward_hook(fwd_hook)
    target_layer.register_full_backward_hook(bwd_hook)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0)

    out = model(x)
    pred = out.argmax().item()
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

    return cam, img, pred_label



def generate_cam_overlay(image_path):
    cam, img, label = get_gradcam(image_path)
    return cam, img, label
