import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


MEMORY_DIR = Path("memory")
INDEX_FILE = MEMORY_DIR / "index.jsonl"


imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])


def ensure_memory_dir() -> None:
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)


def _compute_embedding(model: torch.nn.Module, image: Image.Image) -> torch.Tensor:
    model.eval()
    x = transform(image).unsqueeze(0)
    features = {}

    def hook(_m, _i, o):
        features["emb"] = torch.flatten(o, 1)

    handle = model.avgpool.register_forward_hook(hook)
    with torch.no_grad():
        _ = model(x)
    handle.remove()
    return features["emb"].squeeze(0)


def save_memory_example(
    model: torch.nn.Module,
    image: Image.Image,
    label: str,
    source_path: Optional[str] = None,
    bbox: Optional[Tuple[int, int, int, int]] = None,
    note: str = "",
) -> str:
    ensure_memory_dir()
    label_dir = MEMORY_DIR / label
    label_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    file_name = f"{timestamp}.png"
    save_path = label_dir / file_name
    image.save(save_path)

    emb = _compute_embedding(model, image).detach().cpu().tolist()

    record = {
        "path": str(save_path).replace("\\", "/"),
        "label": label,
        "embedding": emb,
        "note": note,
        "source": source_path,
        "bbox": bbox,
        "created_at": timestamp,
    }

    with INDEX_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return str(save_path)


def _load_index() -> List[Dict]:
    if not INDEX_FILE.exists():
        return []
    entries = []
    with INDEX_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def count_memory_entries() -> int:
    if not INDEX_FILE.exists():
        return 0
    with INDEX_FILE.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def find_best_matches(
    model: torch.nn.Module,
    image: Image.Image,
    top_k: int = 5,
) -> List[Dict[str, float]]:
    entries = _load_index()
    if not entries:
        return []

    query = _compute_embedding(model, image).detach().cpu()
    query = F.normalize(query, dim=0)

    emb_matrix = torch.tensor(
        [entry["embedding"] for entry in entries],
        dtype=torch.float32,
    )
    emb_matrix = F.normalize(emb_matrix, dim=1)
    scores = torch.mv(emb_matrix, query)

    top_scores, top_idx = torch.topk(scores, k=min(top_k, len(entries)))
    results = []
    for score, idx in zip(top_scores.tolist(), top_idx.tolist()):
        entry = entries[idx]
        results.append({
            "label": entry.get("label", "unknown"),
            "score": float(score),
            "path": entry.get("path"),
        })
    return results