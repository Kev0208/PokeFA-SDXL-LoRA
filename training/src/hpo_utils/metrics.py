from __future__ import annotations

import json
from typing import List, Dict, Any

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def _l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12):
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def load_clip_for_adherence(local_path: str, device: torch.device):
    model = CLIPModel.from_pretrained(
        local_path, local_files_only=True, torch_dtype=torch.float16
    ).to(device).eval()
    proc = CLIPProcessor.from_pretrained(local_path, local_files_only=True)
    return model, proc


def clip_cosine_for_batch(
    model: CLIPModel,
    proc: CLIPProcessor,
    images: List[Image.Image],
    text: str,
    device: torch.device,
) -> List[float]:
    with torch.no_grad():
        batch = proc(
            text=[text] * len(images),
            images=images,
            return_tensors="pt",
            padding=True,
        )
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        out = model(**batch)
        img = _l2_normalize(out.image_embeds.float(), dim=-1)
        txt = _l2_normalize(out.text_embeds.float(), dim=-1)
        cos = (img @ txt.T).diag()
        return cos.cpu().tolist()


def load_aesthetic_head(head_path: str, in_dim: int = 768) -> torch.nn.Module:
    sd = torch.load(head_path, map_location="cpu")
    linear = torch.nn.Linear(in_dim, 1)
    try:
        linear.load_state_dict(sd, strict=True)
        return linear
    except Exception:
        pass
    mlp = torch.nn.Sequential(
        torch.nn.Linear(in_dim, 1024),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(1024, 128),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(128, 64),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(64, 16),
        torch.nn.Linear(16, 1),
    )
    if all(k.startswith("layers.") for k in sd.keys()):
        sd = {k.replace("layers.", "", 1): v for k, v in sd.items()}
    mlp.load_state_dict(sd, strict=True)
    return mlp


def load_clip_vision(local_path: str, device: torch.device):
    clip = CLIPModel.from_pretrained(
        local_path, local_files_only=True, torch_dtype=torch.float16
    ).to(device).eval()
    proc = CLIPProcessor.from_pretrained(local_path, local_files_only=True)
    return clip, proc


def aesthetic_scores_for_images(
    clip_vis: CLIPModel,
    proc: CLIPProcessor,
    head: torch.nn.Module,
    images: List[Image.Image],
    device: torch.device,
) -> List[float]:
    head = head.to(device).eval()
    with torch.no_grad():
        batch = proc(images=images, return_tensors="pt")
        pixel = batch["pixel_values"].to(device, dtype=torch.float16)
        vis = clip_vis.vision_model(pixel)
        feats = clip_vis.visual_projection(vis.pooler_output)
        feats = _l2_normalize(feats.float(), dim=-1)
        out = head(feats).squeeze(-1)
        return out.detach().cpu().tolist()


def collapse_proxy(
    adherence: float, aesthetic: float, w_adherence: float = 0.6, w_aesthetic: float = 0.4
) -> float:
    return float(w_adherence) * float(adherence) + float(w_aesthetic) * float(aesthetic) / 10


def summarize_metrics(per_prompt: List[Dict[str, Any]]) -> Dict[str, Any]:
    adhere = sum(d["clip_mean"] for d in per_prompt) / max(1, len(per_prompt))
    aesth = sum(d["aesthetic_mean"] for d in per_prompt) / max(1, len(per_prompt))
    score = collapse_proxy(adhere, aesth)
    return {"adherence_mean": adhere, "aesthetic_mean": aesth, "score": score}


def dump_metrics_json(path: str, summary: Dict[str, Any], per_prompt: List[Dict[str, Any]]):
    payload = {"summary": summary, "prompts": per_prompt}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
