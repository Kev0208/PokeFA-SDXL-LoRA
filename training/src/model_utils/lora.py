# training/src/model_utils/lora.py

from __future__ import annotations
from typing import Dict, Iterable, List
import torch
from torch import nn
from diffusers import UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor2_0

try:
    from peft import LoraConfig, get_peft_model
except Exception as e:
    raise RuntimeError(
        "[lora] PEFT not available. Please `pip install peft` to use LoRA adapters."
    ) from e


def lora_parameters(module: nn.Module) -> Iterable[torch.nn.Parameter]:
    for name, p in module.named_parameters():
        if "lora_" in name:
            yield p


def freeze_non_lora(module: nn.Module):
    for p in module.parameters():
        p.requires_grad_(False)
    for p in lora_parameters(module):
        p.requires_grad_(True)


def _default_unet_target_modules() -> List[str]:
    return ["to_q", "to_k", "to_v", "to_out.0"]


def _default_te_target_modules() -> List[str]:
    return ["q_proj", "k_proj", "v_proj", "out_proj"]


def inject_lora_into_unet(
    unet: UNet2DConditionModel,
    *,
    rank: int = 16,
    alpha: int = 16,
    dropout: float = 0.0,
    adapter_name: str = "default",
    target_modules: List[str] | None = None,
) -> UNet2DConditionModel:
    try:
        unet.set_attn_processor(AttnProcessor2_0())
    except Exception:
        pass

    tm = target_modules or _default_unet_target_modules()
    lcfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        init_lora_weights="gaussian",
        target_modules=tm,
    )
    unet.add_adapter(lcfg, adapter_name=adapter_name)
    try:
        unet.set_adapter(adapter_name)
    except Exception:
        pass
    freeze_non_lora(unet)
    return unet


def inject_lora_into_text_encoder(
    text_encoder: nn.Module,
    *,
    rank: int = 16,
    alpha: int = 16,
    dropout: float = 0.0,
    adapter_name: str = "te",
    target_modules: List[str] | None = None,
) -> nn.Module:
    tm = target_modules or _default_te_target_modules()
    lcfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        init_lora_weights="gaussian",
        target_modules=tm,
    )
    te = get_peft_model(text_encoder, lcfg)
    freeze_non_lora(te)
    return te


def summarize_processors(unet: UNet2DConditionModel) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for _, proc in unet.attn_processors.items():
        k = proc.__class__.__name__
        counts[k] = counts.get(k, 0) + 1
    return counts