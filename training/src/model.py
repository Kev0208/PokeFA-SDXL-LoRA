# training/src/model.py

from typing import Dict, Any, Optional
import os
import torch
from torch import nn
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from model_utils import (
    resolve_dtype,
    set_attention_implementation,
    inject_lora_into_unet,
    inject_lora_into_text_encoder,
    lora_parameters,
    freeze_non_lora,
    summarize_processors,
    build_text_stack,
)


def _verify_prediction_type(noise_scheduler, want: str = "epsilon"):
    pt = getattr(noise_scheduler.config, "prediction_type", None)
    if pt != want:
        print(f"[model] WARNING: scheduler.prediction_type={pt} (expected {want}).")


def _must_exist(path: str, name: str):
    if not os.path.isdir(path):
        raise FileNotFoundError(f"[model] Expected folder missing: {name} -> {path}")


def _normalize_device(val, fallback_device):
    if val is None:
        return torch.device(fallback_device)
    s = str(val).lower()
    if s in ("auto", "same"):
        return torch.device(fallback_device)
    if s == "gpu":
        s = "cuda"
    return torch.device(s)


def _infer_variant_in_dir(dir_path: str) -> Optional[str]:
    if not os.path.isdir(dir_path):
        return None
    try:
        names = set(os.listdir(dir_path))
    except Exception:
        return None
    if "diffusion_pytorch_model.fp16.safetensors" in names:
        return "fp16"
    if "diffusion_pytorch_model.fp16.bin" in names:
        return "fp16"
    return None


def _infer_variant(base_path: str, subfolder: Optional[str]) -> Optional[str]:
    p = os.path.join(base_path, subfolder) if subfolder else base_path
    return _infer_variant_in_dir(p)


def _load_unet_from_path(path: str, dtype: torch.dtype, device: torch.device) -> UNet2DConditionModel:
    sub = os.path.join(path, "unet")
    if os.path.isdir(sub):
        variant = _infer_variant(path, "unet")
        return UNet2DConditionModel.from_pretrained(
            path,
            subfolder="unet",
            variant=variant,
            torch_dtype=dtype,
        ).to(device)

    variant = _infer_variant(path, None)
    try:
        return UNet2DConditionModel.from_pretrained(
            path,
            variant=variant,
            torch_dtype=dtype,
        ).to(device)
    except EnvironmentError:
        if variant is None and _infer_variant_in_dir(path) == "fp16":
            return UNet2DConditionModel.from_pretrained(
                path,
                variant="fp16",
                torch_dtype=dtype,
            ).to(device)
        raise


def _merge_te2_peft_into(te2: nn.Module, peft_dir: str) -> nn.Module:
    try:
        from peft import PeftModel
    except Exception as e:
        raise RuntimeError("[model] PEFT not available; cannot merge TE2 adapter.") from e

    peft_m = PeftModel.from_pretrained(te2, peft_dir)
    merged = peft_m.merge_and_unload()
    return merged


class SDXLBundle(nn.Module):
    def __init__(self, vae, unet, text_stack, noise_scheduler):
        super().__init__()
        self.vae = vae
        self.unet = unet
        self.noise_scheduler = noise_scheduler
        self.text_encoder = text_stack["text_encoder"]
        self.text_encoder_2 = text_stack["text_encoder_2"]
        self._text_stack = text_stack

    @property
    def text_stack(self) -> Dict[str, Any]:
        return {
            "tokenizer": self._text_stack["tokenizer"],
            "tokenizer_2": self._text_stack["tokenizer_2"],
            "text_encoder": self.text_encoder,
            "text_encoder_2": self.text_encoder_2,
        }

    def trainable_parameters(self):
        params = list(lora_parameters(self.unet))
        params.extend(list(lora_parameters(self.text_encoder)))
        params.extend(list(lora_parameters(self.text_encoder_2)))
        return params


def assemble_sdxl(cfg: Dict[str, Any], device: torch.device = torch.device("cuda")) -> SDXLBundle:
    m = cfg["model"]
    stage = str(m.get("stage", "base")).lower()
    base_path = m["base_path"]

    _must_exist(os.path.join(base_path, "vae"), "vae")
    _must_exist(os.path.join(base_path, "scheduler"), "scheduler")
    _must_exist(os.path.join(base_path, "text_encoder"), "text_encoder")
    _must_exist(os.path.join(base_path, "text_encoder_2"), "text_encoder_2")
    _must_exist(os.path.join(base_path, "tokenizer"), "tokenizer")
    _must_exist(os.path.join(base_path, "tokenizer_2"), "tokenizer_2")

    dtype_unet = resolve_dtype(m.get("unet_dtype", "bf16"))
    dtype_vae = resolve_dtype(m.get("vae_dtype", "fp32"))
    te_dtype_name = m.get("te_dtype", "fp32")
    te_device = _normalize_device(m.get("text_encoders_device", "cpu"), fallback_device=device)

    if stage == "refiner":
        ref_unet_path = str(m.get("refiner_unet_path", "")).strip()
        if not ref_unet_path:
            raise ValueError("[model] refiner_unet_path must be set when stage='refiner'.")
        unet = _load_unet_from_path(ref_unet_path, dtype_unet, device)
    else:
        _must_exist(os.path.join(base_path, "unet"), "unet")
        unet_variant = m.get("unet_variant") or _infer_variant(base_path, "unet")
        unet = UNet2DConditionModel.from_pretrained(
            base_path,
            subfolder="unet",
            variant=unet_variant,
            torch_dtype=dtype_unet,
        ).to(device)

    vae_variant = m.get("vae_variant") or _infer_variant(base_path, "vae")
    vae = AutoencoderKL.from_pretrained(
        base_path,
        subfolder="vae",
        variant=vae_variant,
        torch_dtype=dtype_vae,
    ).to(device)

    noise_scheduler = DDPMScheduler.from_pretrained(os.path.join(base_path, "scheduler"))
    _verify_prediction_type(noise_scheduler, want=m.get("prediction_type", "epsilon"))

    text_stack = build_text_stack(
        base_path,
        dtype_name=te_dtype_name,
        device=device,
        encoders_device=te_device,
    )

    if stage == "refiner":
        peft_dir = str(m.get("te2_peft_dir", "")).strip()
        if peft_dir:
            text_stack["text_encoder_2"] = _merge_te2_peft_into(text_stack["text_encoder_2"], peft_dir)
            desired_dtype = resolve_dtype(te_dtype_name)
            text_stack["text_encoder_2"] = text_stack["text_encoder_2"].to(
                device=te_device,
                dtype=desired_dtype,
            ).eval()
        for p in text_stack["text_encoder"].parameters():
            p.requires_grad_(False)
        for p in text_stack["text_encoder_2"].parameters():
            p.requires_grad_(False)

    impl = set_attention_implementation(
        unet,
        requested=m.get("attention_impl", "auto"),
        device=device,
        dtype=next(unet.parameters()).dtype,
    )
    try:
        if m.get("enable_grad_checkpointing", True):
            unet.enable_gradient_checkpointing()
    except Exception as e:
        print("[model] WARNING: failed to enable grad checkpointing:", e)

    def _rank0_print(msg: str):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                print(msg)
        else:
            print(msg)

    te1_dtype = next(text_stack["text_encoder"].parameters()).dtype
    te1_dev = next(text_stack["text_encoder"].parameters()).device
    te2_dtype = next(text_stack["text_encoder_2"].parameters()).dtype
    te2_dev = next(text_stack["text_encoder_2"].parameters()).device
    _rank0_print(
        "[model] attention_impl={}\n"
        "[model] UNet dtype={}, VAE dtype={}\n"
        "[model] TE1 dtype={}, device={} | TE2 dtype={}, device={}".format(
            impl,
            unet.dtype,
            vae.dtype,
            te1_dtype,
            te1_dev,
            te2_dtype,
            te2_dev,
        )
    )

    try:
        _rank0_print(f"[model] attn processors before LoRA: {summarize_processors(unet)}")
    except Exception as e:
        _rank0_print(f"[model] NOTE: could not summarize processors before injection: {e}")

    lora_cfg = cfg.get("lora", {"enable": True, "rank": 16, "alpha": 16, "dropout": 0.0})
    if lora_cfg.get("enable", True):
        unet = inject_lora_into_unet(
            unet,
            rank=int(lora_cfg.get("rank", 16)),
            alpha=int(lora_cfg.get("alpha", 16)),
            dropout=float(lora_cfg.get("dropout", 0.0)),
            adapter_name=str(lora_cfg.get("adapter_name", "default")),
            target_modules=list(lora_cfg.get("target_modules", ["to_q", "to_k", "to_v", "to_out.0"])),
        )
        unet = unet.to(device=device, dtype=dtype_unet)
        try:
            freeze_non_lora(unet)
        except Exception as e:
            _rank0_print(f"[model] WARNING: freeze_non_lora(unet) failed: {e}")
        try:
            _rank0_print(f"[model] attn processors after  LoRA: {summarize_processors(unet)}")
        except Exception as e:
            _rank0_print(f"[model] NOTE: could not summarize processors after injection: {e}")
    else:
        for p in unet.parameters():
            p.requires_grad_(False)

    te_lora_cfg = cfg.get("text_encoder_lora", {})
    te_lora_enabled = (stage == "base") and bool(te_lora_cfg.get("enable", False))

    if te_lora_enabled:
        te_rank = int(te_lora_cfg.get("rank", 16))
        te_alpha = int(te_lora_cfg.get("alpha", 16))
        te_drop = float(te_lora_cfg.get("dropout", 0.0))
        te_adapter_name = str(te_lora_cfg.get("adapter_name", "te"))
        te_targets = list(te_lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "out_proj"]))

        text_stack["text_encoder"] = inject_lora_into_text_encoder(
            text_stack["text_encoder"],
            rank=te_rank,
            alpha=te_alpha,
            dropout=te_drop,
            adapter_name=te_adapter_name,
            target_modules=te_targets,
        )
        text_stack["text_encoder_2"] = inject_lora_into_text_encoder(
            text_stack["text_encoder_2"],
            rank=te_rank,
            alpha=te_alpha,
            dropout=te_drop,
            adapter_name=te_adapter_name,
            target_modules=te_targets,
        )

        freeze_non_lora(text_stack["text_encoder"])
        freeze_non_lora(text_stack["text_encoder_2"])
        text_stack["text_encoder"].train()
        text_stack["text_encoder_2"].train()

        if te_device.type == "cuda":
            desired_dtype = resolve_dtype(te_dtype_name)
            text_stack["text_encoder"] = text_stack["text_encoder"].to(device=te_device, dtype=desired_dtype)
            text_stack["text_encoder_2"] = text_stack["text_encoder_2"].to(
                device=te_device,
                dtype=desired_dtype,
            )
    else:
        for p in text_stack["text_encoder"].parameters():
            p.requires_grad_(False)
        for p in text_stack["text_encoder_2"].parameters():
            p.requires_grad_(False)

    if str(m.get("vae_dtype", "fp32")).lower() in ("fp32", "float32"):
        if vae.dtype != torch.float32:
            vae = vae.float()

    return SDXLBundle(vae=vae, unet=unet, text_stack=text_stack, noise_scheduler=noise_scheduler)