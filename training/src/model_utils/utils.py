from typing import Dict, Any, Tuple, Iterable, Literal, Optional
import torch


def resolve_dtype(name: str):
    n = (name or "").lower()
    if n in ("bf16", "bfloat16"):
        return torch.bfloat16
    if n in ("fp16", "float16", "half"):
        return torch.float16
    return torch.float32


def set_attention_implementation(
    unet,
    requested: str = "auto",
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> str:
    req = (requested or "auto").lower()
    errors = []

    from diffusers.models.attention_processor import AttnProcessor2_0
    try:
        from diffusers.models.attention_processor import FusedAttnProcessor2_0  # type: ignore
    except Exception:
        FusedAttnProcessor2_0 = None
    try:
        from diffusers.models.attention_processor import XFormersAttnProcessor  # type: ignore
    except Exception:
        XFormersAttnProcessor = None

    def _resync():
        if device is not None or dtype is not None:
            ref_param = next(unet.parameters(), None)
            cur_dev = ref_param.device if ref_param is not None else None
            cur_dt = ref_param.dtype if ref_param is not None else None
            unet.to(device=(device or cur_dev), dtype=(dtype or cur_dt))

    def _try_fused():
        if FusedAttnProcessor2_0 is None:
            raise RuntimeError("FusedAttnProcessor2_0 not available in this diffusers/torch build.")
        unet.set_attn_processor(FusedAttnProcessor2_0())
        _resync()
        return "fused"

    def _try_sdpa():
        unet.set_attn_processor(AttnProcessor2_0())
        _resync()
        return "sdpa"

    def _try_xformers():
        if XFormersAttnProcessor is None:
            raise RuntimeError("XFormersAttnProcessor not available.")
        import xformers  # noqa: F401

        unet.set_attn_processor(XFormersAttnProcessor())
        _resync()
        return "xformers"

    if req == "fused":
        try_order = [_try_fused]
    elif req == "sdpa":
        try_order = [_try_sdpa]
    elif req == "xformers":
        try_order = [_try_xformers]
    else:
        try_order = [_try_fused, _try_sdpa, _try_xformers]

    for attempt in try_order:
        try:
            impl = attempt()
            if req != impl and req != "auto":
                print(f"[model] WARNING: requested '{req}' but installed '{impl}' (fallback).")
            return impl
        except Exception as e:
            errors.append(f"{attempt.__name__}: {e}")

    print(
        "[model] WARNING: failed to set requested attention processors; "
        "leaving UNet defaults. Errors:\n  - " + "\n  - ".join(errors)
    )
    _resync()
    return "default"


def make_time_ids(
    sample: Dict[str, Any],
    device: torch.device,
    *,
    order: Literal["diffusers", "wfirst"] = "diffusers",
    out_dtype: Optional[torch.dtype] = torch.float32,
) -> torch.Tensor:
    try:
        ow, oh = sample["original_size"]
        cx, cy = sample["crop_coords_top_left"]
        tw, th = sample["target_size"]
    except Exception as e:
        raise KeyError(
            "Sample must contain keys 'original_size', 'crop_coords_top_left', 'target_size', "
            "each a tuple of length 2. "
            f"Got: original_size={sample.get('original_size')}, "
            f"crop_coords_top_left={sample.get('crop_coords_top_left')}, "
            f"target_size={sample.get('target_size')}"
        ) from e

    if order == "diffusers":
        vec = [int(oh), int(ow), int(cy), int(cx), int(th), int(tw)]
    else:
        vec = [int(ow), int(oh), int(cx), int(cy), int(tw), int(th)]

    t = torch.tensor(vec, device=device)
    if out_dtype is not None:
        t = t.to(dtype=out_dtype)
    return t


def batch_make_time_ids(
    samples: Iterable[Dict[str, Any]],
    device: torch.device,
    *,
    order: Literal["diffusers", "wfirst"] = "diffusers",
    out_dtype: Optional[torch.dtype] = torch.float32,
) -> torch.Tensor:
    tids = [make_time_ids(s, device, order=order, out_dtype=out_dtype) for s in samples]
    return torch.stack(tids, dim=0)