# training/src/model_utils/text.py

from typing import Dict, Any, Tuple, List, Optional, Literal
import torch
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection

_DTYPE_MAP = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


@torch.no_grad()
def build_text_stack(
    base_path: str,
    dtype_name: str = "fp32",
    device: torch.device = torch.device("cpu"),
    *,
    encoders_device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    te1_path = f"{base_path}/text_encoder"
    te2_path = f"{base_path}/text_encoder_2"
    tok1_path = f"{base_path}/tokenizer"
    tok2_path = f"{base_path}/tokenizer_2"

    tok1 = CLIPTokenizer.from_pretrained(tok1_path, subfolder=None)
    tok2 = CLIPTokenizer.from_pretrained(tok2_path, subfolder=None)

    dtype = _DTYPE_MAP.get(dtype_name, torch.float32)
    enc_dev = encoders_device if encoders_device is not None else device

    te1 = CLIPTextModel.from_pretrained(te1_path).to(device=enc_dev, dtype=dtype).eval()
    te2 = CLIPTextModelWithProjection.from_pretrained(te2_path).to(device=enc_dev, dtype=dtype).eval()

    return {
        "tokenizer": tok1,
        "tokenizer_2": tok2,
        "text_encoder": te1,
        "text_encoder_2": te2,
    }


def encode_batch_with_cond_dropout(
    text_stack: Dict[str, Any],
    prompts: List[str],
    device: torch.device,
    cond_dropout_prob: float = 0.10,
    *,
    dropout_rng: Optional[torch.Generator] = None,
    out_dtype: Optional[torch.dtype] = None,
    train_te: bool = False,
    concat_mode: Literal["both", "te2_only"] = "both",
) -> Tuple[torch.Tensor, torch.Tensor]:
    tok1, tok2 = text_stack["tokenizer"], text_stack["tokenizer_2"]
    te1, te2 = text_stack["text_encoder"], text_stack["text_encoder_2"]

    B = len(prompts)
    if cond_dropout_prob > 0:
        g = dropout_rng if dropout_rng is not None else torch.Generator()
        mask = torch.rand(B, generator=g) > cond_dropout_prob
        masked_prompts = [p if bool(mask[i].item()) else "" for i, p in enumerate(prompts)]
    else:
        masked_prompts = prompts

    max_len_1 = getattr(tok1, "model_max_length", 77)
    max_len_2 = getattr(tok2, "model_max_length", 77)

    t1 = tok1(masked_prompts, padding="max_length", truncation=True, max_length=max_len_1, return_tensors="pt")
    t2 = tok2(masked_prompts, padding="max_length", truncation=True, max_length=max_len_2, return_tensors="pt")

    enc_dev = next(te2.parameters()).device
    t1 = {k: v.to(enc_dev) for k, v in t1.items()}
    t2 = {k: v.to(enc_dev) for k, v in t2.items()}

    with torch.set_grad_enabled(train_te):
        te2_out = te2(**t2)
        if concat_mode == "both":
            te1_out = te1(**t1)

    if concat_mode == "both":
        hs1 = te1_out.last_hidden_state
        hs2 = te2_out.last_hidden_state
        prompt_embeds = torch.cat([hs1, hs2], dim=-1)
    else:
        prompt_embeds = te2_out.last_hidden_state

    pooled = te2_out.text_embeds

    if out_dtype is None:
        out_dtype = prompt_embeds.dtype

    prompt_embeds = prompt_embeds.to(device=device, dtype=out_dtype, non_blocking=True)
    pooled = pooled.to(device=device, dtype=out_dtype, non_blocking=True)

    return prompt_embeds, pooled
