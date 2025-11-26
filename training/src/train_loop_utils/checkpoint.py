import os
import json
from typing import Dict, Any, Tuple, Optional

import torch

from train_loop_utils.dist import is_main_process

try:
    import peft  # noqa: F401
    _PEFT_AVAILABLE = True
except Exception:
    _PEFT_AVAILABLE = False


def _fsync_replace(tmp_path: str, dst_path: str):
    with open(tmp_path, "rb+") as f:
        f.flush()
        os.fsync(f.fileno())
    os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
    os.replace(tmp_path, dst_path)
    dirfd = os.open(os.path.dirname(dst_path) or ".", os.O_DIRECTORY)
    try:
        os.fsync(dirfd)
    finally:
        os.close(dirfd)


def save_lora_safetensors(unet, path: str) -> Optional[str]:
    if not is_main_process():
        return None

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    save_dir = os.path.dirname(path)
    weight_name = os.path.basename(path)

    wrote_path: Optional[str] = None

    try:
        unet.save_attn_procs(
            save_dir,
            weight_name=weight_name,
            safe_serialization=True,
        )
        wrote_path = path
    except Exception:
        wrote_path = None

    if wrote_path is None:
        try:
            from safetensors.torch import save_file
            state = unet.attn_processors.state_dict()
            state_cpu = {k: v.detach().float().cpu() for k, v in state.items()}
            tmp = path + ".tmp"
            save_file(state_cpu, tmp)
            _fsync_replace(tmp, path)
            wrote_path = path
        except Exception:
            pt_path = path.replace(".safetensors", ".pt")
            state = {
                k: v.detach().float().cpu()
                for k, v in unet.attn_processors.state_dict().items()
            }
            tmp = pt_path + ".tmp"
            torch.save(state, tmp)
            _fsync_replace(tmp, pt_path)
            wrote_path = pt_path

    return wrote_path


def save_te_lora_safetensors(
    text_encoder, text_encoder_2, path: str
) -> Optional[str]:
    if not is_main_process():
        return None

    state: Dict[str, torch.Tensor] = {}
    for prefix, module in (
        ("text_encoder", text_encoder),
        ("text_encoder_2", text_encoder_2),
    ):
        for name, p in module.named_parameters():
            if "lora_" in name and p.requires_grad:
                state[f"{prefix}.{name}"] = p.detach().float().cpu()

    if not state:
        return None

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    wrote_path: Optional[str] = None

    try:
        from safetensors.torch import save_file
        tmp = path + ".tmp"
        save_file(state, tmp)
        _fsync_replace(tmp, path)
        wrote_path = path
    except Exception:
        pt_path = path.replace(".safetensors", ".pt")
        tmp = pt_path + ".tmp"
        torch.save(state, tmp)
        _fsync_replace(tmp, pt_path)
        wrote_path = pt_path

    return wrote_path


def _load_te_lora_safetensors(bundle, path: str, device):
    if not os.path.isfile(path):
        return

    try:
        if path.endswith(".safetensors"):
            from safetensors.torch import load_file
            sd = load_file(path)
        else:
            sd = torch.load(path, map_location="cpu")
    except Exception as e:
        print(f"[ckpt] WARNING: failed to load TE LoRA from {path}: {e}")
        return

    te1 = bundle.text_encoder
    te2 = bundle.text_encoder_2

    te1_sd: Dict[str, torch.Tensor] = {}
    te2_sd: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if k.startswith("text_encoder."):
            te1_sd[k[len("text_encoder."):]] = v
        elif k.startswith("text_encoder_2."):
            te2_sd[k[len("text_encoder_2."):]] = v

    try:
        if te1_sd:
            s = te1.state_dict()
            s.update(te1_sd)
            te1.load_state_dict(s, strict=False)
            te1.to(device=device, dtype=next(te1.parameters()).dtype)
        if te2_sd:
            s = te2.state_dict()
            s.update(te2_sd)
            te2.load_state_dict(s, strict=False)
            te2.to(device=device, dtype=next(te2.parameters()).dtype)
    except Exception as e:
        print(f"[ckpt] WARNING: failed to apply TE LoRA from {path}: {e}")


def save_training_state(
    dst_pt: str,
    optimizer,
    lr_sched,
    opt_step: int,
    micro_step: int,
    cfg: Dict[str, Any],
):
    if not is_main_process():
        return

    state = {
        "opt": optimizer.state_dict(),
        "sched": lr_sched.state_dict() if lr_sched is not None else None,
        "opt_step": int(opt_step),
        "micro_step": int(micro_step),
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else [],
        },
        "py_random": None,
        "np_random": None,
        "cfg": cfg,
    }
    try:
        import random
        import numpy as np

        state["py_random"] = random.getstate()
        state["np_random"] = np.random.get_state()
    except Exception:
        pass

    os.makedirs(os.path.dirname(dst_pt) or ".", exist_ok=True)
    tmp = dst_pt + ".tmp"
    torch.save(state, tmp)
    _fsync_replace(tmp, dst_pt)

    try:
        meta = {"opt_step": int(opt_step), "micro_step": int(micro_step)}
        with open(
            os.path.join(os.path.dirname(dst_pt), "state_meta.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def load_training_state(
    resume_path: str, bundle, optimizer, lr_sched, device
) -> Tuple[int, int]:
    if resume_path.endswith(".safetensors") or resume_path.endswith(".pt"):
        base_dir = os.path.dirname(resume_path)
        fname = os.path.basename(resume_path)

        unet_lora_path = resume_path
        if "unet-lora" in fname:
            te_fname = fname.replace("unet-lora", "te-lora")
        else:
            te_fname = fname.replace(
                ".safetensors", "-te-lora.safetensors"
            ).replace(".pt", "-te-lora.pt")
        te_lora_path = os.path.join(base_dir, te_fname)
        state_pt = os.path.join(base_dir, "state.pt")
    else:
        base_dir = resume_path
        unet_lora_path = os.path.join(base_dir, "last-unet-lora.safetensors")
        te_lora_path = os.path.join(base_dir, "last-te-lora.safetensors")
        if not os.path.isfile(unet_lora_path):
            unet_lora_path = os.path.join(base_dir, "last.safetensors")
        if not os.path.isfile(unet_lora_path):
            unet_lora_path = os.path.join(base_dir, "last.pt")
        state_pt = os.path.join(base_dir, "state.pt")

    if os.path.isfile(unet_lora_path):
        try:
            if unet_lora_path.endswith(".safetensors"):
                bundle.unet.load_attn_procs(
                    os.path.dirname(unet_lora_path),
                    weight_name=os.path.basename(unet_lora_path),
                    local_files_only=True,
                )
            else:
                sd = torch.load(unet_lora_path, map_location="cpu")
                ap = bundle.unet.attn_processors.state_dict()
                ap.update(sd)
                bundle.unet.attn_processors.load_state_dict(ap, strict=False)
        except Exception:
            try:
                from safetensors.torch import load_file
                sd = load_file(unet_lora_path)
                ap = bundle.unet.attn_processors.state_dict()
                ap.update(sd)
                bundle.unet.attn_processors.load_state_dict(ap, strict=False)
            except Exception as e:
                print(
                    f"[ckpt] WARNING: failed to load UNet LoRA from {unet_lora_path}: {e}"
                )

        bundle.unet.to(
            device=device, dtype=next(bundle.unet.parameters()).dtype
        )

    if os.path.isfile(te_lora_path):
        _load_te_lora_safetensors(bundle, te_lora_path, device=device)

    opt_step = micro_step = 0
    if os.path.isfile(state_pt):
        ckpt = torch.load(state_pt, map_location=device)
        try:
            optimizer.load_state_dict(ckpt["opt"])
        except Exception as e:
            print(f"[ckpt] WARNING: optimizer state failed to load: {e}")

        if lr_sched is not None and ckpt.get("sched") is not None:
            try:
                lr_sched.load_state_dict(ckpt["sched"])
            except Exception as e:
                print(f"[ckpt] WARNING: scheduler state failed to load: {e}")

        opt_step = int(ckpt.get("opt_step", 0))
        micro_step = int(ckpt.get("micro_step", 0))

        try:
            torch.set_rng_state(ckpt["rng"]["torch"])
        except Exception:
            pass
        if torch.cuda.is_available():
            try:
                for i, s in enumerate(ckpt["rng"]["cuda"]):
                    torch.cuda.set_rng_state(s, device=i)
            except Exception:
                pass
        try:
            import random
            import numpy as np

            if ckpt.get("py_random") is not None:
                random.setstate(ckpt["py_random"])
            if ckpt.get("np_random") is not None:
                np.random.set_state(ckpt["np_random"])
        except Exception:
            pass

    return opt_step, micro_step
