from __future__ import annotations
import os
import time
from typing import Dict, Any

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

from peft import PeftModel

from dataloader import build_loaders
from model import assemble_sdxl
from model_utils import encode_batch_with_cond_dropout, apply_min_snr_weight
from train_loop_utils.dist import (
    ddp_init_if_needed,
    ddp_destroy_if_needed,
    is_main_process,
    barrier,
    get_world_size,
    rank0_print,
    seed_all,
)
from train_loop_utils.sched import build_cosine_with_warmup
from train_loop_utils.checkpoint import (
    save_lora_safetensors,
    save_training_state,
    load_training_state,
)
from train_loop_utils.validation import run_val
from train_loop_utils.utils import atomic_copy, write_jsonl, set_speed_toggles


def _cast_lora_master_fp32_module(module, label: str = "module"):
    num = 0
    for n, p in module.named_parameters():
        if "lora_" in n and p.requires_grad:
            if p.dtype != torch.float32:
                p.data = p.data.float()
                num += 1
    return num


def _assert_lora_has_trainable_AB(module: torch.nn.Module, tag: str):
    saw_A = saw_B = 0
    for n, p in module.named_parameters():
        if "lora_A" in n and p.requires_grad:
            saw_A += 1
        if "lora_B" in n and p.requires_grad:
            saw_B += 1
    assert saw_A > 0 and saw_B > 0, f"{tag}: LoRA A/B not both trainable (A={saw_A}, B={saw_B})"


def _expected_timeid_count_from_unet(unet) -> int:
    add_emb = getattr(unet, "add_embedding", None)
    if add_emb is None:
        return 6
    lin1 = getattr(add_emb, "linear_1", None)
    if lin1 is None or not hasattr(lin1, "in_features"):
        return 6
    in_feat = int(lin1.in_features)
    txt_dim = int(getattr(unet.config, "text_embed_dim", 1280))
    if in_feat <= txt_dim:
        return 6
    diff = in_feat - txt_dim
    if diff % 256 != 0:
        return 6
    return diff // 256


def _pack_time_ids(batch, device, *, want_k: int) -> torch.Tensor:
    rows = []
    for i in range(len(batch)):
        orig = batch[i].get("original_size", [1024, 1024])
        crop = batch[i].get("crop_coords", [0, 0])
        targ = batch[i].get("target_size", [1024, 1024])

        if torch.is_tensor(orig):
            orig = orig.tolist()
        if torch.is_tensor(crop):
            crop = crop.tolist()
        if torch.is_tensor(targ):
            targ = targ.tolist()

        six = [
            int(orig[0]),
            int(orig[1]),
            int(crop[0]),
            int(crop[1]),
            int(targ[0]),
            int(targ[1]),
        ]
        if want_k <= 6:
            vec = six[:want_k]
        else:
            vec = six + [0] * (want_k - 6)
        rows.append(vec)
    return torch.tensor(rows, device=device, dtype=torch.long)


def _save_te_peft_adapters(ckpt_dir: str, tag: str, te1, te2, rank0: bool):
    def _save_if_peft(model, step_dir: str, last_dir: str, label: str):
        base_model = model.module if isinstance(model, DDP) else model
        if isinstance(base_model, PeftModel):
            base_model.save_pretrained(step_dir)
            if os.path.isdir(last_dir):
                try:
                    for root, dirs, files in os.walk(last_dir, topdown=False):
                        for f in files:
                            try:
                                os.remove(os.path.join(root, f))
                            except Exception:
                                pass
                        for d in dirs:
                            try:
                                os.rmdir(os.path.join(root, d))
                            except Exception:
                                pass
                    os.rmdir(last_dir)
                except Exception:
                    pass
            base_model.save_pretrained(last_dir)
        else:
            if rank0:
                rank0_print(f"[warn] {label} is not a PeftModel; skipping TE PEFT save.")

    te1_step = os.path.join(ckpt_dir, f"{tag}-te1-peft")
    te2_step = os.path.join(ckpt_dir, f"{tag}-te2-peft")
    te1_last = os.path.join(ckpt_dir, "last-te1-peft")
    te2_last = os.path.join(ckpt_dir, "last-te2-peft")

    _save_if_peft(te1, te1_step, te1_last, "TE1")
    _save_if_peft(te2, te2_step, te2_last, "TE2")


def train(cfg: Dict[str, Any]):
    ddp_init_if_needed()
    seed_all(int(cfg.get("seed", 42)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    world_size = get_world_size()
    rank0 = is_main_process()
    set_speed_toggles()

    bundle = assemble_sdxl(cfg, device=device)
    unet, vae, text_stack, sched = bundle.unet, bundle.vae, bundle.text_stack, bundle.noise_scheduler
    unet_dtype = next(unet.parameters()).dtype

    stage = str(cfg["model"].get("stage", "base")).lower()
    is_refiner = stage == "refiner"

    lora_master_dtype = str(cfg.get("lora", {}).get("master_dtype", "fp32")).lower()
    te_lora_cfg = cfg.get("text_encoder_lora", {})
    te_lora_enabled = (stage == "base") and bool(te_lora_cfg.get("enable", False))

    if lora_master_dtype == "fp32":
        moved_unet = _cast_lora_master_fp32_module(unet, label="unet")
        moved_te = 0
        if te_lora_enabled:
            moved_te += _cast_lora_master_fp32_module(bundle.text_encoder, label="te1")
            moved_te += _cast_lora_master_fp32_module(bundle.text_encoder_2, label="te2")
        if rank0:
            rank0_print(
                f"[lora] Cast LoRA parameters to fp32 for optimizer stability: "
                f"unet={moved_unet}, text_encoders={moved_te}"
            )
    else:
        if rank0:
            rank0_print(
                f"[lora] Keeping LoRA parameters in UNet/TE dtype ({unet_dtype}); "
                f"master_dtype={lora_master_dtype}"
            )

    if te_lora_enabled:
        bundle.text_encoder.train()
        bundle.text_encoder_2.train()
    else:
        bundle.text_encoder.eval()
        bundle.text_encoder_2.eval()

    micro_step, opt_step = 0, 0

    def step_getter():
        return opt_step

    train_loader, val_loader = build_loaders(cfg["dataset"], step_getter=step_getter)

    unet_lora_params, te_lora_params = [], []
    for n, p in unet.named_parameters():
        if "lora_" in n and p.requires_grad:
            unet_lora_params.append(p)

    if te_lora_enabled:
        for n, p in bundle.text_encoder.named_parameters():
            if "lora_" in n and p.requires_grad:
                te_lora_params.append(p)
        for n, p in bundle.text_encoder_2.named_parameters():
            if "lora_" in n and p.requires_grad:
                te_lora_params.append(p)

    lora_params = unet_lora_params + te_lora_params
    try:
        num_tensors = len(lora_params)
        num_params = sum(p.numel() for p in lora_params)
        rank0_print(
            f"[train] LoRA tensors={num_tensors} LoRA params={num_params:,} "
            f"trainable_total={num_params:,}"
        )
        unet_lora_named = [
            (n, p.numel())
            for n, p in unet.named_parameters()
            if ("lora_" in n and p.requires_grad)
        ]
        if unet_lora_named:
            preview = ", ".join(f"{n}({cnt})" for n, cnt in unet_lora_named[:8])
            rank0_print(f"[train] UNet LoRA preview: {preview}")
    except Exception as _e:
        rank0_print(f"[train] NOTE: could not summarize LoRA params: {_e}")

    assert len(lora_params) > 0, "No LoRA parameters found (check LoRA injection)."
    if te_lora_enabled:
        _assert_lora_has_trainable_AB(bundle.text_encoder, "TE1")
        _assert_lora_has_trainable_AB(bundle.text_encoder_2, "TE2")

    base_lr = float(cfg["lora"].get("lr", 1e-4))
    te_lr_scale = float(te_lora_cfg.get("lr_scale", 0.3))
    te_lr = base_lr * te_lr_scale if te_lora_params else base_lr
    if rank0:
        rank0_print(
            f"[train] LRs: unet_lora_lr={base_lr:.3e}, "
            f"te_lora_lr={'N/A' if not te_lora_params else f'{te_lr:.3e}'} "
            f"(scale={te_lr_scale:.3f})"
        )

    betas = tuple(cfg["optim"].get("betas", (0.9, 0.999)))
    eps = float(cfg["optim"].get("eps", 1e-8))

    param_groups = []
    if unet_lora_params:
        param_groups.append({"params": unet_lora_params, "lr": base_lr})
    if te_lora_params:
        param_groups.append({"params": te_lora_params, "lr": te_lr})

    optimizer = torch.optim.AdamW(param_groups, betas=betas, eps=eps, weight_decay=0.0)

    total_steps = int(cfg["train"].get("total_steps", 30_000))
    warmup_steps = int(cfg["train"].get("warmup_steps", max(1, int(0.01 * total_steps))))
    lr_sched = build_cosine_with_warmup(
        optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        last_epoch=-1,
    )

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if device.type == "cuda":
        torch.cuda.set_device(local_rank)

    if world_size > 1:
        ddp_unet = DDP(unet, device_ids=[local_rank], find_unused_parameters=False)
        if te_lora_enabled:
            bundle.text_encoder = DDP(
                bundle.text_encoder, device_ids=[local_rank], find_unused_parameters=False
            )
            bundle.text_encoder_2 = DDP(
                bundle.text_encoder_2, device_ids=[local_rank], find_unused_parameters=False
            )
            if hasattr(text_stack, "text_encoder"):
                text_stack.text_encoder = bundle.text_encoder
            if hasattr(text_stack, "text_encoder_2"):
                text_stack.text_encoder_2 = bundle.text_encoder_2
    else:
        ddp_unet = unet

    accum = int(cfg["train"].get("effective_bsz_micro", 8))
    log_every = int(cfg["train"].get("log_every", 50))
    ckpt_every = int(cfg["train"].get("ckpt_every", 500))
    ckpt_after_steps = int(cfg["train"].get("ckpt_after_steps", 0))
    val_every = int(cfg["train"].get("val_every", 1000))

    min_snr_g = float(cfg["model"].get("min_snr_gamma", 5.0))
    cond_drop = float(cfg["model"].get("cond_dropout_prob", 0.10))
    pred_type = str(cfg["model"].get("prediction_type", "epsilon"))
    t_band = tuple(cfg["model"].get("refiner_timestep_band", [0, 200]))

    run_dir = cfg["train"].get("run_dir", "training/outputs/runs/exp-default")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    resume_path = cfg["train"].get("resume", None)
    if resume_path:
        opt_step, micro_step = load_training_state(
            resume_path, bundle, optimizer, lr_sched, device
        )
        rank0_print(f"[train] Resumed at opt_step={opt_step}, micro_step={micro_step}")

    pbar = (
        tqdm(total=total_steps, desc="train (opt steps)", leave=True)
        if (tqdm is not None and rank0)
        else None
    )

    try:
        last_log_time = time.time()
        running_loss = 0.0
        stop = False

        ca_dim = int(getattr(unet.config, "cross_attention_dim", 1280))
        txt_dim_expected = int(getattr(unet.config, "text_embed_dim", 1280))
        want_k_time_ids = _expected_timeid_count_from_unet(unet)
        if rank0:
            rank0_print(
                f"[dbg] UNet expects {want_k_time_ids} raw time-id scalars "
                f"(→ {want_k_time_ids}×256 add-channels)."
            )

        while opt_step < total_steps and not stop:
            for batch in train_loader:
                ddp_unet.train()
                if te_lora_enabled:
                    bundle.text_encoder.train()
                    bundle.text_encoder_2.train()
                else:
                    bundle.text_encoder.eval()
                    bundle.text_encoder_2.eval()

                images = torch.stack([b["image"] for b in batch]).to(device)
                with torch.no_grad():
                    latents = vae.encode(images).latent_dist.sample() * 0.13025
                    latents = latents.to(dtype=unet_dtype)

                captions = [b["caption"] for b in batch]
                concat_mode = "te2_only" if is_refiner else "both"
                prompt_embeds, pooled = encode_batch_with_cond_dropout(
                    text_stack,
                    captions,
                    device,
                    cond_dropout_prob=cond_drop,
                    out_dtype=unet_dtype,
                    train_te=te_lora_enabled,
                    concat_mode=concat_mode,
                )

                if prompt_embeds.shape[-1] != ca_dim:
                    if prompt_embeds.shape[-1] > ca_dim:
                        prompt_embeds = prompt_embeds[..., :ca_dim]
                    else:
                        pad = ca_dim - prompt_embeds.shape[-1]
                        prompt_embeds = torch.nn.functional.pad(
                            prompt_embeds, (0, pad)
                        )

                if pooled.shape[-1] != txt_dim_expected:
                    if pooled.shape[-1] > txt_dim_expected:
                        pooled = pooled[..., :txt_dim_expected]
                    else:
                        pad = txt_dim_expected - pooled.shape[-1]
                        pooled = torch.nn.functional.pad(pooled, (0, pad))

                time_ids = _pack_time_ids(batch, device=device, want_k=want_k_time_ids)

                noise = torch.randn_like(latents)
                if is_refiner:
                    t_min, t_max = tuple(
                        cfg["model"].get("refiner_timestep_band", [0, 200])
                    )
                    t_min = max(0, int(t_min))
                    t_max = min(
                        int(sched.config.num_train_timesteps) - 1, int(t_max)
                    )
                    if t_min > t_max:
                        t_min, t_max = t_max, t_min
                    t = torch.randint(
                        t_min,
                        t_max + 1,
                        (latents.shape[0],),
                        device=device,
                        dtype=torch.long,
                    )
                else:
                    t = torch.randint(
                        0,
                        sched.config.num_train_timesteps,
                        (latents.shape[0],),
                        device=device,
                        dtype=torch.long,
                    )

                noisy = sched.add_noise(latents, noise, t)

                with torch.autocast(
                    device_type="cuda",
                    dtype=unet_dtype,
                    enabled=(device.type == "cuda"),
                ):
                    pred = ddp_unet(
                        noisy,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        added_cond_kwargs={"text_embeds": pooled, "time_ids": time_ids},
                    ).sample

                target = (
                    noise
                    if pred_type == "epsilon"
                    else sched.get_velocity(latents, noise, t)
                )
                per_ex = torch.mean((pred - target) ** 2, dim=(1, 2, 3))

                if is_refiner:
                    loss = per_ex.mean() / accum
                else:
                    loss = apply_min_snr_weight(
                        per_ex,
                        t,
                        sched,
                        gamma=min_snr_g,
                        prediction_type=pred_type,
                    ) / accum

                try:
                    loss.backward()
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        optimizer.zero_grad(set_to_none=True)
                        torch.cuda.empty_cache()
                        if rank0:
                            print("[WARN] OOM recovered; skipping micro step.")
                        continue
                    raise

                micro_step += 1
                running_loss += float(loss.item())

                if micro_step % accum == 0:
                    torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    lr_sched.step()
                    opt_step += 1

                    if pbar is not None:
                        pbar.update(1)
                        pbar.set_postfix(
                            {
                                "loss": f"{running_loss * accum / max(1, log_every):.4f}",
                                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                            }
                        )

                    if rank0 and (opt_step % log_every == 0):
                        now = time.time()
                        dt = now - last_log_time
                        last_log_time = now
                        log_row = {
                            "step": opt_step,
                            "micro_step": micro_step,
                            "loss/train": running_loss * accum / log_every,
                            "lr": optimizer.param_groups[0]["lr"],
                            "time/it": dt / max(1, log_every),
                            "gpu_mem": (
                                torch.cuda.max_memory_allocated() / (1024**2)
                            )
                            if torch.cuda.is_available()
                            else 0.0,
                            "stage": stage,
                            "dbg/ca_dim": ca_dim,
                            "dbg/pooled_dim": int(pooled.shape[-1]),
                            "dbg/want_k_time_ids": want_k_time_ids,
                        }
                        write_jsonl(os.path.join(run_dir, "metrics.jsonl"), log_row)
                        running_loss = 0.0

                    if (
                        ckpt_every > 0
                        and opt_step % ckpt_every == 0
                        and opt_step >= ckpt_after_steps
                    ):
                        barrier()
                        if rank0:
                            tag = f"step-{opt_step:07d}"
                            unet_step_path = os.path.join(
                                ckpt_dir, f"{tag}-unet-lora.safetensors"
                            )
                            save_lora_safetensors(unet, unet_step_path)
                            atomic_copy(
                                unet_step_path,
                                os.path.join(
                                    ckpt_dir, "last-unet-lora.safetensors"
                                ),
                            )

                            if te_lora_enabled:
                                _save_te_peft_adapters(
                                    ckpt_dir=ckpt_dir,
                                    tag=tag,
                                    te1=bundle.text_encoder,
                                    te2=bundle.text_encoder_2,
                                    rank0=rank0,
                                )

                            save_training_state(
                                os.path.join(ckpt_dir, "state.pt"),
                                optimizer,
                                lr_sched,
                                opt_step,
                                micro_step,
                                cfg,
                            )
                        barrier()

                    if (opt_step % val_every == 0) or (opt_step == total_steps):
                        barrier()
                        val_max_batches = cfg["train"].get(
                            "val_max_batches", None
                        )
                        if isinstance(val_max_batches, int) and val_max_batches <= 0:
                            val_max_batches = None

                        val_loss = run_val(
                            val_loader,
                            bundle,
                            device,
                            min_snr_g,
                            prediction_type=pred_type,
                            seed=1234,
                            max_batches=val_max_batches,
                            show_progress=True,
                            stage=stage,
                            timestep_band=t_band,
                        )
                        if rank0:
                            write_jsonl(
                                os.path.join(run_dir, "metrics.jsonl"),
                                {
                                    "step": opt_step,
                                    "loss/val": float(val_loss),
                                    "stage": stage,
                                },
                            )
                        barrier()

                    if opt_step >= total_steps:
                        stop = True
                        break

    finally:
        if pbar is not None:
            pbar.close()

        barrier()
        if rank0:
            tag = f"step-{opt_step:07d}"
            unet_step_path = os.path.join(
                ckpt_dir, f"{tag}-unet-lora.safetensors"
            )
            save_lora_safetensors(unet, unet_step_path)
            atomic_copy(
                unet_step_path,
                os.path.join(ckpt_dir, "last-unet-lora.safetensors"),
            )

            if te_lora_enabled:
                _save_te_peft_adapters(
                    ckpt_dir=ckpt_dir,
                    tag=tag,
                    te1=bundle.text_encoder,
                    te2=bundle.text_encoder_2,
                    rank0=rank0,
                )

            save_training_state(
                os.path.join(ckpt_dir, "state.pt"),
                optimizer,
                lr_sched,
                opt_step,
                micro_step,
                cfg,
            )
        barrier()
        ddp_destroy_if_needed()
