from typing import Any, Optional, Tuple, List, Dict

import torch
import torch.distributed as dist

from model_utils import encode_batch_with_cond_dropout
from model_utils.snr import _compute_snr

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def _sample_timesteps_band_det(
    sched,
    batch_size: int,
    device: torch.device,
    band: Tuple[int, int],
    *,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    t_min, t_max = int(band[0]), int(band[1])
    t_min = max(0, t_min)
    t_max = min(int(sched.config.num_train_timesteps) - 1, t_max)
    if t_min > t_max:
        t_min, t_max = t_max, t_min
    return torch.randint(
        t_min,
        t_max + 1,
        (batch_size,),
        device=device,
        dtype=torch.long,
        generator=generator,
    )


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


def _pack_time_ids(
    batch: List[Dict[str, Any]],
    device: torch.device,
    *,
    want_k: int,
    dtype: torch.dtype = torch.long,
) -> torch.Tensor:
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
    return torch.tensor(rows, device=device, dtype=dtype)


@torch.no_grad()
def run_val(
    val_loader,
    bundle,
    device: torch.device,
    min_snr_gamma: float,
    *,
    prediction_type: str = "v_prediction",
    seed: int = 1234,
    max_batches: Optional[int] = None,
    show_progress: bool = False,
    stage: str = "base",
    timestep_band: Optional[Tuple[int, int]] = None,
) -> float:
    unet = bundle.unet
    vae = bundle.vae
    sched = bundle.noise_scheduler
    text_stack = bundle.text_stack

    unet_was_train = unet.training
    vae_was_train = vae.training
    unet.eval()
    vae.eval()

    if prediction_type == "v_prediction":
        pt = getattr(sched.config, "prediction_type", None)
        if pt != "v_prediction":
            print(
                f"[val] WARNING: scheduler.prediction_type={pt}, but validation assumes v_prediction."
            )

    gen = torch.Generator(device=device).manual_seed(seed)
    unet_dtype = next(unet.parameters()).dtype

    total_weighted_loss = 0.0
    total_weight = 0.0
    rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
    pbar = (
        tqdm(total=max_batches, desc="val (rank 0)", leave=False)
        if (show_progress and tqdm is not None and rank == 0)
        else None
    )

    want_k_time_ids = _expected_timeid_count_from_unet(unet)

    batches_seen = 0
    for batch in val_loader:
        if (max_batches is not None) and (batches_seen >= max_batches):
            break
        batches_seen += 1
        if pbar is not None:
            pbar.update(1)

        images = torch.stack([b["image"] for b in batch]).to(device)
        latents = vae.encode((images + 1) / 2).latent_dist.mean * 0.18215
        latents = latents.to(dtype=unet_dtype)

        captions = [b["caption"] for b in batch]
        concat_mode = "te2_only" if str(stage).lower() == "refiner" else "both"
        prompt_embeds, pooled = encode_batch_with_cond_dropout(
            text_stack,
            captions,
            device,
            cond_dropout_prob=0.0,
            out_dtype=unet_dtype,
            train_te=False,
            concat_mode=concat_mode,
        )

        time_ids = _pack_time_ids(
            batch, device=device, want_k=want_k_time_ids, dtype=torch.long
        )

        bsz = latents.shape[0]
        if str(stage).lower() == "refiner" and timestep_band is not None:
            t = _sample_timesteps_band_det(
                sched, bsz, device, timestep_band, generator=gen
            )
        else:
            t = torch.randint(
                0,
                sched.config.num_train_timesteps,
                (bsz,),
                device=device,
                dtype=torch.long,
                generator=gen,
            )

        noise = torch.randn(
            latents.shape,
            dtype=latents.dtype,
            device=latents.device,
            generator=gen,
        )
        noisy = sched.add_noise(latents, noise, t)

        with torch.autocast(
            device_type="cuda",
            dtype=unet_dtype,
            enabled=(device.type == "cuda"),
        ):
            pred = unet(
                noisy,
                t,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs={"text_embeds": pooled, "time_ids": time_ids},
            ).sample

        if prediction_type == "v_prediction":
            target = sched.get_velocity(latents, noise, t)
        elif prediction_type == "epsilon":
            target = noise
        else:
            target = sched.get_velocity(latents, noise, t)

        per_ex = torch.mean((pred - target) ** 2, dim=(1, 2, 3))

        if str(stage).lower() == "refiner":
            w = torch.ones_like(per_ex, dtype=torch.float32)
        else:
            snr = _compute_snr(sched, t).clamp(min=1e-8)
            numer = snr.clamp(max=min_snr_gamma)
            denom = snr if prediction_type == "epsilon" else (snr + 1.0)
            w = (numer / denom).float()

        total_weighted_loss += float((per_ex * w).sum().item())
        total_weight += float(w.sum().item())

    if pbar is not None:
        pbar.close()

    if dist.is_available() and dist.is_initialized():
        t_pair = torch.tensor(
            [total_weighted_loss, total_weight],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(t_pair, op=dist.ReduceOp.SUM)
        total_weighted_loss, total_weight = t_pair.tolist()

    unet.train(unet_was_train)
    vae.train(vae_was_train)

    return float(total_weighted_loss) / max(1.0, float(total_weight))
