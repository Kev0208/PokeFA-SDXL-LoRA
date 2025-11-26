from __future__ import annotations
import os, io, json, random
from typing import Dict, Any, Tuple, Callable
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import webdataset as wds
from webdataset import WebLoader

from dataloader_utils import BuildTransform, load_species_fracs, make_species_accept_fn


def _decode_tuple(sample, image_key: str, caption_key: str, meta_key: str):
    img_bytes, cap_bytes, meta_bytes = sample
    try:
        img = Image.open(io.BytesIO(img_bytes))

        has_transparency = (
            (img.mode in ("P", "LA"))
            or ("transparency" in getattr(img, "info", {}))
            or (img.mode == "RGBA")
        )
        if has_transparency:
            if img.mode != "RGBA":
                img = img.convert("RGBA")
            bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(bg, img).convert("RGB")
        else:
            img = img.convert("RGB")

        cap = cap_bytes.decode("utf-8", errors="ignore")
        meta = json.loads(meta_bytes.decode("utf-8", errors="ignore"))
        img_id = meta.get("image_id") or meta.get("id") or ""
        return {"image": img, "caption": cap, "meta": meta, "id": img_id}
    except Exception as e:
        raise e


def _ensure_list_of_dicts(batched):
    if isinstance(batched, dict):
        keys = list(batched.keys())
        n = 0
        for v in batched.values():
            n = len(v)
            break
        return [{k: batched[k][i] for k in keys} for i in range(n)]
    return batched


def _wds_pipeline(urls: str, wds_cfg: Dict[str, Any], is_train: bool):
    shuffle_samples = int(wds_cfg.get("shuffle_samples", 10_000))
    resampled = bool(wds_cfg.get("resample_shards", True if is_train else False))

    shardshuffle = False if resampled else bool(is_train)
    shard_shuffle_buf = int(wds_cfg.get("shard_shuffle_buf", 100))

    image_key = wds_cfg["image_key"]
    caption_key = wds_cfg["caption_key"]
    meta_key = wds_cfg["meta_key"]

    shard_source = wds.ResampledShards(urls) if resampled else wds.SimpleShardList(urls)
    stages = [shard_source]

    if shardshuffle and shard_shuffle_buf > 0:
        stages.append(wds.shuffle(shard_shuffle_buf))

    stages.extend(
        [
            wds.split_by_node,
            wds.split_by_worker,
            wds.tarfile_to_samples(handler=wds.handlers.warn_and_continue),
        ]
    )

    if is_train and shuffle_samples > 0:
        stages.append(wds.shuffle(shuffle_samples))

    stages.extend(
        [
            wds.to_tuple(image_key, caption_key, meta_key),
            wds.map(lambda s: _decode_tuple(s, image_key, caption_key, meta_key)),
        ]
    )

    return wds.DataPipeline(*stages)


def _build_lambda_provider(
    cfg_lambda: Dict[str, Any], step_getter: Callable[[], int]
) -> Callable[[], float]:
    kind = cfg_lambda.get("type", "linear")
    start = float(cfg_lambda.get("start", 0.0))
    end = float(cfg_lambda.get("end", 1.0))
    steps = max(1, int(cfg_lambda.get("steps", 1)))

    if kind == "linear":
        def f():
            t = min(max(step_getter(), 0), steps)
            return start + (end - start) * (t / steps)

        return f
    elif kind == "cosine":
        import math

        def f():
            t = min(max(step_getter(), 0), steps)
            alpha = 0.5 * (1 - math.cos(math.pi * t / steps))
            return start + (end - start) * alpha

        return f
    else:
        return lambda: end


def _seed_worker(worker_id: int):
    rnd = (os.getpid() + worker_id) & 0xFFFFFFFF
    random.seed(rnd)


def build_loaders(
    cfg: Dict[str, Any],
    *,
    step_getter: Callable[[], int] | None = None,
) -> Tuple[WebLoader, WebLoader]:
    wds_cfg = cfg["wds"]
    ld_cfg = cfg["loader"]
    tfm_cfg = cfg["transform"]
    sp_cfg = cfg.get("sampler", {})
    sch_cfg = cfg.get("schedule", {}).get(
        "lambda", {"type": "linear", "start": 0.0, "end": 1.0, "steps": 1}
    )

    Tsize = int(tfm_cfg["size"])
    policy = tfm_cfg["policy"]
    normalize = tfm_cfg.get("normalize", "sdxl")
    deterministic_val = bool(tfm_cfg.get("deterministic_val", True))

    transform_train = BuildTransform(
        Tsize, policy, normalize=normalize, deterministic_val=False
    )
    transform_val = BuildTransform(
        Tsize, policy, normalize=normalize, deterministic_val=deterministic_val
    )

    _step = step_getter or (lambda: 0)

    accept_fn = None
    if sp_cfg.get("mode") == "species_accept":
        species_csv = sp_cfg["species_csv"]
        species_fracs = load_species_fracs(species_csv)
        lam_provider = _build_lambda_provider(sch_cfg, _step)
        accept_fn = make_species_accept_fn(
            species_fracs,
            lambda_provider=lam_provider,
            choose_strategy=sp_cfg.get("choose_strategy", "first"),
            species_field=sp_cfg.get("species_field", "species_list"),
        )

    train_ds = _wds_pipeline(wds_cfg["train_urls"], wds_cfg, is_train=True)
    if accept_fn is not None:
        train_ds = wds.DataPipeline(
            train_ds, wds.select(lambda sample: accept_fn(sample["meta"]))
        )

    def _apply_train_transform(sample):
        return transform_train(sample, step=_step(), is_train=True)

    train_ds = wds.DataPipeline(
        train_ds,
        wds.map(_apply_train_transform),
        wds.batched(int(ld_cfg["batch_size"]), partial=False),
        wds.map(_ensure_list_of_dicts),
    )

    val_ds = _wds_pipeline(
        wds_cfg["val_urls"], {**wds_cfg, "resample_shards": False}, is_train=False
    )

    def _apply_val_transform(sample):
        step_arg = -1 if deterministic_val else _step()
        return transform_val(sample, step=step_arg, is_train=False)

    val_ds = wds.DataPipeline(
        val_ds,
        wds.map(_apply_val_transform),
        wds.batched(int(ld_cfg["batch_size"]), partial=True),
        wds.map(_ensure_list_of_dicts),
    )

    nw = int(ld_cfg["num_workers"])
    pin_memory = bool(ld_cfg.get("pin_memory", True))
    prefetch_factor = int(ld_cfg.get("prefetch_factor", 2))
    persistent_workers = bool(ld_cfg.get("persistent_workers", True))

    train_loader = WebLoader(
        train_ds,
        num_workers=nw,
        batch_size=None,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if nw > 0 else None,
        persistent_workers=persistent_workers if nw > 0 else False,
        worker_init_fn=_seed_worker if nw > 0 else None,
    )

    val_loader = WebLoader(
        val_ds,
        num_workers=max(0, nw // 2),
        batch_size=None,
        pin_memory=pin_memory if nw > 0 else False,
        prefetch_factor=prefetch_factor if nw > 0 else None,
        persistent_workers=(persistent_workers and nw > 0),
        worker_init_fn=_seed_worker if nw > 0 else None,
    )

    return train_loader, val_loader
