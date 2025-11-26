from __future__ import annotations
import os
from datetime import timedelta
from typing import Optional

import torch
import torch.distributed as dist


def _maybe_map_slurm_env():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        return
    if "SLURM_PROCID" in os.environ:
        os.environ.setdefault("RANK", os.environ["SLURM_PROCID"])
    if "SLURM_NTASKS" in os.environ:
        os.environ.setdefault("WORLD_SIZE", os.environ["SLURM_NTASKS"])
    if "SLURM_LOCALID" in os.environ:
        os.environ.setdefault("LOCAL_RANK", os.environ["SLURM_LOCALID"])
    os.environ.setdefault("LOCAL_RANK", os.environ.get("LOCAL_RANK", "0"))


def _pick_backend() -> str:
    return "nccl" if torch.cuda.is_available() else "gloo"


def ddp_init_if_needed(timeout_sec: int = 600) -> bool:
    if not dist.is_available() or dist.is_initialized():
        if torch.cuda.is_available():
            lr = int(os.environ.get("LOCAL_RANK", "0"))
            if lr < torch.cuda.device_count():
                torch.cuda.set_device(lr)
        return False

    _maybe_map_slurm_env()

    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        if torch.cuda.is_available():
            lr = int(os.environ.get("LOCAL_RANK", "0"))
            if lr < torch.cuda.device_count():
                torch.cuda.set_device(lr)
        return False

    backend = _pick_backend()
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    if torch.cuda.is_available():
        lr = int(os.environ.get("LOCAL_RANK", "0"))
        if lr < torch.cuda.device_count():
            torch.cuda.set_device(lr)

    dist.init_process_group(
        backend=backend,
        init_method="env://",
        timeout=timedelta(seconds=timeout_sec),
    )
    return True


def ddp_destroy_if_needed():
    if is_dist():
        dist.destroy_process_group()


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_main_process() -> bool:
    return get_rank() == 0


def barrier(timeout_sec: Optional[int] = None):
    if not is_dist():
        return
    if timeout_sec is None:
        dist.barrier()
    else:
        dist.barrier(torch.device("cuda") if torch.cuda.is_available() else None)


def rank0_print(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)


def seed_all(base_seed: int):
    r = get_rank()
    seed = int(base_seed) + int(r)

    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
