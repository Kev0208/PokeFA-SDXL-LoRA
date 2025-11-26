from __future__ import annotations
import os
import json
import shutil
from typing import Any, Dict

import torch
import torch.distributed as dist


def set_speed_toggles(tf32: bool = True, matmul_precision: str = "high"):
    if tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        try:
            torch.set_float32_matmul_precision(matmul_precision)
        except Exception:
            pass


def set_determinism_toggles(enable: bool = True):
    if not enable:
        return
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _fsync_dir(dirpath: str):
    try:
        d = os.open(dirpath or ".", os.O_RDONLY)
        try:
            os.fsync(d)
        finally:
            os.close(d)
    except Exception:
        pass


def atomic_copy(src: str, dst: str):
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    tmp = dst + ".tmp"
    shutil.copy2(src, tmp)
    with open(tmp, "rb+") as f:
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, dst)
    _fsync_dir(os.path.dirname(dst) or ".")


def write_jsonl(path: str, row: Dict[str, Any], *, fsync_each: bool = False):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    line = json.dumps(row, ensure_ascii=False) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
        if fsync_each:
            f.flush()
            os.fsync(f.fileno())


def write_jsonl_rank_aware(
    basedir: str, filename: str, row: Dict[str, Any], *, fsync_each: bool = False
):
    rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
    name = f"{os.path.splitext(filename)[0]}.rank{rank}.jsonl"
    path = os.path.join(basedir, name)
    write_jsonl(path, row, fsync_each=fsync_each)
