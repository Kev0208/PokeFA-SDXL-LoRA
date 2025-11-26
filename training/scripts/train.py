from __future__ import annotations
import argparse
import json
import os
import sys
import pathlib
import shutil
import traceback
import logging
import atexit
import time
from typing import Dict, Any

THIS = pathlib.Path(__file__).resolve()
TRAIN_DIR = THIS.parents[1]
SRC_DIR = TRAIN_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

import yaml  

from train_loop import train as run_train  
from train_loop_utils.dist import ddp_init_if_needed, is_main_process, barrier  # noqa: E402
from train_loop_utils.utils import set_speed_toggles  


def deep_merge(base: Dict[str, Any], *overlays: Dict[str, Any]) -> Dict[str, Any]:
    import copy

    out = copy.deepcopy(base)
    for ov in overlays:
        stack = [(out, ov)]
        while stack:
            dst, src = stack.pop()
            for k, v in src.items():
                if isinstance(v, dict) and isinstance(dst.get(k), dict):
                    stack.append((dst[k], v))
                else:
                    dst[k] = v
    return out


def _yaml_sanitize(x):
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, dict):
        return {str(k): _yaml_sanitize(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_yaml_sanitize(v) for v in x]
    return str(x)


def atomic_dump_yaml(obj: Dict[str, Any], dst: pathlib.Path):
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    dst.parent.mkdir(parents=True, exist_ok=True)
    clean = _yaml_sanitize(obj)
    with tmp.open("w", encoding="utf-8") as f:
        yaml.safe_dump(clean, f, sort_keys=False, allow_unicode=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, dst)


def load_yaml(path: pathlib.Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        raise SystemExit(f"[train] ERROR: YAML not found: {path}")
    except yaml.YAMLError as e:
        raise SystemExit(f"[train] ERROR: Bad YAML in {path}:\n{e}")


def tiny_schema_check(cfg: Dict[str, Any]):
    req = [
        ("model.base_path", cfg.get("model", {}).get("base_path")),
        ("dataset.wds.train_urls", cfg.get("dataset", {}).get("wds", {}).get("train_urls")),
        ("dataset.wds.val_urls", cfg.get("dataset", {}).get("wds", {}).get("val_urls")),
        ("train.total_steps", cfg.get("train", {}).get("total_steps")),
    ]
    missing = [k for k, v in req if v in (None, "")]
    if missing:
        pretty = "\n  - ".join(missing)
        raise SystemExit(f"[train] ERROR: Missing required config keys:\n  - {pretty}")


def resolve_path(p: str | None, base: pathlib.Path) -> pathlib.Path | None:
    if not p:
        return None
    pp = pathlib.Path(p)
    return (pp if pp.is_absolute() else (base / pp)).resolve()


def maybe_infer_resume_path(run_dir: pathlib.Path, resume_value: str | None) -> str | None:
    if not resume_value:
        return None
    cand = pathlib.Path(resume_value)
    if not cand.is_absolute():
        cand = (run_dir if run_dir.is_dir() else TRAIN_DIR) / resume_value
    if cand.is_dir() and (cand / "checkpoints" / "last.safetensors").exists():
        return str((cand / "checkpoints").resolve())
    if cand.exists():
        return str(cand.resolve())
    print(f"[train] WARNING: resume target not found: {resume_value} (continuing fresh)")
    return None


def parse_dot_overrides(dot_list: list[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for kv in dot_list or []:
        if "=" not in kv:
            print(f"[train] WARNING: ignoring malformed override: {kv}")
            continue
        k, v = kv.split("=", 1)
        cursor = out
        parts = k.split(".")
        for p in parts[:-1]:
            cursor = cursor.setdefault(p, {})
        try:
            cursor[parts[-1]] = json.loads(v)
        except Exception:
            cursor[parts[-1]] = v
    return out


class _Tee:
    def __init__(self, stream, file_path: pathlib.Path, also_console: bool):
        self._stream = stream
        self._fh = file_path.open("a", encoding="utf-8", buffering=1)
        self._also_console = also_console

    def write(self, data):
        try:
            self._fh.write(data)
        except Exception:
            pass
        if self._also_console:
            try:
                self._stream.write(data)
            except Exception:
                pass

    def flush(self):
        try:
            self._fh.flush()
        except Exception:
            pass
        if self._also_console:
            try:
                self._stream.flush()
            except Exception:
                pass

    def close(self):
        try:
            self._fh.close()
        except Exception:
            pass


def _get_rank():
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))


def _setup_run_logging(run_dir: pathlib.Path, enable_console_rank0: bool = True):
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    rank = _get_rank()

    out_path = logs_dir / f"rank{rank}.out"
    err_path = logs_dir / f"rank{rank}.err"
    stdout_tee = _Tee(
        sys.__stdout__,
        out_path,
        also_console=(enable_console_rank0 and rank == 0),
    )
    stderr_tee = _Tee(
        sys.__stderr__,
        err_path,
        also_console=(enable_console_rank0 and rank == 0),
    )
    sys.stdout = stdout_tee
    sys.stderr = stderr_tee

    log = logging.getLogger()
    log.setLevel(logging.INFO)
    for h in list(log.handlers):
        log.removeHandler(h)

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [rank%(rank)d] %(message)s".replace(
            "%(rank)d", str(rank)
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(logs_dir / f"train_rank{rank}.log", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    log.addHandler(fh)

    if enable_console_rank0 and rank == 0:
        sh = logging.StreamHandler(sys.__stdout__)
        sh.setLevel(logging.INFO)
        sh.setFormatter(fmt)
        log.addHandler(sh)

    def _excepthook(exc_type, exc, tb):
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        tb_txt = "".join(traceback.format_exception(exc_type, exc, tb))
        err_txt_path = logs_dir / f"error_rank{rank}.log"
        err_json_path = logs_dir / f"error_rank{rank}.json"
        try:
            with err_txt_path.open("a", encoding="utf-8") as f:
                f.write(f"[{ts}] Uncaught exception on rank {rank}:\n{tb_txt}\n")
            with err_json_path.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "time_utc": ts,
                        "rank": rank,
                        "type": str(exc_type.__name__),
                        "message": str(exc),
                        "traceback": tb_txt,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        except Exception:
            pass
        logging.error("Uncaught exception:\n%s", tb_txt)
        sys.__excepthook__(exc_type, exc, tb)

    sys.excepthook = _excepthook

    atexit.register(stdout_tee.close)
    atexit.register(stderr_tee.close)

    logging.info(
        "Logging initialized for rank %d. Logs dir: %s",
        rank,
        str(logs_dir),
    )
    return stdout_tee, stderr_tee


def main():
    ap = argparse.ArgumentParser(description="PokéFA – launch SDXL LoRA training")
    ap.add_argument("--dataset", required=True, help="Path to training/configs/dataset.yaml")
    ap.add_argument("--model", required=True, help="Path to training/configs/model.yaml")
    ap.add_argument("--train", required=True, help="Path to training/configs/train_*.yaml")
    ap.add_argument(
        "--override",
        action="append",
        default=[],
        help=(
            "Dot overrides, e.g. "
            "--override train.total_steps=2000 "
            "--override wandb.enable=false"
        ),
    )
    ap.add_argument(
        "--paths-relative-to-cwd",
        action="store_true",
        help="Resolve config paths relative to CWD instead of training/ dir",
    )
    ap.add_argument(
        "--no-console",
        dest="no_console",
        action="store_true",
        help="Don’t echo rank0 logs to console (files only).",
    )
    args = ap.parse_args()

    base = pathlib.Path.cwd() if args.paths_relative_to_cwd else TRAIN_DIR
    ds_yaml = resolve_path(args.dataset, base)
    model_yaml = resolve_path(args.model, base)
    train_yaml = resolve_path(args.train, base)

    ds_cfg = load_yaml(ds_yaml)
    model_cfg = load_yaml(model_yaml)
    run_cfg = load_yaml(train_yaml)

    merged = deep_merge(
        ds_cfg,
        {"model": {}},
        model_cfg,
        run_cfg,
        parse_dot_overrides(args.override),
    )

    tiny_schema_check(merged)

    ddp_init_if_needed()

    run_dir = pathlib.Path(
        merged["train"].get("run_dir", "training/outputs/runs/exp-default")
    ).resolve()
    cfg_out = run_dir / "cfg.yaml"

    if is_main_process():
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "configs_src").mkdir(exist_ok=True)
        try:
            shutil.copy2(ds_yaml, run_dir / "configs_src" / "dataset.yaml")
            shutil.copy2(model_yaml, run_dir / "configs_src" / "model.yaml")
            shutil.copy2(train_yaml, run_dir / "configs_src" / "train.yaml")
        except Exception:
            pass
        atomic_dump_yaml(merged, cfg_out)
    barrier()

    _setup_run_logging(run_dir, enable_console_rank0=(not args.no_console))

    resume_val = merged.get("train", {}).get("resume")
    resume_norm = maybe_infer_resume_path(run_dir, resume_val)
    if resume_norm:
        merged["train"]["resume"] = resume_norm

    set_speed_toggles()

    try:
        logging.info("Starting training…")
        run_train(merged)
        logging.info("Training finished normally.")
    except SystemExit as e:
        logging.error("SystemExit: %s", e)
        raise
    except BaseException:
        tb = traceback.format_exc()
        logging.error("Fatal error:\n%s", tb)
        raise


if __name__ == "__main__":
    main()
