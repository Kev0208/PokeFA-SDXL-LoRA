from __future__ import annotations

import argparse
import json
import os
import pathlib
import shlex
import shutil
import subprocess
import sys

import yaml


def sm_env():
    hosts = json.loads(os.environ["SM_HOSTS"])
    current = os.environ["SM_CURRENT_HOST"]
    num_gpus = int(os.environ["SM_NUM_GPUS"])
    model_dir = os.environ["SM_MODEL_DIR"]
    code_root = "/opt/ml/code"
    return hosts, current, num_gpus, model_dir, code_root


def run(cmd: list[str], env=None, cwd=None, check=True):
    print(f"[entry] $ {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    return subprocess.run(cmd, env=env, cwd=cwd, check=check)


def human_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def dir_size_bytes(path: pathlib.Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for p, _, files in os.walk(path, followlinks=False):
        for f in files:
            try:
                total += (pathlib.Path(p) / f).stat().st_size
            except FileNotFoundError:
                pass
    return total


def free_bytes(path: str = "/opt/ml") -> int:
    usage = shutil.disk_usage(path)
    return usage.free


def aws_sync_or_cp(src: str, dst: str, is_dir=False):
    if is_dir:
        run(["aws", "s3", "sync", "--no-progress", src, dst])
    else:
        run(["aws", "s3", "cp", "--no-progress", src, dst])


def load_runtime_config(code_root: str) -> dict:
    repo_root = pathlib.Path(code_root).resolve()
    cfg_path = repo_root / "training" / "configs" / "aws_sm.yaml"
    with cfg_path.open("r") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("runtime", {}) or {}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["train", "hpo"],
        default="train",
        help="Which pipeline to run inside the container.",
    )
    args = parser.parse_args()
    mode = args.mode

    hosts, current, num_gpus, model_dir, code_root = sm_env()
    print(f"[entry] hosts={hosts} current={current} num_gpus={num_gpus}")
    print(f"[entry] model_dir={model_dir}")
    print(f"[entry] mode={mode}")

    runtime_cfg = load_runtime_config(code_root)

    S3_MODEL_BASE = runtime_cfg.get("s3_model_base")
    S3_SPECIES_CSV = runtime_cfg.get("s3_species_csv")
    S3_TRAIN_PREFIX = runtime_cfg.get("s3_train_prefix")
    S3_VAL_PREFIX = runtime_cfg.get("s3_val_prefix")

    if not all([S3_MODEL_BASE, S3_TRAIN_PREFIX, S3_VAL_PREFIX]):
        raise ValueError(
            "Missing one of s3_model_base / s3_train_prefix / s3_val_prefix in aws_sm.yaml.runtime"
        )

    local_cache = pathlib.Path("/opt/ml/input/cache")
    local_model = local_cache / "sdxl_base"
    local_species_csv = local_cache / "manifests" / "train_species.csv"
    local_data_root = pathlib.Path("/opt/ml/input/data")
    local_train_dir = local_data_root / "train"
    local_val_dir = local_data_root / "val"

    local_cache.mkdir(parents=True, exist_ok=True)
    local_species_csv.parent.mkdir(parents=True, exist_ok=True)
    local_train_dir.mkdir(parents=True, exist_ok=True)
    local_val_dir.mkdir(parents=True, exist_ok=True)

    aws_sync_or_cp(S3_MODEL_BASE, str(local_model), is_dir=True)
    if S3_SPECIES_CSV:
        aws_sync_or_cp(S3_SPECIES_CSV, str(local_species_csv), is_dir=False)
        print(f"[entry] downloaded species csv -> {local_species_csv}")
    print(f"[entry] synced model -> {local_model}")

    aws_sync_or_cp(S3_TRAIN_PREFIX, str(local_train_dir), is_dir=True)
    aws_sync_or_cp(S3_VAL_PREFIX, str(local_val_dir), is_dir=True)

    shard_bytes = dir_size_bytes(local_train_dir) + dir_size_bytes(local_val_dir)
    free_now = free_bytes("/opt/ml")
    headroom = 1.5
    need_bytes = int(shard_bytes * headroom)

    print(f"[storage] shard_size(train+val) = {human_bytes(shard_bytes)}")
    print(f"[storage] required headroom (x{headroom}) = {human_bytes(need_bytes)}")
    print(f"[storage] free_now = {human_bytes(free_now)}")

    if free_now < need_bytes:
        print(
            "[storage][FATAL] Not enough free space for safe run. "
            f"Need {human_bytes(need_bytes)}, have {human_bytes(free_now)}. Exiting."
        )
        sys.exit(2)

    repo_root = pathlib.Path(code_root).resolve()
    train_dir = repo_root / "training"

    dataset_rel = runtime_cfg.get("dataset_config", "configs/dataset.yaml")
    model_rel = runtime_cfg.get("model_config", "configs/model.yaml")
    train_rel = runtime_cfg.get("train_config", "configs/train.yaml")
    hpo_rel = runtime_cfg.get("hpo_config", "configs/hpo.yaml")

    ds_yaml = train_dir / dataset_rel
    model_yaml = train_dir / model_rel
    train_yaml = train_dir / train_rel
    hpo_yaml = train_dir / hpo_rel

    train_script = train_dir / "scripts" / "train.py"
    hpo_script = train_dir / "scripts" / "hpo.py"

    if mode == "train":
        required = [train_script, ds_yaml, model_yaml, train_yaml]
    else:
        required = [hpo_script, ds_yaml, model_yaml, train_yaml, hpo_yaml]

    for p in required:
        assert p.exists(), f"[entry] missing: {p}"
    print("[entry] config & script paths verified")

    train_shard_pattern = runtime_cfg.get(
        "train_shard_pattern", "train-{00000..00015}.tar"
    )
    val_shard_pattern = runtime_cfg.get("val_shard_pattern", "val-{00000..00000}.tar")

    overrides = [
        f"model.base_path={json.dumps(str(local_model))}",
        f"dataset.wds.train_urls={json.dumps(str(local_train_dir / train_shard_pattern))}",
        f"dataset.wds.val_urls={json.dumps(str(local_val_dir / val_shard_pattern))}",
        f"train.run_dir={json.dumps(str(model_dir))}",
    ]
    if S3_SPECIES_CSV:
        overrides.append(
            f"dataset.sampler.species_csv={json.dumps(str(local_species_csv))}"
        )

    is_multi_node = len(hosts) > 1
    node_rank = hosts.index(current) if is_multi_node else 0

    env = os.environ.copy()
    env["TORCH_NCCL_BLOCKING_WAIT"] = "1"
    env.setdefault("NCCL_DEBUG", "WARN")
    env.setdefault("NCCL_IB_DISABLE", "0")
    env.setdefault("NCCL_P2P_DISABLE", "0")
    env.setdefault("OMP_NUM_THREADS", "1")

    base_cmd = ["torchrun", "--nproc_per_node", str(num_gpus)]
    if is_multi_node:
        base_cmd += [
            "--nnodes",
            str(len(hosts)),
            "--node_rank",
            str(node_rank),
            "--rdzv_backend",
            "c10d",
            "--rdzv_endpoint",
            f"{hosts[0]}:29400",
        ]
    else:
        base_cmd += ["--standalone"]

    if mode == "train":
        cli = base_cmd + [
            str(train_script),
            "--dataset",
            str(ds_yaml),
            "--model",
            str(model_yaml),
            "--train",
            str(train_yaml),
        ]
        for ov in overrides:
            cli += ["--override", ov]
    else:
        cli = base_cmd + [
            str(hpo_script),
            "--dataset",
            str(ds_yaml),
            "--model",
            str(model_yaml),
            "--train",
            str(train_yaml),
            "--hpo",
            str(hpo_yaml),
        ]
        for ov in overrides:
            cli += ["--override", ov]

    run(cli, env=env, cwd=str(repo_root))
