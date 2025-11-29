from __future__ import annotations

import os
import json
import yaml
import time
import gc
import hashlib
import tempfile
import subprocess
import shlex
import socket
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import torch
import torch.distributed as dist
import optuna
from tqdm.auto import tqdm

from train_loop_utils.dist import (
    ddp_init_if_needed,
    is_main_process,
    barrier,
    get_rank,
    get_world_size,
    rank0_print,
    get_local_rank,
)
from hpo_utils.prompt_probes import PROMPT_PANEL
from hpo_utils.metrics import (
    load_clip_for_adherence,
    clip_cosine_for_batch,
    load_clip_vision,
    load_aesthetic_head,
    aesthetic_scores_for_images,
    summarize_metrics,
    dump_metrics_json,
    collapse_proxy,
)
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel

HPO_ROOT = os.environ.get("POKEFA_HPO_ROOT", os.path.join("outputs", "hpo"))
HPO_TRIALS = os.environ.get("POKEFA_HPO_TRIALS", os.path.join(HPO_ROOT, "trials"))
RUNS_ROOT = os.environ.get("POKEFA_RUNS_ROOT", os.path.join(HPO_ROOT, "runs"))


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _sha256_obj(obj: Dict[str, Any]) -> str:
    s = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _all_equal_string(s: str) -> bool:
    if not dist.is_initialized():
        return True
    outs = [None for _ in range(get_world_size())]
    dist.all_gather_object(outs, s)
    return all(x == outs[0] for x in outs)


def _fsync_replace(tmp_path: str, dst_path: str):
    with open(tmp_path, "rb+") as f:
        f.flush()
        os.fsync(f.fileno())
    os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
    os.replace(tmp_path, dst_path)
    dfd = os.open(os.path.dirname(dst_path) or ".", os.O_DIRECTORY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


def _write_yaml_atomic(path: str, obj: Dict[str, Any]):
    if not is_main_process():
        return
    dirpath = os.path.dirname(path) or "."
    os.makedirs(dirpath, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", delete=False, dir=dirpath, prefix=".tmp_", suffix=".yaml"
    ) as tf:
        yaml.safe_dump(obj, tf, sort_keys=False)
        tmp = tf.name
    _fsync_replace(tmp, path)


def _merge_cfgs(ds: Dict[str, Any], model: Dict[str, Any], train: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "dataset": ds.get("dataset", {}),
        "model": model.get("model", {}),
        "lora": model.get("lora", {}),
        "text_encoder_lora": model.get("text_encoder_lora", {}),
        "train": train.get("train", {}),
        "optim": train.get("optim", {}),
    }


def _override_for_trial(
    cfg: Dict[str, Any],
    *,
    lr: float,
    rank: int,
    p_uncond: float,
    total_steps: int,
    run_dir: str,
    te_rank: int | None = None,
):
    cfg.setdefault("lora", {})
    cfg.setdefault("model", {})
    cfg.setdefault("train", {})
    cfg["lora"]["rank"] = int(rank)
    cfg["lora"]["alpha"] = int(rank)
    cfg["lora"]["lr"] = float(lr)
    cfg["lora"]["force_rebuild"] = True
    cfg["model"]["cond_dropout_prob"] = float(p_uncond)
    cfg["train"]["total_steps"] = int(total_steps)
    cfg["train"]["warmup_steps"] = max(1, int(round(0.03 * total_steps)))
    cfg["train"]["run_dir"] = run_dir
    cfg["train"]["resume"] = False

    if te_rank is not None:
        cfg.setdefault("text_encoder_lora", {})
        cfg["text_encoder_lora"]["rank"] = int(te_rank)
        cfg["text_encoder_lora"]["alpha"] = int(te_rank)


def _broadcast_obj(obj: Dict[str, Any]) -> Dict[str, Any]:
    s = json.dumps(obj) if is_main_process() else ""
    buf = [s]
    if dist.is_initialized():
        dist.broadcast_object_list(buf, src=0)
    return json.loads(buf[0]) if buf[0] else {}


def _apply_env_overrides(
    ds_t: Dict[str, Any],
    md_t: Dict[str, Any],
    tr_t: Dict[str, Any],
    overrides: List[str] | None,
) -> None:
    for kv in overrides or []:
        if "=" not in kv:
            continue
        key, val = kv.split("=", 1)
        try:
            v = json.loads(val)
        except Exception:
            v = val
        parts = key.split(".")
        if not parts:
            continue
        head, rest = parts[0], parts[1:]

        if head == "dataset":
            root = ds_t.setdefault("dataset", {})
            target = ds_t
            target_key = "dataset"
        elif head in ("model", "lora", "text_encoder_lora"):
            if head == "model":
                root = md_t.setdefault("model", {})
                target_key = "model"
            elif head == "lora":
                root = md_t.setdefault("lora", {})
                target_key = "lora"
            else:
                root = md_t.setdefault("text_encoder_lora", {})
                target_key = "text_encoder_lora"
            target = md_t
        elif head in ("train", "optim"):
            if head == "train":
                root = tr_t.setdefault("train", {})
                target_key = "train"
            else:
                root = tr_t.setdefault("optim", {})
                target_key = "optim"
            target = tr_t
        else:
            continue

        if not rest:
            target[target_key] = v
            continue

        cursor = root
        for p in rest[:-1]:
            cursor = cursor.setdefault(p, {})
        cursor[rest[-1]] = v


@dataclass
class EvalCfg:
    base_path: str
    steps: int
    cfg: float
    height: int
    width: int
    negative_prompt: str
    clip_txt_path: str
    clip_vis_path: str
    aesthetic_head_path: str
    pipe_dtype: str = "bf16"


def _build_pipe(device: torch.device, base: str, dtype_name: str):
    dt = torch.bfloat16 if dtype_name == "bf16" else torch.float16
    unet = UNet2DConditionModel.from_pretrained(
        base, subfolder="unet", variant="fp16", torch_dtype=dt, local_files_only=True
    ).to(device)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base, unet=unet, use_safetensors=True, local_files_only=True
    ).to(device)
    pipe.to(dtype=dt)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def _load_lora_into_pipe(pipe, lora_dir: str):
    try:
        pipe.load_lora_adapter(
            lora_dir,
            weight_name="last.safetensors",
            adapter_name="default_0",
            local_files_only=True,
        )
    except Exception:
        pipe.unet.load_attn_procs(
            lora_dir, weight_name="last.safetensors", local_files_only=True
        )
    try:
        pipe.set_adapters(["default_0"], adapter_weights=[0.5])
    except Exception:
        pipe.set_adapters("default_0")


def _partition_tasks(
    n_prompts: int, seeds: List[int], world: int, rank: int
) -> List[Tuple[int, int]]:
    tasks = [(i, s) for i in range(n_prompts) for s in seeds]
    return [t for k, t in enumerate(tasks) if (k % world) == rank]


def _rank_eval_trial(
    cfg_run: Dict[str, Any],
    eval_cfg: EvalCfg,
    prompts: List[Dict[str, Any]],
    seeds: List[int],
) -> List[Dict[str, Any]]:
    if torch.cuda.is_available():
        local = get_local_rank()
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
    else:
        device = torch.device("cpu")

    pipe = _build_pipe(device, eval_cfg.base_path, eval_cfg.pipe_dtype)
    run_dir = cfg_run["train"]["run_dir"]
    _load_lora_into_pipe(pipe, os.path.join(run_dir, "checkpoints"))

    clip_txt_model, clip_txt_proc = load_clip_for_adherence(
        eval_cfg.clip_txt_path, device
    )
    clip_vis_model, clip_vis_proc = load_clip_vision(
        eval_cfg.clip_vis_path, device
    )
    aest_head = load_aesthetic_head(eval_cfg.aesthetic_head_path).to(device).eval()

    my_tasks = _partition_tasks(len(prompts), seeds, get_world_size(), get_rank())

    bar = tqdm(
        total=len(my_tasks),
        desc=f"[eval r{get_rank()}/l{get_local_rank()}]",
        leave=False,
    )
    results: Dict[str, List[Tuple[float, float]]] = {}
    run_n = 0
    run_adh = 0.0
    run_aes = 0.0

    for pidx, seed in my_tasks:
        item = prompts[pidx]
        g = torch.Generator(device=device).manual_seed(int(seed))
        img = pipe(
            prompt=item["gen_prompt"],
            negative_prompt=eval_cfg.negative_prompt,
            num_inference_steps=eval_cfg.steps,
            guidance_scale=eval_cfg.cfg,
            height=eval_cfg.height,
            width=eval_cfg.width,
            generator=g,
        ).images[0]
        clip_vals = clip_cosine_for_batch(
            clip_txt_model, clip_txt_proc, [img], item["clip_prompt"], device
        )
        aest_vals = aesthetic_scores_for_images(
            clip_vis_model, clip_vis_proc, aest_head, [img], device
        )
        rec = results.setdefault(item["id"], [])
        rec.append((clip_vals[0], aest_vals[0]))

        run_n += 1
        run_adh += clip_vals[0]
        run_aes += aest_vals[0]
        bar.set_postfix(
            {
                "adh": f"{(run_adh / run_n):.3f}",
                "aes": f"{(run_aes / run_n):.3f}",
                "score": f"{collapse_proxy(run_adh / run_n, run_aes / run_n):.3f}",
            }
        )
        bar.update(1)

    bar.close()

    per_prompt_partial: List[Dict[str, Any]] = []
    for item in prompts:
        vals = results.get(item["id"], [])
        if vals:
            cmean = sum(v[0] for v in vals) / len(vals)
            amean = sum(v[1] for v in vals) / len(vals)
            per_prompt_partial.append(
                {"id": item["id"], "clip_mean": cmean, "aesthetic_mean": amean}
            )
    return per_prompt_partial


def _gather_per_prompt(parts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if dist.is_initialized():
        out = [None for _ in range(get_world_size())]
        dist.all_gather_object(out, parts)
        merged: Dict[str, Tuple[float, float, int]] = {}
        for chunk in out:
            for row in (chunk or []):
                acc = merged.get(row["id"], (0.0, 0.0, 0))
                merged[row["id"]] = (
                    acc[0] + row["clip_mean"],
                    acc[1] + row["aesthetic_mean"],
                    acc[2] + 1,
                )
        final = []
        for k, (cs, as_, n) in merged.items():
            final.append({"id": k, "clip_mean": cs / n, "aesthetic_mean": as_ / n})
        final.sort(key=lambda x: x["id"])
        return final
    return parts


def _trial_dir(trial_number: int) -> str:
    return os.path.join(HPO_TRIALS, f"trial-{trial_number:04d}")


def _trial_paths(trial_number: int) -> Tuple[str, str, str, str]:
    tdir = _trial_dir(trial_number)
    return (
        os.path.join(tdir, "dataset.yaml"),
        os.path.join(tdir, "model.yaml"),
        os.path.join(tdir, "train.yaml"),
        os.path.join(tdir, "READY"),
    )


def _trial_yaml_paths(trial_number: int) -> Tuple[str, str, str]:
    ds_p, md_p, tr_p, _ = _trial_paths(trial_number)
    return ds_p, md_p, tr_p


def _wait_all_ready(trial_number: int, poll_s: float = 0.1, timeout_s: float = 600.0):
    ds_p, md_p, tr_p, rd_p = _trial_paths(trial_number)
    start = time.time()
    while True:
        local_ready = int(
            os.path.exists(ds_p)
            and os.path.exists(md_p)
            and os.path.exists(tr_p)
            and os.path.exists(rd_p)
        )
        if dist.is_initialized():
            t = torch.tensor(
                local_ready,
                device="cuda" if torch.cuda.is_available() else "cpu",
                dtype=torch.int32,
            )
            dist.all_reduce(t, op=dist.ReduceOp.MIN)
            all_ready = int(t.item())
        else:
            all_ready = local_ready
        if all_ready == 1:
            break
        if (time.time() - start) > timeout_s:
            raise TimeoutError(
                f"[r{get_rank()}] Timeout waiting for all ranks to see trial-{trial_number:04d} files"
            )
        time.sleep(poll_s)


def _clean_env_for_subproc(env: Dict[str, str]) -> Dict[str, str]:
    bad = {
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "GROUP_RANK",
        "NODE_RANK",
        "TORCHELASTIC_RUN_ID",
    }
    new = {k: v for k, v in env.items() if k not in bad}
    new.setdefault("PYTHONNOUSERSITE", "1")
    new.setdefault("OMP_NUM_THREADS", "1")
    return new


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _launch_torchrun_train_for_trial(
    trial_number: int,
    nproc: int,
    log_path: str,
    extra_args: List[str] | None = None,
) -> None:
    ds_p, md_p, tr_p = _trial_yaml_paths(trial_number)
    port = _pick_free_port()

    cmd = [
        "torchrun",
        "--nproc_per_node",
        str(nproc),
        "--master_port",
        str(port),
        "scripts/train.py",
        "--dataset",
        ds_p,
        "--model",
        md_p,
        "--train",
        tr_p,
    ]
    if extra_args:
        cmd += list(extra_args)

    env = _clean_env_for_subproc(os.environ.copy())
    env["MASTER_ADDR"] = env.get("MASTER_ADDR", "127.0.0.1")
    env["MASTER_PORT"] = str(port)

    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "w") as lf:
        print(
            f"[rank0] launching: {' '.join(shlex.quote(x) for x in cmd)}",
            file=lf,
            flush=True,
        )
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(
            f"Child training failed for trial {trial_number} (rc={rc}). See {log_path}."
        )


def _train_and_eval_one(
    cfg_run: Dict[str, Any],
    eval_cfg: EvalCfg,
    seeds: List[int],
    *,
    trial_number: int,
    child_nproc: int | None = None,
) -> List[Dict[str, Any]]:
    run_dir = cfg_run["train"]["run_dir"]
    cfg_hash = _sha256_obj(cfg_run)
    if not _all_equal_string(cfg_hash):
        raise RuntimeError(
            f"[r{get_rank()}] Config hash mismatch across ranks before training."
        )

    barrier()
    time.sleep(0.2)

    if is_main_process():
        nproc = int(child_nproc if child_nproc is not None else get_world_size())
        log_path = os.path.join(run_dir, "train.log")
        _launch_torchrun_train_for_trial(
            trial_number, nproc, log_path, extra_args=None
        )
    barrier()
    time.sleep(0.2)

    parts = _rank_eval_trial(cfg_run, eval_cfg, PROMPT_PANEL, seeds)
    barrier()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return parts


def _write_trial_yamls(trial_number: int, cfg: Dict[str, Any]):
    tdir = _trial_dir(trial_number)
    _ensure_dir(tdir)
    _write_yaml_atomic(os.path.join(tdir, "dataset.yaml"), {"dataset": cfg["dataset"]})
    _write_yaml_atomic(
        os.path.join(tdir, "model.yaml"),
        {
            "model": cfg["model"],
            "lora": cfg["lora"],
            "text_encoder_lora": cfg.get("text_encoder_lora", {}),
        },
    )
    _write_yaml_atomic(
        os.path.join(tdir, "train.yaml"),
        {"train": cfg["train"], "optim": cfg["optim"]},
    )
    if is_main_process():
        ready_tmp = os.path.join(tdir, ".ready.tmp")
        ready_dst = os.path.join(tdir, "READY")
        with open(ready_tmp, "w") as f:
            f.write("ok\n")
            f.flush()
            os.fsync(f.fileno())
        _fsync_replace(ready_tmp, ready_dst)


def _read_trial_cfg_from_yamls(trial_number: int) -> Dict[str, Any]:
    tdir = _trial_dir(trial_number)
    ds = _read_yaml(os.path.join(tdir, "dataset.yaml")) or {}
    md = _read_yaml(os.path.join(tdir, "model.yaml")) or {}
    tr = _read_yaml(os.path.join(tdir, "train.yaml")) or {}
    return _merge_cfgs(ds, md, tr)


def run_study(args):
    os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("PYTHONHASHSEED", "0")
    os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
    os.environ.pop("NCCL_BLOCKING_WAIT", None)
    os.environ.setdefault("POKEFA_FORCE_RESET_ATTN", "1")

    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(get_local_rank())
        except Exception:
            pass

    ddp_init_if_needed()
    rank0 = is_main_process()

    if rank0:
        _ensure_dir(HPO_ROOT)
        _ensure_dir(HPO_TRIALS)
        _ensure_dir(RUNS_ROOT)
    barrier()

    if rank0:
        ds_t = _read_yaml(args.dataset)
        md_t = _read_yaml(args.model)
        tr_t = _read_yaml(args.train)
        hp_t = _read_yaml(args.hpo)
        _apply_env_overrides(ds_t, md_t, tr_t, getattr(args, "override", []))
    else:
        ds_t = md_t = tr_t = hp_t = None
    ds_t = _broadcast_obj(ds_t or {})
    md_t = _broadcast_obj(md_t or {})
    tr_t = _broadcast_obj(tr_t or {})
    hp_t = _broadcast_obj(hp_t or {})

    hpo_root = hp_t.get("hpo", {}) or {}
    stage = str(hpo_root.get("stage", md_t.get("model", {}).get("stage", "base"))).lower()
    if stage not in ("base", "refiner"):
        stage = "base"
    search_te_lora = bool(hpo_root.get("search_te_lora", True))
    te_lora_enabled = bool(md_t.get("text_encoder_lora", {}).get("enable", False))

    eval_cfg = EvalCfg(
        base_path=md_t["model"]["base_path"],
        steps=int(hpo_root.get("eval", {}).get("steps", 28)),
        cfg=float(hpo_root.get("eval", {}).get("cfg", 6.5)),
        height=int(hpo_root.get("eval", {}).get("height", 1024)),
        width=int(hpo_root.get("eval", {}).get("width", 1024)),
        negative_prompt=str(hpo_root.get("eval", {}).get("negative_prompt", "")),
        clip_txt_path=str(hpo_root.get("clip", {}).get("model_path")),
        clip_vis_path=str(hpo_root.get("aesthetic", {}).get("clip_path")),
        aesthetic_head_path=str(hpo_root.get("aesthetic", {}).get("head_path")),
        pipe_dtype=str(hpo_root.get("eval", {}).get("pipe_dtype", "bf16")),
    )
    seeds = list(hpo_root.get("seeds", [0, 1, 2, 3]))
    total_steps = int(hpo_root.get("total_steps", 600))
    n_trials = int(hpo_root.get("n_trials", 10))

    if not rank0:
        expected_trial = 0
        while True:
            barrier()

            stop_flag = os.path.join(HPO_ROOT, "STOP")
            local_stop = int(os.path.exists(stop_flag))
            if dist.is_initialized():
                t = torch.tensor(
                    local_stop,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    dtype=torch.int32,
                )
                dist.all_reduce(t, op=dist.ReduceOp.MAX)
                if int(t.item()) == 1:
                    break
            elif local_stop == 1:
                break

            _wait_all_ready(expected_trial)

            cfg_recv = _read_trial_cfg_from_yamls(expected_trial)
            for k in ("dataset", "model", "lora", "train"):
                if k not in cfg_recv:
                    raise RuntimeError(
                        f"[worker r{get_rank()}] trial {expected_trial}: missing key {k}"
                    )

            trial_token = _sha256_obj(cfg_recv)[:16]
            os.environ["POKEFA_CLEAR_LORA"] = "1"
            os.environ["POKEFA_TRIAL_TOKEN"] = trial_token

            parts = _train_and_eval_one(
                cfg_recv,
                eval_cfg,
                seeds,
                trial_number=expected_trial,
                child_nproc=None,
            )
            _ = _gather_per_prompt(parts)
            barrier()
            time.sleep(0.25)
            expected_trial += 1
        return

    sampler = optuna.samplers.TPESampler(seed=hpo_root.get("seed", 42))
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name="pokefa_lora_hpo",
    )

    def objective(trial: optuna.trial.Trial) -> float:
        trial_id = f"trial-{trial.number:04d}"
        run_dir = os.path.join(RUNS_ROOT, trial_id)
        tdir = _trial_dir(trial.number)
        _ensure_dir(run_dir)
        _ensure_dir(os.path.join(run_dir, "checkpoints"))
        _ensure_dir(tdir)

        if stage == "base":
            lr = trial.suggest_float("lora_lr", 1e-4, 8e-4, log=True)
            rnk = trial.suggest_categorical("lora_rank", [16, 32, 48, 64])
            if te_lora_enabled and search_te_lora:
                te_rnk = trial.suggest_categorical("te_lora_rank", [8, 16, 24, 32])
            else:
                te_rnk = None
        elif stage == "refiner":
            lr = trial.suggest_float("lora_lr", 5e-5, 3e-4, log=True)
            rnk = trial.suggest_categorical(
                "lora_rank", [8, 16, 24, 32, 48, 64]
            )
            te_rnk = None
        else:
            raise RuntimeError(f"Unsupported HPO stage: {stage}")

        p_uncond = trial.suggest_categorical("p_uncond", [0.05, 0.10, 0.15])

        base_cfg = _merge_cfgs(ds_t, md_t, tr_t)
        _override_for_trial(
            base_cfg,
            lr=lr,
            rank=rnk,
            p_uncond=p_uncond,
            total_steps=total_steps,
            run_dir=run_dir,
            te_rank=te_rnk,
        )

        _write_trial_yamls(trial.number, base_cfg)

        barrier()

        _wait_all_ready(trial.number)

        cfg_recv = _read_trial_cfg_from_yamls(trial.number)
        cfg_hash = _sha256_obj(cfg_recv)
        te_rank_log = cfg_recv.get("text_encoder_lora", {}).get("rank")
        rank0_print(
            f"[HPO] {trial_id} cfg_hash={cfg_hash} "
            f"run_dir={cfg_recv['train']['run_dir']} "
            f"stage={stage} "
            f"lora_rank={cfg_recv['lora'].get('rank')} "
            f"te_lora_rank={te_rank_log} "
            f"lora_lr={cfg_recv['lora'].get('lr')} "
            f"p_uncond={cfg_recv['model'].get('cond_dropout_prob')}"
        )

        os.environ["POKEFA_CLEAR_LORA"] = "1"
        os.environ["POKEFA_TRIAL_TOKEN"] = cfg_hash[:16]

        parts0 = _train_and_eval_one(
            cfg_recv,
            eval_cfg,
            seeds,
            trial_number=trial.number,
            child_nproc=None,
        )
        per_prompt = _gather_per_prompt(parts0)
        barrier()

        summary = summarize_metrics(per_prompt)
        dump_metrics_json(os.path.join(tdir, "metrics.json"), summary, per_prompt)
        trial.set_user_attr("adherence_mean", summary["adherence_mean"])
        trial.set_user_attr("aesthetic_mean", summary["aesthetic_mean"])
        trial.set_user_attr("score", summary["score"])
        trial.set_user_attr("stage", stage)
        if te_rnk is not None:
            trial.set_user_attr("te_lora_rank", te_rnk)
        rank0_print(
            f"[HPO] {trial_id} score={summary['score']:.4f}  "
            f"adh={summary['adherence_mean']:.4f}  "
            f"aesth={summary['aesthetic_mean']:.4f}"
        )
        return summary["score"]

    study.optimize(objective, n_trials=n_trials)

    best = {
        "stage": stage,
        "lora_rank": study.best_params["lora_rank"],
        "lora_lr": float(study.best_params["lora_lr"]),
        "p_uncond": float(study.best_params["p_uncond"]),
        "score": float(study.best_value),
    }
    if "te_lora_rank" in study.best_params:
        best["te_lora_rank"] = int(study.best_params["te_lora_rank"])

    with open(os.path.join(HPO_ROOT, "best_params.yaml"), "w") as f:
        yaml.safe_dump(best, f, sort_keys=False)
    rank0_print("[HPO] Best:", best)

    with open(os.path.join(HPO_ROOT, "STOP"), "w") as f:
        f.write("done\n")
    barrier()
