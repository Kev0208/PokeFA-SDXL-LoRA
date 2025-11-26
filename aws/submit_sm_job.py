import os
import pathlib

import sagemaker
import yaml
from sagemaker.pytorch import PyTorch


def load_sm_config() -> dict:
    this = pathlib.Path(__file__).resolve()
    repo_root = this.parents[1]
    cfg_path = repo_root / "training" / "configs" / "aws_sm.yaml"
    with cfg_path.open("r") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg, repo_root


if __name__ == "__main__":
    cfg, repo_root = load_sm_config()

    sm_cfg = cfg.get("sagemaker", {}) or {}

    role_arn = sm_cfg.get("role_arn")
    if not role_arn:
        raise ValueError("Missing sagemaker.role_arn in aws_sm.yaml")

    instance_type = sm_cfg.get("instance_type", "ml.g5.12xlarge")
    instance_count = int(sm_cfg.get("instance_count", 1))
    framework_version = sm_cfg.get("framework_version", "2.2")
    py_version = sm_cfg.get("py_version", "py310")
    output_path = sm_cfg.get("output_path", "s3://your-bucket/outputs")
    code_location = sm_cfg.get("code_location", "s3://your-bucket/code")
    base_job_name = sm_cfg.get("base_job_name", "sdxl-lora")
    volume_size = int(sm_cfg.get("volume_size", 200))
    max_run = int(sm_cfg.get("max_run", 72 * 3600))
    mode = sm_cfg.get("mode", "train")

    region = sm_cfg.get("region")
    if region:
        os.environ.setdefault("AWS_DEFAULT_REGION", region)

    sess = sagemaker.Session()

    estimator = PyTorch(
        entry_point="aws/sm_entry.py",
        source_dir=str(repo_root),
        role=role_arn,
        instance_count=instance_count,
        instance_type=instance_type,
        framework_version=framework_version,
        py_version=py_version,
        sagemaker_session=sess,
        code_location=code_location,
        hyperparameters={"mode": mode},
        output_path=output_path,
        base_job_name=base_job_name,
        volume_size=volume_size,
        disable_profiler=True,
        debugger_hook_config=False,
        max_run=max_run,
    )
    
    estimator.fit(wait=True, logs=True)

