import argparse
import sys, pathlib

THIS = pathlib.Path(__file__).resolve()
TRAIN_DIR = THIS.parents[1]           
SRC_DIR   = TRAIN_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from hpo import run_study

def parse_args():
    ap = argparse.ArgumentParser(description="PokéFA – launch SDXL LoRA HPO")
    ap.add_argument(
        "--dataset",
        required=True,
        help="Path to training/configs/dataset.yaml",
    )
    ap.add_argument(
        "--model",
        required=True,
        help="Path to training/configs/model.yaml",
    )
    ap.add_argument(
        "--train",
        required=True,
        help="Path to training/configs/train.yaml",
    )
    ap.add_argument(
        "--hpo",
        required=True,
        help="Path to training/configs/hpo.yaml",
    )
    ap.add_argument(
        "--override",
        action="append",
        default=[],
        help=(
            "Dot overrides, e.g. "
            '--override model.base_path="/opt/ml/input/cache/sdxl_base"'
        ),
    )
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_study(args)
