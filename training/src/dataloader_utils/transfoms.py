from typing import Dict, Any, Callable
from PIL import Image
import torch
from torchvision.transforms.functional import to_tensor
from . import crops


def _normalize_sdxl(t: torch.Tensor) -> torch.Tensor:
    return t * 2.0 - 1.0


_POLICY_REGISTRY: Dict[str, Callable] = {
    "random_ar_crop": crops.random_ar_crop,
    "center_crop": crops.center_crop,
}


class BuildTransform:
    def __init__(
        self,
        size: int,
        policy_name: str,
        *,
        normalize: str = "sdxl",
        deterministic_val: bool = True,
    ):
        self.T = size
        if policy_name not in _POLICY_REGISTRY:
            raise ValueError(
                f"Unknown policy '{policy_name}'. Available: {list(_POLICY_REGISTRY)}"
            )
        self.policy = _POLICY_REGISTRY[policy_name]
        self.normalize = normalize
        self.deterministic_val = deterministic_val

    def __call__(
        self, sample: Dict[str, Any], *, step: int, is_train: bool
    ) -> Dict[str, Any]:
        img: Image.Image = sample["image"]
        img_id = sample.get("id", sample.get("image_id", "unknown"))

        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        step_arg = step if (is_train or not self.deterministic_val) else -1
        proc_img, geom = self.policy(img, self.T, image_id=str(img_id), step=step_arg)

        t = to_tensor(proc_img)
        if self.normalize == "sdxl":
            t = _normalize_sdxl(t)

        sample["image"] = t
        sample["original_size"] = (int(geom["orig_w"]), int(geom["orig_h"]))
        sample["crop_coords_top_left"] = (int(geom["crop_x"]), int(geom["crop_y"]))
        sample["target_size"] = (int(geom["target_w"]), int(geom["target_h"]))

        return sample
