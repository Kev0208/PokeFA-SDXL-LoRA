from typing import Tuple, Dict, Any
from PIL import Image
import hashlib


def _hash_u32(*parts) -> int:
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8"))
    return int.from_bytes(h.digest()[:4], "big")


def _hash_bit(*parts) -> int:
    return _hash_u32(*parts) & 1


def _rand_symm(image_id: str, step: int) -> float:
    if step < 0:
        return 0.0
    u = _hash_u32(image_id, step) / 2**32
    return u * 2.0 - 1.0


def random_ar_crop(
    img: Image.Image,
    T: int,
    *,
    image_id: str,
    step: int,
) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Return (processed_img, geom) with SDXL geometry fields:

      geom = {
        "orig_w": int, "orig_h": int,
        "crop_x": int, "crop_y": int,
        "target_w": int, "target_h": int
      }

    Policy:

      1) Downscale so min_side -> T.
      2) On the resized image (min_side == T), apply AR-aware cropping:
         - If AR ≤ 2.0 → T×T window with center-biased jitter along the long axis.
         - If AR > 2.0 → center vs edge anchors chosen deterministically by (image_id, step).
         - Validation: step < 0 → no jitter; center anchor.
      3) The short side is never cropped; only the long side is cropped.
      4) crop_x / crop_y are in ORIGINAL pixel coordinates.
    """
    orig_w, orig_h = img.size
    geom: Dict[str, Any] = {
        "orig_w": orig_w,
        "orig_h": orig_h,
        "crop_x": 0,
        "crop_y": 0,
        "target_w": T,
        "target_h": T,
    }

    if min(orig_w, orig_h) == 0:
        return Image.new("RGB", (T, T), (0, 0, 0)), geom

    min_side = min(orig_w, orig_h)
    scale = T / float(min_side)

    if abs(scale - 1.0) > 1e-8:
        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))
        img = img.resize((new_w, new_h), Image.BICUBIC)
        w, h = new_w, new_h
    else:
        w, h = orig_w, orig_h

    landscape = w >= h
    long_side, short_side = (w, h) if landscape else (h, w)
    ar = long_side / max(short_side, 1)

    inv_scale = 1.0 / scale

    if ar <= 2.0:
        jf = 0.06 + 0.12 * max(0.0, min(1.0, (ar - 1.0) / 0.7))
        r = _rand_symm(str(image_id), int(step))

        if landscape:
            margin = max(0, w - T)
            jitter_px = int(round(jf * margin))
            base_left = (w - T) // 2
            left = max(0, min(w - T, base_left + int(round(r * jitter_px))))
            top = 0
            crop = img.crop((left, top, left + T, top + T))
        else:
            margin = max(0, h - T)
            jitter_px = int(round(jf * margin))
            base_top = (h - T) // 2
            top = max(0, min(h - T, base_top + int(round(r * jitter_px))))
            left = 0
            crop = img.crop((left, top, left + T, top + T))

        geom["crop_x"] = int(round(left * inv_scale))
        geom["crop_y"] = int(round(top * inv_scale))

        return (
            crop if crop.size == (T, T) else crop.resize((T, T), Image.BICUBIC),
            geom,
        )

    choose_edge = _hash_bit(image_id, step) if step >= 0 else 0

    if landscape:
        left = 0 if choose_edge else max(0, (w - T) // 2)
        top = 0
        crop = img.crop((left, top, left + T, top + T))
    else:
        top = 0 if choose_edge else max(0, (h - T) // 2)
        left = 0
        crop = img.crop((left, top, left + T, top + T))

    geom["crop_x"] = int(round(left * inv_scale))
    geom["crop_y"] = int(round(top * inv_scale))

    return (
        crop if crop.size == (T, T) else crop.resize((T, T), Image.BICUBIC),
        geom,
    )


def center_crop(
    img: Image.Image,
    T: int,
    *,
    image_id: str,
    step: int,
) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Deterministic resize (shortest side -> T, bicubic) then a center T×T crop.
    Geometry matches ar_continuous_mod's contract.
    """
    orig_w, orig_h = img.size
    geom: Dict[str, Any] = {
        "orig_w": orig_w,
        "orig_h": orig_h,
        "crop_x": 0,
        "crop_y": 0,
        "target_w": T,
        "target_h": T,
    }

    if min(orig_w, orig_h) == 0:
        return Image.new("RGB", (T, T), (0, 0, 0)), geom

    min_side = min(orig_w, orig_h)
    scale = T / float(min_side)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))
    if new_w != orig_w or new_h != orig_h:
        img = img.resize((new_w, new_h), Image.BICUBIC)
    w, h = img.size

    left = max(0, (w - T) // 2)
    top = max(0, (h - T) // 2)
    crop = img.crop((left, top, left + T, top + T))
    if crop.size != (T, T):
        crop = crop.resize((T, T), Image.BICUBIC)

    inv_scale = 1.0 / scale
    geom["crop_x"] = int(round(left * inv_scale))
    geom["crop_y"] = int(round(top * inv_scale))

    return crop, geom
