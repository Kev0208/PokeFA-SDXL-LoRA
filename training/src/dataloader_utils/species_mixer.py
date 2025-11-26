from __future__ import annotations
from typing import Dict, Callable, Any
import csv
import random


def load_species_fracs(csv_path: str) -> Dict[str, float]:
    fracs: Dict[str, float] = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            s = row["pokemon"]
            q = float(row.get("fraction") or 0.0)
            if q > 0:
                fracs[s] = q
    total = sum(fracs.values())
    if total <= 0:
        raise ValueError("species fractions sum to zero.")
    for k in fracs:
        fracs[k] /= total
    return fracs


def _pick_species(meta: Dict[str, Any], field: str, strategy: str) -> str | None:
    val = meta.get(field)
    if val is None:
        return None
    if isinstance(val, list):
        if not val:
            return None
        return str(val[0]) if strategy != "random" else str(random.choice(val))
    if isinstance(val, str):
        parts = [p.strip() for p in val.split(",") if p.strip()]
        return parts[0] if parts else None
    return None


def make_species_accept_fn(
    species_fracs: Dict[str, float],
    *,
    lambda_provider: Callable[[], float],
    choose_strategy: str = "first",
    species_field: str = "species_list",
) -> Callable[[Dict[str, Any]], bool]:
    """
    Acceptance sampling to approximate p_t(s) = (1-λ_t)U + λ_t q(s), with λ_t from a callable.
    """
    K = len(species_fracs)
    if K == 0:
        raise ValueError("Empty species_fracs")
    U = 1.0 / K

    last_lam = None
    last_c = 1.0

    def _refresh_constants(lam: float):
        nonlocal last_lam, last_c
        if last_lam is not None and abs(lam - last_lam) < 1e-9:
            return
        c_vals = []
        for s, q in species_fracs.items():
            pt = (1.0 - lam) * U + lam * q
            if pt > 0 and q > 0:
                c_vals.append(q / pt)
        last_c = min(c_vals) if c_vals else 1.0
        last_c = max(1e-6, min(1.0, last_c))
        last_lam = lam

    def accept(meta: Dict[str, Any]) -> bool:
        lam = max(0.0, min(1.0, float(lambda_provider())))
        _refresh_constants(lam)
        s = _pick_species(meta, species_field, choose_strategy)
        if s is None or s not in species_fracs:
            return random.random() < 0.05
        q = species_fracs[s]
        pt = (1.0 - lam) * U + lam * q
        if q <= 0.0:
            return random.random() < 0.05
        prob = last_c * (pt / q)
        prob = max(0.0, min(1.0, prob))
        return random.random() < prob

    return accept
