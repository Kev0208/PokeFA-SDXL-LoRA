import math
from typing import Optional, Sequence
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def build_cosine_with_warmup(
    optimizer: Optimizer,
    *,
    total_steps: int,
    warmup_steps: Optional[int] = None,
    warmup_ratio: Optional[float] = None,
    min_lrs: Optional[Sequence[float]] = None,
    group_scales: Optional[Sequence[float]] = None,
    last_epoch: int = -1,
) -> LambdaLR:
    total_steps = max(1, int(total_steps))

    if warmup_steps is None and warmup_ratio is None:
        warmup_steps = max(1, int(0.01 * total_steps))
    if warmup_ratio is not None:
        warmup_steps = int(round(total_steps * float(warmup_ratio)))
    warmup_steps = max(0, int(warmup_steps))

    n_groups = len(optimizer.param_groups)
    if min_lrs is None:
        min_lrs = [0.0] * n_groups
    if group_scales is None:
        group_scales = [1.0] * n_groups
    assert len(min_lrs) == n_groups and len(group_scales) == n_groups

    base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def lr_lambda_factory(gidx: int):
        base_lr = base_lrs[gidx] * group_scales[gidx]
        min_lr = float(min_lrs[gidx])

        def lr_lambda(step: int) -> float:
            step = max(0, step)
            if total_steps <= warmup_steps:
                frac = step / max(1, warmup_steps)
                lr = base_lr * frac
                return (lr / base_lrs[gidx]) if base_lrs[gidx] > 0 else 1.0

            if step < warmup_steps:
                lr = base_lr * (step / max(1, warmup_steps))
            else:
                prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                prog = max(0.0, min(prog, 1.0))
                cosv = 0.5 * (1.0 + math.cos(math.pi * prog))
                lr = min_lr + (base_lr - min_lr) * cosv

            return (lr / base_lrs[gidx]) if base_lrs[gidx] > 0 else 1.0

        return lr_lambda

    lambdas = [lr_lambda_factory(i) for i in range(n_groups)]
    return LambdaLR(optimizer, lr_lambda=lambdas, last_epoch=last_epoch)
