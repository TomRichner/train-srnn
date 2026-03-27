"""Warmup-Hold-Cosine learning rate schedule for PyTorch."""

import math

import torch


class WarmupHoldCosineSchedule(torch.optim.lr_scheduler._LRScheduler):
    """Three-phase LR schedule:

    Phase 1 (0% -> warmup_frac):   Linear warmup from start_lr to max_lr
    Phase 2 (warmup_frac -> hold_frac): Hold at max_lr
    Phase 3 (hold_frac -> 100%):   Cosine decay from max_lr to end_lr

    Args:
        optimizer: PyTorch optimizer.
        total_steps: Total number of training steps.
        max_lr: Peak learning rate (default 5e-3).
        start_lr: Initial LR at the beginning of warmup (default 1e-8).
        end_lr: Final LR after cosine decay.  Defaults to max_lr / 20.
        warmup_frac: Fraction of total_steps used for warmup (default 0.2).
        hold_frac: Fraction of total_steps at which hold phase ends and cosine
            decay begins (default 0.7).
        last_epoch: The index of the last epoch (default -1).
    """

    def __init__(
        self,
        optimizer,
        total_steps,
        max_lr=5e-3,
        start_lr=1e-8,
        end_lr=None,
        warmup_frac=0.2,
        hold_frac=0.7,
        last_epoch=-1,
    ):
        self.total_steps = total_steps
        self.max_lr = max_lr
        self.start_lr = start_lr
        self.end_lr = end_lr if end_lr is not None else max_lr / 20.0
        self.warmup_frac = warmup_frac
        self.hold_frac = hold_frac
        # Must call super().__init__ last because it calls get_lr() immediately
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        warmup_end = int(self.total_steps * self.warmup_frac)
        hold_end = int(self.total_steps * self.hold_frac)

        if step < warmup_end:
            # Phase 1: linear warmup
            t = step / max(1, warmup_end)
            lr = self.start_lr + (self.max_lr - self.start_lr) * t
        elif step < hold_end:
            # Phase 2: hold at max_lr
            lr = self.max_lr
        else:
            # Phase 3: cosine decay
            decay_steps = self.total_steps - hold_end
            progress = (step - hold_end) / max(1, decay_steps)
            lr = self.end_lr + 0.5 * (self.max_lr - self.end_lr) * (
                1.0 + math.cos(math.pi * progress)
            )

        return [lr for _ in self.base_lrs]
