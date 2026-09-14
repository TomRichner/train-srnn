"""Train one model on one task.

    python train.py task=cheetah100 model=srnn model.variants=[srnn,srnn-no-adapt] epochs=100

See train_srnn/config.py for every option. Outputs land in
``$SRNN_HOME/results/<task>/<run_name>/``.
"""
from __future__ import annotations

import logging
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from train_srnn.data import build_task
from train_srnn.models.factory import build_model
from train_srnn.training import TRAINERS

log = logging.getLogger(__name__)


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def check_amp(cfg: DictConfig, device: torch.device) -> None:
    if cfg.amp == "fp32":
        return
    if cfg.amp != "bf16":
        raise ValueError(f"Unknown amp mode {cfg.amp!r}; expected fp32 or bf16")
    if device.type != "cuda" or not torch.cuda.is_bf16_supported():
        raise RuntimeError("amp=bf16 needs an Ampere-or-newer CUDA device")
    log.info("AMP: bf16 autocast")


def freeze_params(model, groups: list[str]) -> None:
    """Pin logical parameter groups at their initial values."""
    if not groups:
        return
    frozen = model.cell.freeze([g for g in groups if g != "W_out_gain"])
    if "W_out_gain" in groups:
        model.W_out_gain.requires_grad_(False)
        frozen.append("W_out_gain")
    log.info("Frozen params: %s -> %s", list(groups), frozen)


def maybe_compile(cfg: DictConfig, model, device: torch.device):
    """Model-level torch.compile; the ring trainer compiles just the cell itself."""
    if not (cfg.compile.enabled and device.type == "cuda"):
        return model
    if cfg.compile.log_recompiles:
        import torch._logging
        torch._logging.set_logs(recompiles=True)
    if cfg.compile.cell_only and cfg.task.trainer == "continuous":
        return model
    kwargs = {}
    if cfg.compile.mode:
        kwargs["mode"] = str(cfg.compile.mode)
    if cfg.compile.dynamic is not None:
        kwargs["dynamic"] = bool(cfg.compile.dynamic)
    log.info("torch.compile(model) kwargs=%s", kwargs or "(defaults)")
    return torch.compile(model, **kwargs)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = resolve_device(cfg.device)
    check_amp(cfg, device)
    log.info("Device: %s", device)

    task = build_task(cfg)
    data = task.load(Path(cfg.task.data_dir))
    task.validate(data)

    model = build_model(cfg).to(device)
    freeze_params(model, list(cfg.freeze_params))
    log.info("Parameters: %d total, %d trainable",
             sum(p.numel() for p in model.parameters()),
             sum(p.numel() for p in model.parameters() if p.requires_grad))
    model = maybe_compile(cfg, model, device)

    trainer = TRAINERS[cfg.task.trainer](cfg, model, task, data, device, Path(cfg.output_dir))
    trainer.fit()


if __name__ == "__main__":
    main()
