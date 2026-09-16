"""Helpers shared by the analysis scripts: run caches, GCS, rebuilding models from checkpoints."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import torch

from train_srnn import paths
from train_srnn.config import TASK_CONFIGS, compose_config
from train_srnn.models.factory import build_model

REPO = Path(__file__).resolve().parents[1]


def cache_dir() -> Path:
    """Where downloaded runs live: ``$SRNN_CACHE_DIR`` (default ``$SRNN_HOME/cache``)."""
    return paths.cache_dir()


def read_bucket() -> str:
    for line in (REPO / "cloud" / "config.gpu.env").read_text().splitlines():
        if line.strip().startswith("GCP_BUCKET="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("GCP_BUCKET not found in cloud/config.gpu.env")


def gcloud_storage(*args: str, capture: bool = True) -> subprocess.CompletedProcess:
    if not shutil.which("gcloud"):
        sys.exit("ERROR: gcloud not found in PATH. Install the Google Cloud SDK.")
    return subprocess.run(["gcloud", "storage", *args], capture_output=capture, text=True, check=False)


def variant_names(ckpt: dict) -> list[str]:
    """Names of the K networks in a checkpoint (accepts the pre-refactor key too)."""
    return list(ckpt.get("variant_names") or ckpt.get("ablation_names") or [])


def config_overrides(ckpt: dict) -> list[str]:
    """Hydra overrides that rebuild the checkpoint's model geometry.

    Works for checkpoints written before the typed config (their model
    block has no ``variants``; the names come from ``ablation_names``).
    """
    import json
    cfg = ckpt["config"]
    model_cfg = cfg["model"]
    if model_cfg.get("model_version") != 2:
        raise RuntimeError("Historical SRNN checkpoint: rebuild with its recorded original commit")
    ov = [f"task={cfg['task']['name']}", "model=srnn", f"seed={cfg['seed']}"]
    def flatten(prefix, values):
        for key, value in values.items():
            path = prefix + "." + key
            if isinstance(value, dict):
                flatten(path, value)
            else:
                ov.append(path + "=" + json.dumps(value))
    flatten("model", model_cfg)
    for key in ("input_size", "output_size", "task_type", "h", "ode_unfolds"):
        ov.append("task." + key + "=" + json.dumps(cfg["task"][key]))
    # Canonical checkpoint names already carry explicit seeds.
    ov.append("model.variant_seeds=null")
    ov.append("model.variants=" + json.dumps(variant_names(ckpt)))
    return ov


def rebuild_model(ckpt: dict, device: str = "cpu"):
    """Model with the checkpoint's weights loaded; returns ``(model, cfg, names)``."""
    names = variant_names(ckpt)
    if not names:
        raise RuntimeError("checkpoint has no variant names; not an SRNN run")
    cfg = compose_config(config_overrides(ckpt))
    model = build_model(cfg)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=True)
    stale = [k for k in unexpected if k.startswith("cell._")]     # old gradient-hook masks
    if set(unexpected) - set(stale) or set(missing) - {"cell.per_neuron_mask"}:
        raise RuntimeError(f"state_dict mismatch: missing={missing} unexpected={unexpected}")
    return model.to(device).eval(), cfg, names


def load_train_trace(cfg) -> "np.ndarray":
    """The z-scored training trace of the checkpoint's task, from ``$SRNN_DATA_DIR``."""
    from train_srnn.data import build_task
    task = build_task(cfg)
    return task.load(Path(cfg.task.data_dir)).train_trace
