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
    cfg = ckpt["config"]
    task, model = cfg["task"], cfg["model"]
    task_name = task["name"] if task["name"] in TASK_CONFIGS else "cheetah100"
    rmt = model.get("rmt", {})
    ov = [
        f"task={task_name}", "model=srnn", f"seed={cfg['seed']}",
        f"task.input_size={task['input_size']}", f"task.output_size={task['output_size']}",
        f"task.task_type={task['task_type']}", f"task.h={model['h']}",
        f"task.ode_unfolds={model['ode_unfolds']}",
        f"model.num_units={model['num_units']}", f"model.solver={model['solver']}",
        f"model.readout={model.get('readout', 'synaptic')}",
        f"model.tau_global_init={model.get('tau_global_init', 1.0)}",
        f"model.tau_a_lo_init={model.get('tau_a_lo_init', 0.25)}",
        f"model.tau_a_hi_init={model.get('tau_a_hi_init', 4.0)}",
        f"model.std_zero_floor={str(model.get('std_zero_floor', True)).lower()}",
        f"model.rmt.density={rmt.get('density', model.get('alpha', 1.0 / 3.0))}",
        f"model.rmt.level_of_chaos={rmt.get('level_of_chaos', model.get('level_of_chaos', 1.0))}",
        "model.variants=[" + ",".join(variant_names(ckpt)) + "]",
    ]
    for key in ("dales", "n_a_E", "n_a_I", "n_b_E", "n_b_I", "per_neuron", "echo", "skip"):
        if key in model and "variants" in model:      # typed config: base flags matter
            ov.append(f"model.{key}={str(model[key]).lower()}")
    return ov


def rebuild_model(ckpt: dict, device: str = "cpu"):
    """Model with the checkpoint's weights loaded; returns ``(model, cfg, names)``."""
    names = variant_names(ckpt)
    if not names:
        raise RuntimeError("checkpoint has no variant names; not an SRNN run")
    cfg = compose_config(config_overrides(ckpt))
    model = build_model(cfg)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    stale = [k for k in unexpected if k.startswith("cell._")]     # old gradient-hook masks
    if set(unexpected) - set(stale) or set(missing) - {"cell.per_neuron_mask"}:
        raise RuntimeError(f"state_dict mismatch: missing={missing} unexpected={unexpected}")
    return model.to(device).eval(), cfg, names


def load_train_trace(cfg) -> "np.ndarray":
    """The z-scored training trace of the checkpoint's task, from ``$SRNN_DATA_DIR``."""
    from train_srnn.data import build_task
    task = build_task(cfg)
    return task.load(Path(cfg.task.data_dir)).train_trace
