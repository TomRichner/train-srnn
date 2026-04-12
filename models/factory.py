"""Model factory — build cells and sequence models from Hydra config."""

from __future__ import annotations

from dataclasses import fields
from typing import Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from train_srnn.models.sequence_model import LSTMCellWrapper, SequenceModel
from train_srnn.models.ltc_cell import LTCCell, LTCConfig
from train_srnn.models.srnn_cell import (
    SRNNCell,
    SRNNConfig,
    BatchedSRNNCell,
    SRNN_PRESETS,
)
from train_srnn.models.rmt_matrix import RMTMatrix
from train_srnn.models.ctrnn_cell import (
    CTRNNCell,
    CTRNNConfig,
    NODECell,
    NODEConfig,
    CTGRUCell,
    CTGRUConfig,
)
from train_srnn.utils.io_masks import generate_neuron_partition, make_input_mask


def _cfg_to_dataclass(model_cfg: DictConfig, dc_cls):
    """Extract fields matching a dataclass from an OmegaConf config."""
    valid_keys = {f.name for f in fields(dc_cls)}
    raw = OmegaConf.to_container(model_cfg, resolve=True)
    filtered = {k: v for k, v in raw.items() if k in valid_keys}
    return dc_cls(**filtered)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_cell(
    cfg: DictConfig,
    W_in_mask: Optional[torch.Tensor] = None,
) -> nn.Module:
    """Build an RNN cell from Hydra config.

    Supported ``cfg.model.type`` values:
        lstm, ltc, ltc_rk, ltc_ex, ctrnn, node, ctgru, srnn (+ all ablations)
    """
    model_type: str = cfg.model.type
    input_size: int = cfg.task.input_size
    num_units: int = cfg.model.num_units

    if model_type == "lstm":
        return LSTMCellWrapper(input_size, num_units)

    if model_type in ("ltc", "ltc_rk", "ltc_ex"):
        ltc_cfg = _cfg_to_dataclass(cfg.model, LTCConfig)
        if model_type == "ltc_rk":
            ltc_cfg.solver = "rk4"
        elif model_type == "ltc_ex":
            ltc_cfg.solver = "explicit"
        return LTCCell(input_size, ltc_cfg, W_in_mask=W_in_mask)

    if model_type == "srnn":
        srnn_cfg = _cfg_to_dataclass(cfg.model, SRNNConfig)
        rmt = RMTMatrix(
            n=num_units,
            density=cfg.model.get("alpha", 1.0 / 3.0),
            seed=cfg.seed,
            level_of_chaos=cfg.model.get("level_of_chaos", 1.0),
        )
        rmt.build()
        rmt_export = rmt.export_for_srnn(dales=srnn_cfg.dales)
        return SRNNCell(srnn_cfg, input_size, rmt_export, W_in_mask=W_in_mask)

    if model_type == "ctrnn":
        ctrnn_cfg = _cfg_to_dataclass(cfg.model, CTRNNConfig)
        return CTRNNCell(input_size, ctrnn_cfg, W_in_mask=W_in_mask)

    if model_type == "node":
        node_cfg = _cfg_to_dataclass(cfg.model, NODEConfig)
        return NODECell(input_size, node_cfg, W_in_mask=W_in_mask)

    if model_type == "ctgru":
        ctgru_cfg = _cfg_to_dataclass(cfg.model, CTGRUConfig)
        return CTGRUCell(input_size, ctgru_cfg, W_in_mask=W_in_mask)

    raise ValueError(f"Unknown model type: {model_type!r}")


def build_model(cfg: DictConfig) -> SequenceModel:
    """Build a full :class:`SequenceModel` from Hydra config."""
    num_units = cfg.model.num_units
    seed = cfg.seed

    # Create W_in_mask from neuron partition (same seed used by SequenceModel
    # for the output mask, ensuring consistent partitioning).
    input_idx, _, _ = generate_neuron_partition(num_units, seed)
    W_in_mask = torch.tensor(
        make_input_mask(num_units, input_idx), dtype=torch.float32
    )

    cell = build_cell(cfg, W_in_mask=W_in_mask)
    return SequenceModel(
        cell=cell,
        input_size=cfg.task.input_size,
        output_size=cfg.task.output_size,
        num_units=num_units,
        task_type=cfg.task.task_type,
        io_mask_seed=seed,
    )


def build_batched_model(
    cfg: DictConfig, ablation_names: list[str]
) -> SequenceModel:
    """Build a batched ablation model for parallel SRNN variants.

    Each name in *ablation_names* is looked up in :data:`SRNN_PRESETS` to
    obtain an :class:`SRNNConfig`.  All configs get ``num_units`` overridden
    from ``cfg.model.num_units``.
    """
    input_size: int = cfg.task.input_size
    num_units: int = cfg.model.num_units
    seed = cfg.seed

    from dataclasses import replace

    configs = []
    for name in ablation_names:
        if name not in SRNN_PRESETS:
            raise ValueError(
                f"Unknown ablation preset: {name!r}. "
                f"Available: {list(SRNN_PRESETS.keys())}"
            )
        preset = SRNN_PRESETS[name]
        configs.append(replace(preset, num_units=num_units))

    # Build ONE RMTMatrix (same seed → same W for all ablations)
    rmt = RMTMatrix(
        n=num_units,
        density=cfg.model.get("alpha", 1.0 / 3.0),
        seed=seed,
        level_of_chaos=cfg.model.get("level_of_chaos", 1.0),
    )
    rmt.build()

    # Export per variant (same underlying W, different dales modes)
    rmt_exports = [rmt.export_for_srnn(dales=c.dales) for c in configs]

    # Create W_in_mask from neuron partition
    input_idx, _, _ = generate_neuron_partition(num_units, seed)
    W_in_mask = torch.tensor(
        make_input_mask(num_units, input_idx), dtype=torch.float32
    )

    batched_cell = BatchedSRNNCell(configs, input_size, rmt_exports, W_in_mask=W_in_mask)

    return SequenceModel(
        cell=batched_cell,
        input_size=input_size,
        output_size=cfg.task.output_size,
        num_units=num_units,
        task_type=cfg.task.task_type,
        io_mask_seed=seed,
    )
