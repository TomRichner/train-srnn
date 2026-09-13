"""Build cells and sequence models from the typed config."""

from __future__ import annotations

from dataclasses import fields, replace
from typing import Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from train_srnn.models import variants as V
from train_srnn.models.ctrnn_cell import (
    CTGRUCell, CTGRUConfig, CTRNNCell, CTRNNConfig, NODECell, NODEConfig,
)
from train_srnn.models.ltc_cell import LTCCell, LTCConfig
from train_srnn.models.rmt_matrix import RMTMatrix
from train_srnn.models.sequence_model import LSTMCellWrapper, SequenceModel
from train_srnn.models.srnn_cell import BatchedSRNNCell, SRNNCell, SRNNConfig
from train_srnn.utils.io_masks import generate_neuron_partition, make_input_mask


def _dataclass_from_cfg(model_cfg: DictConfig, dc_cls):
    """Instantiate a cell config dataclass from the matching model-config fields."""
    valid = {f.name for f in fields(dc_cls)}
    raw = OmegaConf.to_container(model_cfg, resolve=True)
    return dc_cls(**{k: v for k, v in raw.items() if k in valid})


def _srnn_config(cfg: DictConfig, variant: V.SRNNVariant) -> SRNNConfig:
    m = cfg.model
    return SRNNConfig(
        num_units=m.num_units, dales=variant.dales,
        n_a_E=variant.n_a_E, n_a_I=variant.n_a_I,
        n_b_E=variant.n_b_E, n_b_I=variant.n_b_I,
        per_neuron=variant.per_neuron, echo=variant.echo, skip=variant.skip,
        solver=m.solver, h=m.h, ode_unfolds=m.ode_unfolds, readout=m.readout,
        tau_global_init=m.tau_global_init, tau_a_lo_init=m.tau_a_lo_init,
        tau_a_hi_init=m.tau_a_hi_init, std_zero_floor=m.std_zero_floor,
    )


def _rmt(cfg: DictConfig, seed: int) -> RMTMatrix:
    rmt = RMTMatrix(n=cfg.model.num_units, density=cfg.model.rmt.density,
                    seed=seed, level_of_chaos=cfg.model.rmt.level_of_chaos)
    rmt.build()
    return rmt


def _input_mask(num_units: int, seed: int) -> torch.Tensor:
    input_idx, _, _ = generate_neuron_partition(num_units, seed)
    return torch.tensor(make_input_mask(num_units, input_idx), dtype=torch.float32)


def srnn_variants(cfg: DictConfig, names: Optional[list[str]] = None,
                  seeds: Optional[list[int]] = None) -> list[V.SRNNVariant]:
    """Resolve ``model.variants`` x ``model.variant_seeds`` (or explicit lists)."""
    m = cfg.model
    names = list(names if names is not None else m.variants)
    seeds = list(seeds) if seeds is not None else (
        list(m.variant_seeds) if m.variant_seeds else None)
    return V.expand(names, seeds, m, default_seed=int(cfg.seed))


def build_cell(cfg: DictConfig, W_in_mask: Optional[torch.Tensor] = None) -> nn.Module:
    """Build a single-network cell for ``cfg.model.type``."""
    model_type: str = cfg.model.type
    input_size: int = cfg.task.input_size

    if model_type == "lstm":
        return LSTMCellWrapper(input_size, cfg.model.num_units)
    if model_type == "ltc":
        return LTCCell(input_size, _dataclass_from_cfg(cfg.model, LTCConfig), W_in_mask=W_in_mask)
    if model_type == "ctrnn":
        return CTRNNCell(input_size, _dataclass_from_cfg(cfg.model, CTRNNConfig), W_in_mask=W_in_mask)
    if model_type == "node":
        return NODECell(input_size, _dataclass_from_cfg(cfg.model, NODEConfig), W_in_mask=W_in_mask)
    if model_type == "ctgru":
        return CTGRUCell(input_size, _dataclass_from_cfg(cfg.model, CTGRUConfig), W_in_mask=W_in_mask)
    if model_type == "srnn":
        variant = srnn_variants(cfg)[0]
        srnn_cfg = _srnn_config(cfg, variant)
        rmt_export = _rmt(cfg, variant.seed).export_for_srnn(dales=srnn_cfg.dales)
        return SRNNCell(srnn_cfg, input_size, rmt_export, W_in_mask=W_in_mask)
    raise ValueError(f"Unknown model type: {model_type!r}")


def _wrap(cfg: DictConfig, cell: nn.Module) -> SequenceModel:
    return SequenceModel(
        cell=cell,
        input_size=cfg.task.input_size,
        output_size=cfg.task.output_size,
        num_units=cfg.model.num_units,
        task_type=cfg.task.task_type,
        io_mask_seed=cfg.seed,
    )


def build_model(cfg: DictConfig) -> SequenceModel:
    """Single-network model (for SRNN: the first entry of ``model.variants``)."""
    W_in_mask = _input_mask(cfg.model.num_units, cfg.seed)
    return _wrap(cfg, build_cell(cfg, W_in_mask=W_in_mask))


def build_batched_model(
    cfg: DictConfig,
    names: Optional[list[str]] = None,
    seeds: Optional[list[int]] = None,
) -> SequenceModel:
    """K SRNN variants in one batched cell.

    Variants sharing a recurrent-matrix seed share one ``RMTMatrix``, so a
    comparison across variants at the same seed is paired on connectivity.
    The neuron partition (input/output masks) is built from ``cfg.seed`` and
    shared by all K networks.
    """
    variants = srnn_variants(cfg, names, seeds)
    configs = [_srnn_config(cfg, v) for v in variants]

    rmt_cache: dict[int, RMTMatrix] = {}
    for v in variants:
        if v.seed not in rmt_cache:
            rmt_cache[v.seed] = _rmt(cfg, v.seed)
    rmt_exports = [rmt_cache[v.seed].export_for_srnn(dales=c.dales)
                   for c, v in zip(configs, variants)]

    W_in_mask = _input_mask(cfg.model.num_units, cfg.seed)
    cell = BatchedSRNNCell(configs, cfg.task.input_size, rmt_exports, W_in_mask=W_in_mask)
    model = _wrap(cfg, cell)
    model.ablation_names = [v.name for v in variants]
    return model
