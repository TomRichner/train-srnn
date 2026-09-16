"""Build cells and sequence models from the typed config."""

from __future__ import annotations

from dataclasses import fields
import math
from typing import Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from train_srnn.models import variants as V
from train_srnn.models.ctrnn_cell import CTGRUCell, CTGRUConfig, CTRNNCell, CTRNNConfig, NODECell
from train_srnn.models.lstm_cell import LSTMCell
from train_srnn.models.ltc_cell import LTCCell, LTCConfig
from train_srnn.models.rmt_matrix import RMTMatrix
from train_srnn.models.sequence_model import SequenceModel
from train_srnn.models.base import RNNCell
from train_srnn.models.srnn_cell import SRNNCell, SRNNConfig
from train_srnn.utils.io_masks import generate_neuron_partition, make_input_mask


def _dataclass_from_cfg(model_cfg: DictConfig, dc_cls):
    """Instantiate a cell config dataclass from the matching model-config fields."""
    valid = {f.name for f in fields(dc_cls)}
    raw = OmegaConf.to_container(model_cfg, resolve=True)
    return dc_cls(**{k: v for k, v in raw.items() if k in valid})


def _srnn_config(cfg: DictConfig, variant: V.SRNNVariant) -> SRNNConfig:
    if cfg.model.model_version != SRNNCell.MODEL_VERSION:
        raise ValueError("SRNN model_version must be 2; use the historical commit for old checkpoints")
    cell_cfg = _dataclass_from_cfg(cfg.model, SRNNConfig)
    for key in V.FLAGS:
        setattr(cell_cfg, key, getattr(variant, key))
    cell_cfg.init_seed = variant.seed
    return cell_cfg


def _rmt(cfg: DictConfig, seed: int) -> RMTMatrix:
    m = cfg.model.rmt
    n = cfg.model.num_units if m.F_tracks_network else m.F_ref_n
    alpha = round(m.density * n) / n if m.F_tracks_network else m.F_ref_indegree / n
    if not 0 < alpha <= 1:
        raise ValueError("Reference connection probability must be in (0, 1]")
    scale = 1 / math.sqrt(n * alpha * (2 - alpha))
    rmt = RMTMatrix(n=cfg.model.num_units, density=m.density, seed=seed,
                    level_of_chaos=m.level_of_chaos,
                    mu_E_tilde=m.mu_E_relative * scale, mu_I_tilde=m.mu_I_relative * scale,
                    sigma_E_tilde=m.sigma_E_relative * scale, sigma_I_tilde=m.sigma_I_relative * scale)
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


def build_cell(cfg: DictConfig, W_in_mask: Optional[torch.Tensor] = None,
               variants: Optional[list[str]] = None, seeds: Optional[list[int]] = None) -> RNNCell:
    """Build the cell for ``cfg.model.type``; SRNN gets one network per resolved variant."""
    model_type: str = cfg.model.type
    input_size: int = cfg.task.input_size

    if model_type == "lstm":
        return LSTMCell(input_size, cfg.model.num_units)
    if model_type == "ltc":
        return LTCCell(input_size, _dataclass_from_cfg(cfg.model, LTCConfig), W_in_mask=W_in_mask)
    if model_type == "ctrnn":
        return CTRNNCell(input_size, _dataclass_from_cfg(cfg.model, CTRNNConfig), W_in_mask=W_in_mask)
    if model_type == "node":
        return NODECell(input_size, _dataclass_from_cfg(cfg.model, CTRNNConfig), W_in_mask=W_in_mask)
    if model_type == "ctgru":
        return CTGRUCell(input_size, _dataclass_from_cfg(cfg.model, CTGRUConfig), W_in_mask=W_in_mask)
    if model_type == "srnn":
        resolved = srnn_variants(cfg, variants, seeds)
        configs = [_srnn_config(cfg, v) for v in resolved]
        # Variants sharing a seed share one recurrent matrix, so comparisons
        # across variants at the same seed are paired on connectivity.
        rmt_cache: dict[int, RMTMatrix] = {}
        for v in resolved:
            if v.seed not in rmt_cache:
                rmt_cache[v.seed] = _rmt(cfg, v.seed)
        exports = [rmt_cache[v.seed].export_for_srnn(dales=c.dales, dales_init=cfg.model.rmt.dales_init) for c, v in zip(configs, resolved)]
        cell = SRNNCell(configs, input_size, exports, W_in_mask=W_in_mask)
        cell.variant_names = [v.name for v in resolved]
        return cell
    raise ValueError(f"Unknown model type: {model_type!r}")


def build_model(cfg: DictConfig, variants: Optional[list[str]] = None,
                seeds: Optional[list[int]] = None) -> SequenceModel:
    """Cell plus readout. The neuron partition (input/output masks) comes from ``cfg.seed``."""
    W_in_mask = _input_mask(cfg.model.num_units, cfg.seed)
    cell = build_cell(cfg, W_in_mask=W_in_mask, variants=variants, seeds=seeds)
    return SequenceModel(
        cell=cell,
        input_size=cfg.task.input_size,
        output_size=cfg.task.output_size,
        num_units=cfg.model.num_units,
        task_type=cfg.task.task_type,
        io_mask_seed=cfg.seed,
    )
