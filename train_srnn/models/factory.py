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


#: Separator between a preset name and its recurrent-matrix seed in a
#: decorated ablation name, e.g. ``srnn-no-dales-skip-seed3``. Chosen to be
#: safe as a directory name, in pandoc markdown, and in LaTeX, since decorated
#: names become per-variant output directories and report headings.
_SEED_SEP = "-seed"


def parse_ablation_name(name: str, default_seed: int) -> tuple[str, int]:
    """Split a (possibly decorated) ablation name into ``(preset, seed)``.

    ``"srnn-skip"``        -> ``("srnn-skip", default_seed)``
    ``"srnn-skip-seed3"``  -> ``("srnn-skip", 3)``

    Decorated names are what get stored in ``model.ablation_names`` and hence
    in every checkpoint, so scripts that rebuild a model from a checkpoint
    (``build_batched_model(cfg, ckpt["ablation_names"])``) reconstruct the
    exact same per-variant seeds without needing the seed list.
    """
    if _SEED_SEP not in name:
        return name, int(default_seed)
    preset, _, seed_str = name.rpartition(_SEED_SEP)
    try:
        return preset, int(seed_str)
    except ValueError:
        raise ValueError(
            f"Malformed ablation name {name!r}: expected "
            f"'<preset>{_SEED_SEP}<int>'."
        ) from None


def expand_ablation_specs(
    ablation_names: list[str], ablation_seeds: list[int] | None
) -> list[str]:
    """Cross variants with seeds, variant-major, decorating each name.

    ``(["a", "b"], [1, 2])`` -> ``["a-seed1", "a-seed2", "b-seed1", "b-seed2"]``.

    With *ablation_seeds* falsy the names pass through unchanged (and each
    variant then uses ``cfg.seed``).
    """
    if not ablation_seeds:
        return list(ablation_names)
    out = []
    for name in ablation_names:
        if _SEED_SEP in name:
            raise ValueError(
                f"batched_ablation_seeds was given, but {name!r} already "
                f"carries a '{_SEED_SEP}' seed suffix — pick one or the other."
            )
        for s in ablation_seeds:
            out.append(f"{name}{_SEED_SEP}{int(s)}")
    return out


def build_batched_model(
    cfg: DictConfig,
    ablation_names: list[str],
    ablation_seeds: list[int] | None = None,
) -> SequenceModel:
    """Build a batched ablation model for parallel SRNN variants.

    Each name in *ablation_names* is looked up in :data:`SRNN_PRESETS` to
    obtain an :class:`SRNNConfig`.  All configs get ``num_units`` overridden
    from ``cfg.model.num_units``.

    A name may carry a ``-seed<int>`` suffix selecting the seed of the recurrent
    random matrix for that variant; without one, ``cfg.seed`` is used.  Passing
    *ablation_seeds* expands the plain names against those seeds (see
    :func:`expand_ablation_specs`) — e.g. two variants x five seeds gives K=10
    networks trained in one batch.

    Variants sharing a seed share one ``RMTMatrix``, so a comparison across
    variants at the same seed stays paired on the recurrent connectivity.  The
    other random draws (``cell.W_in``, ``readout_weight``) are independent per
    variant either way, since they are allocated ``(K, ...)`` in one draw.

    The neuron partition (input/output masks) is always built from ``cfg.seed``
    and shared by all K variants — ``BatchedSRNNCell`` broadcasts a single
    ``W_in_mask`` and ``SequenceModel`` a single ``output_mask``, so I/O
    routing is held fixed across seeds by construction.
    """
    input_size: int = cfg.task.input_size
    num_units: int = cfg.model.num_units
    seed = cfg.seed

    from dataclasses import replace

    names = expand_ablation_specs(list(ablation_names), ablation_seeds)

    configs, variant_seeds = [], []
    for name in names:
        preset_name, variant_seed = parse_ablation_name(name, seed)
        if preset_name not in SRNN_PRESETS:
            raise ValueError(
                f"Unknown ablation preset: {preset_name!r} (from {name!r}). "
                f"Available: {list(SRNN_PRESETS.keys())}"
            )
        preset = SRNN_PRESETS[preset_name]
        configs.append(replace(
            preset,
            num_units=num_units,
            solver=cfg.model.get("solver", preset.solver),
            h=cfg.model.get("h", preset.h),
            ode_unfolds=cfg.model.get("ode_unfolds", preset.ode_unfolds),
            tau_a_lo_init=cfg.model.get("tau_a_lo_init", preset.tau_a_lo_init),
            tau_a_hi_init=cfg.model.get("tau_a_hi_init", preset.tau_a_hi_init),
        ))
        variant_seeds.append(variant_seed)

    # One RMTMatrix per distinct seed; variants at the same seed share W.
    rmt_cache: dict[int, RMTMatrix] = {}
    for s in variant_seeds:
        if s not in rmt_cache:
            rmt = RMTMatrix(
                n=num_units,
                density=cfg.model.get("alpha", 1.0 / 3.0),
                seed=s,
                level_of_chaos=cfg.model.get("level_of_chaos", 1.0),
            )
            rmt.build()
            rmt_cache[s] = rmt

    # Export per variant (same underlying W per seed, different dales modes)
    rmt_exports = [
        rmt_cache[s].export_for_srnn(dales=c.dales)
        for c, s in zip(configs, variant_seeds)
    ]

    # Create W_in_mask from neuron partition
    input_idx, _, _ = generate_neuron_partition(num_units, seed)
    W_in_mask = torch.tensor(
        make_input_mask(num_units, input_idx), dtype=torch.float32
    )

    batched_cell = BatchedSRNNCell(configs, input_size, rmt_exports, W_in_mask=W_in_mask)

    model = SequenceModel(
        cell=batched_cell,
        input_size=input_size,
        output_size=cfg.task.output_size,
        num_units=num_units,
        task_type=cfg.task.task_type,
        io_mask_seed=seed,
    )
    model.ablation_names = names
    return model
