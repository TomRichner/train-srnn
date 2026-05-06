"""Regression test for the log_* -> isp_* parameter rename.

Captures a golden file before the rename and re-checks against it after the
rename. The "before" capture stores per-parameter signatures keyed by the
old (log_*) names; the "after" run renames the *current* parameter names
back to the old names via LOG_TO_ISP_RENAME_MAP (inverse), then compares.

The forward outputs and final state are bit-identical regardless of the
attribute name — the rename only changes Python-level attribute names,
not the underlying tensor data or the ODE arithmetic. So the assertions
below are exact equality (torch.equal), not allclose.

Usage
-----
  PYTHONPATH=. python scripts/test_isp_rename_regression.py --capture
      → writes scripts/_isp_rename_golden.pt

  PYTHONPATH=. python scripts/test_isp_rename_regression.py
      → loads the golden, applies the rename map, asserts equivalence
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from train_srnn.models.rmt_matrix import RMTMatrix          # noqa: E402
from train_srnn.models.srnn_cell import (                   # noqa: E402
    SRNNCell,
    SRNNConfig,
    BatchedSRNNCell,
    SRNN_PRESETS,
)


GOLDEN_PATH = REPO_ROOT / "scripts" / "_isp_rename_golden.pt"


# Forward map: old (log_*) -> new (isp_*). Used at compare-time to map the
# golden's old keys onto the current model's keys, regardless of which side
# of the rename is "current."
LOG_TO_ISP_RENAME_MAP: dict[str, str] = {
    # SRNNCell + BatchedSRNNCell
    "log_tau_global":         "isp_tau_global",
    # SRNNCell single-cell
    "log_tau_d":              "isp_tau_d",
    "log_tau_a_E":            "isp_tau_a_E",
    "log_tau_a_E_lo":         "isp_tau_a_E_lo",
    "log_tau_a_E_hi":         "isp_tau_a_E_hi",
    "log_tau_a_I":            "isp_tau_a_I",
    "log_tau_a_I_lo":         "isp_tau_a_I_lo",
    "log_tau_a_I_hi":         "isp_tau_a_I_hi",
    "log_c_E":                "isp_c_E",
    "log_c_I":                "isp_c_I",
    "log_tau_b_rec_E":        "isp_tau_b_rec_E",
    "log_tau_b_rel_E":        "isp_tau_b_rel_E",
    "log_tau_b_rec_I":        "isp_tau_b_rec_I",
    "log_tau_b_rel_I":        "isp_tau_b_rel_I",
    # BatchedSRNNCell *_vec
    "log_tau_d_vec":          "isp_tau_d_vec",
    "log_tau_a_E_vec":        "isp_tau_a_E_vec",
    "log_tau_a_I_vec":        "isp_tau_a_I_vec",
    "log_c_E_vec":            "isp_c_E_vec",
    "log_c_I_vec":            "isp_c_I_vec",
    "log_tau_b_rec_E_vec":    "isp_tau_b_rec_E_vec",
    "log_tau_b_rel_E_vec":    "isp_tau_b_rel_E_vec",
    "log_tau_b_rec_I_vec":    "isp_tau_b_rec_I_vec",
    "log_tau_b_rel_I_vec":    "isp_tau_b_rel_I_vec",
}


# ---------------------------------------------------------------------------
# Builders — each picks a different code path on purpose
# ---------------------------------------------------------------------------

def _build_rmt(num_units: int, seed: int, dales: bool):
    rmt = RMTMatrix(n=num_units, density=1.0/3.0, seed=seed, level_of_chaos=1.0)
    rmt.build()
    return rmt.export_for_srnn(dales=dales)


def build_single_singletier() -> tuple[SRNNCell, str]:
    """SRNNCell with n_a_E=1, n_a_I=1 (single-tier SFA), STD on, per_neuron=False.

    Exercises log_tau_global, log_tau_d, log_tau_a_E, log_tau_a_I (single-tier
    branches), log_c_E, log_c_I, log_tau_b_rec_E/I, log_tau_b_rel_E/I.
    """
    cfg = SRNNConfig(
        num_units=12, dales=True,
        n_a_E=1, n_a_I=1, n_b_E=1, n_b_I=1,
        per_neuron=False, solver="semi_implicit", h=0.02, ode_unfolds=1,
    )
    rmt_export = _build_rmt(cfg.num_units, seed=42, dales=cfg.dales)
    return SRNNCell(cfg, input_size=4, rmt_export=rmt_export), "single_singletier"


def build_single_multitier() -> tuple[SRNNCell, str]:
    """SRNNCell with n_a_E=3, n_a_I=3 (multi-tier SFA), STD on, per_neuron=False.

    The only configuration that creates log_tau_a_{E,I}_{lo,hi} attrs.
    """
    cfg = SRNNConfig(
        num_units=12, dales=True,
        n_a_E=3, n_a_I=3, n_b_E=1, n_b_I=1,
        per_neuron=False, solver="semi_implicit", h=0.02, ode_unfolds=1,
    )
    rmt_export = _build_rmt(cfg.num_units, seed=43, dales=cfg.dales)
    return SRNNCell(cfg, input_size=4, rmt_export=rmt_export), "single_multitier"


def build_batched_mixed() -> tuple[BatchedSRNNCell, str]:
    """BatchedSRNNCell K=3 mixing three presets.

    Picks variants that exercise:
    - srnn (default 3-tier SFA, STD, no per_neuron)
    - srnn-e-only-per-neuron (per_neuron=True hits the *_vec gradient hooks)
    - srnn-multi-sfa (n_a_E=2 multi-tier batched init at SFA tier index 2)
    """
    from dataclasses import replace
    names = ["srnn", "srnn-e-only-per-neuron", "srnn-multi-sfa"]
    num_units = 12
    configs = []
    for n in names:
        preset = SRNN_PRESETS[n]
        configs.append(replace(preset, num_units=num_units))
    rmt_exports = [
        _build_rmt(num_units, seed=44, dales=c.dales) for c in configs
    ]
    return BatchedSRNNCell(
        configs, input_size=4, rmt_exports=rmt_exports
    ), "batched_mixed"


CONFIGS = [build_single_singletier, build_single_multitier, build_batched_mixed]


# ---------------------------------------------------------------------------
# Capture / verify core
# ---------------------------------------------------------------------------

def _perturb_in_place(cell: torch.nn.Module, seed: int) -> None:
    """Reproducibly perturb every Parameter so gradients are non-trivial."""
    g = torch.Generator().manual_seed(seed)
    for _, p in cell.named_parameters():
        noise = torch.empty_like(p).normal_(generator=g) * 0.3
        with torch.no_grad():
            p.add_(noise)


def _capture_one(builder) -> dict:
    # Seed the global RNG before construction so cell-internal randn calls
    # (W_in, init_state's x) are reproducible across runs.
    torch.manual_seed(2026)
    cell, label = builder()
    _perturb_in_place(cell, seed=123)

    # Determine input batching shape per cell type
    if isinstance(cell, BatchedSRNNCell):
        K = cell.K
        B = 2
        T = 5
        D = cell.input_size
        N = cell.N
        # state shape (K, B, max_state_dim)
        state = cell.init_state(B, device=torch.device("cpu"))
        # Override init_state's randn for determinism
        torch.manual_seed(456)
        x = state[..., -N:].clone()
        x.normal_().mul_(0.1)
        state = state.clone()
        state[..., -N:] = x
        # inputs: (B, D) per step — broadcast over K inside _batched_input_drive
        torch.manual_seed(789)
        inputs_seq = torch.randn(T, B, D)
    else:
        B = 2
        T = 5
        D = cell.input_size
        N = cell.config.num_units
        state = cell.init_state(B, device=torch.device("cpu"))
        torch.manual_seed(456)
        x = state[..., -N:].clone()
        x.normal_().mul_(0.1)
        state = state.clone()
        state[..., -N:] = x
        torch.manual_seed(789)
        inputs_seq = torch.randn(T, B, D)

    state.requires_grad_(False)  # state is a hidden, not a leaf param
    outputs = []
    for t in range(T):
        out, state = cell(inputs_seq[t], state)
        outputs.append(out)
    out_stack = torch.stack(outputs, dim=0)  # (T, ...) or (T, K, B, N)

    loss = out_stack.pow(2).sum()
    loss.backward()

    grad_norms = {}
    data_norms = {}
    for n, p in cell.named_parameters():
        data_norms[n] = p.detach().double().norm().item()
        if p.grad is None:
            grad_norms[n] = None
        else:
            grad_norms[n] = p.grad.detach().double().norm().item()

    return {
        "label": label,
        "output": out_stack.detach().clone(),
        "final_state": state.detach().clone(),
        "grad_norms": grad_norms,
        "data_norms": data_norms,
    }


def capture() -> dict:
    return {b.__name__: _capture_one(b) for b in CONFIGS}


def _apply_rename(name: str) -> str:
    """Map an old log_* name onto the new isp_* name where applicable.

    Only matches at the *attribute* boundary. Since cell-level params live
    at the top-level (no module nesting inside the cell), a simple full-name
    lookup suffices.
    """
    return LOG_TO_ISP_RENAME_MAP.get(name, name)


def verify(golden: dict) -> None:
    current = capture()
    if set(current.keys()) != set(golden.keys()):
        raise AssertionError(
            f"config-set mismatch: golden={sorted(golden)} current={sorted(current)}"
        )
    for cfg_key in golden:
        g = golden[cfg_key]
        c = current[cfg_key]
        if g["label"] != c["label"]:
            raise AssertionError(f"{cfg_key}: label mismatch")

        # Forward output: bit-identical
        if not torch.equal(g["output"], c["output"]):
            diff = (g["output"] - c["output"]).abs().max().item()
            raise AssertionError(
                f"{cfg_key}: forward output mismatch (max abs diff {diff:.3e})"
            )
        # Final state: bit-identical
        if not torch.equal(g["final_state"], c["final_state"]):
            diff = (g["final_state"] - c["final_state"]).abs().max().item()
            raise AssertionError(
                f"{cfg_key}: final_state mismatch (max abs diff {diff:.3e})"
            )

        # Normalize both sides to the canonical isp_* namespace before
        # comparing — we don't know which side is post-rename. Pre-rename
        # everything is log_*; post-rename current is isp_* but golden may
        # still be log_*. Either way both sides land at isp_* after rename.
        renamed_grad = {_apply_rename(k): v for k, v in g["grad_norms"].items()}
        renamed_data = {_apply_rename(k): v for k, v in g["data_norms"].items()}
        cur_grad = {_apply_rename(k): v for k, v in c["grad_norms"].items()}
        cur_data = {_apply_rename(k): v for k, v in c["data_norms"].items()}

        if set(renamed_grad) != set(cur_grad):
            missing = set(renamed_grad) - set(cur_grad)
            extra = set(cur_grad) - set(renamed_grad)
            raise AssertionError(
                f"{cfg_key}: param-name mismatch\n"
                f"  in golden, missing from current: {sorted(missing)}\n"
                f"  in current, missing from golden: {sorted(extra)}"
            )

        for k, v_golden in renamed_grad.items():
            v_current = cur_grad[k]
            if (v_golden is None) != (v_current is None):
                raise AssertionError(
                    f"{cfg_key}.{k}: grad presence mismatch "
                    f"(golden None={v_golden is None}, current None={v_current is None})"
                )
            if v_golden is None:
                continue
            if v_golden != v_current:
                raise AssertionError(
                    f"{cfg_key}.{k}: grad-norm mismatch "
                    f"(golden={v_golden!r} current={v_current!r})"
                )

        for k, v_golden in renamed_data.items():
            v_current = cur_data[k]
            if v_golden != v_current:
                raise AssertionError(
                    f"{cfg_key}.{k}: data-norm mismatch "
                    f"(golden={v_golden!r} current={v_current!r})"
                )

        print(f"  ok: {cfg_key}  ({len(renamed_grad)} params)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", action="store_true",
                    help="Write a fresh golden file at " + str(GOLDEN_PATH))
    args = ap.parse_args()

    if args.capture:
        snap = capture()
        torch.save(snap, GOLDEN_PATH)
        print(f"captured -> {GOLDEN_PATH}")
        for k, v in snap.items():
            print(f"  {k}: {len(v['grad_norms'])} params, "
                  f"output {tuple(v['output'].shape)}")
        return

    if not GOLDEN_PATH.exists():
        sys.exit(f"missing golden file at {GOLDEN_PATH}; run with --capture first")
    golden = torch.load(GOLDEN_PATH, weights_only=False)
    verify(golden)
    print("ALL OK")


if __name__ == "__main__":
    main()
