# CLAUDE.md

Guidance for AI coding agents working in this repository.

## What this is

PyTorch training and analysis code for SRNN, a continuous-time recurrent
network with Dale's law, spike-frequency adaptation, and short-term
synaptic depression, plus the Hasani et al. LTC/CTRNN/NODE/CT-GRU baselines.
Start with `README.md`; the equations are in `docs/model.md`, the code map
in `docs/architecture.md`.

## Environment

- `uv sync` creates `.venv` (Python 3.12). Run everything with `uv run`.
- Data, results, and analysis caches live outside the repo under
  `$SRNN_HOME` (see `train_srnn/paths.py`). Set it before running
  `train.py` or the scripts.
- Never install packages on your own and never read `.env` files.

## Conventions

- Config is code: every task and model is a dataclass in
  `train_srnn/config.py`, registered with Hydra. Unknown keys fail at
  startup. `conf/config.yaml` only lists defaults.
- SRNN variants are names parsed by `train_srnn/models/variants.py`;
  add a token there rather than a preset table. K variants always run in
  one `SRNNCell` with a leading K axis (K = 1 for a single variant).
- Tensors are batch-first `(B, T, F)`; cell state is `(K, B, S)`.
- New cells subclass `train_srnn/models/base.py:RNNCell` and get a config
  dataclass and a `MODEL_CONFIGS` entry; new tasks subclass `Task` and
  register in `TASKS`.
- Files ported from Hasani et al. carry a provenance header; keep it.

## Tests

`uv run pytest -q` runs in about a minute on CPU and must pass before a
commit. `tests/golden/*.pt` pin the numerics of every cell and both
trainers; a numerical change must either keep them bitwise or regenerate
them on purpose with `uv run python tests/golden/make_golden.py` and say so
in the commit message. The end-to-end signal is a GPU run on a cloud VM
(`docs/cloud.md`), not a local CPU run.

## Commits

One logical change per commit with a descriptive message. Do not push.
