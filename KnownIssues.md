# Known Issues

Tracked limitations and bugs. When working in these areas, either avoid the pitfall or plan around it explicitly.

---

## 1. `per_neuron=False` is not enforced in `BatchedSRNNCell`

**Summary.** Every variant trained through `BatchedSRNNCell` (i.e. anything using `batched_ablations=...` via `build_batched_model`) effectively runs as **per_neuron=True**, regardless of each variant's `SRNNConfig.per_neuron` flag. The `per_neuron=False` presets (`srnn`, `srnn-echo`, `srnn-no-adapt`, etc.) all silently learn per-neuron time constants in batched mode.

**Why this happens.** In the single-cell `SRNNCell` (`train_srnn/models/srnn_cell.py:194, 199, 212, 224, 236, 248, 257`), the time-constant parameters have shape `(1,)` when `per_neuron=False` and `(n_E,)` / `(n_I,)` / `(N,)` when `True`. Shape `(1,)` structurally forces all neurons to share the parameter, so "non-per-neuron" is enforced by construction.

In `BatchedSRNNCell` (same file, constructor from line 729), **all** time-constant parameters are stored at per-neuron shape unconditionally:

- `log_tau_d`: `(K, N)` (line 805)
- `log_tau_a_E`, `log_c_E`, `c_0_E`: `(K, n_E, max_n_a_E)` (lines 809–825)
- `log_tau_a_I`, `log_c_I`, `c_0_I`: `(K, n_I, max_n_a_I)`
- `log_tau_b_rec_E`, `log_tau_b_rel_E`: `(K, n_E)`
- `log_tau_b_rec_I`, `log_tau_b_rel_I`: `(K, n_I)`

There is no mechanism (no post-step tying, no mean broadcast, no frozen-shared parameter) that keeps a non-per-neuron variant's neurons tied together during training. At init the `torch.full(...)` calls set all N entries identically, but the optimizer pushes them apart from the first gradient step.

**Grepping `per_neuron` in `srnn_cell.py`** confirms the flag is referenced only inside `SRNNCell` and the preset dictionary. It never appears inside `BatchedSRNNCell`.

**What this means for mixing.** Nothing special — the behavior is the same whether:
- All variants in the batch are `per_neuron=True` (works as intended),
- All variants are `per_neuron=False` (each trains as per-neuron — wrong),
- The batch mixes the two (per-neuron ones work, non-per-neuron ones train as per-neuron).

Mixing per-neuron and non-per-neuron presets in a batch is not a new bug; the non-per-neuron ones were already running as per-neuron.

**Practical consequences.**
- Running `srnn-per-neuron` side-by-side with `srnn` in a batched ablation is redundant — both will end up training as per-neuron.
- A rigorous `per_neuron=True` vs `per_neuron=False` comparison **must** use two separate single-cell training runs (`build_model`, not `build_batched_model`).
- Results from prior batched ablation runs that included non-per-neuron variants should be interpreted with this in mind: the "srnn" etc. baselines in those runs are effectively `srnn-per-neuron`.

**If/when a fix is attempted,** options include:
- Register a post-step hook per variant that averages the per-neuron values and broadcasts the mean back for `per_neuron=False` variants.
- Add a buffer that masks gradients for non-per-neuron variants so only a shared component updates.
- Accept the current behavior as "batched ablations are per-neuron by nature" and update docs/presets to reflect it (drop non-per-neuron presets from the batched catalog).
