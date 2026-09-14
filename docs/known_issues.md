# Known issues

Open limitations in the current code. Each entry: what happens, why, impact,
suggested fix. Paths are relative to the repo root.

## 1. `echo` freezes only the recurrent matrix

`echo=True` variants (`srnn-echo`, `srnn-e-only-echo`, ...) were meant to be
reservoirs: recurrent matrix and all intrinsic dynamics fixed, only `W_in` and
the readout trained. `SRNNCell._effective_W` (`train_srnn/models/srnn_cell.py`)
detaches `W_raw` for echo variants and nothing else. `W_raw_gain`, `W_in_gain`,
`isp_tau_global`, every `log_*_gain`, `a_0_scalar`, `c_0_*_scalar` and, for
`per_neuron=True`, every `*_vec` still train.

Impact: "echo" results are frozen-structure, trainable-gain, trainable-timescale
hybrids, not classical reservoirs. Do not cite them against ESN literature
without qualification.

Fix: add an `echo` branch to `_linked` (or a second mask) that detaches the
scalar and gain parameters as well as the vectors, so the per-variant detach
covers everything in `FREEZE_GROUPS` except `W_in` and the readout. `cell.freeze`
cannot do this: it pins whole `(K, ...)` tensors, i.e. all variants at once.
`tests/test_linked_params.py` already checks the echo `W_raw` slice; extend it.

## 2. `smnist` is row-wise, not pixel-wise sequential MNIST

`SmnistConfig` (`train_srnn/config.py`) sets `input_size=28, seq_len=28`, and
`SmnistTask` (`train_srnn/data/classic.py`) feeds each image as 28 rows. This
is the setup of the LTC paper (Hasani et al. 2021) and is kept so those numbers
stay comparable. Canonical sequential MNIST (Le et al. 2015; uRNN, LMU, S4)
is 784 steps of one pixel and is a long-memory benchmark; the 28-step version
is not (LSTMs exceed 99%).

Impact: do not report `smnist` results as "sequential MNIST" without saying
row-wise.

Fix: add a `SerialSmnistConfig` (`input_size=1, seq_len=784`) to
`TASK_CONFIGS` and a loader flag that flattens `(28, 28)` to `(784, 1)`. Expect
784-step BPTT to be much slower per epoch.

## 3. The trainable initial condition gets no gradient under truncated BPTT

`SequenceModel.forward` runs the prefix before `bptt_start_idx` under
`torch.no_grad()` and then detaches the state, so `TrainableIC.ic` is cut out
of the graph whenever `window_len > bptt_len` (every windowed task). In the ring
trainer `ContinuousTrainer._start_state` freezes the IC outright and the state
is never reset. `freeze_ic_after_burnin=True` (default in `TrainConfig`) makes
this explicit: after `Trainer.refresh_ic` copies the burn-in fixed point into
the IC, `requires_grad_(False)` is set so it no longer counts as a trainable
parameter. `burn_in_every` (default 1) re-runs burn-in each epoch for the
windowed trainer so the frozen IC tracks the current network's fixed point
(`rebuild_ic_each_epoch=False` for the ring trainer).

Impact: `trainable_ic=True` is equivalent to using the burn-in state as a
buffer. Any "trained IC" seen in checkpoints is burn-in output.

Fix: only relevant if the IC should learn. Set `bptt_len == window_len` so
there is no no-grad prefix, and note that `bptt_chunk_len` detaches still
limit the IC's gradient to the first chunk.

## 4. `W_in` initialisation is not fan-in scaled

`SRNNCell._init_recurrent` draws `W_in = randn(K, N, input_size) * 0.1`
regardless of `input_size`. For unit-variance inputs (the trace tasks are
z-scored) the per-neuron drive std at init is `0.1 * sqrt(input_size)`:
har 2.4, ozone 0.85, gesture 0.57, smnist 0.53, cheetah100_act 0.48,
cheetah100 0.41, person 0.26, occupancy 0.22. `W_in_mask` does not change this
(applied at forward time). The readout uses the `nn.Linear` Kaiming default
and is fine.

Impact: tasks start from different operating points for reasons unrelated to
the model; `W_in_gain` absorbs part of the mismatch during training.

Fix: `randn(...) / sqrt(input_size)` (or `/ sqrt(n_input_neurons)` if the
masked fan-in is the better target). Changes the init of every fresh run, so
do it at the start of a new comparison family, not mid-series.

## 5. `torch.compile` CUDA-graph modes fail in training

`compile.mode=reduce-overhead` or `max-autotune` (`CompileConfig`,
`train_srnn/config.py`) wrap each compiled cell call in a CUDA graph via
`cudagraph_trees`. The training pattern is T cell calls followed by one
`backward()`. `SequenceModel._step` calls `mark_cudagraph_step()`
(`train_srnn/utils/cell_loop.py`) as PyTorch requires, which fixes the
forward aliasing errors but invalidates intermediates the deferred backward
needs: `accessing tensor output of CUDAGraphs that has been overwritten`.
Upstream: pytorch/pytorch #148439, #158551, #169545, open as of PyTorch 2.11.

Impact: only the default Inductor mode (`compile.mode=null`) works; it is what
production runs use.

Fix (if launch overhead is ever measured above ~20% of step time): capture the
whole chunk forward+backward manually with `torch.cuda.graph` inside a custom
`autograd.Function` (NVIDIA RNN-T pattern), or compile the whole chunk loop so
Dynamo emits one forward and one backward graph.

## 6. Per-step host syncs in the loss and metric path

`Trainer.loss_and_metrics` (`train_srnn/training/trainer.py`) returns
`[l.item() for l in losses]` and `Task.metric` (`train_srnn/data/task.py`)
returns a Python `float`, so every optimizer step forces `2K` device syncs
that are only needed for per-epoch logging. Both trainers call it per step;
closed loop adds one `u.item()` per alpha sample.

Impact: bounded. At K=6, N=300, B=48 the GPU is at ~90% utilisation, so the
win is a few percent of wall-clock; larger at small scale where the queue
drains between syncs.

Fix: return `torch.stack(losses)` and a `(K,)` metric tensor, accumulate on
device across the epoch, and call `.tolist()` once per epoch.

## 7. Mixed `softplus` / `exp` parameterisation of positive scalars

In `SRNNCell` every positive quantity is
`tau_global * exp(log_*_gain) * softplus(isp_*_vec)` (`_tau_d`, `_tau_a`,
`_tau_b`, `_c`; `tau_global` is itself `softplus(isp_tau_global)`). Per-neuron
bases use inverse-softplus (`isp_`), per-variant gains use true log (`log_`).
Softplus is near-linear around the small init values (`tau_d` 0.1 s, `c` 0.05),
which makes Adam steps roughly additive in seconds; the gain is centred on 1
and explores log-space symmetrically. Both are conventions, not requirements.

Impact: raw-parameter plots mix two scales and every conversion to effective
values (`scripts/postprocess.py`, `docs/equations.md`) must spell out which
transform applies. Not a correctness problem.

Fix (not planned): all-`exp` so the three factors add in log-space. Changes the
optimisation geometry and invalidates checkpoints, so only with a fresh
comparison family.

## 8. `--skip-refresh` is all-or-nothing

`cloud/submit.sh --skip-refresh` writes one `skip-refresh` metadata key that
`cloud/startup_gpu.sh` uses to gate three unrelated steps: git fetch and reset
to `--branch`, `gcloud storage cp -r` of the dataset into `$SRNN_HOME/data/`,
and `pip3 install` of the Hydra stack. There is no way to refresh the code but
keep the dataset, or the reverse.

Impact: every code-only re-dispatch re-copies the dataset (tens of seconds and
bucket egress); a data-only refresh cannot skip git.

Fix: replace the dataset copy with `gcloud storage rsync`, which makes that
step free when nothing changed; optionally split into `--skip-git` and
`--skip-data` with `--skip-refresh` as the alias for both.

## 9. `a_0` and `c_0_E` / `c_0_I` share an exact flat direction

In `SRNNCell._drive` the rate is `r = f(x - sum_j c_j a_j - a_0)` and each SFA
channel relaxes to `c_0_j + r`. Substituting `a_j -> a_j - c_0_j` removes `c_0`
from the dynamics and leaves the threshold `theta = a_0 + sum_j c_j c_0_j`, so
the loss depends on the two parameters only through `theta_E` and `theta_I`.
The direction is exactly flat (verified to 3e-16 on a float64 rollout), so
Adam does not drift along it. With `per_neuron=False` the three scalars
`a_0_scalar`, `c_0_E_scalar`, `c_0_I_scalar` carry two degrees of freedom;
with `per_neuron=True` the whole `c_0_*_vec` (about 3N entries) is redundant.

Impact: interpretive only. `a_0`, `c_0_E`, `c_0_I` are not individually
meaningful; `scripts/postprocess.py:plot_offsets_evolution` plots the raw
three and should plot `theta_E`, `theta_I` instead or as well.

Fix: `freeze_params=[c_0_E]` gives an identifiable parameterisation with no
lost expressiveness (`FREEZE_GROUPS` covers it; `c_0_E_vec` starts at zero).
Do not pin `a_0`: no-adapt variants have no `c_0` and would lose their only
threshold. Changing this breaks comparability with existing runs.

## 10. CTGRU decay factor uses `ln tau` in place of `tau`

`CTGRUCell.__init__` (`train_srnn/models/ctrnn_cell.py`) registers
`exp_decay = exp(-1 / ln_tau_table)`. The upstream TF code
(`experiments_with_ltcs/ctrnn_model.py`, `np.exp(-1.0/self.ln_tau_table)`)
does exactly this, and the port keeps it for fidelity. With `tau_base = 1` the
first timescale has `ln tau = 0`, so its decay is `exp(-inf) = 0` and that
channel is zeroed every step; the others decay by `exp(-1 / (i ln sqrt(10)))`,
which is not a physical time constant.

Impact: CTGRU effectively has `M - 1` usable timescales with an unintended
decay schedule. Results match upstream, which is the point of the baseline.

Fix: if a corrected CTGRU is wanted, add a config flag selecting
`exp(-dt / tau)` and keep the default at upstream behaviour.

## 11. `RMTMatrix.export_for_srnn(dales=True)` flips a few sampled signs

`RMTMatrix.build` samples `W = mu + sigma * randn` per column type, so a small
fraction of entries have the opposite sign from their column: measured 24, 22
and 17 of ~30000 nonzeros at N=300 (seeds 1 to 3; 0.06 to 0.08%), 3 of 7427 at
N=150. The export stores `inv_softplus(|W|)` and a per-column `dales_sign`, so
`SRNNCell._effective_W` reconstructs `sign * |W|`, i.e. those entries change
sign relative to the sampled matrix (max entry change 0.15).
`tests/test_rmt_matrix.py::test_export_round_trip_with_dales` asserts the
Dale-compliant reconstruction, not equality with `W`. `dales=False` exports the
signed matrix unchanged.

Impact: the Dale and no-Dale variants of one seed start from matrices that
differ in ~0.07% of entries, and the spectral radius reported by `RMTMatrix`
is that of the sampled `W`, not the initial effective matrix.

Fix: sample with the sign constraint (clip the Gaussian at zero, or resample
disagreeing entries), or document that the Dale init is `sign * |W|`.

## 12. Legacy gradient-hook linking masks (audited)

Before the unified `SRNNCell`, per-neuron linking and echo freezing were
`Tensor.register_hook` callbacks on the `*_vec` and `W_raw` parameters. Such
hooks are dropped silently by `copy.deepcopy` and are not guaranteed to fire on
the recompute pass of `torch.utils.checkpoint(use_reentrant=False)`, so runs
with `grad_checkpoint=True` could in principle have trained per-neuron vectors
for `per_neuron=False` variants.

Audit of the archived `ring2x5-100e` run (K=10, N=300, all variants
`per_neuron=False`, `grad_checkpoint=True`, `compile_cell=True`, fp32,
100 epochs), comparing `model_state_dict` of `init.pt` and `last.pt`:
`cell.a_0_vec`, `cell.isp_tau_d_vec`, `cell.isp_tau_b_rec_E_vec` (also
`isp_tau_a_E_vec`, `isp_c_E_vec`, `c_0_E_vec`, `isp_tau_b_rel_E_vec`) have
max |last - init| = 0.0 for every variant, and across-neuron std is 0 to 1e-7
(float32 rounding of a constant vector) at both init and last. The paired
scalars moved (`a_0_scalar` to -0.096..+0.016, `log_tau_d_gain` to
-0.38..-0.61). The hooks fired; that run trained shared parameters as intended.

The current code applies the masks in the forward pass (`SRNNCell._linked`,
`_effective_W`), so the failure mode cannot recur; `tests/test_linked_params.py`
checks the gradients with and without checkpointing. Checkpoints from the old
cell carry `cell._*_vec_mask` buffers and lack `cell.per_neuron_mask`, so a
strict `load_state_dict` (the `init_ckpt` path) rejects them; reading their
state dicts directly, as `scripts/postprocess.py` does, still works.

## 13. With `burn_in=0` the initial condition starts synapses fully depressed

`TrainableIC` initialises the state to zeros, and `cell.init_state` (which
sets `b = 1`) is only used inside `compute_burn_in`. With `burn_in=0` every
window therefore starts at `b = 0` and `x = 0`; under `std_zero_floor` the
synaptic gain is then negative, `-b_min / (1 - b_min)`, until `b` recovers
over a few `tau_b_rec`. Production runs use the default `burn_in=10`, which
overwrites the IC with the settled unforced state, so they are unaffected.
Fix: seed `TrainableIC` from `cell.init_state` (a = 0, b = 1, x = 0) rather
than zeros.

## 14. Readout timing differs between output modes

The `synaptic` and `rate` outputs use `r` and `b_full` evaluated at the
state entering the last sub-step, while `dendritic` returns the post-update
`x`. The one-sub-step offset is documented in `docs/model.md`; it matters
only when comparing readout modes at the same step.
