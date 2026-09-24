# Known issues

Open limitations in the current code. Each entry: what happens, why, impact,
suggested fix. Paths are relative to the repo root.

## 1. `echo` freezes only the recurrent matrix

`echo=True` variants (`srnn-echo`, `srnn-e-only-echo`, ...) were meant to be
reservoirs: recurrent matrix and all intrinsic dynamics fixed, only `W_in` and
the readout trained. `SRNNCell._effective_W` (`train_srnn/models/srnn_cell.py`)
detaches `W_raw` for echo variants and nothing else. `log_W_raw_gain`, `W_in_gain`,
every intrinsic `log_*_gain`, `a_0_scalar` and, for `per_neuron=True`,
every `*_vec` still train. Version 2 removed `tau_global` and the SFA-state
offsets `c_0`; their removal does not change this remaining echo limitation.

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
`exp(log_*_gain) * softplus(isp_*_vec)` (`_tau_d`, `_tau_a`,
`_tau_b`, `_c`). Version 2 removed the redundant global time multiplier. Per-neuron
bases use inverse-softplus (`isp_`), per-variant gains use true log (`log_`).
Softplus approaches a linear transform for large raw parameters and an
exponential transform for negative raw parameters; initialization includes
`tau_d=0.1` s and a total SFA budget `c=0.5`. The gain is centred on 1
and explores log-space symmetrically. Both are conventions, not requirements.

Impact: raw-parameter plots mix two scales and every conversion to effective
values (`scripts/postprocess.py`, `docs/equations.md`) must spell out which
transform applies. Not a correctness problem.

Fix (not planned): all-`exp` so the two factors add in log-space. Changes the
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

## 9. Resolved in version 2: redundant SFA offsets

The old `a_0` / `c_0_E` / `c_0_I` flat direction is removed. SFA states now
relax toward the raw firing rate, without an additive state offset. The rate
uses the threshold `a_0` and the normalized total adaptation budget
`c / K * sum(a_k)`. Parameter reports show `a_0` and total `c`; historical
checkpoints require their recorded original code revision.

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

## 11. Dale projection changes rare sampled recurrent-weight signs

`RMTMatrix.build` samples a Gaussian per presynaptic population, so rare
entries can disagree with that population's sign. The default
`export_for_srnn(..., dales_init=True)` projects both enforced and unenforced
training variants to `sign * abs(W)`. They now share the same initial effective
matrix; the previous between-condition initialization discrepancy is resolved.
The `dales` argument independently controls sign enforcement during training.

Remaining limitation: the spectral radius reported directly by `RMTMatrix`
is for the sampled matrix, before projection. Use `cell._effective_W()` for
spectral properties of the initialized trained model. MATLAB parity imports
the exact unprojected matrix separately, rather than claiming projection is
numerically identical to the MATLAB draw.

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
strict `load_state_dict` (the `init_ckpt` path) rejects them. Version 2 also
checks an explicit model-version buffer; reconstruct or postprocess historical
runs using their recorded original code revision.

## 13. With `burn_in=0` the initial condition starts synapses fully depressed

`TrainableIC` initializes the state to zeros, and `cell.init_state` (which
sets `b = 1`) is used inside `compute_burn_in`. With `burn_in=0`, training
therefore starts at `b = 0` and `x = 0`. Version 2 uses the product of STD
states directly, so synaptic output starts at zero and recovers; the former
negative-gain issue from `std_zero_floor` is resolved because that transform
was removed. Production runs retain `burn_in=10`, which overwrites the IC
with the settled unforced state.

Possible fix: seed `TrainableIC` from `cell.init_state` (a = 0, b = 1,
seeded dendritic draws) rather than zeros. This would change no-burn-in runs.

## 14. Resolved in version 2: readout timing

All readout modes now use the completed integration step. Synaptic and rate
outputs are recomputed from the updated state, matching the timing of the
dendritic output. The former last-substep offset no longer applies.

## 15. Resolved: resuming an interrupted run

`init_ckpt` is a warm start: it restores model, optimizer and scheduler state,
then trains a new run from epoch 0 with a fresh burn-in, so a long run continued
this way overruns its learning-rate schedule. Before `resume=true` existed, that
was the only way to continue, and it lost the epoch counter, RNG streams and the
ring trainer's carried state.

`resume=true` now continues `output_dir` exactly (see `docs/architecture.md`,
"Run directory"), bitwise on CPU. Checkpoints written before this change have
no `trainer_state` and cannot be resumed; they remain valid for analysis and for
`init_ckpt`. On GPU a resumed run matches an uninterrupted one only up to normal
run-to-run variation, and the continuation restarts from the latest checkpoint,
so up to `checkpoint_interval` epochs are recomputed.
