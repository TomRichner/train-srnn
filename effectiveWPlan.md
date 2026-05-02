# Plan: hoist `_effective_W` out of the per-step cell call

## Why this exists

Tracked in `KnownIssues.md` §10. Both `SRNNCell._effective_W`
(`train_srnn/models/srnn_cell.py:296`) and `BatchedSRNNCell._effective_W`
(`train_srnn/models/srnn_cell.py:1108`) materialize the full effective
recurrent weight matrix on every call to `cell.forward(...)`. The trainer
calls `cell(...)` once per BPTT timestep, so at `chunk_len=250` the
matrix is rebuilt 250× per training step even though `W_raw` only changes
once, at `optimizer.step()`.

Cost at production K=6 N=300 bf16: `W_eff` is `(6, 300, 300) ≈ 1.08 MB`
per materialization → ~270 MB of avoidable scratch traffic per chunk
step, ~4 GB per epoch, plus the redundant softplus / sign / mask
element-wise ops running 250×.

This plan does §10 option (1): hoist the construction up one level so it
runs once per BPTT segment, not once per timestep.

## Scope

**Independent of the chunk-compile spike.** This change is its own
worktree (`hoist-effective-w`) so it can land on `main` cleanly without
the Option-3 unknowns. Once it's in, the chunk-compile branch will
rebase / merge from main and inherit a smaller per-step trace — which
incidentally helps Option 3's compile-time risk (one of its three big
risks; see `CompileChunkOutline.md` and `tmp/notes/cuda_graphs_options.md`).

**In scope:**

1. New cell-side API: optional pre-computed `W_eff` argument.
2. Refactor `SRNNCell.forward` and `BatchedSRNNCell.forward` to use it
   when provided, fall back to recomputing otherwise.
3. Hoist the call out of the four per-step loop sites:
   - `train_srnn/models/sequence_model.py:_run_segment` (windowed open-loop)
   - `train_srnn/models/sequence_model.py:_cl_run_segment` (windowed closed-loop)
   - `train_srnn/training/continuous.py:_forward_chunk_pure_tf` (continuous open-loop)
   - `train_srnn/training/continuous.py:_forward_chunk_closed_loop` (continuous closed-loop)
4. A targeted equivalence test: same `(W_raw, state, inputs)` produces
   byte-identical forward output and per-parameter gradient with vs.
   without the hoist (within fp32 numerical noise; bf16 may drift by a
   few ULPs, which is fine).
5. A real-config smoke run on `seeg srnn` confirming a non-regression in
   train-loss curve at K=2 size=30 vs. the cell-compile baseline of
   ~11.6 s/epoch.

**Out of scope:**

- §10 option (2) (cached W_eff with version counter). Don't touch.
- §10 option (3) (verify Inductor isn't already CSE'ing this). Done as
  Step 0 below — if true, we abandon and write the verdict.
- Any change to LSTM, LTC, CTRNN cells. None of them have `_effective_W`.
- §9 (per-step host syncs from `.item()` calls). Tracked separately.
- The chunk-compile spike itself.

## Step 0 — Verify Inductor isn't already CSE'ing this

**This is a kill condition.** If `torch.compile`'s default-mode Inductor
codegen already deduplicates the 250 redundant `_effective_W()` calls
within one compiled graph, the hoisting buys nothing on the trained
path (it would still help the eager-eval path, but eval is
no-grad+windowed so the cost there is small).

Procedure (15 minutes, on local CUDA box or short cloud dispatch):

```bash
# Compile-on, single chunk, dump the generated kernel code to a file.
TORCH_LOGS=output_code python - <<'EOF'
import torch
from train_srnn.models.factory import build_model
from omegaconf import OmegaConf
cfg = OmegaConf.load("conf/config.yaml")
cfg = OmegaConf.merge(cfg, OmegaConf.load("conf/task/seeg.yaml"),
                     OmegaConf.load("conf/model/srnn.yaml"))
cfg.size = 30
cfg.batched_ablations = ["srnn-e-only-per-neuron"]
model = build_model(cfg).cuda()
cell = torch.compile(model.cell)
B = 24; T = 250; N = 30
state = torch.zeros(1, B, cell.state_size, device="cuda")
x = torch.randn(B, 1, device="cuda")
for t in range(T):
    out, state = cell(x, state)
EOF
```

Inspect the dumped kernel code (logged to stderr) for a single
`softplus` of `W_raw` that gets reused, vs. 250 separate softplus
invocations. If reused → Inductor is CSE'ing → **abandon**, write
verdict, close branch.

Reasonable prior: Inductor *might* CSE within one graph but cannot do so
across the 250 separate compile-cell graphs (one per timestep), and the
current per-step compile boundary forces a fresh graph each timestep.
So expectation is no CSE today. Verify before doing the work.

## Step 1 — API: optional `W_eff` arg on SRNN cells

Modify two forward signatures:

```python
# train_srnn/models/srnn_cell.py

class SRNNCell(nn.Module):
    def forward(
        self,
        inputs: torch.Tensor,
        state: torch.Tensor,
        W_eff: Optional[torch.Tensor] = None,   # NEW
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ...
        if W_eff is None:
            W_eff = self._effective_W()
        # rest unchanged

class BatchedSRNNCell(nn.Module):
    def forward(
        self,
        inputs: torch.Tensor,
        state: torch.Tensor,
        W_eff: Optional[torch.Tensor] = None,   # NEW
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ...
        if W_eff is None:
            W_eff = self._effective_W()
        # rest unchanged
```

**Why optional, not mandatory.** Callers that don't know about the
hoist (the smoke test, ad-hoc analysis scripts, future code) keep
working with the current signature. The cell stays self-sufficient.
Eager-eval paths that do one cell call per timestep don't need to be
refactored — paying the construction cost once isn't a problem.

**Why not generalize across all cells.** LSTMCellWrapper, LTC, CTRNN
don't have an `_effective_W`. Adding a no-op kwarg to them is noise.
Callers that hoist will feature-detect via `hasattr(cell, '_effective_W')`.

**Compile cache impact.** Under `compile_cell=true` (the production
default), the cell is wrapped with `torch.compile`. Adding a
`Tensor`-typed positional/kwarg argument changes the input signature,
which changes the compile cache key. Verify after the change that we're
not getting double-compilation: log epoch 0 wall time and confirm only
one compile event (search for `[INFO] Compiling cell` followed by no
recompile messages — `compile_log_recompiles=true` flushes them).

## Step 2 — Hoist at all four call sites

### 2a. `_forward_chunk_pure_tf` (continuous, open-loop)

`train_srnn/training/continuous.py:205`

```python
def _forward_chunk_pure_tf(model, cell, chunk_x, state):
    T = chunk_x.shape[1]
    # Hoist _effective_W: SRNN-flavored cells reconstruct it from W_raw +
    # softplus + Dale signs + sparsity_mask. W_raw is fixed across the
    # chunk (only changes at optimizer.step), so compute once and pass in.
    W_eff = cell._effective_W() if hasattr(cell, "_effective_W") else None
    hidden_seq = None
    for t in range(T):
        mark_cudagraph_step()
        if W_eff is not None:
            h_t, state = cell(chunk_x[:, t, :], state, W_eff=W_eff)
        else:
            h_t, state = cell(chunk_x[:, t, :], state)
        state = state.clone()
        if hidden_seq is None:
            hidden_seq = empty_time_buffer(h_t, T)
        hidden_seq[..., t, :] = h_t
    ...
```

### 2b. `_forward_chunk_closed_loop` (continuous, closed-loop)

`train_srnn/training/continuous.py:247` — same treatment, hoist before
the per-step loop.

### 2c. `SequenceModel._run_segment` (windowed, open-loop)

`train_srnn/models/sequence_model.py:171` — same pattern. This is the
trickier site because `_run_segment` is called from inside
`torch.utils.checkpoint.checkpoint(...)` when `grad_checkpoint=true`.
Two options:

**Option α: hoist inside `_run_segment` (per-segment scope).** Pay
construction once per segment. With `grad_checkpoint_segment_len=5`
and `bptt_chunk_len=25` we have 5 segments per chunk → 5 reconstructions
per chunk vs the current 25. 5× win, not 250×, but no API contortion.

**Option β: hoist in `SequenceModel.forward` and pass W_eff through to
`_run_segment` as an explicit arg.** Pay construction once per chunk.
Composes cleanly with `checkpoint(...)` (W_eff becomes a saved input
tensor; the recompute uses the saved value rather than rebuilding).
Cost: tiny activation save (~360 KB per segment).

**Pick: option β.** It matches the cleaner per-chunk scope of the
continuous trainer and gives the full 250× win on the windowed path
too. The activation overhead is trivial.

```python
# sequence_model.py

def _run_segment(self, x_seg, state, W_eff=None):    # NEW arg
    T_seg = x_seg.shape[1]
    out_seq = None
    with self._no_autocast_cache_ctx():
        for t in range(T_seg):
            mark_cudagraph_step()
            if W_eff is not None:
                out, state = self.cell(x_seg[:, t, :], state, W_eff=W_eff)
            else:
                out, state = self.cell(x_seg[:, t, :], state)
            ...
```

In `SequenceModel.forward` — compute W_eff once and pass it down to all
`_run_segment` calls (warmup region too — same parameter values, same
W_eff):

```python
# Compute W_eff once for the whole forward pass; pass it to every
# segment call. Skip for cells that don't define _effective_W.
W_eff = self.cell._effective_W() if hasattr(self.cell, "_effective_W") else None
...
outs, state = self._run_segment(x_seg, state, W_eff=W_eff)
# and for checkpointed segments:
outs, state = torch.utils.checkpoint.checkpoint(
    self._run_segment, x_seg, state, W_eff,
    use_reentrant=False,
)
```

### 2d. `SequenceModel._cl_run_segment` (windowed, closed-loop)

`train_srnn/models/sequence_model.py:247` — mirror of 2c. Same `W_eff`
threaded through `_forward_closed_loop` to all `_cl_run_segment` calls.

## Step 3 — Tests

### 3a. Equivalence unit test (new)

Add `scripts/test_effective_w_hoist.py`. On CPU with fp32, build one
SRNNCell and one BatchedSRNNCell, fix all params, fix a state +
inputs, and assert:

```python
# Pre-hoist path
out_a, state_a = cell(inputs, state)              # cell computes W_eff internally

# Hoist path
W_eff = cell._effective_W()
out_b, state_b = cell(inputs, state, W_eff=W_eff)

torch.testing.assert_close(out_a, out_b, rtol=0, atol=0)
torch.testing.assert_close(state_a, state_b, rtol=0, atol=0)
```

**Both paths must be byte-identical** in fp32 because the math is
identical and the same W_eff tensor is consumed. (If we observe any
difference, something deeper is wrong — investigate before merging.)

Also test gradient equivalence: backward through both paths against
the same loss target, compare per-parameter gradients with `rtol=0
atol=0`.

### 3b. Existing smoke (already on path)

```bash
bash smoke_test.sh
```

Catches model×task regressions in 2-epoch runs. Should pass unchanged.

### 3c. End-to-end loss curve check (cloud dispatch)

A 4-epoch K=2 size=30 seeg srnn dispatch **on the same VM as the spike**
(`continuous-k2-smoke-srnn-seeg-seed1`):

```bash
git push origin hoist-effective-w
bash cloud/submit.sh continuous-k2-smoke-srnn-seeg-seed1 \
    hoist-effective-w-smoke seeg srnn 1 \
    "epochs=4 size=30 continuous_profile=true \
     checkpoint_interval=2 burn_in=0 freeze_ic_after_burnin=false \
     grad_checkpoint=false \
     compile=true compile_cell=true \
     compile_dynamic=false compile_log_recompiles=true \
     task.train_trace_max_len=89989 lr=1e-4 closed_loop.enabled=false \
     batched_ablations=[srnn-e-only-per-neuron,srnn-e-only-skip-per-neuron]"
```

**Success criteria:**
- Run completes (exit=0).
- Train loss at epoch 3 within ~5% of `a747471`'s baseline (this is a
  short run with a small seed; exact-match isn't expected, but a
  >5% divergence at epoch 3 suggests we changed semantics and not
  just performance).
- Steady-state epoch wall ≤ 11.6 s/epoch (no regression). The hope is
  modest improvement (5-10%, since the per-step cost includes more
  than just `_effective_W`); a real win would be a nice surprise.
- No new `recompile` log lines (compile cache is single-keyed).

**Failure modes:**
- Loss curve diverges → a hoisting bug (unlikely given byte-identical
  unit tests, but possible if the W_eff is captured on a different
  graph than the gradients flow through). Bisect against the unit
  test until reproduced locally.
- Wall-clock regresses → unexpected; investigate before merging.
  Likely cause: compile cache fragmentation from the new kwarg.

## Step 4 — Verdict and merge

Append a `## Verdict` section to this file with:

- Step-0 result (Inductor was/wasn't CSE'ing).
- Unit-test pass/fail.
- 4-epoch wall-clock and final loss vs. baseline.
- One of: `merge`, `iterate`, `abandon`.

If `merge`: open a PR from `hoist-effective-w` → `main`. The
chunk-compile branch will then merge from main to pick up the change.

If `abandon` (Inductor was CSE'ing or some other negative outcome):
write the reason, `git worktree remove` the branch, leave §10 in
`KnownIssues.md` as documentation of the dead end.

## Estimate

- Step 0: 15 min (Inductor CSE check).
- Step 1: 20 min (API change on two cells).
- Step 2: 30 min (four call-site hoists, including Sequence-Model
  threading through `forward → _run_segment` and the closed-loop
  variant).
- Step 3: 30 min (write unit test, run smoke, dispatch cloud check).
- Step 4: 15 min (verdict + PR or cleanup).

**Total: ~2 hours of engineering, 1 short L4 dispatch.**

The tight bound is the same as the chunk-compile spike's, and unlike
that spike this one is high-confidence: the math is provably
equivalent, the only risk is API plumbing or a missed call site.
