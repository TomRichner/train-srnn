# Spike: compile the BPTT chunk, not just the cell

## Why this exists

This is **Option 3** from `tmp/notes/cuda_graphs_options.md` (read it first
for the full landscape; it's at the repo root in this worktree). The
preceding work already proved:

- `compile=true compile_cell=true compile_mode=null` works in production
  (~11.6 s/epoch at K=2 size=30 on L4) and is the new default.
- `compile_mode=reduce-overhead` and `max-autotune` **do not work** with
  the autograd-over-Python-loop BPTT pattern because PyTorch's
  `cudagraph_trees` cannot reconcile "recycle outputs between
  iterations" with "keep saved-for-backward alive." See `KnownIssues.md`
  §8 for the diagnosis.

This spike asks one specific question: **if we move `torch.compile` from
`model.cell` up to the function `_forward_chunk_pure_tf` (and its
closed-loop sibling), will Dynamo trace the whole 250-step loop into a
single graph, and does that single graph let `cudagraph_trees` handle
the chunk as one forward + one backward?**

If yes, we get CUDA graphs in training for free.

The principal risk is **compile time**. A 250-step unrolled trace is
roughly 12,500 ops; we've previously seen the toy 50-step version stick
for 13 min before completing. At production size=300 K=6 it could be
worse. The first goal of this spike is measuring how bad that is, not
shipping anything.

## Scope

**This is a 1–2 hour experiment, not a feature.** Do the minimum
refactor needed to test the hypothesis on a tiny config (K=2, size=30,
half-T = 89989). Do not try to ship to production. Do not rewrite the
training loop. Do not generalize to other compile modes.

The deliverables are:

1. A working code path on this branch where the chunk function (not the
   cell) is compiled in the continuous trainer.
2. Two timing numbers from cloud GPU dispatches at K=2 size=30:
   - `compile_mode=null` (default mode) — does it trace at all? In how
     long? Steady-state vs. the cell-compile baseline of 11.6 s/epoch.
   - `compile_mode=reduce-overhead` — does it run cleanly? If yes,
     steady-state vs. the same baseline.
3. A short verdict in this file (or a follow-up note) recommending one
   of: ship to production, scale-test on K=6 size=300, or abandon.

If the trace-time alone is intolerable at K=2 size=30, abandon early.
The full Option 3 implementation (production-scale, gradient
equivalence tests, closed-loop, eval-path coexistence) is **out of
scope** for this spike. Defer to a follow-up if the spike is
encouraging.

## What to change (concretely)

All edits live in **`train_srnn/training/continuous.py`**. Don't touch
`train_srnn/models/sequence_model.py` (eval still uses the
uncompiled cell — keeping that path eager is part of why this works).

### Step 1. Pre-allocate `hidden_seq` before the loop

The `if out_seq is None: out_seq = empty_time_buffer(out, T)` lazy-alloc
pattern in `_forward_chunk_pure_tf` and `_forward_chunk_closed_loop`
introduces a Python conditional inside the loop body. Dynamo would
either graph-break on the first iteration's None branch or specialize
on it (recompile every chunk). Replace with up-front allocation:

```python
def _forward_chunk_pure_tf(model, cell, chunk_x, state):
    T = chunk_x.shape[1]
    # Pre-allocate using cell metadata. State.shape[:-1] = leading dims
    # (B,) or (K, B); cell.num_units = output feature dim. Dtype tracks
    # state to follow autocast. (When AMP is bf16 and state stays fp32,
    # this allocates fp32; the compiled cell will downcast as needed.)
    hidden_seq = torch.empty(
        state.shape[:-1] + (T, cell.num_units),
        device=state.device, dtype=state.dtype,
    )
    for t in range(T):
        h_t, state = cell(chunk_x[:, t, :], state)
        state = state.clone()
        hidden_seq[..., t, :] = h_t
    if hidden_seq.dim() == 4:
        x_in_seq = chunk_x.unsqueeze(0).expand(hidden_seq.shape[0], -1, -1, -1)
    else:
        x_in_seq = chunk_x
    return _readout_chunk(model, hidden_seq, x_in_seq), state
```

Mirror in `_forward_chunk_closed_loop`. For the closed-loop path you
also need `x_in_seq` pre-allocated — it has the same leading dims as
`y_prev` (broadcast result of the alpha blend). Do a single `cell(...)`
call before the loop *only* if you need to determine the leading-dim
broadcast; otherwise compute the leading shape from `y_prev` directly:

```python
hidden_seq = torch.empty(
    y_prev.shape[:-1] + (T, cell.num_units),
    device=state.device, dtype=state.dtype,
)
x_in_seq = torch.empty(
    y_prev.shape[:-1] + (T, chunk_x.shape[-1]),
    device=state.device, dtype=state.dtype,
)
```

Note `cell.num_units` works for both `SRNNCell` and `BatchedSRNNCell`
(both expose it).

### Step 2. Remove the inner-loop `mark_cudagraph_step()` calls

Inside the loop body, `mark_cudagraph_step()` was telling the allocator
"previous output is dead" between cell calls. When the *whole chunk* is
one compiled graph, that boundary doesn't exist anymore — there are no
intermediate compile-graph outputs to mark dead. Drop those calls from
both chunk functions.

(Don't remove `state.clone()`. It's still needed: under reduce-overhead
the chunk's compiled output buffers still alias when the chunk is
called repeatedly across BPTT steps, exactly the original problem at
the *chunk* level instead of the *cell* level. The clone breaks
chunk-to-chunk aliasing.)

If you want, add a `mark_cudagraph_step()` call **between chunks** in
the trainer's hot loop (just before each `_forward_chunk_pure_tf(...)`
call in `run_continuous_training`) to signal the chunk-level boundary
to `cudagraph_trees`. Probably required for reduce-overhead to work
correctly across chunks.

### Step 3. Compile the chunk functions in the trainer

Currently around `continuous.py:387` the trainer compiles `model.cell`:

```python
cell = model.cell
if (bool(cfg.get("compile", False))
        and bool(cfg.get("compile_cell", False))
        and device.type == "cuda"):
    cell = torch.compile(cell, **compile_kwargs)
```

Add a parallel knob for compiling the chunk functions instead. Suggest
a new flag `compile_chunk: false` (default) in the same code path:

```python
cell = model.cell
chunk_pure_tf_fn = _forward_chunk_pure_tf
chunk_closed_loop_fn = _forward_chunk_closed_loop

if bool(cfg.get("compile", False)) and device.type == "cuda":
    if bool(cfg.get("compile_chunk", False)):
        # Compile the whole chunk loop. Do NOT also compile the cell —
        # the chunk compile will inline it.
        log.info("Compiling forward chunk functions; kwargs=%s",
                 compile_kwargs or "(defaults)")
        chunk_pure_tf_fn = torch.compile(_forward_chunk_pure_tf, **compile_kwargs)
        chunk_closed_loop_fn = torch.compile(_forward_chunk_closed_loop, **compile_kwargs)
    elif bool(cfg.get("compile_cell", False)):
        log.info("Compiling cell (continuous trainer scope only); kwargs=%s",
                 compile_kwargs or "(defaults)")
        cell = torch.compile(cell, **compile_kwargs)
```

Then in the training step (`continuous.py:480` area), call
`chunk_pure_tf_fn(model, cell, chunk_x, state)` and
`chunk_closed_loop_fn(...)` instead of the bare module-level functions.

Add the flag to `conf/config.yaml`:

```yaml
compile_chunk: false  # EXPERIMENTAL (CompileChunkOutline.md). When true,
  # compile _forward_chunk_pure_tf instead of model.cell so Dynamo traces
  # the whole BPTT loop into one graph. Hypothesis: enables reduce-overhead.
  # Risk: very long compile time at production scale. Mutually exclusive
  # with compile_cell — when both true, compile_chunk wins.
```

### Step 4. Tracing-friendly cleanup

A few small things that will likely cause graph breaks if you skip them:

- Inside `_forward_chunk_pure_tf`, the `if hidden_seq.dim() == 4:` branch
  resolves at trace time on a static shape, so it should specialize
  cleanly. Leave it.
- `_readout_chunk` calls `model._readout_one` in its own loop (T
  iterations); when the *outer* function is compiled, that inner loop
  also gets traced. Same pre-alloc treatment may help: replace
  `outs.append(...) + torch.stack(...)` with `out = torch.empty(...);
  out[..., t, :] = model._readout_one(...)` to avoid append/stack inside
  a compiled trace. Mirror the existing `cell_loop.empty_time_buffer`
  helper. (This is a 5-line change in `_readout_chunk`.)
- `state.clone()` inside the loop is fine inside a compiled graph (it
  just becomes a copy op). Keep it.
- Ensure the trace is **not** wrapped in `torch.utils.checkpoint`. The
  trainer's hot loop doesn't currently checkpoint chunks (that's
  `sequence_model._run_segment`'s job in the windowed trainer). You
  shouldn't run into checkpoint composition here — just verify.

## How to test

The keep-alive VM is `continuous-k2-smoke-srnn-seeg-seed1`
(us-central1-a, project `liquidneuralnets`, cleanup=keep). Push your
branch first; `cloud/submit.sh` defaults to `skip-refresh=0` so it
fetches the latest commit on each dispatch.

### T1. Smoke: does it even trace? (compile_mode=null, default mode)

```bash
git push origin compile-chunk
bash cloud/submit.sh continuous-k2-smoke-srnn-seeg-seed1 \
    spike-chunk-default seeg srnn 1 \
    "epochs=4 size=30 continuous_profile=true \
     checkpoint_interval=2 burn_in=0 freeze_ic_after_burnin=false \
     grad_checkpoint=false \
     compile=true compile_cell=false compile_chunk=true \
     compile_dynamic=false compile_log_recompiles=true \
     task.train_trace_max_len=89989 lr=1e-4 closed_loop.enabled=false \
     batched_ablations=[srnn-e-only-per-neuron,srnn-e-only-skip-per-neuron]"
```

**Watch the log via**:
```bash
gcloud compute ssh continuous-k2-smoke-srnn-seeg-seed1 \
    --zone=us-central1-a --project=liquidneuralnets -- \
    sudo tail -f /var/log/training-spike-chunk-default-1.log
```

**Success criteria**:
- Compile completes in **< 10 minutes** wall-clock (epoch 0 includes
  compile time; the prior cell-compile baseline was ~37 s for epoch 0).
- Run completes (exit=0).
- Steady-state epoch wall (epoch 2 train-only) is recorded.

If compile takes > 30 min, kill the run and **abandon Option 3** —
production scale would be untenable. Note this in the verdict and
move on.

### T2. The actual prize: reduce-overhead

Only run this if T1 succeeded. Same config + `compile_mode=reduce-overhead`:

```bash
bash cloud/submit.sh continuous-k2-smoke-srnn-seeg-seed1 \
    spike-chunk-reduce-overhead seeg srnn 1 \
    "epochs=4 size=30 continuous_profile=true \
     checkpoint_interval=2 burn_in=0 freeze_ic_after_burnin=false \
     grad_checkpoint=false \
     compile=true compile_cell=false compile_chunk=true \
     compile_mode=reduce-overhead compile_dynamic=false compile_log_recompiles=true \
     task.train_trace_max_len=89989 lr=1e-4 closed_loop.enabled=false \
     batched_ablations=[srnn-e-only-per-neuron,srnn-e-only-skip-per-neuron]" \
    --skip-refresh
```

**Success criteria**:
- No `accessing tensor output of CUDAGraphs that has been overwritten`
  in the log.
- No `static input data pointer changed` aliasing error.
- Run completes (exit=0).
- Steady-state epoch wall **lower than 11.6 s/epoch** (the
  cell-compile default-mode baseline). If it's higher, the chunk
  compile didn't help — write the verdict and stop.

If T2 errors at the same `cudagraph_trees` boundary as the previous
attempts (commit `065dfb6` V5), Option 3 is dead — the underlying
PyTorch limitation extends past the per-cell granularity. Document
that and abandon. If it succeeds, the cell-compile path can be
deprecated and a follow-up will plumb production-scale validation.

## Reference points

The cell-compile baseline numbers from `main` (commit `a747471`),
K=2 size=30, B=24, half-T (89989), L4 fp32:

| Config | Epoch 0 | Steady-state |
|---|---|---|
| compile=true compile_cell=true ckpt=False | 36.9 s | 11.6 s |
| compile=true compile_cell=true ckpt=True seg_len=5 | 36.9 s | 11.7 s |

A successful chunk-compile + reduce-overhead steady-state of, say,
9 s/epoch would be a clear win (~25% faster) and worth the follow-up
investment. 10–11 s would be marginal; 12 s+ would mean the chunk
compile didn't unlock anything that matters at this scale, and we
revisit only if launch overhead is later proven to dominate at
production scale.

## What to write at the end

Append a short `## Verdict` section to this file once T1 (and possibly
T2) have run, with: compile time, steady-state wall, error trace if
any, and one of: `ship`, `scale-test`, or `abandon`. Then either open
a PR (if `ship` or `scale-test`) or `git worktree remove` this branch
(if `abandon`).

## Out of scope

- Production-size validation (size=300 K=6). Only if the spike
  recommends `scale-test`.
- Closed-loop verification. Yes, the closed-loop chunk is also
  refactored, but T1/T2 don't enable it. Closed-loop coverage is
  follow-up work.
- Eval-path changes. Eval keeps the eager cell — windowed
  `_run_segment` is not touched.
- `torch.utils.checkpoint` composition. The trainer's hot loop doesn't
  checkpoint chunks today; if it ever does, that's separate.
- Updating the rest of the codebase (other tasks, `train.py`, etc.).
- `make_graphed_callables` or the NVIDIA RNN-T pattern (those are
  Option 2, not Option 3 — see `tmp/notes/cuda_graphs_options.md`).
