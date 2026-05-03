# Chunk-compile spike — post-mortem

**Branch:** `compile-chunk` (preserved; archival tag `spike/chunk-compile`).
**Period:** 2026-05-01 → 2026-05-02.
**Verdict in this branch:** `CompileChunkOutline.md` `## Verdict` section.
**This file:** the broader post-mortem including the production-scale scale-test that landed *after* the verdict was written, and the resulting recommendation to abandon shipping the spike.

## Why this branch exists

`tmp/notes/cuda_graphs_options.md` enumerates strategies for unlocking
CUDA graphs in autograd-over-Python-loop BPTT training. Option 3 is the
"compile the chunk function instead of the cell" approach. This branch
implemented Option 3 end-to-end:

1. Refactored the continuous trainer's chunk forwards
   (`_forward_chunk_pure_tf`, `_forward_chunk_closed_loop`) for
   torch.compile-friendliness.
2. Added a `compile_chunk: false` (default) config knob.
3. Discovered three layered preconditions for `compile_mode="reduce-overhead"`
   to actually work with this autograd pattern:
   - chunk-output `state` must be `.detach().clone()`'d (not just
     `.detach()`'d) at the chunk-to-chunk boundary, otherwise it aliases
     graph-owned static buffers (PyTorch issue #104435).
   - loss + per-K metric must be computed *inside* the compiled chunk
     so autograd's backward graph stays fully under cudagraph_trees'
     control (pytorch/pytorch issues #148439, #158551, #169545).
   - per-step `.item()` calls must be removed from the trainer's hot
     loop (host syncs interfere with cudagraph capture/replay; this
     also fixes `KnownIssues.md §9`).
4. Added cloud infra: `--branch=<name>` knob on `cloud/submit.sh` +
   `cloud/startup_gpu.sh` so feature branches can be dispatched without
   manually SSH'ing into the keep-alive VM.

Three GPU dispatches were needed to satisfy preconditions 1, 2, and 3 —
each one fell over with a different error that taught us the next
constraint. T2 #3 finally landed clean.

## Numerical results

All runs at `task=seeg model=srnn seed=1`, half-T (89989) for K=2, full
T (179989) for K=6, B=24, bptt_chunk_len=250, fp32, `closed_loop.enabled=false`
unless noted, on the L4 GPU keep-alive VM
(`continuous-k2-smoke-srnn-seeg-seed1`).

### K=2 size=30 (smoke scale, 4 epochs)

| Config | Compile (epoch 0) | Steady-state |
|---|---|---|
| cell-compile + default (commit `a747471` baseline) | 36.9 s | 11.6 s/epoch |
| chunk-compile + default (T1, commit `369d856`) | 33:46 | 10.5 s/epoch (-9.5%) |
| chunk-compile + reduce-overhead (T2 #1) | — | **failed**: `accessing tensor output of CUDAGraphs that has been overwritten` |
| chunk-compile + reduce-overhead + chunk-boundary `.detach().clone()` (T2 #2, `254e454`) | — | **failed**: `Trying to backward through the graph a second time` |
| chunk-compile + reduce-overhead + clone + loss-in-chunk (T2 #3, `4be34da`) | 37:48 | **8.0 s/epoch (-31%)** |

### K=6 size=300 (production scale, 20 epochs, closed_loop.enabled=true)

| Config | Compile (epoch 0) | Steady-state mean | Total wall |
|---|---|---|---|
| chunk-compile + reduce-overhead + clone + loss-in-chunk (`254e454` + `4be34da`) | 55:34 | **49.0 s/epoch** | ~70 min |
| cell-compile + default (spike branch, with §9 fix) | 78.7 s | 37.2 s/epoch | ~13 min |
| **cell-compile + default (main branch, no §9 fix)** | **78.0 s** | **35.6 s/epoch** | **~15 min** |

## Key findings

1. **Chunk-compile + reduce-overhead works** — the spike validates that
   the autograd-over-Python-loop pattern *can* be made to play nicely
   with cudagraph_trees, contrary to `KnownIssues.md §8`'s pessimism.
   The four-precondition checklist above is the durable artifact.

2. **At K=2 size=30 (small dev-scale ablation runs), chunk-compile
   gives a real ~31% steady-state win** over cell-compile, but only
   after a 38-min compile tax that's amortized only across hundreds
   of epochs. **Useful only for K=2 dev runs ≥ ~50 epochs.** Even then
   the iteration cost of recompiling on every code change is brutal.

3. **At K=6 size=300 (production scale), chunk-compile *loses* by 38%**
   per epoch (49.0 s vs cell-compile's 35.6 s) and adds 55 minutes of
   compile time. Total wall: 70 min vs 15 min for the same workload.
   The launch-overhead-elimination win (the whole point of CUDA graphs)
   becomes irrelevant once kernels are big enough to be compute-bound.

4. **The `KnownIssues.md §9` per-step-`.item()` fix that came with the
   loss-in-chunk refactor (commit `4be34da`) doesn't help at production
   scale.** Expected 5–10% win from sync elimination; measured ~3%
   *slowdown* on the spike branch vs. main at K=6 size=300 (37.2 s vs
   35.6 s). The trainer-loop wrapper overhead probably outweighs the
   sync wins. The per-step `.item()` cost gets amortized into the GPU
   queue at this scale.

5. **Result: production should stay on main's cell-compile + default.**
   The complex chunk-compile machinery on this branch isn't worth it.

## What gets cherry-picked from this branch

Just the cloud infra. New branch `cloud-branch-knob` off main
contains:

- `0027cf9` — `cloud/submit.sh` + `cloud/startup_gpu.sh`: accept
  `--branch=<name>` (default `main`) so feature branches can be
  dispatched to the keep-alive VM without SSH-and-checkout dance.
- `8d289a8` — fix the FETCH_HEAD reset for shallow clones (`git fetch
  --depth 1` doesn't set up a remote-tracking ref for the new branch,
  so `git reset --hard FETCH_HEAD` is the only thing that works).

These are pure infra, backwards-compatible, ~50 LOC. Worth shipping
to main on their own.

## What stays on this branch (intentionally not cherry-picked)

- `46bfd62` chunk-compile spike refactor + tests
- `369d856` train.py model-level compile defer when `compile_chunk=true`
- `254e454` chunk-boundary `.detach().clone()`
- `4be34da` loss-in-chunk refactor + tensor accumulators (KnownIssues §9 fix)
- `c8335d4` `CompileChunkOutline.md` verdict

These all hang together — separating §9 from the loss-in-chunk machinery
is structurally awkward, and §9 alone didn't help anyway. If any future
PyTorch release fixes cudagraph_trees' production-scale limitations
enough to make chunk-compile competitive at K=6 size=300, this branch
is the place to start the re-evaluation.

## Preservation

- Branch `compile-chunk` stays on origin (`origin/compile-chunk` at
  `c8335d4`).
- An annotated tag `spike/chunk-compile` is created at the same commit
  for archival clarity (tags don't move; branches can drift). To find
  this work later: `git fetch --tags && git checkout spike/chunk-compile`.
- This file (`tmp/notes/chunk_compile_spike_postmortem.md`) lives on
  the branch and points to `CompileChunkOutline.md` for the operational
  detail.

## Lessons for the next person

- **Don't believe scale-up extrapolations from K=2 numbers.** Launch-
  overhead-dominated kernels at K=2 size=30 become compute-bound at
  production K=6 size=300. The win profile inverts.
- **Always do the small + large scale-test pair before changing
  defaults.** This spike's verdict at K=2 (`ship` with scale-test
  follow-up) was correct in spirit — but the scale-test is what
  produced the actual ship/no-ship decision.
- **The cudagraph_trees + autograd interaction has many failure modes.**
  Each precondition (clone, loss-in-chunk, no syncs) was discovered
  by hitting a different error. There may be more failure modes the
  spike didn't surface; the four-precondition list is necessary but
  may not be sufficient for other autograd patterns.
- **Bare `.detach()` is a footgun under reduce-overhead.** It returns
  a new view of the same storage, suppresses the canonical aliasing
  error message, and silently allows the tensor to be overwritten.
  Always `.detach().clone()` when crossing a compile-region boundary.

## Pointers

- `CompileChunkOutline.md` — original spike outline + verdict (after
  T2 #3, before scale-test data).
- `tmp/notes/cuda_graphs_options.md` — strategy survey that motivated
  the spike (Option 3 of 5).
- `KnownIssues.md §8` — the original blocker documentation.
- `KnownIssues.md §9` — per-step host syncs (fixed on this branch
  but doesn't help in practice at production scale).
- `effectiveWPlan.md` — the (separate) `_effective_W` hoist that
  landed on main as commit `47f0da8` *before* this spike began.
- `scripts/test_compile_chunk.py` — 13 tests covering chunk-compile
  internals; all pass on local CPU.
