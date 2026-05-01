# CUDA Graphs for autograd-over-Python-loop RNN training: state of the art

**Question.** Is `torch.compile(mode="reduce-overhead")` truly impossible to satisfy when training an SRNN/CTRNN whose forward is a Python loop calling a compiled cell across `T` BPTT timesteps?

**Short answer.** With current PyTorch (≤ 2.11) it is *practically* impossible to make `reduce-overhead` work on this exact pattern. The issue is well-documented, has multiple open issues, and is acknowledged by maintainers as a structural limitation of `cudagraph_trees` for autograd workloads with retained intermediates. NVIDIA's own RNN-T case study uses **manual `torch.cuda.graph()` capture wrapped in a custom autograd Function**, not `torch.compile`. Below: why the easy path fails, what alternatives exist, and which are worth pursuing for this codebase.

---

## Why reduce-overhead fails on autograd + Python-loop RNN

The conflict is between two things `cudagraph_trees` cannot reconcile in current PyTorch:

| Need | Mechanism | Conflict |
|------|-----------|----------|
| Recycle compiled-cell output buffers between iterations (the whole point of CUDA graphs) | `cudagraph_mark_step_begin()` invalidates previous-iteration outputs | But autograd has saved those outputs (or activations derived from them) for the eventual backward |
| Keep saved-for-backward tensors alive until the chunk's loss.backward() runs | Don't invalidate previous outputs | But then static-buffer aliasing kicks in: next iteration's forward writes to the same address, causing the "static input data pointer changed" error |

PyTorch documents this directly: *"Memory for activations that are saved in the forward cannot be reclaimed in the backward."* And separately: *"To prevent overwriting, clone the tensor outside of torch.compile() or call torch.compiler.cudagraph_mark_step_begin() before each model invocation."* The two pieces of guidance are mutually exclusive when forward and backward straddle multiple iterations of the loop.

The error we hit on V5 — `accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run` fired during `_engine_run_backward` — is exactly this conflict. Our `cudagraph_mark_step_begin()` calls between cell calls invalidated activations that the backward pass still needed. Removing those calls would re-trigger the original forward-pass aliasing error.

PyTorch's own docs note the heuristic: *"in training we [start a new iteration on each invocation] so long as there is not a pending backward that has not been invoked."* Our pattern violates this — we run T forwards, *then* one backward. The cudagraph trees machinery expects forward + backward + step *per* invocation, not T forwards followed by one backward.

This is a known and tracked limitation:
- pytorch/pytorch #146569 — `fullgraph=True` + recurrent invocation
- pytorch/pytorch #148439 — explicit "accessing overwritten output" issue (open)
- pytorch/pytorch #158551 — "despite clone and mark_step_begin" (open, no maintainer fix)
- pytorch/pytorch #169545 — `compile + cudagraph + gradient accumulation fails` (open)

There is no maintainer-accepted workaround within `torch.compile`. The fundamental redesign is in `cudagraph_trees` itself, and as of mid-2026 that work has not landed.

---

## The options, ranked by realism for this codebase

### Option 0. **Status quo: default mode + the slice-assign refactor**

What we have on `main` today after commit `065dfb6`. Compile-on with default mode (kernel fusion via Inductor codegen, no CUDA graphs). The cell_loop refactor is independently a code-quality win and is neutral here.

**Pros**
- Works. Confirmed steady-state ~11s/epoch at K=2 size=30.
- No further engineering risk.
- Future-compatible if PyTorch ever fixes the underlying limitation: the slice-assign + `mark_cudagraph_step()` pattern is what the docs *recommend*, so we're aligned with the API contract; the bug is below us.

**Cons**
- Leaves an estimated 5–25% of step time on the table (kernel launch overhead, scaling inversely with K · size · B).

**Recommendation**: keep this as the default. Document the failure mode in `KnownIssues.md`. Move on.

---

### Option 1. **Compile only the eval path with reduce-overhead**

Eval runs under `no_grad()` and discards intermediates → no saved-for-backward, no aliasing conflict. The trainer's forward chunks remain on default mode.

**Pros**
- Eval is currently the second-largest wall-clock contributor on super-epochs (~30s of windowed eval at K=6).
- Self-contained change in the eval call site only.

**Cons**
- We just made eval cheap by gating it to checkpoint boundaries (per-reader-sweep semantics). The win is small in absolute wall-clock — maybe 5–10s per super-epoch.
- Adds complexity to track two compile modes in one process (potential for compile-cache fragmentation).

**Recommendation**: skip unless eval becomes a bottleneck again.

---

### Option 2. **Manual `torch.cuda.graph()` capture with a custom `autograd.Function`** (NVIDIA RNN-T pattern)

The only documented working pattern for CUDA graphs on autograd-trained recurrent models. NVIDIA's `dl-cuda-graph` examples explicitly demonstrate this for RNN-T:

> "Use PyTorch's Native API. … manual capture is necessary for fine-grained control that torch.compile doesn't provide."

Their pattern:
1. Pre-allocate static input + output + state buffers for a single chunk of fixed length.
2. Warm up on a side stream (3+ eager iterations).
3. Capture the **forward** of the entire chunk (the Python loop becomes recorded GPU work) into one `cuda.CUDAGraph`.
4. Capture the **backward** of the entire chunk into a separate `cuda.CUDAGraph`, sharing the same memory pool so saved activations persist across the forward→backward boundary.
5. Wrap both in a `torch.autograd.Function` so the chunk looks like a single op to the rest of the model.
6. To run, copy live inputs into static buffers and `graph.replay()`.

**Pros**
- Works for autograd training. Proven in production at NVIDIA scale.
- Largest possible win — eliminates ALL launch overhead for the chunk, not just per-cell.
- Bucketing handles variable-length sequences (we have fixed `chunk_len`, so no bucketing needed).

**Cons**
- Significant engineering: ~300–500 LOC of new code in `continuous.py` plus a new module for graph management.
- Requires `chunk_len` and `B` to be static at graph-capture time. Already true for us, so cheap.
- Requires pre-allocated buffers for `chunk_x`, `chunk_y`, `state`, `y_prev`, `alpha_chunk`, `logits` — manageable.
- Can't easily mix with `grad_checkpoint=True` (the captured graph already implements the chunk; checkpointing would re-capture). Would need a version-flag for "graph mode" vs "checkpoint mode".
- AMP autocast inside a captured graph requires care (autocast state must be applied before capture).

**Recommendation**: **prototype** if Option 0's launch overhead becomes the proven bottleneck. The right time to do this is *after* we've measured wall-clock breakdown at production K/size/B and confirmed launch overhead ≥ 20% of step time. Until then, premature optimization.

**References for the implementation**:
- NVIDIA `dl-cuda-graph/examples/rnnt.html` — full code pattern
- PyTorch blog `Accelerating PyTorch with CUDA Graphs` — warm-up + side-stream capture pattern
- `torch.cuda.make_graphed_callables` — high-level wrapper (works for autograd, drops-in to nn.Module). Constraint: only parameters trainable, no buffers with `requires_grad`. Captures one fixed-shape forward+backward as a fused autograd op.

`make_graphed_callables` is the easier subset of Option 2: pass `model.cell` plus a sample input, get back a graphed callable. It would graph one cell call's forward+backward as a fused unit. **But** it would have to be called T times in our loop, and each call captures fresh static buffers — the same chunk_t-vs-chunk_(t+1) aliasing problem returns. Not a win for our usage.

---

### Option 3. **Larger compile region: graph the entire chunk loop**

Instead of compiling only `cell`, compile the *entire* `_forward_chunk_pure_tf` function. Dynamo traces the Python `for t in range(T):` loop and unrolls it into a single graph; the compiled chunk becomes one graph with no Python loop control between cell calls.

Inductor would see ~250 cell calls back-to-back, fuse them into one big graph, and reduce-overhead could capture the whole thing as one CUDA graph. Saved-for-backward becomes a single forward→backward boundary, which `cudagraph_trees` *does* handle.

**Pros**
- No custom autograd machinery — let `torch.compile` do the work.
- One CUDA graph per chunk → maximum launch-overhead reduction.

**Cons**
- Massive trace: 250-step unroll × ~50 ops/cell = ~12,500 ops in one graph. **Compile time** could be very long (we already saw a 13-min stuck compile on the smaller version of this earlier in development).
- Dynamic shapes are killers here. If `chunk_len` ever varies, recompile.
- The slice-assign `if out_seq is None: ...` branch becomes a graph-break unless Dynamo can specialize on it. Restructuring the loop to allocate up-front (no None check) would help.
- The loop body contains `state.clone()` and `mark_cudagraph_step()` — both interact with capture. We'd want to remove both for this option, but neither hurts in the larger-graph world (clone becomes a graph-internal copy; mark_step_begin is a no-op inside a captured region).
- Closed-loop's data-dependent path (the `(1 - alpha) * x_real + alpha * y_prev` blend) is fine; no control flow there.

**Recommendation**: try this *after* Option 0 is solid. It's a one-line code change (move `torch.compile` from `model.cell` to `_forward_chunk_pure_tf`) plus careful handling of compile-time. If compile completes in reasonable time at production size, this is the fastest path to CUDA graphs working.

**Risk**: compile time can blow up at production scale (size=300, K=6 means much larger per-cell graphs). Worth a 1-day spike.

---

### Option 4. **Skip CUDA graphs; reduce launch overhead another way**

If launch overhead is the bottleneck, attack it directly without involving graphs:

a. **Bigger batches (B↑)** — every kernel launch processes more data, fixed launch overhead amortized. Already on the user's roadmap. The strongest single lever.

b. **Bigger K (parallel ablations)** — same effect, K is a bmm batch axis.

c. **Bigger model (size↑)** — kernels become compute-bound; launch fraction drops naturally.

d. **bf16 AMP** — smaller kernels, but kernels run faster, *increasing* launch-overhead fraction. Bad if launch overhead is the issue. Skip.

e. **Custom fused CUDA kernel for the SRNN cell** — replace the ~50 separate ops in the cell forward (Dale's law mask, multi-timescale SFA, ODE step, activation) with one Triton or CUDA kernel. Eliminates ~50× launches in one stroke.

   - **Pros**: largest possible win. Compatible with autograd if written via `torch.autograd.Function`. No interaction with cuda graphs at all.
   - **Cons**: ~weeks of engineering. Hard to maintain as the cell evolves (we still tune ablations regularly). Triton helps — one Triton kernel per cell variant or one parametrized kernel.

**Recommendation**: pursue (a)–(c) opportunistically — they don't require code changes, just config exploration. Defer (e) until cell architecture is stable.

---

### Option 5. **Wait it out**

Track pytorch/pytorch issues #148439 #158551 #169545. The maintainers acknowledge the limitation; a fix would require redesigning `cudagraph_trees` to handle multi-invocation forward + single backward. There's no public ETA, but the pattern (autograd RL, RNN training, gradient accumulation) is common enough that someone will eventually fix it.

**Pros**: zero work.
**Cons**: indefinite timeline.

**Recommendation**: subscribe to the issues; revisit every few PyTorch releases. Currently the safe assumption is that this is unsolved through PyTorch 2.11.

---

## Concrete next-step recommendation

1. **Land** the cell_loop refactor as the new default (already done in `065dfb6`).
2. **Document** in `KnownIssues.md` that `compile_mode=reduce-overhead` and `compile_mode=max-autotune` do not work in continuous training because of the autograd-over-Python-loop CUDA-graph limitation. Reference issues #148439 / #158551.
3. **Run** the K=2 size=30 default-mode baseline on the new commit to confirm zero regression vs. `c2974ae`.
4. **Defer** Options 2/3 until we have a measured launch-overhead number at production K=6 size=300 B=24 (or larger B). If the number is ≥ 20% of step time, prototype Option 3 first (cheaper) and Option 2 second.
5. **Pursue Option 4(a)–(c)** opportunistically. Larger B is already valuable.

---

## Sources

- [CUDAGraph Trees — PyTorch docs](https://docs.pytorch.org/docs/stable/torch.compiler_cudagraph_trees.html)
- [How CUDA Graph Works in torch.compile — fkong.tech](https://fkong.tech/posts/2025-12-23-cuda-graph-in-torch-compile/)
- [RNN-T (RNN Transducer) CUDA Graph Best Practice — NVIDIA](https://docs.nvidia.com/dl-cuda-graph/examples/rnnt.html)
- [CUDA Graph Best Practice for PyTorch (overview) — NVIDIA](https://docs.nvidia.com/dl-cuda-graph/latest/)
- [Accelerating PyTorch with CUDA Graphs — pytorch.org/blog](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/)
- [torch.cuda.make_graphed_callables — PyTorch docs](https://docs.pytorch.org/docs/stable/generated/torch.cuda.make_graphed_callables.html)
- [Issue #148439 — accessing tensor output of CUDAGraphs that has been overwritten](https://github.com/pytorch/pytorch/issues/148439)
- [Issue #158551 — RuntimeError despite clone and cudagraph_mark_step_begin](https://github.com/pytorch/pytorch/issues/158551)
- [Issue #169545 — torch.compile + CUDAGraph + gradient accumulation fails](https://github.com/pytorch/pytorch/issues/169545)
- [PyTorch Forum — torch.compile max-autotune overwriting error](https://discuss.pytorch.org/t/torch-compile-max-autotune-to-prevent-overwriting-clone-the-tensor-outside-of-torch-compile-or-call-torch-compiler-cudagraph-mark-step-begin-before-each-model-invocation/195151)
