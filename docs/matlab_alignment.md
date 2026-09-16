# MATLAB-aligned three-condition experiment

This protocol compares no adaptation, one SFA/one STD timescale, and three
SFA/two STD timescales. The current approved manuscript experiment uses 15
paired seeds per condition (45 networks), 100 epochs and 2,000 optimizer steps,
on a fresh VM deployed through the repository cloud scripts. An eight-seed
one-epoch preflight passed; the user then selected 15 seeds for the manuscript run.
This is a fixed user-selected count, not a measured maximum-capacity count.
The previous 100-epoch three-seed pilot was cancelled at the user's request;
its maximum-capacity search and automatic final-run pipeline will not run.
The current protocol preserves ring-state carry and periodic test evaluation. Historical regression results
and checkpoints remain historical references, not expected numerical outputs
of the revised equations.

The scientific reference is FractionalReservoir's
`docs/EquationsParametersDocs/Equations_stability_paper.md`, instantiated by
`scripts/paper/sfaEI_mu7_fast_config.m` and its named preset. The generated
`figs/sfaEI_mu7_fast/doc_tables/equation_table.md` documents that run.

Both populations use raw firing rate to drive SFA and STD. Recurrent output
is rate times the product of the active depression states. The total SFA
budget is c, so the feedback is c/K times the sum of K adaptation states.
The deterministic comparison omits Wiener noise and STF, retaining two
populations and shared presynaptic depression. One-timescale SFA starts at
0.25 s; the three-timescale ladder spans 0.25 to 10 s with MATLAB endpoint
jitter. STD recovery/release pairs are 2/0.25 s and, for MTS, 4/0.5 s.
Single- and multiple-timescale STD are not strength-matched.

Integration is MATLAB's zero-noise SRA1: k1=f(y),
k2=f(y+3h k1/4), y_new=y+h(k1+2k2)/3. Four 2.5-ms substeps advance each
100-Hz observation, with fixed input over the substeps. The readout uses
the completed state.

## Running

Use the existing environment. Set `SRNN_HOME` to the external data/results
root. Each command creates a fresh output directory and refuses collisions.
The current run uses 15 seeds, 500 neurons, Dale enforcement, and no skip.
The standalone runner remains available for local or manually managed execution:

```bash
python scripts/run_matlab_aligned.py --dry-run
python scripts/run_matlab_aligned.py --profile --seeds 15
python scripts/run_matlab_aligned.py --seeds 15
python scripts/summarize_matlab_aligned.py "$SRNN_HOME/results/cheetah100/<run>"
```

Optional comparisons use `--skip` or `--no-dales`, separately or together.
They are not part of the primary three-condition experiment. Use
`--run-name NAME` for a unique explicit name. Profiles and full experiments
run in fresh subprocesses. A profile executes one full epoch (20 optimizer
steps), full validation/test evaluation, checkpoint save/reload, and the
same compilation path, while retaining the 100-epoch learning-rate schedule.

Training retains 24 stateful ring readers and 250-sample BPTT chunks.
State carries across chunks and epochs; ring wraparound is intentional.
`grad_checkpoint=true` is honored by the ring trainer, recomputing five-sample
segments during backward without detaching state inside a BPTT chunk. Tests
verify identical outputs, gradients, updates, and carried reader states with
checkpointing enabled and disabled, including closed-loop training.
The 10-second unforced burn-in initializes and freezes the initial condition.
Evaluation uses the existing windowed teacher-forced protocol, and test
performance is inspected periodically. This is not an untouched-test-set
protocol. The run has 100 epochs/2,000 optimizer steps, 60 warmup steps,
Adam at 0.0005, no decay, FP32, and per-variant gradient clipping at one.
Validation at initialization and every 100 steps is required by the summary.

The optional capacity-sizing utility increases whole seed counts with all
three conditions in one cell; it is not part of the approved 15-seed run. Keep model size, precision, readers, chunk length, and solver fixed.
Select the largest tested count whose whole-device sampled usage is below
80% of VRAM, with successful compilation, training, evaluation and checkpoint
reload. GPU samples are one second apart and may miss transient peaks;
PyTorch allocator peaks are recorded separately. Do not use profiles for
scientific conclusions. The final experiment starts fresh with consecutive
seeds chosen independently of results.

## Artifacts and interpretation

`training.log`, `resolved_config.yaml`, `experiment_metadata.json`, and
`gpu_memory.json` capture execution, source and dataset checksums, runtime,
completion, and memory. Metadata parameter counts are allocated trainable
storage, including inactive padding. A separate per-variant active count
excludes masked connections, inactive adaptation entries, and the frozen IC.
It includes optional per-neuron entries and shared gains, counting optimized
scalar entries rather than identifiable degrees of freedom. `effective_params.pt` records final transformed
cell parameters. Standard trainer checkpoints/histories are retained.

The summary writes SVG/300-dpi PNG learning curves and a JSON statistical
report. Individual seeds and arithmetic mean validation curves are plotted
on a log axis. The primary score is the trapezoidal integral of natural-log
validation loss over steps 0–2,000, divided by 2,000. Comparisons use paired
seeds, 2,000 paired bootstrap replicates, paired effect size dz, and two-sided
sign-flip tests (exact through 20 seeds, 100,000 deterministic Monte Carlo
samples above that). Three-seed pilot intervals are descriptive; three
pairwise P values are unadjusted. Lower scores are better, regardless of
which condition achieves them. Model correctness does not require an MTS
advantage. Final test losses are reported separately from learning scores.

MATLAB parity and step-refinement checks must pass before the manuscript
experiment. Numerical agreement checks deterministic equations; it does not
establish fit quality or biological validity. Manuscript prose and figures
are integrated in a separate approved editing step.

## Optional capacity search (superseded for this manuscript run)

On the deployed GPU environment, the sizing orchestrator runs profiles only:

```bash
python scripts/size_matlab_batch.py --output-dir "$SRNN_HOME/results/capacity-unique" \
  --known-profile "$SRNN_HOME/results/cheetah100/<successful-profile>"
```

Omit `--known-profile` to start at three seeds. The script doubles the whole
seed count until a memory failure, then binary-searches the largest passing
count and repeats that count once. Only explicit CUDA out-of-memory errors
or a measured memory-limit exceedance bound the search; nonfinite metrics,
compilation errors, and other failures stop it. The conservative acceptance
measure is the larger of the sampled whole-device peak and PyTorch's reserved
peak plus 1 GiB, at most 80% of device capacity. The allowance is bookkeeping,
not a guarantee against unsampled transient peaks. An optional `--max-seeds`
sets a resource cap; a passing cap is not claimed to be the hardware maximum.
`capacity_summary.json` records every attempt and the final training command;
the final training run is not launched automatically.

## Earlier sequential pipeline (cancelled)

After a successful checkpointed three-seed profile, the complete approved
sequence can run under `nohup` using:

```bash
python scripts/run_matlab_pipeline.py \
  --profile-dir "$SRNN_RESULTS_DIR/cheetah100/<checkpointed-profile>" \
  --output-dir "$SRNN_RESULTS_DIR/<unique-pipeline>" --run-prefix matlab-v2
```

It runs the fresh pilot, pilot summary, capacity search, fresh final training,
and final summary sequentially. `pipeline_status.json` records stage commands,
logs, timestamps, selected seed count, and output paths using atomic writes.
A failed stage stops the pipeline; it never changes packages, source, VM state,
or the scientific protocol to recover silently. Source and dataset checksums
are checked between stages. Keep the deployed runtime source fixed until the
sequence completes.
