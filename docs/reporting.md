# Detailed SRNN training reports

The rich report survived the refactor in `scripts/postprocess.py::build_report`.
Its history includes the automatic PDF report (8629265, April 28), per-timescale
plots (60c5501, August 27), and weight-spread and report improvements (e8e5099
and d8b7575, August 27). The latest pre-September-11 revision is d3031ba.
The September 13 refactor (6acd943) retained it. The separate
`summarize_matlab_aligned.py` learning-curve summary does not invoke that report.

`scripts/report_srnn.py` provides a local, version-2 report for the refactored
model and large paired-seed experiments. It reuses the effective-parameter
transforms from postprocess and reads checkpoints one at a time, avoiding
retaining every optimizer state in memory. No training or cloud job is started.

## Usage

```bash
uv run python scripts/report_srnn.py "$SRNN_HOME/cache/myrun" --pdf
# Detailed pages for every network (potentially a very long PDF):
uv run python scripts/report_srnn.py "$SRNN_HOME/cache/myrun" --variants all --pdf
# Or choose exact variant names and a separate output directory:
uv run python scripts/report_srnn.py "$SRNN_HOME/cache/myrun" \
  --variants srnn-sfa3-std2-seed1 --output-dir /path/to/report --pdf
```

Inputs are trusted local SRNN version-2 checkpoints (`init.pt`, `last.pt`,
optional `epoch_*.pt`), `training_history.csv`, and `test_history.csv`.
Download these from the run's result directory first. Execution metadata JSON
files are included when present. Version-1 checkpoints need their historical
source revision; the new report rejects them rather than reinterpreting them.
The performance section currently assumes regression histories (MSE and MAE).

Markdown, PNG plots, CSV tables, and JSON are generated without PDF tools.
`--pdf` calls the installed `md2pdf` skill wrapper; override its location with
`--pdf-wrapper /path/to/md2pdf.sh` or `MD2PDF_WRAPPER`. It requires the wrapper's
Pandoc/LuaLaTeX dependencies. PDF failures propagate as errors while preserving
the Markdown and figures. Existing reports without this feature's generation
marker are protected; use `--output-dir` to preserve a historical report.

## Contents and interpretation

- Validation curves for every seed and condition means, plus recorded learning
  rates and final prediction metrics.
- Recurrent E/I weight means, magnitudes, relative change from initialization,
  wrong-sign connection counts, and input/output weight spread.
- Total SFA budgets and the rate-nonlinearity offset, including that offset in
  the no-adaptation condition.
- Dendritic time constants and every active SFA, STD recovery, and STD release
  timescale, separately for E and I. Inactive padded slots are excluded.
- Initial/final effective-parameter tables and per-network evolution panels.
- Saved solver, skip, Dale, adaptation-count, and execution provenance.

Aggregates and CSVs include **all networks**. Detailed pages default to the
lowest-numbered seed of each condition, chosen without consulting outcomes.
Aggregate bands are sample SD across network means; detail bands are population
SD within a network. Weight-spread panels summarize within-network weight SD.
Structural zero recurrent connections are excluded. Time constants use seconds;
SFA c is the total budget, not the budget divided by the timescale count.
Initialization is epoch 0; saved zero-based epoch indices become completed
epoch counts. `last.pt` replaces an intermediate snapshot at the same epoch.

Outputs are `report.md`, optional `report.pdf`, `report_figures/`,
`report_parameter_history.csv`, `report_parameter_initial_final.csv`,
`report_performance.csv`, and `report_summary.json`. Keep generated artifacts
outside the repository. The CSVs contain every network even when the PDF uses
only representative detail pages.

The manuscript run `m5-15s-100e-0916` has 45 networks (15 paired seeds in each
of three conditions), no skip, Dale enforcement, and 100 epochs. Its report
lives beside its checkpoints in `$SRNN_HOME/cache/m5-15s-100e-0916/`.

On installations where LuaLaTeX records `Producer: LuaTeX` instead of
`luahbtex`, the wrapper may refuse to overwrite its own previous PDF. Preserve
that PDF and render into a new `--output-dir`; the report command deliberately
propagates the wrapper's protection rather than overriding it.
