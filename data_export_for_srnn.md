# Stage 6.0 — SEEG Data Export for SRNN Training

This document describes the per-block `.mat` files produced by stage 6.0
(`run_srnn_export.m`) for use as input to a PyTorch Stable Recurrent Neural
Network (SRNN) regression model.

## Output location

```
{XLTEK_bad_chs}/{subject_id}/srnn_export/
```

One `.mat` file per analysis block, per subject.

## Filename schema

```
seeg_{subject_id}_b{block_number}_{awake|asleep}_{baseline|stim}.mat
```

Examples:

```
seeg_939_b4_awake_baseline.mat
seeg_939_b9_asleep_stim.mat
seeg_1533_b1_awake_baseline.mat
```

Field meanings:
- `subject_id` — string identifier matching `subjects(i).subject_id`
- `block_number` — 1-based block index in the subject's TEA block list
- `awake|asleep` — sourced from `subjects(i).isAsleep(block_number)` in
  `dataset_configs/main_dataset_config/subject_structure.m`
- `baseline|stim` — sourced from `ica_info.stim_block_ids` /
  `ica_info.baseline_block_ids` (which originate in the stage 3.9 block
  selection)

A simple Python regex to parse the filename:

```python
import re
PATTERN = re.compile(
    r"^seeg_(?P<sid>[^_]+)_b(?P<block>\d+)_"
    r"(?P<sleep>awake|asleep)_(?P<cond>baseline|stim)\.mat$"
)
m = PATTERN.match(filename)
sid    = m.group("sid")
block  = int(m.group("block"))
asleep = m.group("sleep") == "asleep"
stim   = m.group("cond") == "stim"
```

## File contents

Each `.mat` is HDF5 (MATLAB v7.3, **uncompressed** — chosen for fast random
access from Python). It contains exactly two variables:

| Variable    | Shape                  | Dtype     | Meaning                                      |
|-------------|------------------------|-----------|----------------------------------------------|
| `data_filt` | `[n_samples, n_chan]`  | float64   | Sensor-space, HP-filtered, ICA-cleaned SEEG  |
| `SR`        | scalar                 | float64   | Sampling rate (Hz)                           |

Shape convention is **time × channels** (so PyTorch can index `[t, ch]`
directly without a transpose at the inner loop).

> Note: HDF5 stores arrays in C order while MATLAB writes in Fortran order.
> When loaded with `h5py`, `data_filt` will appear as `[n_chan, n_samples]`
> and you must transpose. See loading example below.

## Preprocessing chain (provenance)

The data in `data_filt` is produced by, in order:

1. **Raw TEA load** — read full block (or `max_seconds` if set) from the
   subject's TEA `.mat` file via `TEA.read()`.
2. **Good-channel selection** — keep only channels in
   `ica_info.channels.good_ch_names` (rejected by stage 4.1 bad-channel
   review).
3. **High-pass filter** — Butterworth, order 2, cutoff = `filter_hp` Hz
   (default 3 Hz), applied with `filtfilt` (zero-phase).
4. **ICA decomposition (precomputed in stage 5.0)** — Picard ICA was fit on
   HP-filtered (default 2 Hz) raw data of the same channels.
5. **Component selection (stage 5.1)** — user reviewed components and split
   them into `components_kept` (neural) and `components_removed` (artifact).
6. **Clean projection** — apply only the kept-component sub-rows of the
   unmixing matrix, then back-project to sensor space:

   ```
   X            = block_data'                     % n_ch × time
   ic           = ica_weights(kept, :) * X        % n_kept × time
   data_filt    = ica_winv(:, kept)  * ic         % n_ch × time
   ```

   Then transposed and cast to `double`. **No sign convention is applied**
   in stage 6.0 (unlike stage 5.2 which negates input and un-negates
   output). The negation in 5.0/5.2 cancels through the projection so the
   omission is mathematically equivalent for `data_filt`.

The HP-filter cutoff used in step 3 (3 Hz) is intentionally slightly
different from the cutoff used during ICA fitting in stage 5.0 (2 Hz). The
ICA decomposition is robust to small differences in low-frequency content
of the projected data. If you want exact match, rerun stage 5.0 with
`'filter_hp', 3` (or rerun this stage with `'filter_hp', 2`).

## Channel ordering

The columns of `data_filt` are in the order given by
`ica_info.channels.good_ch_names`, which is the surviving subset of
`ica_info.channels.all_ch_names` (in the original recording order).

**Channel names are NOT saved per-file** to keep export files lean. They live
in the upstream `full_ica.mat`. To recover them in Python:

```python
import h5py
with h5py.File('full_ica.mat', 'r') as f:
    # MATLAB cell of strings → HDF5 references; resolve each
    refs = f['ica_info']['channels']['good_ch_names'][:].squeeze()
    good_ch_names = ["".join(chr(c[0]) for c in f[r][:]) for r in refs]
```

(Loading nested struct/cell from v7.3 .mat in Python is awkward; if you
need channel names often, consider re-exporting them as a sidecar JSON.)

## Loading from Python

### Minimal `h5py` example

```python
import h5py
import numpy as np

def load_seeg_block(path):
    """Returns (data, sr) where data is float32 [time × channels]."""
    with h5py.File(path, "r") as f:
        # h5py loads MATLAB-saved arrays in reversed dim order
        data = np.array(f["data_filt"]).T          # [time × channels]
        sr   = float(np.array(f["SR"]).squeeze())
    return data.astype(np.float32), sr
```

### PyTorch `Dataset` skeleton

```python
import os, re, glob, torch
from torch.utils.data import Dataset

PATTERN = re.compile(
    r"^seeg_(?P<sid>[^_]+)_b(?P<block>\d+)_"
    r"(?P<sleep>awake|asleep)_(?P<cond>baseline|stim)\.mat$"
)

class SrnnSeegDataset(Dataset):
    def __init__(self, root, subjects=None, condition=None, sleep=None,
                 window_samples=2000, stride=1000):
        self.window = window_samples
        self.stride = stride
        self.index = []   # list of (path, start_sample, end_sample)
        for path in glob.glob(os.path.join(root, "*", "srnn_export", "seeg_*.mat")):
            m = PATTERN.match(os.path.basename(path))
            if not m: continue
            if subjects  and m["sid"]   not in subjects:  continue
            if condition and m["cond"]  != condition:     continue
            if sleep     and m["sleep"] != sleep:         continue
            data, _ = load_seeg_block(path)
            n = data.shape[0]
            for s in range(0, n - window_samples + 1, stride):
                self.index.append((path, s, s + window_samples))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        path, s, e = self.index[i]
        data, sr = load_seeg_block(path)
        return torch.from_numpy(data[s:e])     # [window × channels]
```

For higher throughput at scale, cache the loaded arrays (or memory-map them
with `h5py.File(..., 'r').get('data_filt')[s:e].T`) instead of reloading
the whole block on every `__getitem__`.

## Differences from stage 5.2 (`full_clean.mat`)

| Aspect                | Stage 5.2 (`full_clean.mat`)                          | Stage 6.0 (per-block `.mat`)               |
|-----------------------|-------------------------------------------------------|--------------------------------------------|
| Files per subject     | 1 consolidated                                        | 1 per analysis block                       |
| Variables saved       | `data_ica`, `data_filt`, `SR`, `clean_info`, `channel_names` | Just `data_filt` and `SR`                  |
| Sign convention       | Negate before, un-negate after (matches 5.0)          | None (mathematically equivalent for output) |
| HP filter cutoff      | Configurable (3 Hz used for current subjects)         | Configurable (3 Hz default)                |
| Compression           | v7.3, uncompressed                                    | v7.3, uncompressed                         |
| Intended consumer     | MATLAB downstream stages (5.3 VAR, 5.7 Kreiss)        | Python / PyTorch SRNN training             |

## Re-running

Stage 6.0 skips per-block files that already exist. To force a rebuild for
one subject:

```matlab
run_pipeline(env_file, [6.0], 'subject_id', '939', 'force', true);
```

To rebuild all subjects:

```matlab
run_pipeline(env_file, [6.0], 'force', true);
```

To use a different HP cutoff (e.g., 1 Hz):

```matlab
run_pipeline(env_file, [6.0], 'filter_hp', 1, 'force', true);
```
