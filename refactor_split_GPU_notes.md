# GPU VM + Repo Split Notes

Notes from the session where we:
1. Validated a GCP L4 GPU VM for the PyTorch refactor.
2. Split `pytorch_refactor/` out of the parent repo into its own repo `train-srnn`.

Use this as context when restarting work in the `train-srnn` repo.

---

## 1. GCP Project / Quota State

- **Project**: `liquidneuralnets`
- **Preferred region**: `us-central1` (Iowa)
- **L4 zones in us-central1**: `us-central1-a`, `us-central1-b`, `us-central1-c` (note: `-f` does NOT offer L4)
- **Quotas (as of session end)**:
  - `NVIDIA_L4_GPUS` (regional us-central1): **1**
  - `PREEMPTIBLE_NVIDIA_L4_GPUS`: **1**
  - `GPUS_ALL_REGIONS` (global): **1** — had to request increase from 0; GCP enforces BOTH the regional and global GPU quota.
- **Capacity note**: L4s in us-central1 are frequently in stockout. Expect `ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS` and be prepared to retry across zones. There is no public API to pre-check capacity; the create request itself IS the check. Spot (`PREEMPTIBLE_NVIDIA_L4_GPUS`) often has better availability and is ~65% cheaper.

## 2. Validated VM Configuration

Successfully launched and verified:
- **Machine type**: `g2-standard-8` (1× L4, 8 vCPU, 32 GB RAM)
- **Base image**: `pytorch-2-7-cu128-ubuntu-2204-nvidia-570` from `deeplearning-platform-release` (Google official DLVM)
  - Torch **2.7.1+cu128** pre-installed
  - CUDA **12.8**, NVIDIA driver **570.211.01**
  - Ubuntu 22.04 LTS
  - System Python **3.10.12** (at `/usr/bin/python3` — no conda env in this image variant)
  - Pre-installed: `torch`, `torchvision`, `numpy`, `tqdm`
- **Boot disk**: `pd-balanced`, **minimum 100 GB** (DLVM image requires ≥100 GB — 60 GB was rejected)
- **Required flags**: `--maintenance-policy=TERMINATE` for GPU VMs
- **L4 GPU detected correctly**: `torch.cuda.is_available() = True`, device name `NVIDIA L4`, 23 GB VRAM

### Exact create command that worked

```bash
gcloud compute instances create <name> \
  --project=liquidneuralnets \
  --zone=us-central1-b \
  --machine-type=g2-standard-8 \
  --accelerator=type=nvidia-l4,count=1 \
  --image-family=pytorch-2-7-cu128-ubuntu-2204-nvidia-570 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-balanced \
  --maintenance-policy=TERMINATE \
  --metadata="install-nvidia-driver=True"
```

## 3. Dependencies to install on top of the DLVM base

Already baked into the image: `torch`, `torchvision`, `numpy`, `tqdm`.

Need to install via `sudo pip3 install`:
- `hydra-core` (tested: 1.3.2)
- `omegaconf` (tested: 2.3.0)
- `h5py` (tested: 3.16.0)
- `scipy` (tested: 1.15.3)
- `pandas` (tested: 2.3.3)

Command that works end-to-end:
```bash
sudo pip3 install hydra-core omegaconf h5py scipy pandas
```

No venv — install directly into system Python. Rationale: DLVM is already an isolated environment, the VM is single-purpose and ephemeral, and the pre-built CUDA-matched `torch` lives in system Python. A venv would either shadow it or force re-install.

## 4. SSH / Auth Gotchas (Worth Remembering)

These ate a large chunk of the session. Next time, budget for them or fix once.

### Two GPU quotas, not one
- Regional `NVIDIA_L4_GPUS` is necessary but NOT sufficient.
- `GPUS_ALL_REGIONS` (global) must ALSO be ≥ the number of GPUs you want.
- First-time GPU users always hit this.

### SSH key passphrase blocks `gcloud compute ssh`
- `~/.ssh/google_compute_engine` was passphrase-protected.
- `gcloud compute ssh` fails silently with `Permission denied (publickey)` if the key isn't loaded in ssh-agent (in non-interactive contexts it can't prompt for the passphrase).
- **Fix**: `ssh-add --apple-use-keychain ~/.ssh/google_compute_engine` to cache the passphrase in macOS Keychain persistently.

### OS Login split-brain
- This project has `enable-oslogin` unset (metadata mode) at project level.
- But the Google account already had an OS Login profile registered from prior work.
- Modern `gcloud` pushes keys to OS Login by default when the API is available, but unless the VM has `enable-oslogin=TRUE`, the VM doesn't read OS Login — it reads metadata. Result: key goes to OS Login, VM looks in metadata, auth fails.
- **Fix**: Either (a) enable OS Login on the instance:
  ```bash
  gcloud compute instances add-metadata <vm> --metadata=enable-oslogin=TRUE
  ```
  then add the local public key to the OS Login profile:
  ```bash
  gcloud compute os-login ssh-keys add --key-file=$HOME/.ssh/google_compute_engine.pub
  ```
  (b) Or set `enable-oslogin=TRUE` at the project level so all future VMs are consistent.
- **Symptom of this issue**: username on SSH attempt becomes `tomrichner_gmail_com@` instead of `tom@` once OS Login activates. That's the signal that OS Login is now in use.

### Phone / browser SSH uses a different path
- Cloud Shell / mobile gcloud SSH uses short-lived IAP-tunneled certs, bypasses the entire laptop keypair + metadata/OS Login story.
- Useful as a fallback when laptop SSH is broken.

### `gcloud compute ssh --troubleshoot`
- Non-invasive diagnostic that checks network path, IAM, VPC, VM boot state.
- Doesn't detect client-side key problems, but is useful to rule out everything else.

### Service account scopes
- Default Compute service account has limited scopes; running `gcloud` commands inside the VM gets "insufficient authentication scopes" errors.
- For self-delete + GCS uploads, VM needs `--scopes=cloud-platform` at creation time. Already solved in the old `cloud/` scripts (commit `082a4b1`).

## 5. Repo Split: How It Was Done

The refactor lived at `liquid_time_constant_networks/pytorch_refactor/` in the parent TF1 research repo. We extracted it into its own repo with full history.

### Outcome

- **New repo**: `git@github.com:TomRichner/train-srnn.git` (private)
- **Local path**: `/Users/tom/Desktop/local_code/train-srnn/`
- **Default branch**: `main` (was `master` in parent, renamed post-filter)
- **Contents**: Former `pytorch_refactor/` contents promoted to root — `train.py`, `conf/`, `models/`, `data/`, `utils/`, `cloud/`, `requirements.txt`, `README.md`, `smoke_test.sh`, etc.
- **History**: 103 commits preserved, only those that touched `pytorch_refactor/`. Top commit still `47795bf fix: occupancy and ozone task configs had per_timestep_labels: false`.
- **Parent repo `liquid_time_constant_networks`**: completely untouched. Still has `pytorch_refactor/` as a subdirectory. The split was non-destructive to the parent.

### Method used (`git filter-repo`, Option B)

1. Clone **from GitHub remote** (not from local path — avoids the hardlink issue that blocks filter-repo):
   ```bash
   git clone git@github.com:TomRichner/liquid_time_constant_networks.git train-srnn
   ```
   Note: `git clone --no-local <local_path>` also works, but cloning from remote is cleaner.
2. Run filter-repo (installed via `brew install git-filter-repo`):
   ```bash
   cd train-srnn && git filter-repo --subdirectory-filter pytorch_refactor
   ```
   This rewrites history to contain only pytorch_refactor/ commits, promotes subdirectory to root, and removes the `origin` remote as a safety measure.
3. Rename branch:
   ```bash
   git branch -m master main
   ```
4. Attach the new GitHub repo as origin:
   ```bash
   git remote add origin git@github.com:TomRichner/train-srnn.git
   ```
5. Push (will prompt for SSH key passphrase if not cached):
   ```bash
   git push -u origin main
   ```

### `cloud/` directory resolution
- Parent repo had **two** `cloud/` directories: `liquid_time_constant_networks/cloud/` (TF1 cloud tooling) and `liquid_time_constant_networks/pytorch_refactor/cloud/` (refactor cloud tooling).
- The refactor's `cloud/` (inside `pytorch_refactor/`) was the active one for this project.
- It was automatically included in the filter because it was a child of `pytorch_refactor/`. It now lives at `train-srnn/cloud/`.
- The parent-level `cloud/` was **excluded** and remains only in the parent repo, where it belongs.

## 6. Housekeeping Still TODO in the Parent Repo

These are not urgent but should happen before the parent repo gets confusing:

1. **Add a breadcrumb** at `liquid_time_constant_networks/pytorch_refactor/README.md` or a top-level note: "This code has moved to github.com/TomRichner/train-srnn".
2. **Delete `liquid_time_constant_networks/pytorch_refactor/`** once confident nothing was missed. Do this as a single commit:
   ```bash
   cd /Users/tom/Desktop/local_code/liquid_time_constant_networks
   git rm -r pytorch_refactor
   git commit -m "move pytorch_refactor to its own repo: github.com/TomRichner/train-srnn"
   git push
   ```
   Recommend waiting a few days / a session or two to be sure.
3. The empty placeholder file `pytorch_refactor/refactor_split_GPU_notes.md` in the parent repo is orphaned; it can be deleted whenever.

## 7. Cloud Tooling Rewrite: Plan for Next Session

Context to resume the cloud work in `train-srnn/`:

### Architecture (agreed)

Two-layer split between build-time and boot-time:

| Phase | What it does | Run by |
|---|---|---|
| **Image build** (offline, rare) | Install OS packages, CUDA (inherited from DLVM base), PyTorch (inherited), Python deps, optionally pre-clone repo | `cloud/build_image.sh` |
| **VM startup** (every launch) | Pull latest code, read experiment config from instance metadata, run training, upload results to GCS, self-delete | `cloud/startup.sh` |

Goal: `startup.sh` should do as little as possible. Target: from VM boot to training starts in <60 seconds.

### Private repo auth on the VM

Since `train-srnn` is private, the VM needs read credentials. Recommended options:
1. **Deploy key in instance metadata** — simple. Generate an ed25519 keypair, add pub to repo Settings → Deploy Keys (read-only), bake private half into instance metadata at launch, `startup.sh` reads it from metadata, writes to `~/.ssh/id_ed25519`, clones, deletes.
2. **GCP Secret Manager** — store deploy key or fine-grained PAT in Secret Manager, give VM service account `secretmanager.secretAccessor`, `startup.sh` fetches at boot via `gcloud secrets versions access`. Cleaner, rotatable, auditable. Needs `--scopes=cloud-platform` at VM creation (already doing this for self-delete).

Decision deferred to next session. Start with Option 1 for simplicity; upgrade to Option 2 if/when it matters.

### Concrete next steps

1. **Read existing** `train-srnn/cloud/build_image.sh`, `startup.sh`, `config.env`, `launch_run.sh` to see what's already there and what needs updating for the new DLVM base + new repo URL + private-repo auth.
2. **Update `build_image.sh`** to use `pytorch-2-7-cu128-ubuntu-2204-nvidia-570` as base, install the deps in section 3 of this doc, produce a new `srnn-pytorch` (or renamed) image family.
3. **Update `startup.sh`** to clone from `git@github.com:TomRichner/train-srnn.git` using a deploy key, run training via Hydra, upload results to GCS, self-delete with a `trap` on EXIT so failed runs still clean up.
4. **Update `config.env`** with the new image family name, new repo URL, etc.
5. **Build the new image once**, then test-launch a single VM from it via `launch_run.sh` to validate the full pipeline.

### Important: self-delete safety pattern
Do not rely on `set -e` + self-delete-at-end, because a failed training would leave the VM running and billing forever. Use an EXIT trap:

```bash
trap 'gsutil cp /var/log/startup.log gs://<bucket>/failed/$(hostname)/ 2>/dev/null; \
      gcloud compute instances delete $(hostname) --zone=$(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/zone | awk -F/ "{print \$NF}") --quiet' EXIT
```

(Pattern only — the actual command needs the project, zone extraction, correct log paths, etc.)

## 8. Things That Worked Well and Should Be Reused

- DLVM `pytorch-2-7-cu128-ubuntu-2204-nvidia-570` as base image. Google-official, fresh CUDA 12.8, driver 570, torch 2.7.1 pre-baked against CUDA 12.8. No driver install dance needed. Python 3.10 is fine for the project.
- OS Login enabled at instance level works cleanly once the key mismatch is resolved.
- `gcloud compute ssh --troubleshoot` as first diagnostic for SSH problems.
- `git filter-repo --subdirectory-filter` for clean history extraction.

## 9. Session End State

- **GCP**: `g2-l4-vm-test` deleted. No VMs running, no ongoing GPU billing.
- **Repo**: `train-srnn` created, populated, pushed to GitHub. Parent repo unchanged.
- **Local dev environment**: laptop gcloud SSH working via Keychain-cached passphrase. No further fixes needed.
- **What's NOT done**: cloud tooling rewrite (section 7), parent-repo cleanup (section 6), image rebuild, first end-to-end training run on the new stack.
