#!/usr/bin/env bash
# VM startup script (GPU pipeline) — runs on boot to execute a single training run
# Reads experiment config from VM metadata tags. Assumes Google DLVM base image
# with torch + CUDA pre-installed in system Python.
set -euo pipefail

echo "=== Startup $(date -Iseconds) (GPU pipeline) ==="

# Read metadata
META_URL="http://metadata.google.internal/computeMetadata/v1/instance/attributes"
META_HEADER="Metadata-Flavor: Google"
RUN_NAME=$(curl -sf -H "$META_HEADER" "$META_URL/run-name")
EXPERIMENT=$(curl -sf -H "$META_HEADER" "$META_URL/experiment")
MODEL=$(curl -sf -H "$META_HEADER" "$META_URL/model")
SEED=$(curl -sf -H "$META_HEADER" "$META_URL/seed")
TRAIN_ARGS=$(curl -sf -H "$META_HEADER" "$META_URL/train-args" || echo "")
BUCKET=$(curl -sf -H "$META_HEADER" "$META_URL/bucket")
CLEANUP=$(curl -sf -H "$META_HEADER" "$META_URL/cleanup" || echo "delete")
case "$CLEANUP" in
    delete|stop|keep) ;;
    *) echo "WARN: unknown cleanup=$CLEANUP, defaulting to delete"; CLEANUP="delete" ;;
esac
SKIP_REFRESH=$(curl -sf -H "$META_HEADER" "$META_URL/skip-refresh" || echo "0")
BRANCH=$(curl -sf -H "$META_HEADER" "$META_URL/branch" || echo "main")

# Per-run log path so re-dispatches don't accumulate into one file.
LOG="/var/log/training-${RUN_NAME}-${SEED}.log"
exec > >(tee -a "$LOG") 2>&1

RESULTS_PREFIX="$BUCKET/results-pytorch/$RUN_NAME/$MODEL/$EXPERIMENT/seed$SEED"
VM_NAME=$(hostname)

# WORKDIR persists for stop/keep so re-dispatches reuse the on-disk repo.
if [ "$CLEANUP" = "delete" ]; then
    WORKDIR="/tmp/workdir"
else
    WORKDIR="/opt/train-srnn"
    sudo mkdir -p /opt && sudo chown "$(id -u):$(id -g)" /opt 2>/dev/null || true
fi
WATCHER_PID=""

echo "Run: $RUN_NAME | Experiment: $EXPERIMENT | Model: $MODEL | Seed: $SEED"
echo "Cleanup: $CLEANUP | Skip-refresh: $SKIP_REFRESH | Branch: $BRANCH | Workdir: $WORKDIR"

# Cleanup handler
cleanup() {
    local exit_code=$?
    local end_time
    end_time=$(date +%s)
    local duration=$(( end_time - ${START_TIME:-$end_time} ))

    echo "=== Cleanup (exit=$exit_code) $(date -Iseconds) ==="

    # Kill background upload watcher
    if [ -n "$WATCHER_PID" ]; then
        kill "$WATCHER_PID" 2>/dev/null || true
        wait "$WATCHER_PID" 2>/dev/null || true
    fi

    # Upload final results (only this run's output dir; per-run scoping
    # avoids re-uploading stale artifacts from prior runs on the same VM).
    if [ -n "${RUN_OUTPUT:-}" ] && [ -d "$WORKDIR/$RUN_OUTPUT" ]; then
        gcloud storage cp -r "$WORKDIR/$RUN_OUTPUT/*" "$RESULTS_PREFIX/" 2>/dev/null || true
    fi

    # Upload log
    gcloud storage cp "$LOG" "$RESULTS_PREFIX/training_log.txt" 2>/dev/null || true

    # Upload enriched metadata
    python3 -c "
import json, time
meta = {
    'run_name': '$RUN_NAME', 'experiment': '$EXPERIMENT', 'model': '$MODEL',
    'seed': int('$SEED'), 'exit_code': $exit_code,
    'vm_name': '$VM_NAME',
    'hardware': 'gpu-l4',
    'start_time': '${START_TIME_ISO:-unknown}',
    'completed': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
    'duration_seconds': $duration,
    'commit': '${GIT_COMMIT_FULL:-unknown}',
    'train_args': '$TRAIN_ARGS',
    'cleanup_mode': '$CLEANUP',
    'skip_refresh': $SKIP_REFRESH,
}
if $exit_code != 0:
    meta['failed_at'] = time.strftime('%Y-%m-%dT%H:%M:%SZ')
    meta['error'] = True
json.dump(meta, open('/tmp/metadata.json', 'w'), indent=2)
" && gcloud storage cp /tmp/metadata.json "$RESULTS_PREFIX/run_metadata.json" 2>/dev/null || true

    # Lifecycle dispatch based on per-run cleanup metadata.
    case "$CLEANUP" in
        delete)
            echo "cleanup=delete; deleting VM"
            gcloud compute instances delete "$VM_NAME" --zone="$GCP_ZONE" --quiet 2>/dev/null || true
            ;;
        stop)
            echo "cleanup=stop; stopping VM (disk persists, ~\$0.02/hr)"
            gcloud compute instances stop "$VM_NAME" --zone="$GCP_ZONE" --quiet 2>/dev/null || true
            ;;
        keep)
            echo "cleanup=keep; VM stays running. Dispatch next job via cloud/submit.sh"
            ;;
    esac
}
trap cleanup EXIT

# Step 1: Clone or refresh the private repo. Gated by skip-refresh.
GCP_ZONE=$(curl -sf -H "$META_HEADER" "http://metadata.google.internal/computeMetadata/v1/instance/zone" | rev | cut -d/ -f1 | rev)
GCP_PROJECT=$(curl -sf -H "$META_HEADER" "http://metadata.google.internal/computeMetadata/v1/project/project-id")
REPO_URL="git@github.com:TomRichner/train-srnn.git"

if [ "$SKIP_REFRESH" = "1" ] && [ -d "$WORKDIR/.git" ]; then
    echo "skip-refresh=1; using existing $WORKDIR (no git fetch)"
else
    # Fetch deploy key from Secret Manager
    SSH_DIR="/root/.ssh"
    DEPLOY_KEY="$SSH_DIR/deploy_key"
    mkdir -p "$SSH_DIR"
    chmod 700 "$SSH_DIR"

    echo "Fetching deploy key from Secret Manager..."
    if ! gcloud secrets versions access latest \
        --secret="train-srnn-deploy-key" \
        --project="$GCP_PROJECT" > "$DEPLOY_KEY" 2>/dev/null; then
        echo "FATAL: Could not fetch deploy key from Secret Manager"
        echo "Check: (1) secret 'train-srnn-deploy-key' exists in project '$GCP_PROJECT'"
        echo "       (2) VM service account has secretmanager.secretAccessor role"
        echo "       (3) VM was launched with cloud-platform scope"
        exit 1
    fi
    chmod 600 "$DEPLOY_KEY"

    # Configure SSH for GitHub
    ssh-keyscan -t ed25519 github.com >> "$SSH_DIR/known_hosts" 2>/dev/null
    cat > "$SSH_DIR/config" <<SSHEOF
Host github.com
    IdentityFile $DEPLOY_KEY
    StrictHostKeyChecking yes
    IdentitiesOnly yes
SSHEOF
    chmod 600 "$SSH_DIR/config"

    # Clone with retry — or git fetch + reset if a checkout already exists.
    # Honors the per-run "branch" metadata key (default: main); lets feature
    # branches be tested without merging to main first. Reset to FETCH_HEAD
    # (not origin/$BRANCH) because shallow clones only set up a remote-
    # tracking ref for the originally-cloned branch; FETCH_HEAD always
    # points to whatever was just fetched, regardless of which branch the
    # workdir was originally cloned with.
    if [ -d "$WORKDIR/.git" ]; then
        echo "Existing repo at $WORKDIR; refreshing via git fetch + reset to $BRANCH"
        ( cd "$WORKDIR" \
            && git fetch --depth 1 origin "$BRANCH" \
            && git reset --hard FETCH_HEAD )
    else
        for attempt in 1 2 3; do
            if git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$WORKDIR" 2>&1; then
                break
            fi
            echo "Clone attempt $attempt failed, retrying in 30s..."
            sleep 30
        done

        if [ ! -d "$WORKDIR/.git" ]; then
            echo "FATAL: Git clone failed after 3 attempts"
            exit 1
        fi
    fi

    # Scrub deploy key from disk
    rm -f "$DEPLOY_KEY" "$SSH_DIR/config"
fi

cd "$WORKDIR"
GIT_COMMIT=$(git rev-parse --short HEAD)
GIT_COMMIT_FULL=$(git rev-parse HEAD)
echo "Git commit: $GIT_COMMIT  (full: $GIT_COMMIT_FULL)"

# Capture start time (used by cleanup for duration and metadata)
START_TIME=$(date +%s)
START_TIME_ISO=$(date -u +%Y-%m-%dT%H:%M:%SZ)

# Step 2: Download dataset from GCS (skipped on skip-refresh — reuse on-disk copy)
if [ "$SKIP_REFRESH" = "1" ] && [ -d "train_srnn/data/$EXPERIMENT" ]; then
    echo "skip-refresh=1; reusing on-disk dataset at train_srnn/data/$EXPERIMENT"
else
    mkdir -p "train_srnn/data/$EXPERIMENT"
    gcloud storage cp -r "$BUCKET/datasets/$EXPERIMENT/*" "train_srnn/data/$EXPERIMENT/" || true
fi

# Step 3: Python dependencies (skipped on skip-refresh — assume already installed)
# DLVM base ships torch 2.7.1+cu128 in system Python. A venv would shadow the
# CUDA-matched torch, so install the Hydra stack directly into system Python.
if [ "$SKIP_REFRESH" = "1" ]; then
    echo "skip-refresh=1; skipping pip install (assuming Hydra stack present)"
else
    echo "Installing Hydra stack into system Python..."
    sudo pip3 install --quiet hydra-core omegaconf h5py scipy pandas
fi

# Wait for NVIDIA driver to finish installing (DLVM installs it on first boot,
# can take several minutes). Poll nvidia-smi until it succeeds or we time out.
echo "Waiting for NVIDIA driver..."
for i in $(seq 1 60); do
    if nvidia-smi >/dev/null 2>&1; then
        echo "  driver ready after ${i}0s"
        break
    fi
    sleep 10
done

# GPU sanity check — fail fast if CUDA is not available
echo "GPU sanity check:"
python3 -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available — check DLVM image and --accelerator flag'
print(f'  torch: {torch.__version__}')
print(f'  CUDA:  {torch.version.cuda}')
print(f'  GPU:   {torch.cuda.get_device_name(0)}  ({torch.cuda.get_device_properties(0).total_memory // 1024**3} GB)')
"

# Step 4: Run training
echo "=== Training start $(date -Iseconds) ==="

# Per-run output dir keeps re-dispatches isolated on disk so the GCS upload
# only ships *this run's* artifacts, not stale ones from prior runs.
RUN_OUTPUT="results/$EXPERIMENT/${RUN_NAME}_seed${SEED}"
mkdir -p "$RUN_OUTPUT"

# Add parent dir to PYTHONPATH so `train_srnn` package is importable
export PYTHONPATH="$WORKDIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1   # flush logs in real time (stdout is piped via tee)

# Background upload watcher: polls progress.json and uploads periodically
EPOCHS=$(echo "$TRAIN_ARGS" | sed -n 's/.*epochs=\([0-9]*\).*/\1/p')
EPOCHS=${EPOCHS:-200}
if [ "$EPOCHS" -gt 50 ]; then
    UPLOAD_INTERVAL=$(( EPOCHS / 10 ))
else
    UPLOAD_INTERVAL=5
fi
RESULTS_DIR="$WORKDIR/$RUN_OUTPUT"

(
    LAST_UPLOADED=0
    while true; do
        sleep 30
        # Find progress.json under this run's output dir
        PROGRESS_FILE=$(find "$RESULTS_DIR" -name "progress.json" 2>/dev/null | head -1)
        [ -z "$PROGRESS_FILE" ] && continue
        CURRENT_EPOCH=$(python3 -c "import json; print(json.load(open('$PROGRESS_FILE'))['epoch'])" 2>/dev/null || echo "")
        [ -z "$CURRENT_EPOCH" ] && continue

        NEXT_UPLOAD=$(( LAST_UPLOADED + UPLOAD_INTERVAL ))
        if [ "$CURRENT_EPOCH" -ge "$NEXT_UPLOAD" ]; then
            echo "  [periodic-upload] Epoch $CURRENT_EPOCH: uploading to GCS..."
            gcloud storage cp -r "$RESULTS_DIR/*" "$RESULTS_PREFIX/" 2>/dev/null || true
            gcloud storage cp "$LOG" "$RESULTS_PREFIX/training_log.txt" 2>/dev/null || true
            LAST_UPLOADED=$CURRENT_EPOCH
        fi
    done
) &
WATCHER_PID=$!

set -f  # disable globbing so [a,b] in TRAIN_ARGS isn't expanded
python3 train.py \
    model=$MODEL \
    task=$EXPERIMENT \
    seed=$SEED \
    $TRAIN_ARGS \
    output_dir="$RUN_OUTPUT"
set +f

echo "=== Training complete $(date -Iseconds) ==="
echo "SUCCESS" >> "$LOG"
