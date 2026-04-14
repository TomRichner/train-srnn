#!/usr/bin/env bash
# VM startup script (GPU pipeline) — runs on boot to execute a single training run
# Reads experiment config from VM metadata tags. Assumes Google DLVM base image
# with torch + CUDA pre-installed in system Python.
set -euo pipefail

LOG="/var/log/training.log"
exec > >(tee -a "$LOG") 2>&1

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

RESULTS_PREFIX="$BUCKET/results-pytorch/$RUN_NAME/$MODEL/$EXPERIMENT/seed$SEED"
VM_NAME=$(hostname)

WORKDIR="/tmp/workdir"
WATCHER_PID=""

echo "Run: $RUN_NAME | Experiment: $EXPERIMENT | Model: $MODEL | Seed: $SEED"

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

    # Upload final results
    if [ -d "$WORKDIR/results" ]; then
        gcloud storage cp -r "$WORKDIR/results/*" "$RESULTS_PREFIX/" 2>/dev/null || true
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
    'commit': '${GIT_COMMIT:-unknown}',
    'train_args': '$TRAIN_ARGS',
}
if $exit_code != 0:
    meta['failed_at'] = time.strftime('%Y-%m-%dT%H:%M:%SZ')
    meta['error'] = True
json.dump(meta, open('/tmp/metadata.json', 'w'), indent=2)
" && gcloud storage cp /tmp/metadata.json "$RESULTS_PREFIX/run_metadata.json" 2>/dev/null || true

    # Self-delete
    gcloud compute instances delete "$VM_NAME" --zone="$GCP_ZONE" --quiet 2>/dev/null || true
}
trap cleanup EXIT

# Step 1: Clone private repo via deploy key from Secret Manager
GCP_ZONE=$(curl -sf -H "$META_HEADER" "http://metadata.google.internal/computeMetadata/v1/instance/zone" | rev | cut -d/ -f1 | rev)
GCP_PROJECT=$(curl -sf -H "$META_HEADER" "http://metadata.google.internal/computeMetadata/v1/project/project-id")
REPO_URL="git@github.com:TomRichner/train-srnn.git"

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

# Clone with retry
for attempt in 1 2 3; do
    if git clone --depth 1 "$REPO_URL" "$WORKDIR" 2>&1; then
        break
    fi
    echo "Clone attempt $attempt failed, retrying in 30s..."
    sleep 30
done

# Verify clone succeeded
if [ ! -d "$WORKDIR/.git" ]; then
    echo "FATAL: Git clone failed after 3 attempts"
    exit 1
fi

# Scrub deploy key from disk
rm -f "$DEPLOY_KEY" "$SSH_DIR/config"

cd "$WORKDIR"
GIT_COMMIT=$(git rev-parse --short HEAD)
echo "Git commit: $GIT_COMMIT"

# Capture start time (used by cleanup for duration and metadata)
START_TIME=$(date +%s)
START_TIME_ISO=$(date -u +%Y-%m-%dT%H:%M:%SZ)

# Step 2: Download dataset from GCS
mkdir -p "train_srnn/data/$EXPERIMENT"
gcloud storage cp -r "$BUCKET/datasets/$EXPERIMENT/*" "train_srnn/data/$EXPERIMENT/" || true

# Step 3: Python dependencies
# DLVM base ships torch 2.7.1+cu128 in system Python. A venv would shadow the
# CUDA-matched torch, so install the Hydra stack directly into system Python.
echo "Installing Hydra stack into system Python..."
sudo pip3 install --quiet hydra-core omegaconf h5py scipy pandas

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
mkdir -p results/$EXPERIMENT

# Add parent dir to PYTHONPATH so `train_srnn` package is importable
export PYTHONPATH="$WORKDIR:${PYTHONPATH:-}"

# Background upload watcher: polls progress.json and uploads periodically
EPOCHS=$(echo "$TRAIN_ARGS" | sed -n 's/.*epochs=\([0-9]*\).*/\1/p')
EPOCHS=${EPOCHS:-200}
if [ "$EPOCHS" -gt 50 ]; then
    UPLOAD_INTERVAL=$(( EPOCHS / 10 ))
else
    UPLOAD_INTERVAL=5
fi
RESULTS_DIR="$WORKDIR/results"

(
    LAST_UPLOADED=0
    while true; do
        sleep 30
        # Find progress.json anywhere under results dir
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
    output_dir=results
set +f

echo "=== Training complete $(date -Iseconds) ==="
echo "SUCCESS" >> "$LOG"
