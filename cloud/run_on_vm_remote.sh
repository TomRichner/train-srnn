#!/usr/bin/env bash
# Runs ON a keep-alive dev VM. Refreshes the repo and executes one training run.
# Invoked by cloud/run_on_vm.sh via gcloud compute ssh.
# Args: <run_name> <experiment> <model> <seed> <bucket> <train_args_file>
set -euo pipefail

RUN_NAME="$1"
EXPERIMENT="$2"
MODEL="$3"
SEED="$4"
BUCKET="$5"
TRAIN_ARGS_FILE="$6"
TRAIN_ARGS=$(cat "$TRAIN_ARGS_FILE")
rm -f "$TRAIN_ARGS_FILE"

WORKDIR="/opt/train-srnn"
RESULTS_PREFIX="$BUCKET/results-pytorch/$RUN_NAME/$MODEL/$EXPERIMENT/seed$SEED"
LOG="/var/log/training-${RUN_NAME}.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== Run $RUN_NAME on $(hostname) at $(date -Iseconds) ==="
echo "  experiment=$EXPERIMENT model=$MODEL seed=$SEED"
echo "  train_args=$TRAIN_ARGS"

# Refresh repo. Re-fetch deploy key per run, scrub after — same pattern as
# startup_gpu.sh, no long-lived key on disk.
SSH_DIR="/root/.ssh"
DEPLOY_KEY="$SSH_DIR/deploy_key"
mkdir -p "$SSH_DIR" && chmod 700 "$SSH_DIR"

GCP_PROJECT=$(curl -sf -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/project/project-id")

if ! gcloud secrets versions access latest \
    --secret="train-srnn-deploy-key" \
    --project="$GCP_PROJECT" > "$DEPLOY_KEY" 2>/dev/null; then
    echo "FATAL: could not fetch deploy key from Secret Manager"
    exit 1
fi
chmod 600 "$DEPLOY_KEY"

ssh-keyscan -t ed25519 github.com >> "$SSH_DIR/known_hosts" 2>/dev/null
cat > "$SSH_DIR/config" <<SSHEOF
Host github.com
    IdentityFile $DEPLOY_KEY
    StrictHostKeyChecking yes
    IdentitiesOnly yes
SSHEOF
chmod 600 "$SSH_DIR/config"

if [ ! -d "$WORKDIR/.git" ]; then
    echo "FATAL: $WORKDIR is not a git checkout. Was this VM launched with --keep-alive?"
    rm -f "$DEPLOY_KEY" "$SSH_DIR/config"
    exit 1
fi

( cd "$WORKDIR" && git fetch --depth 1 origin main && git reset --hard origin/main )
GIT_COMMIT=$( cd "$WORKDIR" && git rev-parse --short HEAD )
echo "Git commit: $GIT_COMMIT"

# Scrub deploy key
rm -f "$DEPLOY_KEY" "$SSH_DIR/config"

# Refresh dataset (idempotent — gcloud storage cp overwrites)
mkdir -p "$WORKDIR/train_srnn/data/$EXPERIMENT"
gcloud storage cp -r "$BUCKET/datasets/$EXPERIMENT/*" \
    "$WORKDIR/train_srnn/data/$EXPERIMENT/" 2>/dev/null || true

# Ensure deps are present (no-op if already installed)
sudo pip3 install --quiet hydra-core omegaconf h5py scipy pandas

# Sanity-check CUDA before training
python3 -c "import torch; assert torch.cuda.is_available()" || {
    echo "FATAL: CUDA not available on this VM"
    exit 1
}

cd "$WORKDIR"
export PYTHONPATH="$WORKDIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1   # flush logs in real time (stdout is piped via tee)
RUN_OUTPUT="results/$EXPERIMENT/${RUN_NAME}_seed${SEED}"
mkdir -p "$RUN_OUTPUT"

START_TIME=$(date +%s)
START_TIME_ISO=$(date -u +%Y-%m-%dT%H:%M:%SZ)

# set -f protects Hydra list syntax [a,b,c] from shell globbing
set -f
EXIT_CODE=0
python3 train.py task=$EXPERIMENT model=$MODEL seed=$SEED $TRAIN_ARGS \
    output_dir="$RUN_OUTPUT" || EXIT_CODE=$?
set +f

END_TIME=$(date +%s)
DURATION=$(( END_TIME - START_TIME ))

# Upload results, log, and metadata (mirrors startup_gpu.sh layout)
gcloud storage cp -r "$RUN_OUTPUT/*" "$RESULTS_PREFIX/" 2>/dev/null || true
gcloud storage cp "$LOG" "$RESULTS_PREFIX/training_log.txt" 2>/dev/null || true

python3 -c "
import json, time
meta = {
    'run_name': '$RUN_NAME', 'experiment': '$EXPERIMENT', 'model': '$MODEL',
    'seed': int('$SEED'), 'exit_code': $EXIT_CODE,
    'vm_name': '$(hostname)',
    'hardware': 'gpu-keepalive',
    'start_time': '$START_TIME_ISO',
    'completed': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
    'duration_seconds': $DURATION,
    'commit': '$GIT_COMMIT',
    'train_args': '''$TRAIN_ARGS''',
}
if $EXIT_CODE != 0:
    meta['error'] = True
json.dump(meta, open('/tmp/metadata.json', 'w'), indent=2)
" && gcloud storage cp /tmp/metadata.json "$RESULTS_PREFIX/run_metadata.json" 2>/dev/null || true

echo "=== Done (exit=$EXIT_CODE, duration=${DURATION}s). Results: $RESULTS_PREFIX ==="
exit $EXIT_CODE
