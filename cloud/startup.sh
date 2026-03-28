#!/usr/bin/env bash
# VM startup script — runs on boot to execute a single training run
# Reads experiment config from VM metadata tags
set -euo pipefail

LOG="/var/log/training.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== Startup $(date -Iseconds) ==="

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

echo "Run: $RUN_NAME | Experiment: $EXPERIMENT | Model: $MODEL | Seed: $SEED"

# Cleanup handler
cleanup() {
    local exit_code=$?
    echo "=== Cleanup (exit=$exit_code) $(date -Iseconds) ==="

    # Upload final results
    if [ -d "/tmp/workdir/results" ]; then
        gcloud storage cp -r /tmp/workdir/results/* "$RESULTS_PREFIX/" 2>/dev/null || true
    fi

    # Upload log
    gcloud storage cp "$LOG" "$RESULTS_PREFIX/training_log.txt" 2>/dev/null || true

    # Upload metadata
    python3 -c "
import json, time
json.dump({
    'run_name': '$RUN_NAME', 'experiment': '$EXPERIMENT', 'model': '$MODEL',
    'seed': int('$SEED'), 'exit_code': $exit_code,
    'vm_name': '$VM_NAME', 'completed': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
    'train_args': '$TRAIN_ARGS'
}, open('/tmp/metadata.json', 'w'))
" && gcloud storage cp /tmp/metadata.json "$RESULTS_PREFIX/run_metadata.json" 2>/dev/null || true

    # Self-delete
    gcloud compute instances delete "$VM_NAME" --zone="$GCP_ZONE" --quiet 2>/dev/null || true
}
trap cleanup EXIT

# Step 1: Clone repo
GCP_ZONE=$(curl -sf -H "$META_HEADER" "http://metadata.google.internal/computeMetadata/v1/instance/zone" | rev | cut -d/ -f1 | rev)
REPO_URL="https://github.com/TomRichner/liquid_time_constant_networks.git"
WORKDIR="/tmp/workdir"

for attempt in 1 2 3; do
    if git clone --depth 1 "$REPO_URL" "$WORKDIR" 2>/dev/null; then
        break
    fi
    echo "Clone attempt $attempt failed, retrying in 30s..."
    sleep 30
done
cd "$WORKDIR/pytorch_refactor"
echo "Git commit: $(git rev-parse --short HEAD)"

# Step 2: Download dataset (if needed)
if [ "$EXPERIMENT" != "smnist" ]; then
    mkdir -p "data/$EXPERIMENT"
    gcloud storage cp -r "$BUCKET/datasets/$EXPERIMENT/*" "data/$EXPERIMENT/" || true
fi

# Step 3: Setup Python environment
if [ -d "/opt/python-venv" ]; then
    source /opt/python-venv/bin/activate
    pip install --quiet hydra-core omegaconf 2>/dev/null || true
else
    python3 -m venv /tmp/venv
    source /tmp/venv/bin/activate
    pip install --quiet -r requirements.txt
fi

# Step 4: Run training
echo "=== Training start $(date -Iseconds) ==="
mkdir -p results/$EXPERIMENT

python3 train.py \
    model=$MODEL \
    task=$EXPERIMENT \
    seed=$SEED \
    $TRAIN_ARGS \
    output_dir=results/$EXPERIMENT/${MODEL}_\${size} \
    2>&1 | tee -a "$LOG"

echo "=== Training complete $(date -Iseconds) ==="
echo "SUCCESS" >> "$LOG"
