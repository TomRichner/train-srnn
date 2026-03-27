#!/usr/bin/env bash
# Launch a single training run on GCP
# Usage: ./cloud/launch_run.sh <run_name> <experiment> <model> <seed> [extra hydra args...]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/config.env"

RUN_NAME="${1:?Usage: $0 <run_name> <experiment> <model> <seed> [args...]}"
EXPERIMENT="${2:?}"
MODEL="${3:?}"
SEED="${4:?}"
shift 4
EXTRA_ARGS="$*"

VM_NAME="${RUN_NAME}-${MODEL}-${EXPERIMENT}-seed${SEED}"
VM_NAME="${VM_NAME//_/-}"  # GCP doesn't allow underscores

# Check if VM already exists
if gcloud compute instances describe "$VM_NAME" --zone="$GCP_ZONE" --project="$GCP_PROJECT" &>/dev/null; then
    echo "VM $VM_NAME already exists, skipping"
    exit 0
fi

# Check concurrency
RUNNING=$(gcloud compute instances list --project="$GCP_PROJECT" \
    --filter="name~^${RUN_NAME}-" --format="value(name)" 2>/dev/null | wc -l)
if [ "$RUNNING" -ge "$MAX_CONCURRENT_VMS" ]; then
    echo "Concurrency limit reached ($RUNNING/$MAX_CONCURRENT_VMS), waiting..."
    exit 1
fi

# Determine machine type
MACHINE_TYPE="${DEFAULT_MACHINE_TYPE_FAMILY}-${DEFAULT_MACHINE_TIER}"

# Check for experiment-specific overrides
if [ -f "$SCRIPT_DIR/experiments/${EXPERIMENT}.env" ]; then
    source "$SCRIPT_DIR/experiments/${EXPERIMENT}.env"
    MACHINE_TYPE="${DEFAULT_MACHINE_TYPE_FAMILY}-${MACHINE_TIER:-$DEFAULT_MACHINE_TIER}"
    EXTRA_ARGS="${ARGS:-} $EXTRA_ARGS"
fi

echo "Launching $VM_NAME ($MACHINE_TYPE)..."

SCHEDULING_ARGS=""
if [ "$GCP_USE_SPOT" = "true" ]; then
    SCHEDULING_ARGS="--provisioning-model=SPOT --instance-termination-action=STOP"
fi

gcloud compute instances create "$VM_NAME" \
    --project="$GCP_PROJECT" \
    --zone="$GCP_ZONE" \
    --machine-type="$MACHINE_TYPE" \
    --image-family="$GCP_IMAGE_FAMILY" \
    --image-project="$GCP_PROJECT" \
    --boot-disk-size="$BOOT_DISK_SIZE" \
    --boot-disk-type="$BOOT_DISK_TYPE" \
    --scopes=storage-full \
    --metadata="run-name=$RUN_NAME,experiment=$EXPERIMENT,model=$MODEL,seed=$SEED,bucket=$GCP_BUCKET,train-args=$EXTRA_ARGS" \
    --metadata-from-file=startup-script="$SCRIPT_DIR/startup.sh" \
    $SCHEDULING_ARGS \
    --quiet

echo "Created $VM_NAME"
