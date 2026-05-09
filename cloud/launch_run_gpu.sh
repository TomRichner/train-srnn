#!/usr/bin/env bash
# Launch a single GPU training run on GCP (L4 / g2-standard-8)
# Usage: ./cloud/launch_run_gpu.sh <run_name> <experiment> <model> <seed> [extra hydra args...]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/config.gpu.env"

CLEANUP="delete"
BRANCH="main"
while [[ "${1:-}" == --* ]]; do
    case "$1" in
        --cleanup=*) CLEANUP="${1#--cleanup=}"; shift ;;
        --cleanup)   CLEANUP="${2:?--cleanup needs a value}"; shift 2 ;;
        --branch=*)  BRANCH="${1#--branch=}"; shift ;;
        --branch)    BRANCH="${2:?--branch needs a value}"; shift 2 ;;
        *) echo "FATAL: unknown flag '$1'" >&2; exit 1 ;;
    esac
done
case "$CLEANUP" in
    delete|stop|keep) ;;
    *) echo "FATAL: --cleanup must be delete|stop|keep, got '$CLEANUP'" >&2; exit 1 ;;
esac

RUN_NAME="${1:?Usage: $0 [--cleanup=delete|stop|keep] <run_name> <experiment> <model> <seed> [args...]}"
EXPERIMENT="${2:?}"
MODEL="${3:?}"
SEED="${4:?}"
shift 4
EXTRA_ARGS="$*"
# Strip single quotes so Hydra list syntax [a,b] is stored clean in metadata.
# Users type quotes locally (zsh needs them), but the metadata value should not
# contain them — the VM's startup script handles globbing protection.
EXTRA_ARGS="${EXTRA_ARGS//\'/}"

VM_NAME="${RUN_NAME}-${MODEL}-${EXPERIMENT}-seed${SEED}"
VM_NAME="${VM_NAME//_/-}"  # GCP doesn't allow underscores

# Check if VM already exists
if gcloud compute instances describe "$VM_NAME" --zone="$GCP_ZONE" --project="$GCP_PROJECT" &>/dev/null; then
    echo "VM $VM_NAME already exists, skipping"
    exit 0
fi

# Check concurrency (GPU quota is tight — usually 1)
RUNNING=$(gcloud compute instances list --project="$GCP_PROJECT" \
    --filter="name~^${RUN_NAME}-" --format="value(name)" 2>/dev/null | wc -l)
if [ "$RUNNING" -ge "$MAX_CONCURRENT_VMS" ]; then
    echo "Concurrency limit reached ($RUNNING/$MAX_CONCURRENT_VMS), waiting..."
    exit 1
fi

# Determine machine type
MACHINE_TYPE="${DEFAULT_MACHINE_TYPE_FAMILY}-${DEFAULT_MACHINE_TIER}"

# Check for experiment-specific overrides (shared with CPU path).
# Note: the experiments/*.env MACHINE_TIER values target the n4d family
# (e.g. standard-2, highcpu-4) and don't exist in g2. Source for ARGS only
# and keep the GPU default machine type.
if [ -f "$SCRIPT_DIR/experiments/${EXPERIMENT}.env" ]; then
    unset MACHINE_TIER
    source "$SCRIPT_DIR/experiments/${EXPERIMENT}.env"
    MACHINE_TYPE="${DEFAULT_MACHINE_TYPE_FAMILY}-${DEFAULT_MACHINE_TIER}"
    EXTRA_ARGS="${ARGS:-} $EXTRA_ARGS"
fi

echo "Launching $VM_NAME ($MACHINE_TYPE + ${GPU_COUNT}× ${GPU_TYPE})..."

SCHEDULING_ARGS="--maintenance-policy=TERMINATE"
if [ "$GCP_USE_SPOT" = "true" ]; then
    SCHEDULING_ARGS="$SCHEDULING_ARGS --provisioning-model=SPOT --instance-termination-action=STOP"
fi

# Write train-args to a temp file so commas in values (e.g. batched_ablations)
# don't break gcloud's metadata comma-delimited parsing.
TRAIN_ARGS_FILE=$(mktemp)
echo "$EXTRA_ARGS" > "$TRAIN_ARGS_FILE"

gcloud compute instances create "$VM_NAME" \
    --project="$GCP_PROJECT" \
    --zone="$GCP_ZONE" \
    --machine-type="$MACHINE_TYPE" \
    --accelerator="type=${GPU_TYPE},count=${GPU_COUNT}" \
    --image-family="$GCP_IMAGE_FAMILY" \
    --image-project="$GCP_IMAGE_PROJECT" \
    --boot-disk-size="$BOOT_DISK_SIZE" \
    --boot-disk-type="$BOOT_DISK_TYPE" \
    --scopes=cloud-platform \
    --metadata="run-name=$RUN_NAME,experiment=$EXPERIMENT,model=$MODEL,seed=$SEED,bucket=$GCP_BUCKET,cleanup=$CLEANUP,skip-refresh=0,branch=$BRANCH,install-nvidia-driver=True" \
    --metadata-from-file="startup-script=$SCRIPT_DIR/startup_gpu.sh,train-args=$TRAIN_ARGS_FILE" \
    $SCHEDULING_ARGS \
    --quiet

rm -f "$TRAIN_ARGS_FILE"

echo "Created $VM_NAME (cleanup=$CLEANUP)"
if [ "$CLEANUP" != "delete" ]; then
    echo "Subsequent dispatches:"
    echo "  cloud/submit.sh $VM_NAME <run_name> $EXPERIMENT $MODEL <seed> [--cleanup=...] [--skip-refresh] [args...]"
    echo "Manual lifecycle: cloud/stop_vm.sh $VM_NAME  /  cloud/start_vm.sh $VM_NAME"
fi
