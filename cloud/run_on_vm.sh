#!/usr/bin/env bash
# Re-run training on an existing keep-alive GPU dev VM.
# Usage: ./cloud/run_on_vm.sh <vm_name> <run_name> <experiment> <model> <seed> [extra hydra args...]
#
# Assumes the target VM was launched with `launch_run_gpu.sh --keep-alive ...`,
# so /opt/train-srnn already exists and the DLVM driver is up.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/config.gpu.env"

VM_NAME="${1:?Usage: $0 <vm_name> <run_name> <experiment> <model> <seed> [args...]}"
RUN_NAME="${2:?}"
EXPERIMENT="${3:?}"
MODEL="${4:?}"
SEED="${5:?}"
shift 5
EXTRA_ARGS="${*//\'/}"

# Source experiment env for ARGS prefix (mirrors launch_run_gpu.sh)
if [ -f "$SCRIPT_DIR/experiments/${EXPERIMENT}.env" ]; then
    unset MACHINE_TIER
    source "$SCRIPT_DIR/experiments/${EXPERIMENT}.env"
    EXTRA_ARGS="${ARGS:-} $EXTRA_ARGS"
fi

# Resolve VM zone (the VM may have been launched in a different zone than the
# config default).
VM_ZONE=$(gcloud compute instances list --project="$GCP_PROJECT" \
    --filter="name=$VM_NAME" --format="value(zone)" 2>/dev/null | head -1)
if [ -z "$VM_ZONE" ]; then
    echo "FATAL: VM $VM_NAME not found in project $GCP_PROJECT" >&2
    exit 1
fi
VM_ZONE=$(basename "$VM_ZONE")

VM_STATUS=$(gcloud compute instances describe "$VM_NAME" \
    --zone="$VM_ZONE" --project="$GCP_PROJECT" \
    --format="value(status)" 2>/dev/null)
if [ "$VM_STATUS" != "RUNNING" ]; then
    echo "FATAL: VM $VM_NAME is $VM_STATUS (need RUNNING). Try cloud/start_vm.sh $VM_NAME" >&2
    exit 1
fi

TRAIN_ARGS_FILE=$(mktemp)
echo "$EXTRA_ARGS" > "$TRAIN_ARGS_FILE"
REMOTE_ARGS_NAME="train-args-$(date +%s)-$$.txt"

echo "Pushing runner + train-args to $VM_NAME ($VM_ZONE)..."
gcloud compute scp --zone="$VM_ZONE" --project="$GCP_PROJECT" \
    "$SCRIPT_DIR/run_on_vm_remote.sh" "$VM_NAME:/tmp/run_on_vm_remote.sh"
gcloud compute scp --zone="$VM_ZONE" --project="$GCP_PROJECT" \
    "$TRAIN_ARGS_FILE" "$VM_NAME:/tmp/$REMOTE_ARGS_NAME"
rm -f "$TRAIN_ARGS_FILE"

echo "Running $RUN_NAME on $VM_NAME..."
gcloud compute ssh "$VM_NAME" --zone="$VM_ZONE" --project="$GCP_PROJECT" \
    --command="sudo bash /tmp/run_on_vm_remote.sh '$RUN_NAME' '$EXPERIMENT' '$MODEL' '$SEED' '$GCP_BUCKET' '/tmp/$REMOTE_ARGS_NAME'"

echo "Done. Results: $GCP_BUCKET/results-pytorch/$RUN_NAME/$MODEL/$EXPERIMENT/seed$SEED/"
