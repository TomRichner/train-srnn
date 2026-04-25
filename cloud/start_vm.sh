#!/usr/bin/env bash
# Start a previously stopped keep-alive dev VM.
# Capacity is re-checked at start time — same risk as a fresh launch.
# Usage: ./cloud/start_vm.sh <vm_name>
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/config.gpu.env"

VM_NAME="${1:?Usage: $0 <vm_name>}"

VM_ZONE=$(gcloud compute instances list --project="$GCP_PROJECT" \
    --filter="name=$VM_NAME" --format="value(zone)" 2>/dev/null | head -1)
if [ -z "$VM_ZONE" ]; then
    echo "VM $VM_NAME not found" >&2
    exit 1
fi
VM_ZONE=$(basename "$VM_ZONE")

echo "Starting $VM_NAME in $VM_ZONE..."
gcloud compute instances start "$VM_NAME" \
    --zone="$VM_ZONE" --project="$GCP_PROJECT" --quiet

STATUS=$(gcloud compute instances describe "$VM_NAME" \
    --zone="$VM_ZONE" --project="$GCP_PROJECT" --format="value(status)")
echo "Status: $STATUS"
echo "SSH:    gcloud compute ssh $VM_NAME --zone=$VM_ZONE --project=$GCP_PROJECT"
echo "Re-run: cloud/run_on_vm.sh $VM_NAME <run_name> <experiment> <model> <seed> [args...]"
