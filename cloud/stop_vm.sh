#!/usr/bin/env bash
# Stop a keep-alive dev VM (preserves disk; ~$0.02/hr while stopped).
# Usage: ./cloud/stop_vm.sh <vm_name>
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

echo "Stopping $VM_NAME in $VM_ZONE..."
gcloud compute instances stop "$VM_NAME" \
    --zone="$VM_ZONE" --project="$GCP_PROJECT" --quiet

echo "Stopped. Restart with: cloud/start_vm.sh $VM_NAME"
