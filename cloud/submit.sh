#!/usr/bin/env bash
# Dispatch a training run to an existing keep-alive (or stopped) GPU dev VM.
# VM-native: writes per-run knobs to instance metadata, then `start` (if
# stopped) or `reset` (if running) to re-trigger startup_gpu.sh. No SSH
# session held open by the local laptop while the run is in flight.
#
# Usage:
#   cloud/submit.sh <vm_name> <run_name> <experiment> <model> <seed> \
#       [--cleanup=delete|stop|keep] [--skip-refresh] [extra hydra args...]
#
# Defaults: --cleanup=keep (since you're submitting to a live dev VM,
# you presumably want it to stay live).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/config.gpu.env"

VM_NAME="${1:?Usage: $0 <vm_name> <run_name> <experiment> <model> <seed> [--cleanup=...] [--skip-refresh] [args...]}"
RUN_NAME="${2:?}"
EXPERIMENT="${3:?}"
MODEL="${4:?}"
SEED="${5:?}"
shift 5

CLEANUP="keep"
SKIP_REFRESH=0
BRANCH="main"
EXTRA_ARGS=""
while [ $# -gt 0 ]; do
    case "$1" in
        --cleanup=*)    CLEANUP="${1#--cleanup=}"; shift ;;
        --cleanup)      CLEANUP="${2:?--cleanup needs a value}"; shift 2 ;;
        --skip-refresh) SKIP_REFRESH=1; shift ;;
        --branch=*)     BRANCH="${1#--branch=}"; shift ;;
        --branch)       BRANCH="${2:?--branch needs a value}"; shift 2 ;;
        *)              EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done
case "$CLEANUP" in
    delete|stop|keep) ;;
    *) echo "FATAL: --cleanup must be delete|stop|keep, got '$CLEANUP'" >&2; exit 1 ;;
esac
EXTRA_ARGS="${EXTRA_ARGS//\'/}"

# Source experiment env for ARGS prefix (mirrors launch_run_gpu.sh)
if [ -f "$SCRIPT_DIR/experiments/${EXPERIMENT}.env" ]; then
    unset MACHINE_TIER
    source "$SCRIPT_DIR/experiments/${EXPERIMENT}.env"
    EXTRA_ARGS="${ARGS:-} $EXTRA_ARGS"
fi

# Locate VM (any zone — VM may have been launched in a different zone than config default)
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

# Write train-args via temp file (commas in [a,b,c] would break --metadata=)
TRAIN_ARGS_FILE=$(mktemp)
echo "$EXTRA_ARGS" > "$TRAIN_ARGS_FILE"

# Update per-run metadata. add-metadata works on RUNNING and TERMINATED VMs.
echo "Updating metadata on $VM_NAME ($VM_ZONE)..."
gcloud compute instances add-metadata "$VM_NAME" \
    --zone="$VM_ZONE" --project="$GCP_PROJECT" \
    --metadata="run-name=$RUN_NAME,experiment=$EXPERIMENT,model=$MODEL,seed=$SEED,bucket=$GCP_BUCKET,cleanup=$CLEANUP,skip-refresh=$SKIP_REFRESH,branch=$BRANCH,repo-url=$REPO_URL" \
    --metadata-from-file="train-args=$TRAIN_ARGS_FILE,startup-script=$SCRIPT_DIR/startup_gpu.sh" --quiet
rm -f "$TRAIN_ARGS_FILE"

# Dispatch: start (cold) or reset (warm). Both re-trigger startup-script.
case "$VM_STATUS" in
    TERMINATED)
        echo "Starting $VM_NAME (was TERMINATED)..."
        gcloud compute instances start "$VM_NAME" \
            --zone="$VM_ZONE" --project="$GCP_PROJECT" --quiet
        ;;
    RUNNING)
        echo "Resetting $VM_NAME (will kill any in-flight job)..."
        gcloud compute instances reset "$VM_NAME" \
            --zone="$VM_ZONE" --project="$GCP_PROJECT" --quiet
        ;;
    *)
        echo "FATAL: VM is $VM_STATUS; cannot dispatch" >&2
        exit 1
        ;;
esac

RESULTS_PREFIX="$GCP_BUCKET/results-pytorch/$RUN_NAME/$MODEL/$EXPERIMENT/seed$SEED"
LOG_PATH="/var/log/training-${RUN_NAME}-${SEED}.log"
echo
echo "Dispatched. cleanup=$CLEANUP skip-refresh=$SKIP_REFRESH branch=$BRANCH"
echo "Results:   $RESULTS_PREFIX/"
echo "Tail log:  gcloud compute ssh $VM_NAME --zone=$VM_ZONE -- sudo tail -f $LOG_PATH"
echo "Status:    gcloud storage cat $RESULTS_PREFIX/run_metadata.json   # appears at end of run"
