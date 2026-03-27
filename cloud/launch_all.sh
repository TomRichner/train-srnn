#!/usr/bin/env bash
# Launch all experiments: models × tasks × seeds
# Usage: ./cloud/launch_all.sh <run_name> [--seeds N] [--models "m1 m2"] [--dry-run] [extra hydra args]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/config.env"

RUN_NAME="${1:?Usage: $0 <run_name> [--seeds N] [--models 'm1 m2'] [--dry-run]}"
shift

# Parse args
N_SEEDS=5
MODELS="$ALL_MODELS"
DRY_RUN=false
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --seeds) N_SEEDS="$2"; shift 2 ;;
        --models) MODELS="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

# Build job list
JOBS=()
for exp in $ALL_EXPERIMENTS; do
    for model in $MODELS; do
        for seed in $(seq 1 "$N_SEEDS"); do
            JOBS+=("$exp $model $seed")
        done
    done
done

echo "Total jobs: ${#JOBS[@]} ($N_SEEDS seeds × $(echo $MODELS | wc -w) models × $(echo $ALL_EXPERIMENTS | wc -w) experiments)"
$DRY_RUN && { echo "(dry run)"; for j in "${JOBS[@]}"; do echo "  $j"; done; exit 0; }

# Phase 1: Burst launch
BATCH_SIZE=8
launched=0
for job in "${JOBS[@]}"; do
    read -r exp model seed <<< "$job"
    "$SCRIPT_DIR/launch_run.sh" "$RUN_NAME" "$exp" "$model" "$seed" $EXTRA_ARGS &
    ((launched++))

    if ((launched % BATCH_SIZE == 0)); then
        wait
        echo "Launched $launched/${#JOBS[@]}, pausing 15s..."
        sleep 15
    fi

    # Check concurrency
    RUNNING=$(gcloud compute instances list --project="$GCP_PROJECT" \
        --filter="name~^${RUN_NAME}-" --format="value(name)" 2>/dev/null | wc -l)
    if [ "$RUNNING" -ge "$MAX_CONCURRENT_VMS" ]; then
        echo "Hit concurrency limit, switching to sequential mode..."
        break
    fi
done
wait

# Phase 2: Sequential (poll for free slots)
for job in "${JOBS[@]:$launched}"; do
    read -r exp model seed <<< "$job"

    while true; do
        RUNNING=$(gcloud compute instances list --project="$GCP_PROJECT" \
            --filter="name~^${RUN_NAME}-" --format="value(name)" 2>/dev/null | wc -l)
        [ "$RUNNING" -lt "$MAX_CONCURRENT_VMS" ] && break
        sleep 30
    done

    "$SCRIPT_DIR/launch_run.sh" "$RUN_NAME" "$exp" "$model" "$seed" $EXTRA_ARGS
done

echo "All ${#JOBS[@]} jobs launched."
