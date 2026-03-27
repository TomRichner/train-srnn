#!/usr/bin/env bash
# Monitor running experiments
# Usage: ./cloud/monitor.sh <run_name> [experiment] [model]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/config.env"

RUN_NAME="${1:?Usage: $0 <run_name> [experiment] [model]}"
FILTER_EXP="${2:-}"
FILTER_MODEL="${3:-}"

echo "=== Monitor: $RUN_NAME ($(date)) ==="

# Running VMs
echo -e "\n--- Running VMs ---"
gcloud compute instances list --project="$GCP_PROJECT" \
    --filter="name~^${RUN_NAME}-" \
    --format="table(name,zone,machineType.basename(),status,scheduling.preemptible)" 2>/dev/null

# Completion status
echo -e "\n--- Completion Status ---"
printf "%-20s" ""
for model in $ALL_MODELS; do
    [ -n "$FILTER_MODEL" ] && [ "$model" != "$FILTER_MODEL" ] && continue
    printf "%-8s" "${model:0:7}"
done
echo

for exp in $ALL_EXPERIMENTS; do
    [ -n "$FILTER_EXP" ] && [ "$exp" != "$FILTER_EXP" ] && continue
    printf "%-20s" "$exp"

    for model in $ALL_MODELS; do
        [ -n "$FILTER_MODEL" ] && [ "$model" != "$FILTER_MODEL" ] && continue

        # Count completed seeds in GCS
        completed=$(gcloud storage ls "$GCP_BUCKET/results-pytorch/$RUN_NAME/$model/$exp/" 2>/dev/null | grep -c "seed" || echo 0)

        # Check if VM is running
        vm_name="${RUN_NAME}-${model}-${exp}"
        running=$(gcloud compute instances list --project="$GCP_PROJECT" \
            --filter="name~^${vm_name//_/-}" --format="value(name)" 2>/dev/null | wc -l)

        if [ "$completed" -ge 5 ]; then
            printf "%-8s" "done"
        elif [ "$running" -gt 0 ]; then
            printf "%-8s" "${completed}+run"
        elif [ "$completed" -gt 0 ]; then
            printf "%-8s" "${completed}/5"
        else
            printf "%-8s" "."
        fi
    done
    echo
done

echo -e "\nLegend: done=all seeds | N+run=N done+running | N/5=partial | .=not started"
