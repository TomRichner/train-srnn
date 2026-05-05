#!/usr/bin/env bash
# Overnight: two 4-hour production runs back-to-back, with postprocess
# between them.
#
#   Run 1: K=6, B=48, bf16, lr=5e-4, no closed-loop, W-hoist (current main).
#   Run 2: same + closed_loop.enabled=true (default alpha_baseline=0.3).
#
# Epoch budget at K=6 B=48 size=300 on L4 GPU after the W-hoist fix
# (commit 47f0da8):
#   - train-only epoch:  ~17.3 s
#   - eval epoch (every checkpoint_interval=12 epochs): +22 s for
#     valid+test eval
#   - average steady-state:  ~19.1 s/epoch
#   - epoch 0 (compile):  ~40 s extra
#
#   4 h = 14,400 s  →  ~750 epochs no-CL
#                  →  ~660 epochs closed-loop (per-step readout costs more)
#
# Set EPOCHS_NOCL and EPOCHS_CL below if you want different bounds.
# Defaults aim each run at ≤ 4 h with a small safety margin.

set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
VM_NAME="continuous-k2-smoke-srnn-seeg-seed1"
VM_ZONE="us-central1-a"
GCP_PROJECT="liquidneuralnets"

REPO_DIR="/Users/richner.thomas/Desktop/MayoVertex/train-srnn"
VENV="/Users/richner.thomas/Desktop/local_venv/srnn-train/.venv"

RUN1_NAME="overnight-run1-Whoist"
RUN2_NAME="overnight-run2-Whoist-cl"

# Epoch budgets, sized for ~4h each. Tune freely.
EPOCHS_NOCL=700   # ≈ 3.7 h at 19.1 s/epoch.
EPOCHS_CL=700     # closed-loop is slightly slower per epoch; ≈ 4.2 h.
                  # Matched epoch counts so the loss curves overlay cleanly.

ABLATIONS="batched_ablations=[srnn-e-only-per-neuron,srnn-e-only-skip-per-neuron,srnn-sfa-e-only-per-neuron,srnn-sfa-e-only-skip-per-neuron,srnn-std-e-only-per-neuron,srnn-std-e-only-skip-per-neuron]"

# Common knobs (rest come from conf/config.yaml: compile=true, compile_cell=true,
# grad_checkpoint=true, grad_checkpoint_segment_len=5, etc.)
COMMON_ARGS="batch_size=48 amp=bf16 continuous_profile=true lr=5e-4 ${ABLATIONS}"

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
log() { printf '[%s] %s\n' "$(date +%Y-%m-%dT%H:%M:%S%z)" "$*"; }

# Robust SSH wrapper — retries on transient gcloud/IAP failures (we hit
# these regularly when SSH'ing tens of times in a session).
ssh_with_retry() {
    local cmd="$1"
    local tries=0
    until [ $tries -ge 5 ]; do
        if out=$(gcloud compute ssh "$VM_NAME" --zone="$VM_ZONE" \
                --project="$GCP_PROJECT" --command="$cmd" 2>&1); then
            echo "$out"
            return 0
        fi
        tries=$((tries + 1))
        sleep 30
    done
    echo "[ssh_with_retry] gave up after $tries attempts" >&2
    return 1
}

wait_for_run() {
    local run_name="$1"
    local log_file="/var/log/training-${run_name}-1.log"
    log "waiting for ${run_name} to finish (polling every 60s)..."
    while true; do
        if ssh_with_retry "sudo grep -qE 'Training complete|exit=' $log_file 2>/dev/null && echo done"  \
                | grep -q "done"; then
            break
        fi
        sleep 60
    done
    # Report final status line
    local tail_out
    tail_out=$(ssh_with_retry "sudo tail -3 $log_file" || true)
    log "${run_name} terminal lines:"
    printf '%s\n' "$tail_out"
}

dispatch() {
    local run_name="$1"
    local extra_args="$2"
    local epochs="$3"
    log "dispatching ${run_name} (epochs=${epochs})..."
    bash "$REPO_DIR/cloud/submit.sh" "$VM_NAME" "$run_name" seeg srnn 1 \
        "epochs=${epochs} ${COMMON_ARGS} ${extra_args}" --skip-refresh
}

postprocess() {
    local run_name="$1"
    log "postprocess ${run_name}..."
    cd "$REPO_DIR"
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
    # Tail a few lines so the orchestration log doesn't get spammed by all
    # the per-variant plot writes; full output is in stdout-buffered tools.
    python scripts/postprocess.py "$run_name" --task seeg --seed 1 2>&1 \
        | tail -40
    log "postprocess ${run_name} done"
}

# -----------------------------------------------------------------------------
# Run 1: W-hoist, no closed-loop
# -----------------------------------------------------------------------------
log "=== overnight orchestration starting ==="
log "run1=${RUN1_NAME} epochs=${EPOCHS_NOCL}"
log "run2=${RUN2_NAME} epochs=${EPOCHS_CL}  closed_loop.enabled=true"

dispatch "$RUN1_NAME" "closed_loop.enabled=false" "$EPOCHS_NOCL"
wait_for_run "$RUN1_NAME"
postprocess "$RUN1_NAME"

# -----------------------------------------------------------------------------
# Run 2: W-hoist + closed-loop
# -----------------------------------------------------------------------------
# alpha_baseline / teacher_forcing_batch_frac / etc come from conf/config.yaml
# (defaults: alpha_baseline=0.3, teacher_forcing_batch_frac=0.2).
dispatch "$RUN2_NAME" "closed_loop.enabled=true" "$EPOCHS_CL"
wait_for_run "$RUN2_NAME"
postprocess "$RUN2_NAME"

log "=== all done ==="
