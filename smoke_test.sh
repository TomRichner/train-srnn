#!/usr/bin/env bash
# Smoke test: run each model x task combo for 2 epochs
set -euo pipefail

MODELS="lstm ltc ltc_rk ltc_ex ctrnn node ctgru srnn srnn_no_adapt srnn_e_only"
TASKS="har smnist gesture occupancy traffic power ozone person cheetah"
PASS=0; FAIL=0; ERRORS=""

for model in $MODELS; do
    for task in $TASKS; do
        echo -n "Testing model=$model task=$task ... "
        if python train.py model=$model task=$task epochs=2 size=8 compile=false 2>/dev/null; then
            echo "PASS"; ((PASS++))
        else
            echo "FAIL"; ((FAIL++)); ERRORS="$ERRORS\n  $model x $task"
        fi
    done
done

echo "Results: $PASS passed, $FAIL failed"
[ -n "$ERRORS" ] && echo -e "Failures:$ERRORS"
exit $((FAIL > 0))
