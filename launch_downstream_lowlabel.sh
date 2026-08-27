#!/bin/bash
# Downstream-only label-fraction sweep: reuses the pretrained encoders in JEPA_models_pt/lam<LAM>/.
# Usage:  ./launch_downstream_lowlabel.sh <LAM> [W_PER_CLASS ...]
#   e.g.  ./launch_downstream_lowlabel.sh 0.1              -> W in 2 5 10 25 50 100
#         ./launch_downstream_lowlabel.sh 0.3 2 10 100
# Results: results_json/lam<LAM>_n<W>/OPP_fold*_results.json  (full-label point = results_json/lam<LAM>/)
set -euo pipefail

if [ -z "${1:-}" ]; then echo "Usage: $0 <LAM> [W_PER_CLASS ...]"; exit 1; fi
DATASET=OPP
LAM=$1; shift
WS=${@:-2 5 10 25 50 100}

for W in $WS; do
    for FOLD in 1 2 3 4; do
        if ! ls JEPA_models_pt/lam${LAM}/JEPA_model_OPP_fold${FOLD}_seed*.pt >/dev/null 2>&1; then
            echo "ERROR: no pretrained encoders in JEPA_models_pt/lam${LAM}/ for fold ${FOLD}"; exit 1
        fi
        JID=$(sbatch --parsable --job-name=down_lam${LAM}_n${W}_f${FOLD} train_downstream_basilisk.slurm "$DATASET" "$FOLD" "$LAM" "$W")
        echo "lam ${LAM} | ${W} windows/class | fold ${FOLD} -> job ${JID}"
    done
done
