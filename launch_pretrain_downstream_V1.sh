#!/bin/bash

set -euo pipefail

DATASET=OPP
# Lambda values (JEPA-loss weight) to launch. Default list below;
# or override from the command line:  ./launch_pretrain_downstream_V1.sh 0.1 0.3
LAMS=${@:-0 0.1 0.3 0.5 1}

for LAM in $LAMS; do
    for FOLD in 1 2 3 4; do
        echo "Submitting pretraining for lam ${LAM} fold ${FOLD}"

        PRE_ID=$(sbatch --parsable --job-name=pre_lam${LAM}_f${FOLD} pretrain_basilisk.slurm "$DATASET" "$FOLD" "$LAM")
        echo "Pretraining job for lam ${LAM} fold ${FOLD}: ${PRE_ID}"
        echo "Submitting downstream dependent on ${PRE_ID}"

        DOWN_ID=$(sbatch --parsable --job-name=down_lam${LAM}_f${FOLD} --dependency=afterok:${PRE_ID} train_downstream_basilisk.slurm "$DATASET" "$FOLD" "$LAM")
        echo "Downstream job ID for lam ${LAM} fold ${FOLD}: ${DOWN_ID}"
    done
done
