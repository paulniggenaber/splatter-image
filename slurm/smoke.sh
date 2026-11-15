#!/bin/bash
# Test C: short smoke runs of the training loop, symmetry and control.
# Small data subset, and validation / video / checkpoint every 100 steps, so
# every code path of a real run is exercised once. Prints s/it and GPU memory.
# Run from the repo root: mkdir -p out err && sbatch slurm/smoke.sh
#SBATCH --job-name=smoke
#SBATCH --partition=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --time=01:00:00
#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err

module load miniconda3
eval "$(conda shell.bash hook)"
conda activate splatter-image
export PYTHONNOUSERSITE=1
export SHAPENET_DATASET_ROOT=${SHAPENET_DATASET_ROOT:-$HOME/data/SRN}
export WANDB_MODE=offline

CKPT=${CKPT:-$HOME/ckpt/cars}
STEPS=${STEPS:-200}
TAG=${TAG:-$SLURM_JOB_ID}

echo "Job ${SLURM_JOB_ID} on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv

COMMON="+dataset=cars +experiment=lpips_100k opt.pretrained_ckpt=$CKPT opt.base_lr=0.00005 \
    data.subset=24 opt.iterations=$STEPS \
    logging.val_log=100 logging.loop_log=100 logging.ckpt_iterations=100"

for RUN in sym control; do
    echo "=== smoke $RUN ==="; date
    if [ "$RUN" = "sym" ]; then
        # warm-up of 100 steps, so both the true-plane and the predicted-plane path run
        SYM="model.symmetry.enabled=true model.symmetry.gt_plane_iters=100"
    else
        SYM="model.symmetry.enabled=false"
    fi
    python train_network.py $COMMON $SYM hydra.run.dir=$PWD/experiments_out/smoke_${TAG}_$RUN \
        | grep -v "starting iteration\|finished forward\|finished opt\|finished iteration"
    echo "exit code of $RUN: ${PIPESTATUS[0]}"
    ls -la experiments_out/smoke_${TAG}_$RUN/ | grep -v "^total"
done
date
