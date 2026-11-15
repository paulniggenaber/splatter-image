#!/bin/bash
# Go/no-go check for the symmetry idea, no training involved.
# Released cars model, first N test cars, three ways:
#   model        = the released model as it is
#   twin_all     = + a mirrored twin for every Gaussian across the TRUE plane
#   replace_far  = keep the camera-side half, build the other half by mirroring
# Scores are split into near-side and hidden-side views.
# Run from the repo root: mkdir -p out err && sbatch slurm/gonogo.sh
#SBATCH --job-name=gonogo
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
export SHAPENET_DATASET_ROOT=$HOME/data/SRN

N=${N:-100}
CKPT=${CKPT:-$HOME/ckpt/cars}
RES=results/gonogo
mkdir -p $RES

echo "Job ${SLURM_JOB_ID} on $(hostname), N=$N"
nvidia-smi --query-gpu=name,memory.total --format=csv

for MODE in model twin_all replace_far; do
    echo "=== $MODE ==="
    date
    python eval.py cars --experiment_path $CKPT --max_examples $N \
        --sym_mode $MODE --save_vis 3 --out_folder $RES/renders_$MODE \
        --scores_path $RES/scores_$MODE.json --per_view_path $RES/per_view_$MODE.txt
    cp scores.txt $RES/per_object_$MODE.txt
    cat $RES/scores_$MODE.json
done
date
