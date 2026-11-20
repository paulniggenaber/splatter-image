#!/bin/bash
# Level 3 / 3b fine-tuning on Taya's account. Pass the run configuration in ARGS, e.g.
#   sbatch --export=ALL,ARGS="--mask learned --keep 0.10 --hours 10 --out runs/ft10_learned" finetune.sbatch
# The script stops itself after --hours and saves model_best / model_latest; if a job is killed,
# resubmitting the same command resumes from runs/<name>/train_state.pth.
#SBATCH --job-name=ft
#SBATCH --partition=stud
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=11:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
cd ~/dl4cv-project/splatter-image
export PATH="$HOME/dl4cv-project/.conda-env/bin:$PATH"
export PYTHONUNBUFFERED=1
export HF_HOME="$HOME/dl4cv-project/.cache/huggingface"
mkdir -p logs runs
python finetune_budget.py --num_workers 5 $ARGS
