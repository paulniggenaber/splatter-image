#!/bin/bash
# Level 1 on all 704 test cars, split in 4 array jobs of 176 objects.
#   smoke test first (interactive GPU shell):
#     python eval_budget.py cars --max_objects 3 --out results/smoke
#   then:  sbatch budget_eval.sbatch
#SBATCH --job-name=budget_eval
#SBATCH --partition=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/budget_eval_%A_%a.out
set -euo pipefail
cd ~/dl4cv-project/splatter-image
export PATH="$HOME/dl4cv-project/.conda-env/bin:$PATH"
export PYTHONUNBUFFERED=1
export HF_HOME="$HOME/dl4cv-project/.cache/huggingface"
mkdir -p results logs
START=$(( SLURM_ARRAY_TASK_ID * 176 ))
python eval_budget.py cars --start_idx $START --max_objects 176 \
    --criteria random opacity scale opacity_area edge_opacity contribution error_sens error_sens_signed merge --out results/budget_full_part${SLURM_ARRAY_TASK_ID}
