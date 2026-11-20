#!/bin/bash
# Level 3 on top of Paul's SYMMETRY model (budget fine-tuning of the symmetric Splatter Image).
# Runs from a separate checkout of paul/dev (his model code) + our budget scripts:
#   ~/dl4cv-project/splatter-sym
# Usage: sbatch --export=ALL,ARGS="--init_path runs/sym_long --mask hybrid --keep 0.10 ..." finetune_sym.sbatch
#SBATCH --job-name=ftsym
#SBATCH --partition=stud
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=11:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
cd ~/dl4cv-project/splatter-sym
export PATH="$HOME/dl4cv-project/.conda-env/bin:$PATH"
export PYTHONUNBUFFERED=1
export HF_HOME="$HOME/dl4cv-project/.cache/huggingface"
export SHAPENET_DATASET_ROOT="$HOME/dl4cv-project/data"   # Paul's srn.py reads the data root from here
mkdir -p logs runs
python finetune_budget.py --num_workers 5 $ARGS
