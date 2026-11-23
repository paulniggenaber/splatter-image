#!/bin/bash
#SBATCH --job-name=prune50_20
#SBATCH --partition=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=pruning_50_20_%j.out

cd ~/dl4cv-project/splatter-image

export PATH="$HOME/dl4cv-project/.conda-env/bin:$PATH"
export PRUNE_RATIO=0.50
export PRUNING_MAX_OBJECTS=20

python eval_pruning.py cars --split test --save_vis 0
