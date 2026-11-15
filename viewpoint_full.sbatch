#!/bin/bash
#SBATCH --job-name=viewpoint704
#SBATCH --partition=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=viewpoint704_%j.out

cd ~/dl4cv-project/splatter-image

export PATH="$HOME/dl4cv-project/.conda-env/bin:$PATH"
export VIEWPOINT_MAX_OBJECTS=704
export EVAL_START_IDX=0

python eval_viewpoint_gap.py cars --split test --save_vis 0
