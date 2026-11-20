#!/bin/bash
# Cost benchmark + 360-degree comparison video.
# Paper-quality timings need a full A100 (7g.80gb). If it waits too long in the queue, use
#   sbatch --gres=gpu:3g.40gb:1 bench.sbatch      (then timings are "indicative, MIG slice")
#SBATCH --job-name=bench
#SBATCH --partition=stud
#SBATCH --gres=gpu:7g.80gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
cd ~/dl4cv-project/splatter-image
export PATH="$HOME/dl4cv-project/.conda-env/bin:$PATH"
export PYTHONUNBUFFERED=1
export HF_HOME="$HOME/dl4cv-project/.cache/huggingface"
mkdir -p logs results
python benchmark_cost.py --ft_path runs/ft10_hybrid --ft_ckpt model_best.pth --out results/cost
