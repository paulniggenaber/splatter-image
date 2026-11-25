#!/bin/bash
# entirely new code file from Paul
# Quick check that a batch job can use the GPU and the built renderer.
# Run from the repo root: mkdir -p out err && sbatch slurm/check.sh
#SBATCH --job-name=check
#SBATCH --partition=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000
#SBATCH --time=00:10:00
#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err

module load miniconda3
eval "$(conda shell.bash hook)"
conda activate splatter-image
export PYTHONNOUSERSITE=1

echo "Job ${SLURM_JOB_ID} on $(hostname)"
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import diff_gaussian_rasterization; print('Renderer OK')"
