#!/bin/bash
# Train the stand-alone mirror-plane predictor on the cached SRN cars and score it on
# validation cars and on test cars at the benchmark input view.
# Run from the repo root:  mkdir -p out err && sbatch slurm/plane.sh
# Options (env): STEPS (default 4000), ARCH (resnet18 | small), TAG (output folder name), CATS ("cars" or "cars chairs")
#SBATCH --job-name=plane
#SBATCH --partition=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32000
#SBATCH --time=01:00:00
#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err

module load miniconda3
eval "$(conda shell.bash hook)"
conda activate splatter-image
export PYTHONNOUSERSITE=1
export SHAPENET_DATASET_ROOT=${SHAPENET_DATASET_ROOT:-$HOME/data/SRN}
export SRN_CACHE_DIR=${SRN_CACHE_DIR:-$HOME/data/SRN_cache}
export PYTHONUNBUFFERED=1

TAG=${TAG:-plane1}
echo "Job ${SLURM_JOB_ID} on $(hostname): plane predictor $TAG"
nvidia-smi --query-gpu=name,memory.total --format=csv | head -2
date
python train_plane.py --out experiments_out/$TAG --arch ${ARCH:-resnet18} --steps ${STEPS:-4000} \
    --workers 4 --test_cars ${TEST_CARS:-100} --categories ${CATS:-cars}
date
