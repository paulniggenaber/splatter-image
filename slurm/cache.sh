#!/bin/bash
# Fill the SRN disk cache (one file per car). CPU only, no GPU needed.
# Without the cache every training job spends about 2 hours re-reading 250,000
# small files from the network drive before it trains at full speed.
# Run from the repo root: mkdir -p out err && sbatch slurm/cache.sh
# Optional: ARGS="--first 400 --splits train" to cache only a part.
#SBATCH --job-name=cache
#SBATCH --partition=stud
#SBATCH --cpus-per-task=8
#SBATCH --mem=16000
#SBATCH --time=01:00:00
#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err

module load miniconda3
eval "$(conda shell.bash hook)"
conda activate splatter-image
export PYTHONNOUSERSITE=1
export SHAPENET_DATASET_ROOT=${SHAPENET_DATASET_ROOT:-$HOME/data/SRN}
export SRN_CACHE_DIR=${SRN_CACHE_DIR:-$HOME/data/SRN_cache}

date
python -m datasets.cache_srn --category cars --workers 8 ${ARGS:---splits train val}
du -sh $SRN_CACHE_DIR
date
