#!/bin/bash
# Cache all SRN training chairs in chunks and free the raw images of every cached chair.
# Run on the login node:  ssh hpc 'bash -s' < slurm/cache_chairs_full.sh
#
# Why: the disk quota (50 GB) cannot hold raw images (many small files) plus cache for
# 4,612 chairs. Once a chair is in the cache, training reads only its cache file and its
# intrinsics.txt (datasets/srn.py), so rgb/ and pose/ can go. A full copy stays on Paul's Mac.
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate splatter-image
export PYTHONNOUSERSITE=1 SHAPENET_DATASET_ROOT=$HOME/data/SRN SRN_CACHE_DIR=$HOME/data/SRN_cache
cd $HOME/splatter-image-night
RAW=$HOME/data/SRN/srn_chairs/chairs_train
CACHE=$(ls -d $SRN_CACHE_DIR/srn_chairs_*)/train
for FIRST in 2600 3200 3800 4400 4612; do
    nice -n 10 python -m datasets.cache_srn --category chairs --splits train --workers 4 --first $FIRST
    freed=0
    for d in $RAW/*/; do
        id=$(basename $d)
        if [ -s "$CACHE/$id.pt" ] && [ -d "$d/rgb" ]; then
            rm -rf "$d/rgb" "$d/pose"; freed=$((freed+1))
        fi
    done
    echo "chunk up to $FIRST: cache files $(ls $CACHE | wc -l), raw folders freed now $freed"
done
echo CACHE_ALL_DONE
