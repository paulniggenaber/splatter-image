#!/bin/bash
# entirely new code file from Paul
# Run on the cluster LOGIN node (needs internet), not as a job:
#   ssh hpc 'bash -s' < slurm/chairs_full.sh
# Downloads the SRN chairs zip again (3.9 GB), extracts ALL 4085 training chairs
# (plus 100 val and 100 test, as before), then deletes the zip.
# Peak extra disk use about 5 GB, afterwards about 0.8 GB more than now.
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate splatter-image
export PYTHONNOUSERSITE=1
cd $HOME/splatter-image-night
~/gdown-venv/bin/gdown "https://drive.google.com/uc?id=1VWtRZkC4_ON6EBhNNBTag-BwsjPB2Yc8" -O $HOME/data/SRN/srn_chairs.zip && echo DOWNLOADED
python -m datasets.extract_srn_zip $HOME/data/SRN/srn_chairs.zip $HOME/data/SRN --category chairs \
    --train 4085 --val 100 --test 100 --workers 3 && echo EXTRACTED \
    && rm -f $HOME/data/SRN/srn_chairs.zip && echo ZIP_REMOVED
