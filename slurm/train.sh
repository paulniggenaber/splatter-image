#!/bin/bash
# Fine-tune the released cars model, with or without the symmetry extension.
#
# Run from the repo root:
#     mkdir -p out err
#     sbatch --job-name=sym     --export=ALL,RUN=sym     slurm/train.sh
#     sbatch --job-name=control --export=ALL,RUN=control slurm/train.sh
#
# RUN=sym      symmetry extension on
# RUN=control  exactly the same training without it (the fair baseline)
#
# The partition allows 24 h per job. If a run is not finished by then, submit the
# same command again: it continues from experiments_out/<RUN>/model_latest.pth
# (saved every 1000 steps).
#
# Optional: EXTRA="opt.iterations=50001" to pass more Hydra overrides.
#SBATCH --partition=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64000
#SBATCH --time=1-00:00:00
#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err

module load miniconda3
eval "$(conda shell.bash hook)"
conda activate splatter-image
export PYTHONNOUSERSITE=1
export SHAPENET_DATASET_ROOT=${SHAPENET_DATASET_ROOT:-$HOME/data/SRN}
export WANDB_MODE=offline            # GPU nodes have no internet; sync later with `wandb sync`

RUN=${RUN:-sym}
CKPT=${CKPT:-$HOME/ckpt/cars}        # folder that contains model_latest.pth (released cars model)
RUN_DIR=$PWD/experiments_out/$RUN    # fixed folder = a resubmitted job resumes

if [ "$RUN" = "sym" ]; then
    SYM="model.symmetry.enabled=true"
elif [ "$RUN" = "control" ]; then
    SYM="model.symmetry.enabled=false"
else
    echo "RUN must be sym or control"; exit 1
fi

echo "Job ${SLURM_JOB_ID} on $(hostname): RUN=$RUN, dir $RUN_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv
date

# Same settings for both runs. lpips_100k = the paper's fine-tune stage
# (100,001 steps, LPIPS weight 0.01); base_lr raised to the paper's stage-1 rate
# so the network can adapt to the twins.
python train_network.py +dataset=cars +experiment=lpips_100k \
    opt.pretrained_ckpt=$CKPT \
    opt.base_lr=0.00005 \
    $SYM \
    hydra.run.dir=$RUN_DIR \
    $EXTRA
date
