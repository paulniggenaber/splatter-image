#!/bin/bash
# Pilot: a short fine-tune (default 2000 steps on 400 training cars) followed by
# an evaluation on the first 100 test cars. Fits in one hour. It is NOT the real
# run; it only shows which way the numbers move.
#     sbatch --job-name=pilot_sym     --export=ALL,RUN=sym     slurm/pilot.sh
#     sbatch --job-name=pilot_control --export=ALL,RUN=control slurm/pilot.sh
# Another try without overwriting: TAG=sym2 EXTRA="model.symmetry.plane_detach=true"
#SBATCH --partition=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --time=01:00:00
#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err

module load miniconda3
eval "$(conda shell.bash hook)"
conda activate splatter-image
export PYTHONNOUSERSITE=1
export SHAPENET_DATASET_ROOT=${SHAPENET_DATASET_ROOT:-$HOME/data/SRN}
export WANDB_MODE=offline
export SRN_CACHE_DIR=${SRN_CACHE_DIR:-$HOME/data/SRN_cache}   # one file per car, fills itself on first use
export PYTHONUNBUFFERED=1

RUN=${RUN:-sym}
STEPS=${STEPS:-2000}
CKPT=${CKPT:-$HOME/ckpt/cars}
TAG=${TAG:-$RUN}                  # name of this pilot (folders pilot_<TAG>), e.g. TAG=sym2
RUN_DIR=$PWD/experiments_out/pilot_$TAG
RES=results/pilot_$TAG
mkdir -p $RES

if [ "$RUN" = "sym" ]; then
    SYM="model.symmetry.enabled=true"
    MODES="oracle model"
else
    SYM="model.symmetry.enabled=false"
    MODES="model"
fi

echo "Job ${SLURM_JOB_ID} on $(hostname): pilot $RUN, $STEPS steps"
nvidia-smi --query-gpu=name,memory.total --format=csv | head -2
date

python train_network.py +dataset=cars +experiment=lpips_100k \
    opt.pretrained_ckpt=$CKPT opt.base_lr=0.00005 $SYM \
    data.subset=400 opt.iterations=$STEPS \
    logging.val_log=1000000 logging.loop_log=1000000 logging.ckpt_iterations=500 \
    hydra.run.dir=$RUN_DIR $EXTRA \
    | grep --line-buffered -v "starting iteration\|finished forward\|finished opt\|finished iteration"
echo "exit code of training: ${PIPESTATUS[0]}"
date

for MODE in $MODES; do
    echo "=== eval $MODE ==="
    python eval.py cars --experiment_path $RUN_DIR --max_examples ${EVAL_N:-100} \
        --sym_mode $MODE --save_vis 3 --out_folder $RES/renders_$MODE \
        --scores_path $RES/scores_$MODE.json --per_view_path $RES/per_view_$MODE.txt
    cat $RES/scores_$MODE.json
    date
done
