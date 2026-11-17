#!/bin/bash
# Evaluate a trained run on the test cars, with near-side / hidden-side scores.
#     sbatch --export=ALL,RUN_DIR=$PWD/experiments_out/sym,MODES="model oracle" slurm/eval.sh
#     sbatch --export=ALL,RUN_DIR=$PWD/experiments_out/control slurm/eval.sh
# RUN_DIR  folder of the run (contains model_latest.pth and .hydra/)
# MODES    model  = the model as it is (symmetry model: its own predicted plane)
#          oracle = symmetry model mirroring with the TRUE plane (ablation)
# EVAL_EXTRA  extra eval.py flags, e.g. "--plane_predictor experiments_out/plane1/plane_predictor.pth"
# RES_NAME    results folder name (default: name of the run folder)
# N        number of test cars (default: all 704, about 55 minutes per mode)
# Results land in results/<name of the run>/.
#SBATCH --job-name=eval
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
export PYTHONUNBUFFERED=1

RUN_DIR=${RUN_DIR:?set RUN_DIR}
MODES=${MODES:-model}
RES=results/${RES_NAME:-$(basename $RUN_DIR)}
mkdir -p $RES
if [ -n "$N" ]; then LIMIT="--max_examples $N"; else LIMIT=""; fi

echo "Job ${SLURM_JOB_ID} on $(hostname): eval $RUN_DIR, modes: $MODES, N=${N:-all}"
for MODE in $MODES; do
    echo "=== eval $MODE ==="; date
    python eval.py ${DATASET:-cars} --experiment_path $RUN_DIR $LIMIT \
        --sym_mode $MODE $EVAL_EXTRA --save_vis 3 --out_folder $RES/renders_$MODE \
        --scores_path $RES/scores_$MODE.json --per_view_path $RES/per_view_$MODE.txt
    cp scores.txt $RES/per_object_$MODE.txt
    cat $RES/scores_$MODE.json
done
date
