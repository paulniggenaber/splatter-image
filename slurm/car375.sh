#!/bin/bash
# Test car 375 (876d92ce6a0e4bf399588eee976baae) has only blank white images.
# 1) Score every cars model/mode on that car alone, so it can be subtracted from the 704-car means.
# 2) Re-evaluate pure mirroring (predicted plane) on the other 703 cars: its car-375 PSNR is infinite.
#     sbatch --job-name=car375 slurm/car375.sh
#SBATCH --partition=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --time=02:30:00
#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err

module load miniconda3
eval "$(conda shell.bash hook)"
conda activate splatter-image
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export SHAPENET_DATASET_ROOT=${SHAPENET_DATASET_ROOT:-$HOME/data/SRN}

CAR=876d92ce6a0e4bf399588eee976baae
E=$PWD/experiments_out
R=results/car375; mkdir -p $R

one() {  # $1 name, $2 run dir, $3 mode
    PER_OBJECT_PATH=$R/per_object_$1_$3.txt python eval.py cars --experiment_path $2 \
        --sym_mode $3 --only_ids $CAR --scores_path $R/$1_$3.json
}
one released $HOME/ckpt/cars model
one control_long $E/control_long model
for m in model oracle twins_off oracle_on colour_off replace_far twin_all; do one sym_long $E/sym_long $m; done
for m in model oracle; do one sym_fargate $E/sym_fargate $m; done
one sym_purem $E/sym_purem oracle
echo "=== single-car scores"; for f in $R/*.json; do echo "$f $(tr -d '\n ' < $f)"; done

# pure mirroring, predicted plane, on the 703 other cars
RES=results/sym_purem_40k_n703; mkdir -p $RES
PER_OBJECT_PATH=$RES/per_object_model.txt python eval.py cars --experiment_path $E/sym_purem \
    --sym_mode model --exclude_ids $CAR --scores_path $RES/scores_model.json
cat $RES/scores_model.json
date
