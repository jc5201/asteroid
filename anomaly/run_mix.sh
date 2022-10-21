#!/bin/bash
set -e

SEEDS=(42 10 1 100 20 27 50 30 14 5)
MACHINES=("valve" "slider")
GPU=6

export CUDA_VISIBLE_DEVICES=$GPU
PYTHON="/home/kjc/.conda/envs/asteroid/bin/python"
PYTHON_FILE="baseline_mix.py"

for seed in ${SEEDS[@]}; do
   for MACHINE in ${MACHINES[@]}; do
      sed -i "s@^seed.*@seed: ${seed}@g" baseline.yaml

      RESULT_DIR="result_1022_dilate_label"
      mkdir -p ${RESULT_DIR}
      sed -i "s@^result_directory.*@result_directory: ${RESULT_DIR}@g" baseline.yaml

      RESULT_NAME="mixture_baseline_${MACHINE}_seed${seed}.yaml"
      sed -i "s@^result_file.*@result_file: ${RESULT_NAME}@g" baseline.yaml

      sed -i "s@^MACHINE =.*@MACHINE = '${MACHINE}'@g" ${PYTHON_FILE}
      sed -i "s@^S1 =.*@S1 = 'id_00'@g" ${PYTHON_FILE}
      sed -i "s@^S2 =.*@S2 = 'id_02'@g" ${PYTHON_FILE}
      
      $PYTHON ${PYTHON_FILE}
   done
done
