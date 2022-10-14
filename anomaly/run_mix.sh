#!/bin/bash

SEEDS=(0)
MACHINES=("valve" "slider")
GPU=6

export CUDA_VISIBLE_DEVICES=$GPU
PYTHON="/home/kjc/.conda/envs/asteroid/bin/python"

for seed in ${SEEDS[@]}; do
   for MACHINE in ${MACHINES[@]}; do
      sed -i "s/^seed.*/seed: ${seed}/g" baseline.yaml

      RESULT_DIR="result_1015_baseline"
      mkdir -p ${RESULT_DIR}
      sed -i "s/^result_directory.*/result_directory: ${RESULT_DIR}/g" baseline.yaml

      RESULT_NAME="mixture_baseline_${MACHINE}02_seed${seed}.yaml"
      sed -i "s/^result_file.*/result_file: ${RESULT_NAME}/g" baseline.yaml

      sed -i "s/^MACHINE =.*/MACHINE = '${MACHINE}'/g" baseline_mix.py
      sed -i "s/^S1 =.*/S1 = 'id_00'/g" baseline_mix.py
      sed -i "s/^S2 =.*/S2 = 'id_02'/g" baseline_mix.py
      
      $PYTHON baseline_mix.py
   done
done
