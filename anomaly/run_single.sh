#!/bin/bash

SEEDS=(0)
MACHINES=("valve" "slider")
GPU=7

export CUDA_VISIBLE_DEVICES=$GPU
PYTHON="/home/kjc/.conda/envs/asteroid/bin/python"

for seed in ${SEEDS[@]}; do
   for MACHINE in ${MACHINES[@]}; do
      sed -i "s/^seed.*/seed: ${seed}/g" baseline.yaml

      RESULT_DIR="result_1015_baseline"
      mkdir -p ${RESULT_DIR}
      sed -i "s/^result_directory.*/result_directory: ${RESULT_DIR}/g" baseline.yaml

      RESULT_NAME="oracle_baseline_${MACHINE}_seed${seed}.yaml"
      sed -i "s/^result_file.*/result_file: ${RESULT_NAME}/g" baseline.yaml

      sed -i "s/^MACHINE =.*/MACHINE = '${MACHINE}'/g" baseline.py

      $PYTHON baseline.py
   done
done
