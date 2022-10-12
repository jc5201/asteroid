#!/bin/bash

SEEDS=(0, 1, 2, 3)

for seed in ${SEEDS[@]}; do
   RESULT_NAME="slider_id00_02_original_seed${seed}.yaml"
   sed -i "s/^seed.*/seed: ${seed}/g" baseline.yaml
   sed -i "s/^result_file.*/result_file: ${RESULT_NAME}/g" baseline.yaml
   python baseline_src_xumx_shared.py
done
