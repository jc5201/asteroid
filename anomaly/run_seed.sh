#!/bin/bash

SEEDS=(42 10 1 100 20 27 50 30 14 5)

for seed in ${SEEDS[@]}; do
   RESULT_NAME="slider_id00_02_original_seed${seed}.yaml"
   sed -i "s/^seed.*/seed: ${seed}/g" baseline.yaml
   sed -i "s/^result_file.*/result_file: ${RESULT_NAME}/g" baseline.yaml
   python baseline_src_xumx_original.py
done
