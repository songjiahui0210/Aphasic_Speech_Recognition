#!/usr/bin/env bash

# source activate /home/liu.lian1/envs/pylangacq_env
conda activate /home/liu.lian1/envs/pylangacq_env
python -m debugpy --listen 5678 --wait-for-client training_lora/train_lora.py \
--config-path "conf" \
--config-name "prompt.yaml"

