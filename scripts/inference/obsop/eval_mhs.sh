#!/bin/bash

# export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

python inference/obsoperator.py \
    --obs_name=mhs \
    --model_name=train_mhs_obsop_20260610 \
    --start_year=2022 \
    --end_year=2023 \
    --debug=False
    # > slurmlogs/eval_pretrain_xichen_state_forecast_20260526.log 2>&1