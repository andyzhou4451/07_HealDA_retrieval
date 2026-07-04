#!/bin/bash

# export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

python inference/era5_interp_forecast.py --decorrelation_hours=6 \
    --output_dir=/public/home/wangwuxing01/research/XiChen/data/xichen_results/pretrain_xichen_state_forecast_20260526_interp_0p25deg \
    --forecast_name=pretrain_xichen_state_forecast_20260526 \
    > slurmlogs/eval_pretrain_xichen_state_forecast_20260526_interp_0p25deg.log 2>&1