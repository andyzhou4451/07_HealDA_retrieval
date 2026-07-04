#!/bin/bash

# export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

python inference/era5_interp_forecast.py --decorrelation_hours=6 \
    --output_dir=/public/home/wangwuxing01/research/XiChen/data/xichen_results/finetune_xichen_state_forecast_ar15_20260612_interp_0p25deg \
    --forecast_name=finetune_xichen_state_forecast_ar15_20260612 \
    > slurmlogs/eval_finetune_xichen_state_forecast_ar15_20260612_interp_0p25deg.log 2>&1