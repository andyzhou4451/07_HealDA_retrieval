#!/bin/bash

# export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

nohup python inference/era5_forecast.py --decorrelation_hours=6 \
    --output_dir=/public/home/wangwuxing01/research/XiChen/data/xichen_results/finetune_xichen_state_forecast_ar8_20260529 \
    --forecast_name=finetune_xichen_state_forecast_ar8_20260529 \
    > slurmlogs/eval_finetune_xichen_state_forecast_ar8_20260530.log 2>&1 &