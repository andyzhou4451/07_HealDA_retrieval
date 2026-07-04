#!/bin/bash
#SBATCH --job-name=healda_tq13_h100_1gpu
#SBATCH --nodes=1
#SBATCH --partition=zs
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gnode02
#SBATCH --cpus-per-task=16
#SBATCH --mem=250G
#SBATCH --output=slurmlogs/%x_%j.out
#SBATCH --error=slurmlogs/%x_%j.err

source ~/.bashrc
conda activate xichen_v1

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}

python train.py \
  --config configs/train_h100_80gb_single_gpu.yaml \
  --device cuda:0
