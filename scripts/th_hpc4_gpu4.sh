#!/bin/sh
#SBATCH -p gpu1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=8
#SBATCH --job-name=healda_ret_hpc4_4gpu
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -eu

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

module add "${HPC4_PYTORCH_MODULE:-pytorch/1.11.0-cu11.3-py3.9}"

export HPC4_GPUS_PER_NODE="${HPC4_GPUS_PER_NODE:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_GPU:-8}}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

mkdir -p logs outputs tensorboard

: "${HPC4_TRAIN_CMD:=python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=${HPC4_GPUS_PER_NODE} main.py --config-name=train_h100_80gb_single_gpu.yaml hardware.single_gpu=false training.single_gpu=false training.device=cuda task_name=healda_xichen_tq13_hpc4_4gpu}"

echo "TH-HPC4 one-node ${HPC4_GPUS_PER_NODE}-GPU job"
echo "Command: ${HPC4_TRAIN_CMD}"

yhrun sh -lc "$HPC4_TRAIN_CMD"
