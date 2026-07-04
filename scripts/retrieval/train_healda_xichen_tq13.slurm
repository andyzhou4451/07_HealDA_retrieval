#!/bin/bash
#SBATCH --job-name=xichen_retrieval_tq13
#SBATCH --nodes=1
#SBATCH --partition=zs
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --nodelist=gnode02
#SBATCH --cpus-per-task=8
#SBATCH --mem=500G
#SBATCH --output=slurmlogs/%x_%j.out
#SBATCH --error=slurmlogs/%x_%j.err

source ~/.bashrc
conda activate xichen_v1

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_NSOCKS_PERTHREAD=2
export NCCL_SOCKET_NTHREADS=4
export PYTHONFAULTHANDLER=1
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}

torchrun \
  --nproc_per_node=2 \
  --master_port=29501 \
  main.py \
  --config-name=train.yaml \
  datamodule=retrieval/healda_atms_amsua_mhs_hrs4_gdas_tq13.yaml \
  model=retrieval/healda_xichen_tq13.yaml \
  pipeline=retrieval/trainer.yaml \
  training=retrieval/healda_atms_amsua_mhs_hrs4_gdas_tq13.yaml \
  loss_fn=retrieval_tq_huber.yaml \
  paths=retrieval_public02.yaml \
  paths.obs_dir=/public02/data/Observation/observation_npy/ \
  paths.era5_dir=/public02/data/era5_np181x360_level13 \
  paths.scale_dir=/public02/data/era5_np181x360_level13/normalized_mean_std \
  training.device=cuda \
  training.precision.type=bf16 \
  training.gradient_accumulation_steps=2 \
  training.ddp_auto_move_inputs=false \
  training.use_gradient_checkpointing=true \
  datamodule.data.strict_time_index=true \
  datamodule.data.dt_data=1 \
  datamodule.data.max_points_per_sensor=100000 \
  datamodule.num_workers=8 \
  datamodule.prefetch_factor=4 \
  model.model_size=base \
  task_name=healda_xichen_retrieval_atms_amsua_mhs_hrs4_gdas_tq13
