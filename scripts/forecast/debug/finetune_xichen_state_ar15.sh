#!/bin/bash

# export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1
export HYDRA_FULL_ERROR=1
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_NSOCKS_PERTHREAD="${NCCL_NSOCKS_PERTHREAD:-2}"
export NCCL_SOCKET_NTHREADS="${NCCL_SOCKET_NTHREADS:-4}"

if [[ -z "${MASTER_PORT:-}" ]]; then
  MASTER_PORT="$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')"
fi
export MASTER_PORT

python -m torch.distributed.run --nproc_per_node=2 \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    main.py \
    --config-name=train.yaml \
    datamodule=forecast/xichen_state_forecast.yaml \
    datamodule.iter_num=15 \
    datamodule.batch_size=24 \
    datamodule.start_train_year=2016 \
    datamodule.debug=True \
    model=forecast/xichen_state_forecast.yaml \
    loss_fn=crps_gaussian.yaml \
    paths=default.yaml \
    hydra=default.yaml \
    pipeline=forecast/xichen_state_forecast.yaml \
    training=forecast/finetune_forecast.yaml \
    training.epochs=20 \
    training.detach_iter_num=3 \
    training.lr=1e-7 \
    training.scheduler_type=none \
    training.pretrain_ckpt=logs/pretrain_xichen_state_forecast_20260526/runs/checkpoints/best.ckpt \
    task_name=debug_xichen_state_forecast_ar15_20260612 \
    > slurmlogs/debug_xichen_state_forecast_ar15_20260624.log