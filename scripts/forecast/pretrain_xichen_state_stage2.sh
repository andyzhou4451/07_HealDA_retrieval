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

nohup python -m torch.distributed.run --nproc_per_node=2 \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    main.py \
    --config-name=pretrain_xichen_state_stage2.yaml \
    > logs/pretrain_xichen_state_forecast_stage2_20260520.log 2>&1 &