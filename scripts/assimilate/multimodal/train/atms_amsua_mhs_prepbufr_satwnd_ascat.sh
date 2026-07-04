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
export TASK_QUEUE_ENABLE=0

if [[ -z "${MASTER_PORT:-}" ]]; then
  MASTER_PORT="$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')"
fi
export MASTER_PORT

python -m torch.distributed.run --nproc_per_node=2 \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    main.py \
    --config-name=train.yaml \
    datamodule=assimilate/random_bg/atms_amsua_mhs_prepbufr_satwnd_ascat.yaml \
    datamodule.batch_size=4 \
    datamodule.start_train_year=2016 \
    datamodule.debug=False \
    model=assimilate/random_bg/multimodal/atms_amsua_mhs_prepbufr_satwnd_ascat.yaml \
    loss_fn=crps_gaussian.yaml \
    paths=default.yaml \
    hydra=default.yaml \
    pipeline=assimilate/multimodal/random_bg_trainer.yaml \
    training=multimodal_da/random_bg_atms_amsua_mhs_prepbufr_satwnd_ascat.yaml \
    training.precision.type=bf16 \
    training.max_grad_norm=1.0 \
    training.epochs=60 \
    training.lr=1e-3 \
    task_name=train_multimodal_da_randombg_atms_amsua_mhs_prepbufr_satwnd_ascat_20260625 \
    > slurmlogs/train_multimodal_da_randombg_atms_amsua_mhs_prepbufr_satwnd_ascat_20260625.log