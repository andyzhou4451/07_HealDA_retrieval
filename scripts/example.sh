#!/bin/bash
# CUDA/GPU 分布式训练示例 (Hydra + 原生 PyTorch DDP)
#
# 用法:
#   bash scripts/example.sh
#   bash scripts/example.sh --nproc 4
#
# 注意:
#   - data_dir 需在 configs/paths/default.yaml 中配置或设置环境变量 DATA_DIR
#   - 默认训练2010-2021数据，验证2022，测试2023

NPROC=1
EPOCHS=100
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --nproc)
            NPROC="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

cd "$(dirname "$0")/.."

python -m torch.distributed.run \
    --nproc_per_node=$NPROC \
    main.py \
    epochs=$EPOCHS