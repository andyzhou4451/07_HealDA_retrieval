# -*- coding: utf-8 -*-
"""XiChen/HealDA 训练入口。

该入口保留 XiChen 原有 Hydra 配置方式，同时补强单机 2×H100 DDP 训练需要的
rank 解析、CUDA 性能开关、日志目录、debug 参数下传与异常清理逻辑。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    import pyrootutils

    root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)
except ImportError:
    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import hydra
from omegaconf import DictConfig, OmegaConf

from src.utils import setup_logger
from src.utils.device import (
    configure_accelerator_performance,
    get_device,
    init_distributed,
    manual_seed,
    normalize_device_type,
)
from src.utils.tqdm_logger import patch_tqdm_for_logger


def _env_int(name: str, default: int) -> int:
    """从环境变量读取整数；Slurm/torchrun 未设置时使用安全默认值。"""
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


def _resolve_device_type(config: DictConfig) -> str:
    """按 training.device 优先解析设备类型，并把 gpu 统一规范化为 cuda。"""
    training = config.get("training", {})
    value = training.get("device", config.get("device", "cuda")) if hasattr(training, "get") else config.get("device", "cuda")
    return normalize_device_type(str(value))


def _propagate_debug_overrides(config: DictConfig) -> None:
    """把顶层 debug=true 下传到 datamodule/training，保证冒烟测试只跑少量 batch。"""
    if not config.get("debug", False):
        return
    OmegaConf.set_struct(config, False)
    if "datamodule" in config:
        config.datamodule.debug = True
        config.datamodule.batch_size = int(config.datamodule.get("batch_size", 1) or 1)
        config.datamodule.num_workers = int(config.datamodule.get("num_workers", 0) or 0)
    if "training" in config:
        config.training.epochs = int(config.training.get("max_epochs", config.training.get("epochs", 1)) or 1)
        config.training.limit_train_batches = int(config.training.get("limit_train_batches", 2) or 2)
        config.training.limit_val_batches = int(config.training.get("limit_val_batches", 2) or 2)
        config.training.profile = False


@hydra.main(version_base=None, config_path="configs", config_name="train.yaml")
def main(config: DictConfig) -> None:
    """Hydra 主函数：构建 DataModule、Trainer，并执行训练循环。"""
    local_rank = _env_int("LOCAL_RANK", 0)
    global_rank = _env_int("RANK", local_rank)
    world_size = _env_int("WORLD_SIZE", 1)
    hardware_cfg = config.get("hardware", {})
    single_gpu = bool(hardware_cfg.get("single_gpu", False)) if hasattr(hardware_cfg, "get") else False
    if single_gpu and world_size > 1:
        raise RuntimeError(
            "hardware.single_gpu=true 只允许用 python/main.py 单进程启动；"
            "不要用 torchrun --nproc_per_node>1，或显式设置 hardware.single_gpu=false。"
        )
    if single_gpu:
        local_rank = 0
        global_rank = 0
        world_size = 1
    is_main = global_rank == 0
    device_type = _resolve_device_type(config)

    _propagate_debug_overrides(config)
    configure_accelerator_performance(config=config, device_type=device_type, log_fn=None)

    task_name = str(config.get("task_name", "debug"))
    log_dir = str(config.paths.get("log_dir", "logs"))
    output_dir = str(config.paths.get("output_dir", "logs/output"))
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{task_name}.rank{global_rank}.log") if is_main else None
    log = setup_logger(rank=global_rank, log_file=log_file)
    patch_tqdm_for_logger(log)

    if is_main:
        log.info("=" * 80)
        log.info("XiChen HealDA-style T/Q13 retrieval training")
        log.info(f"device={device_type}, global_rank={global_rank}, local_rank={local_rank}, world_size={world_size}, single_gpu={single_gpu}")
        log.info(f"hydra_output_dir={output_dir}")
        log.info("=" * 80)

    if world_size > 1:
        init_distributed(device_type=device_type, local_rank=local_rank)
        if is_main:
            log.info(f"DDP initialized with backend={'nccl' if device_type == 'cuda' else ('hccl' if device_type == 'npu' else 'gloo')} and world_size={world_size}")

    device = get_device(device_type=device_type, local_rank=local_rank)
    manual_seed(device_type=device_type, seed=int(config.get("seed", config.get("training", {}).get("seed", 1024))), rank=global_rank)

    if is_main:
        log.info("Initializing retrieval datamodule")
    datamodule = hydra.utils.instantiate(
        config.datamodule,
        distributed=(world_size > 1),
        num_replicas=world_size,
        rank=global_rank,
        _recursive_=False,
    )
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    if is_main:
        log.info("Initializing retrieval trainer")
    trainer = hydra.utils.instantiate(
        config.pipeline,
        cfg=config,
        device=device,
        local_rank=local_rank,
        world_size=world_size,
        is_main=is_main,
        output_dir=output_dir,
        log_dir=log_dir,
        log=log,
        _recursive_=False,
    )

    if is_main:
        log.info(
            "Training config: "
            f"epochs={config.training.epochs}, lr={config.training.lr}, "
            f"batch_size_per_rank={config.datamodule.get('batch_size', 1)}, "
            f"num_workers_per_rank={config.datamodule.get('num_workers', 0)}"
        )

    try:
        trainer.fit(train_loader=train_loader, val_loader=val_loader)
    finally:
        trainer.cleanup()

    if is_main:
        log.info("Training completed")


if __name__ == "__main__":
    main()
