# -*- coding: utf-8 -*-
"""
CompressionTrainer - 天气压缩 (VAE) 训练器。

本模块定义 ``CompressionTrainer``，基于 ``BaseTrainer`` 实现 ERA5 状态场的
**有损压缩** (Latent Quantization VAE) 训练流程：

- 模型：``XiChenAutoEncoder``，输入 ``[B, Vi, H, W]`` 的状态场，通过
  ``quan_mlp`` / ``post_quan_mlp`` 量化器在 ``z_dim=69`` 的潜空间进行编解码，
  可选 ``ending_norm``（LayerNorm）对末端特征做归一化；
- 数据流：每个 batch 内容为 ``(inps, variables, std)``，其中 ``inps`` 即真值，
  也作为自编码重建的目标；
- 损失：CRPS-Gaussian (默认) 或 L1，对 ``(preds, log_var)`` 与 ``inps`` 计算；
- 训练：单步前向 → 单次反向；不像 forecast 需要 AR 滚动；
- 验证：同结构，额外计算 ``z-500`` 上的加权 RMSE；
- 优化器：``torch.optim.AdamW``（无 NPU 融合回退）；
- 混合精度：通过 ``autocast`` + ``GradScaler`` 组合，支持 ``bf16`` /
  ``fp16`` / ``fp32``。
"""
import os
from abc import ABC, abstractmethod
from typing import Optional
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from omegaconf import DictConfig
import hydra

from src.metrics.weighted_acc_rmse import weighted_rmse_torch_channels

from src.utils.lr_scheduler import CosineSchedulerWithWarmup
from src.utils import get_logger
from src.utils.device import (
    autocast,
    get_grad_scaler,
    get_device,
)
from src.utils.model import count_parameters_detailed, format_number
from src.pipeline.base.trainer import BaseTrainer
from src.utils.device import empty_cache

class CompressionTrainer(BaseTrainer):
    """天气压缩 (VAE) 训练器。

    与 ``BaseTrainer`` 的区别：
    - 遵循 ``BaseTrainer`` 模板方法：``__init__`` 显式调用 ``super().__init__``，
      ``fit`` 继承自基类；
    - 单步非 AR 训练：每个 batch 只需一次前向 + 一次反向，不存在
      ``detach_iter_num`` 分阶段；
    - 优化器构造时直接使用 ``torch.optim.AdamW``（无 NPU 融合回退）；
    - ``validate`` 返回 ``dict``：除 ``val/loss``（平均损失）外，按变量记录
      反归一化加权 RMSE，键名形如 ``val/rmse_<var_name>``。
    """

    def __init__(
        self,
        cfg: DictConfig,
        device,
        local_rank: int,
        world_size: int,
        is_main: bool,
        **kwargs,
    ):
        """初始化 CompressionTrainer — 全部由基类 __init__ 模板驱动."""
        super().__init__(cfg, device, local_rank, world_size, is_main, **kwargs)

    def _build_models(self) -> None:
        """实例化压缩模型 + 加载预训练 (按需).

        Hook contract: 基类 __init__ Phase 1 调一次.
        """
        self.model = hydra.utils.instantiate(
            self.config.model.net, _recursive_=False
        ).to(self.device)

        if self.training_config.get("resume_pretrain", False):
            self._load_pretrain_ckpt()

        if self.is_main:
            total_params, trainable_params, frozen_params = count_parameters_detailed(self.model)
            self.log.info(f"Total parameters: {format_number(total_params)} ({total_params:,})")
            self.log.info(f"Trainable parameters: {format_number(trainable_params)} ({trainable_params:,})")
            self.log.info(f"Frozen parameters: {format_number(frozen_params)} ({frozen_params:,})")
            self.log.info(f"Trainable parameters ratio: {trainable_params/total_params*100:.2f}%")

    def _setup_optimizer_scheduler(self):
        """构造优化器与学习率调度器。

        - 优化器：``torch.optim.AdamW``，参数 ``lr`` / ``betas`` / ``weight_decay`` 由
          ``training_config`` 控制；
        - 调度器：可选 ``cosine_warmup`` / ``cosine`` / ``None``。
        """
        if self.is_main:
            self.log.info("Initializing optimizer...")
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.training_config.get("lr", 1e-4),
            betas=self.training_config.get("betas", [0.9, 0.95]),
            weight_decay=self.training_config.get("weight_decay", 5e-5),
        )

        if self.is_main:
            self.log.info("Initializing scheduler...")
        # 调度器
        self.scheduler = self._build_scheduler(self.optimizer)

    def _load_pretrain_ckpt(self):
        """加载预训练 Checkpoint（仅恢复 ``model_state_dict`` 并把全部参数置为可训练）。"""
        pretrain_ckpt_path = self.training_config.get("pretrain_ckpt")
        if not pretrain_ckpt_path or not os.path.exists(pretrain_ckpt_path):
            if self.is_main:
                self.log.info(f"No {pretrain_ckpt_path} provided, training from scratch.")
            return

        if self.is_main:
            self.log.info(f"Lodaing pretrained checkpoint from: {pretrain_ckpt_path}")

        # 1. 加载字典 (使用 map_location 确保设备无关性)
        ckpt = torch.load(pretrain_ckpt_path, map_location=self.device)

        # 2. 加载模型权重
        self.model.load_state_dict(ckpt["model_state_dict"])

        for name, param in self.model.named_parameters():
            param.requires_grad = True

    def _load_checkpoint(self):
        """断点续训：从 ``{ckpt_dir}/last.ckpt`` 恢复。

        与 ``BaseTrainer._load_checkpoint`` 流程一致，但固定从
        ``self.ckpt_dir/last.ckpt`` 加载。
        """
        if not self.ckpt_dir or not os.path.exists(os.path.join(self.ckpt_dir, "last.ckpt")):
            if self.is_main:
                self.log.info(f"No {self.ckpt_dir} provided, training from scratch.")
            return

        if self.is_main:
            self.log.info(f"Resuming training from: {os.path.join(self.ckpt_dir, 'last.ckpt')}")

        # 1. 加载字典 (使用 map_location 确保设备无关性)
        ckpt = torch.load(os.path.join(self.ckpt_dir, "last.ckpt"), map_location=self.device)

        # 2. 加载模型权重
        self.model.load_state_dict(ckpt["model_state_dict"])

        for name, param in self.model.named_parameters():
            param.requires_grad = True

        # 3. 加载优化器状态
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        # 4. 加载调度器状态 (如果存在)
        if self.scheduler and "scheduler_state_dict" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        # 5. 加载 AMP 标量 (如果存在)
        if self.scaler and "scaler_state_dict" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])

        # 6. 恢复标量状态
        self.start_epoch = ckpt.get("epoch", -1) + 1  # 兼容无 epoch 字段的旧 ckpt
        self.best_loss = ckpt.get("best_loss", float("inf"))  # 兼容无 best_loss 字段的旧 ckpt

        if self.is_main:
            self.log.info(f"Resumed successfully. Start Epoch: {self.start_epoch}, Best Loss: {self.best_loss:.4f}")

    def _wrap_ddp(self):
        """DDP 包装：当 ``world_size > 1`` 时用 ``DistributedDataParallel`` 包裹裸模型。

        注意:
            - 仅当 ``self.model`` 尚未被 ``DDP`` 包装时执行；
            - ``device_ids`` / ``output_device`` 在 CUDA / NPU 上绑定本地 rank；
            - ``find_unused_parameters`` 从 ``training_config.find_unused_parameters``
              读取；
            - ``_save_ckpt`` 中通过
              ``self.model.module if isinstance(self.model, DDP) else self.model``
              取出裸模型。
        """
        if self.world_size > 1 and not isinstance(self.model, DDP):
            self.model = self._wrap_single_ddp(self.model)

    def train_epoch(self, loader, epoch, epochs):
        """单 epoch 训练：单步 VAE 自编码重建。

        Args:
            loader: 训练 DataLoader，batch 内容为 ``(inps, variables, std)``：
                - ``inps``: ``[B, Vi, H, W]`` 状态场（既作输入也作目标）；
                - ``variables``: 当前 batch 的变量名列表；
                - ``std``: ``[Vi]`` 各变量反归一化用标准差。
            epoch (int): 当前 epoch 编号。
            epochs (int): 总 epoch 数。

        Returns:
            dict: ``{"train/loss": float}``
            - ``train/loss``: 本 epoch 平均训练损失
        """
        self.model.train()
        total_loss = 0
        pbar = tqdm(loader, desc=f"Training epoch {epoch}/{epochs}", leave=False, disable=not self.is_main)

        for batch_idx, batch in enumerate(pbar):
            inps, variables, std = batch
            inps = inps.to(self.device)

            self.optimizer.zero_grad()

            with autocast(self.device_type, dtype=self.precision_type):
                loss = 0
                preds, log_var = self.model(inps, variables, use_checkpoint=True)
                loss = loss + self.loss_fn(
                    preds, 
                    log_var, 
                    inps, 
                    torch.ones_like(log_var).to(self.device, dtype=log_var.dtype)
                )

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                max_norm=self.training_config.get("max_grad_norm", 1.0)
            )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss = total_loss + loss.item()
            pbar.set_postfix(
                {
                    "train_loss": f"{loss.item():.4f}",
                }
            )

            if self.profiler:
                self.profiler.step()

        empty_cache(self.device_type)

        return {"train/loss": total_loss / len(loader)}

    def validate(self, loader, epoch, epochs):
        """单 epoch 验证：单步 VAE 自编码重建 + 每个输入变量的反归一化加权 RMSE。

        Args:
            loader: 验证 DataLoader。
            epoch (int): 当前 epoch 编号。
            epochs (int): 总 epoch 数。

        Returns:
            dict: ``{"val/loss": float, "val/rmse_<var_name>": float, ...}``
            - ``val/loss``: 本 epoch 平均验证损失
            - ``val/rmse_<var_name>``: 每个变量反归一化加权 RMSE,共
              ``len(variables)`` 个 key (compression 输入 69 变量)
            空 loader 时返回 ``{"val/loss": float("inf")}``,与 ``BaseTrainer``
            的契约保持一致。
        """
        if len(loader) == 0:
            # 空 dataloader (验证日期范围无文件匹配) — 跳过避免 ZeroDivisionError / ValueError.
            if self.is_main:
                self.log.warning(f"Epoch {epoch}/{epochs}: val_loader is empty, skipping validation.")
            return {"val/loss": float("inf")}
        self.model.eval()
        total_loss, total_mse = 0, 0
        pbar = tqdm(loader, desc=f"Validating epoch {epoch}/{epochs}", leave=False, disable=not self.is_main)

        for batch_idx, batch in enumerate(pbar):
            inps, variables, std = batch
            inps = inps.to(self.device)

            self.optimizer.zero_grad()

            with autocast(self.device_type, dtype=self.precision_type):
                loss = 0
                preds, log_var = self.model(inps, variables, use_checkpoint=True)
                loss = loss + self.loss_fn(
                    preds, 
                    log_var, 
                    inps, 
                    torch.ones_like(log_var).to(self.device, dtype=log_var.dtype)
                )

                total_loss = total_loss + loss.item()

                val_rmse = std.to(self.device, dtype=preds.dtype) * weighted_rmse_torch_channels(preds, inps)
                # val_rmse shape: (B, C) per-batch, per-channel; mean over batch for accumulation
                val_rmse = val_rmse.detach().mean(dim=0)  # (C,)
                total_mse += val_rmse ** 2
                pbar.set_postfix(
                    {
                        "val_loss": f"{loss.item():.4f}",
                        "val_rmse_z500": f"{val_rmse[variables.index('z-500')].item():.4f}"
                    }
                )

        empty_cache(self.device_type)

        rmse_per_var = torch.sqrt(total_mse / len(loader))  # (C,) per-channel
        metrics = {"val/loss": total_loss / len(loader)}
        for var_idx, var in enumerate(variables):
            metrics[f"val/rmse_{var}"] = float(rmse_per_var[var_idx].item())
        return metrics

    def _save_ckpt(self, ckpt_dir, filename, epoch, val_loss, is_best=False):
        """保存 Checkpoint。

        注意:
            - 必须通过 ``self.model.module if isinstance(self.model, DDP) else self.model``
              取出裸模型的 ``state_dict()``；
            - 保存字段：``epoch`` / ``model_state_dict`` / ``optimizer_state_dict`` /
              ``scheduler_state_dict``（可为 ``None``）/ ``scaler_state_dict``（可为
              ``None``）/ ``best_loss``。
        """
        # 1. 获取裸模型 (处理 DDP 情况)
        model_to_save = self.model.module if isinstance(self.model, DDP) else self.model

        ckpt_dict = {
            "epoch": epoch,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
            "best_loss": self.best_loss, # 保存当前的最佳损失，防止重置
        }

        save_path = os.path.join(ckpt_dir, filename)
        torch.save(ckpt_dict, save_path)
        if self.is_main:
            self.log.info(f"Checkpoint saved to {save_path}")
