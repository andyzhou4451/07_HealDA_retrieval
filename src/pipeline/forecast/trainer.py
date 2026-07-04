# -*- coding: utf-8 -*-
"""
ForecastTrainer - 天气预报 (state-forecast) 训练器。

本模块定义 ``ForecastTrainer``，基于 ``BaseTrainer`` 实现 ERA5 状态场的
**自回归 (Auto-Regressive, AR) 多步滚动预报**训练流程：

- 数据流：每个 batch 包含 ``(inps, tgts, lead_times, variables, iter_num, std)``，
  其中 ``tgts`` 形状为 ``[B, T, Vi, H, W]``，对应未来 ``iter_num`` 个时间步的真值；
- 模型：``XiChenForecast``，接受 ``(B, Vi, H, W)`` 输入并返回
  ``(preds, log_var)``，其中 ``log_var`` 为 CRPS-Gaussian 所需的预测对数方差；
- 训练核心：``detach_iter_num`` 步为一阶段 (``stage``)，每阶段重新走前向 + 反向，
  阶段之间对 ``preds`` 做 ``detach`` 以**截断计算图**，避免长 AR 链路的反向显存爆炸；
- 损失：CRPS-Gaussian (默认) 或 L1，对每个时间步独立求损失后取均值；
- 验证：同样以 ``detach_iter_num`` 为单位滚动，每个阶段内部 ``detach`` 切断图；
  末尾通过 ``std * weighted_rmse_torch(preds, tgts[:, -1])`` 计算带物理量纲的 RMSE，
  并按变量聚合输出每个变量对应的 RMSE 字典；
- 优化器：``torch.optim.AdamW``（无 NPU 融合回退）；
  调度器可选 ``cosine_warmup`` / ``cosine`` / ``None``；
- 混合精度：通过 ``src.utils.device.autocast`` + ``GradScaler`` 组合，
  支持 ``bf16`` / ``fp16`` / ``fp32``。
"""
import os
from abc import ABC, abstractmethod
from typing import Optional
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from omegaconf import DictConfig
import hydra

from src.metrics.weighted_acc_rmse import weighted_rmse_torch

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

class ForecastTrainer(BaseTrainer):
    """天气预报训练器，支持iterative prediction。

    与 ``BaseTrainer`` 的区别在于：
    - 直接重写 ``__init__`` / ``fit`` / ``train_epoch`` / ``validate``，不调用
      ``super().__init__``，因此**未继承**基类的 ``_setup_components`` /
      ``_load_checkpoint`` 流程，而是在本类内做了一套**等价的初始化序列**：
      组件实例化 → 预训练加载 → 断点续训 → DDP 包装 → Logger/Profiler；
    - 实现了基于 ``detach_iter_num`` 的分阶段 AR 滚动前向，分阶段反向
      （``stage_loss.backward``）以控制长 AR 链路计算图占用的显存；
    - ``validate`` 返回 ``(val_loss, val_rmse_per_var_dict)``，并把每个变量
      的 RMSE 写入 TensorBoard 的 ``val/rmse_<var>`` 标量；
    - 优化器构造时直接使用 ``torch.optim.AdamW``（无 NPU 融合回退）；
      失败时回退到 ``torch.optim.AdamW``；这与基类通过 Hydra 实例化优化器
      的方式不同，是本类为 NPU 训练做的特殊优化。
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
        """初始化 ForecastTrainer — 全部由基类 __init__ 模板驱动."""
        super().__init__(cfg, device, local_rank, world_size, is_main, **kwargs)

    def _build_models(self) -> None:
        """实例化 model + 加载预训练 (调用 _load_pretrain_ckpt 由子类钩入).

        Hook contract: 基类 __init__ Phase 1 调一次, 模型实例化后子类负责 pretrain 加载.
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

        - 优化器：``torch.optim.AdamW``，参数 ``lr`` / ``betas`` / ``weight_decay``
          由 ``training_config`` 控制；
        - 调度器：可选 ``cosine_warmup``（自定义 ``CosineSchedulerWithWarmup``）、
          ``cosine``（``CosineAnnealingLR``）或 ``None``，其中 ``cosine_warmup``
          需要 ``warmup_epochs`` 与 ``epochs`` 两个超参。
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
        """加载预训练 Checkpoint（用于 finetune 阶段）。

        注意:
            与 ``BaseTrainer._load_checkpoint`` 不同，本方法只加载
            ``model_state_dict`` 并把所有参数 ``requires_grad = True``，
            不恢复优化器 / 调度器 / AMP 标量。
        """
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
        self.model.load_state_dict(ckpt["model_state_dict"], strict=self.training_config.get("load_strict", True))

        for name, param in self.model.named_parameters():
            param.requires_grad = True

    def _load_checkpoint(self):
        """断点续训：从 ``{ckpt_dir}/last.ckpt`` 恢复训练。

        流程：
        1. ``torch.load(..., map_location=self.device)`` 读取 checkpoint 字典；
        2. 加载 ``model_state_dict`` 并把全部参数置为 ``requires_grad=True``；
        3. 加载 ``optimizer_state_dict``；
        4. 可选加载 ``scheduler_state_dict`` 与 ``scaler_state_dict``；
        5. 恢复 ``start_epoch`` / ``best_loss`` 等标量状态。

        注意:
            与基类的差异：基类按 ``config.training.resume_ckpt`` 路径加载，
            而本类固定从 ``self.ckpt_dir/last.ckpt`` 加载。
        """
        if not self.ckpt_dir or not os.path.exists(os.path.join(self.ckpt_dir, "last.ckpt")):
            if self.is_main:
                self.log.info(f"No {self.ckpt_dir}/last.ckpt provided, training from scratch.")
            return

        if self.is_main:
            self.log.info(f"Resuming training from: {os.path.join(self.ckpt_dir, 'last.ckpt')}")

        # 1. 加载字典 (使用 map_location 确保设备无关性)
        ckpt = torch.load(os.path.join(self.ckpt_dir, "last.ckpt"), map_location=self.device)

        # 2. 加载模型权重
        self.model.load_state_dict(ckpt["model_state_dict"], strict=self.training_config.get("load_strict", True))

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
        """DDP 包装：当 ``world_size > 1`` 时用 ``DistributedDataParallel`` 包裹模型。

        注意:
            - 仅当 ``self.model`` 尚未被 ``DDP`` 包装时执行；
            - ``device_ids`` / ``output_device`` 在 CUDA / NPU 上绑定本地 rank，
              CPU 上保持 ``None``；
            - ``find_unused_parameters`` 从 ``training_config.find_unused_parameters``
              读取（默认 ``True``），用于兼容冻结子模块等场景；
            - ``_save_ckpt`` 中通过 ``self.model.module if isinstance(self.model, DDP) else self.model``
              取出裸模型，避免保存的权重带 ``module.`` 前缀。
        """
        if self.world_size > 1 and not isinstance(self.model, DDP):
            self.model = self._wrap_single_ddp(self.model)

    def train_epoch(self, loader, epoch, epochs):
        """单 epoch 训练：AR 多步滚动 + ``detach_iter_num`` 分阶段反向。

        Args:
            loader: 训练 DataLoader，batch 内容为
                ``(inps, tgts, lead_times, variables, iter_num, std)``：
                - ``inps``: ``[B, Vi, H, W]`` 起始场；
                - ``tgts``: ``[B, T, Vi, H, W]`` 未来 ``T`` 个时间步的真值；
                - ``lead_times``: ``[B, T]``（或类似）每个目标步对应的预报时长
                  （小时），用于 lead-time cross-attention 注入；
                - ``variables``: 当前 batch 的变量名列表；
                - ``iter_num``: 当前 batch 的目标步数 ``T``（可为 Tensor）；
                - ``std``: ``[Vi]`` 各变量的标准化方差，用于反归一化指标；
            epoch (int): 当前 epoch 编号。
            epochs (int): 总 epoch 数。

        Returns:
            dict: ``{"train/loss": float}``，本 epoch 训练平均损失。

        实现要点:
            - ``detach_iter_num = training_config.detach_iter_num`` 控制每阶段
              内联跑多少个时间步；
            - ``stage_num = iter_num // detach_iter_num`` 为阶段数；
            - 每个 ``stage`` 起点清零梯度；``stage > 0`` 时 ``preds = preds.detach()``
              以**截断阶段间计算图**，避免长 AR 链反向时显存爆炸；
            - ``stage_loss`` 累加 ``detach_iter_num`` 个 ``iter_loss``，最后除以
              ``detach_iter_num`` 得到阶段平均；
            - ``self.scaler.scale(stage_loss).backward()`` 配合
              ``unscale_`` / ``clip_grad_norm_`` / ``step`` / ``update`` 完成
              AMP 反向；
            - 阶段之间根据 ``self.device_type`` 选择 ``torch.cuda.empty_cache()`` 或
              ``torch.npu.empty_cache()`` 释放缓存（CPU 模式跳过）。
        """
        self.model.train()
        total_loss = 0
        pbar = tqdm(loader, desc=f"Training epoch {epoch}/{epochs}", leave=False, disable=not self.is_main)

        for batch_idx, batch in enumerate(pbar):
            inps, tgts, lead_times, variables, iter_num, std = batch
            preds = inps.to(self.device)
            tgts = tgts.to(self.device)
            lead_times = lead_times.to(self.device)
            iter_num = iter_num.item() if isinstance(iter_num, torch.Tensor) else iter_num

            detach_iter_num = self.training_config.get("detach_iter_num", 4)
            stage_num = iter_num // detach_iter_num

            total_batch_loss = 0

            for stage in range(stage_num):
                # 清零梯度
                self.optimizer.zero_grad()

                # 除了第一个stage，断开梯度连接
                if stage > 0:
                    preds = preds.detach()

                stage_loss = 0
                with autocast(self.device_type, dtype=self.precision_type):
                    for iter_step in range(detach_iter_num):
                        preds, log_var = self.model(
                            preds, 
                            lead_times, 
                            variables, 
                            use_checkpoint=True
                        )
                        iter_loss = self.loss_fn(
                            preds, 
                            log_var, 
                            tgts[:, detach_iter_num * stage + iter_step], 
                            torch.ones_like(log_var).to(self.device, dtype=log_var.dtype)
                        )
                        stage_loss = stage_loss + iter_loss

                # 计算stage平均损失
                stage_loss = stage_loss / detach_iter_num
                total_batch_loss += stage_loss.item()

                # 反向传播和参数更新
                self.scaler.scale(stage_loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    max_norm=self.training_config.get("max_grad_norm", 1.0)
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()

                empty_cache(self.device_type)

            # 计算整个batch的平均损失
            avg_batch_loss = total_batch_loss / stage_num
            total_loss += avg_batch_loss

            pbar.set_postfix({"train_loss": f"{avg_batch_loss:.4f}"})

        return {"train/loss": total_loss / len(loader)}

    def validate(self, loader, epoch, epochs):
        """单 epoch 验证：与训练相同的 AR 滚动流程，但每步 ``detach`` 切断图。

        Args:
            loader: 验证 DataLoader。
            epoch (int): 当前 epoch 编号。
            epochs (int): 总 epoch 数。

        Returns:
            dict: ``{"val/loss": float, "val/rmse_<var_name>": float, ...}``
            - ``val/loss``: 本 epoch 平均验证损失
            - ``val/rmse_<var_name>``: 每个变量反归一化加权 RMSE,共
              ``len(variables)`` 个 key (forecast 输入 69 变量)
            空 loader 时返回 ``{"val/loss": float("inf")}``,与 ``BaseTrainer``
            的契约保持一致。

        实现要点:
            - 阶段内每步前向后立即 ``preds, log_var = preds.detach(), log_var.detach()``
              以节省显存；
            - 验证阶段不调用 ``backward``，亦不调用 ``optimizer.zero_grad``；
            - RMSE 计算使用 ``std * weighted_rmse_torch(preds, tgts[:, -1])``，
              再按变量聚合 ``mse`` 后开方。
        """
        if len(loader) == 0:
            # 空 dataloader (验证日期范围无文件匹配) — 跳过避免 ZeroDivisionError.
            if self.is_main:
                self.log.warning(f"Epoch {epoch}/{epochs}: val_loader is empty, skipping validation.")
            return {"val/loss": float("inf")}
        self.model.eval()
        total_loss = 0
        # 使用字典来累积每个变量的MSE
        total_mse_dict = {}
        pbar = tqdm(loader, desc=f"Validating epoch {epoch}/{epochs}", leave=False, disable=not self.is_main)

        for batch_idx, batch in enumerate(pbar):
            inps, tgts, lead_times, variables, iter_num, std = batch
            inps = inps.to(self.device)
            tgts = tgts.to(self.device)
            lead_times = lead_times.to(self.device)
            iter_num = iter_num.item() if isinstance(iter_num, torch.Tensor) else iter_num

            detach_iter_num = self.training_config.get("detach_iter_num", 4)
            stage_num = iter_num // detach_iter_num

            batch_loss = 0
            with autocast(self.device_type, dtype=self.precision_type):
                preds = inps
                for stage in range(stage_num):
                    stage_loss = 0
                    for iter_step in range(detach_iter_num):
                        preds, log_var = self.model(preds, lead_times, variables, use_checkpoint=True)
                        preds, log_var = preds.detach(), log_var.detach()

                        iter_loss = self.loss_fn(
                            preds, 
                            log_var, 
                            tgts[:, detach_iter_num * stage + iter_step], 
                            torch.ones_like(log_var).to(self.device, dtype=log_var.dtype)
                        )
                        stage_loss = stage_loss + iter_loss.item()

                    batch_loss += stage_loss / detach_iter_num

                total_loss += batch_loss / stage_num

                # 计算所有变量的RMSE
                val_rmse = std.to(self.device, dtype=preds.dtype) * weighted_rmse_torch(preds, tgts[:, -1])

                # 累积每个变量的MSE
                for i, var in enumerate(variables):
                    if var not in total_mse_dict:
                        total_mse_dict[var] = 0
                    total_mse_dict[var] += val_rmse[i].item() ** 2

                # 创建包含所有变量RMSE的字符串用于显示
                rmse_strs = []
                for i, var in enumerate(variables):
                    rmse_strs.append(f"{var}:{val_rmse[i].item():.4f}")
                rmse_str = ", ".join(rmse_strs)

                pbar.set_postfix(
                    {
                        "val_loss": f"{(batch_loss / stage_num):.4f}", 
                        "val_rmse": rmse_str
                    }
                )

        empty_cache(self.device_type)

        # 计算每个变量的平均RMSE
        avg_rmse_per_var = {}
        for var, mse_sum in total_mse_dict.items():
            avg_rmse_per_var[var] = np.sqrt(mse_sum / len(loader))

        # 返回平均损失和所有变量的RMSE字典
        metrics = {"val/loss": total_loss / len(loader)}
        for var, rmse in avg_rmse_per_var.items():
            metrics[f"val/rmse_{var}"] = float(rmse)
        return metrics

    def _save_ckpt(self, ckpt_dir, filename, epoch, val_loss, is_best=False):
        """保存 Checkpoint。

        注意:
            - 必须通过 ``self.model.module if isinstance(self.model, DDP) else self.model``
              取出裸模型的 ``state_dict()``，避免 ``module.`` 前缀污染权重；
            - 保存字段：``epoch`` / ``model_state_dict`` / ``optimizer_state_dict`` /
              ``scheduler_state_dict``（可为 ``None``） / ``scaler_state_dict``（可为
              ``None``，当 ``self.scaler is None`` 时） / ``best_loss``。
        """
        # 1. 获取裸模型 (处理 DDP 情况)
        model_to_save = self.model.module if isinstance(self.model, DDP) else self.model

        ckpt_dict = {
            "epoch": epoch,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state_dict": self.scaler.state_dict() if self.scaler is not None else None,  # 修正这里
            "best_loss": self.best_loss,
        }

        save_path = os.path.join(ckpt_dir, filename)
        torch.save(ckpt_dict, save_path)
        if self.is_main:
            self.log.info(f"Checkpoint saved to {save_path}")
