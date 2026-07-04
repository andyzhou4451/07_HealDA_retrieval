# -*- coding: utf-8 -*-
"""
ObsOperatorTrainer - 卫星观测算子 (Obs-Op) 训练器。

本模块定义 ``ObsOperatorTrainer``，训练卫星亮温观测算子 ``XiChenObsOp``，
将 ERA5 状态场 (含卫星几何辅助场) 映射为各卫星通道的亮温 + 观测误差方差：

- 支持 4 种卫星：``atms`` / ``amsua`` / ``mhs`` / ``hrs4``，每种卫星对应一份
  datamodule / model / pipeline / script；
- 数据流：每个 batch 内容为
  ``(state, sat, sat_mask, tmbrs_std, tmbrs_vars)``：
  - ``state``: ``[B, 69, H, W]`` ERA5 状态场（含 surface + pressure level vars）；
  - ``sat``: 每颗卫星的辅助场（cos(zenith), azimuth, scan/fov/orbit, satellite_height）；
  - ``sat_mask``: ``[B, 1, H, W]``（或类似）扫描掩码；
  - ``tmbrs_std``: 各通道反归一化标准差；
  - ``tmbrs_vars``: 通道名称列表；
- 模型：``XiChenObsOp``，输入 ``(state, sat, sat_mask)``，输出
  ``(out_sat, log_var, tgt_sat)``：
  - ``out_sat``: 预测亮温；
  - ``log_var``: CRPS-Gaussian 所需的对数方差（用于学习观测误差）；
  - ``tgt_sat``: 真实亮温；
- 损失：CRPS-Gaussian，按 ``sat_mask`` 加权；通过
  ``torch.repeat_interleave(sat_mask, repeats=tgt_sat.shape[1], dim=1)`` 把单通道
  mask 广播到所有亮温通道；
- 验证指标：除 ``val_loss`` 外，按模型 ``out_sat_vars`` 顺序计算
  ``val_rmse`` (预测亮温 RMSE) 与 ``val_obserr`` (观测误差预测 RMSE)；
- 优化器：``torch.optim.AdamW``（无 NPU 融合回退）；
- 混合精度：通过 ``autocast`` + ``GradScaler`` 组合，支持 ``bf16`` /
  ``fp16`` / ``fp32``。
"""
import os
from abc import ABC, abstractmethod
from typing import Optional
from tqdm import tqdm
import numpy as np
import torch
try:
    import torch_npu
    import torch_npu.distributed
except ImportError:
    pass
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from omegaconf import DictConfig
import hydra

from src.pipeline.base.trainer import BaseTrainer

from src.utils.lr_scheduler import CosineSchedulerWithWarmup
from src.utils import get_logger
from src.utils.device import (
    autocast,
    get_grad_scaler,
    get_device,
)
from src.utils.model import count_parameters_detailed, format_number
from src.utils.device import empty_cache

class ObsOperatorTrainer(BaseTrainer):
    """Obs-Op 训练器（per-satellite），支持iterative prediction。

    与 forecast / compression 训练器相比，该类的关键差异：
    - **多卫星兼容**：模型侧 ``XiChenObsOp`` 的 ``out_sat_vars`` 与 dataset 的
      ``tmbrs_vars`` 通过名称索引对齐（见 ``validate`` 中的 ``tgt_sat_var_ids``）；
    - **mask 处理**：``sat_mask`` 在训练损失与验证指标中均作为加权权重；
    - **观测误差建模**：同时预测 ``out_sat`` (亮温) 与 ``log_var`` (观测误差方差)，
      ``val_obserr`` 指标专门监控预测误差估计的合理性；
    - 该类继承 ``BaseTrainer``（最终继承 ``ABC``），初始化序列由基类统一管理
      （``_build_models`` / ``_setup_optimizer_scheduler`` / ``_wrap_ddp`` 等），
      本类仅重写 ``train_epoch`` 与 ``validate``。
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
        """初始化 ObsOperatorTrainer — 全部由基类 __init__ 模板驱动."""
        super().__init__(cfg, device, local_rank, world_size, is_main, **kwargs)

    def _build_models(self) -> None:
        """实例化 obsop 模型 + 加载预训练 (strict=False, 允许迁移学习).

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

        - 优化器：``torch.optim.AdamW``（无 NPU 融合回退）；
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
        """加载预训练 Checkpoint（``strict=False``，便于 obs-op 微调时的迁移学习）。"""
        pretrain_ckpt_path = self.training_config.get("pretrain_ckpt")
        if not pretrain_ckpt_path or not os.path.exists(pretrain_ckpt_path):
            if self.is_main:
                self.log.info(f"No {pretrain_ckpt_path} provided, training from scratch.")
            return

        if self.is_main:
            self.log.info(f"Loading pretrained checkpoint from: {pretrain_ckpt_path}")

        # 1. 加载字典 (使用 map_location 确保设备无关性)
        ckpt = torch.load(pretrain_ckpt_path, map_location=self.device)

        # 2. 加载模型权重
        self.model.load_state_dict(ckpt["model_state_dict"], strict=self.training_config.get("load_strict", True))

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
        """DDP 包装：当 ``world_size > 1`` 时用 ``DistributedDataParallel`` 包裹裸模型。

        注意:
            - 仅当 ``self.model`` 尚未被 ``DDP`` 包装时执行；
            - ``device_ids`` / ``output_device`` 在 CUDA / NPU 上绑定本地 rank；
            - ``_save_ckpt`` 中通过
              ``self.model.module if isinstance(self.model, DDP) else self.model``
              取出裸模型。
        """
        if self.world_size > 1 and not isinstance(self.model, DDP):
            self.model = self._wrap_single_ddp(self.model)

    def train_epoch(self, loader, epoch, epochs):
        """单 epoch 训练：单步 obs-op 前向 + 反向。

        Args:
            loader: 训练 DataLoader，batch 内容为
                ``(state, sat, sat_mask, tmbrs_std, tmbrs_vars)``：
                - ``state``: ``[B, 69, H, W]`` ERA5 状态场；
                - ``sat``: 卫星辅助场（cos(zenith), azimuth, scan/fov/orbit, satellite_height）；
                - ``sat_mask``: 扫描掩码（陆地/海冰/质量控制）；
                - ``tmbrs_std``: 各通道反归一化标准差；
                - ``tmbrs_vars``: 通道名称列表。
            epoch (int): 当前 epoch 编号。
            epochs (int): 总 epoch 数。

        Returns:
            dict: ``{"train/loss": float}``

        实现要点:
            - 模型前向返回 ``(out_sat, log_var, tgt_sat)``；
            - 损失按 ``torch.repeat_interleave(sat_mask, repeats=tgt_sat.shape[1], dim=1)``
              把单通道 mask 广播到所有亮温通道再传给 CRPS-Gaussian；
            - 走标准 AMP 反向链路 ``scaler.scale(loss).backward()`` →
              ``unscale_`` → ``clip_grad_norm_`` → ``scaler.step`` → ``scaler.update``。
        """
        self.model.train()
        total_loss = 0
        pbar = tqdm(loader, desc=f"Training epoch {epoch}/{epochs}", leave=False, disable=not self.is_main)

        for batch_idx, batch in enumerate(pbar):
            state, sat, sat_mask, tmbrs_std, tmbrs_vars = batch
            state = state.to(self.device)
            sat = sat.to(self.device)
            sat_mask = sat_mask.to(self.device)
            tmbrs_std = tmbrs_std.to(self.device)

            self.optimizer.zero_grad()

            with autocast(self.device_type, dtype=self.precision_type):
                out_sat, log_var, tgt_sat = self.model(state, sat, sat_mask, use_checkpoint=True)
                loss = self.loss_fn(
                    out_sat, 
                    log_var, 
                    tgt_sat, 
                    torch.repeat_interleave(sat_mask, repeats=tgt_sat.shape[1], dim=1)
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
        """单 epoch 验证：单步 obs-op 前向，按 ``out_sat_vars`` 计算 RMSE 与 obserr。

        Args:
            loader: 验证 DataLoader。
            epoch (int): 当前 epoch 编号。
            epochs (int): 总 epoch 数。

        Returns:
            dict: ``{"val/loss": float, "val/rmse_out_<ch>": float, "val/obserr_pred_<ch>": float, ...}``
            - ``val/loss``: 本 epoch 平均验证损失
            - ``val/rmse_out_<ch>``: 每个 ``out_sat_vars`` 通道反归一化 RMSE
            - ``val/obserr_pred_<ch>``: 每个通道 mask 加权 ``sqrt(exp(log_var))``
            共 ``1 + 2 * len(out_sat_vars)`` 个 key (per-satellite 通道数不同)。
            空 loader 时返回 ``{"val/loss": float("inf")}``,与 ``BaseTrainer``
            的契约保持一致。

        实现要点:
            - ``tgt_sat_var_ids`` 把模型 ``out_sat_vars`` 映射到 dataset
              ``tmbrs_vars`` 的索引，确保只统计模型实际输出的通道；
            - ``val_rmse`` = ``std * sqrt(sum(sat_mask * (out - tgt)**2) / (sum(mask) + eps))``，
              反映预测亮温的物理量纲误差；
            - ``val_obserr`` = ``std * sqrt(sum(exp(log_var) * sat_mask) / (sum(mask) + eps))``，
              反映预测观测误差估计的合理性。
        """
        if len(loader) == 0:
            # 空 dataloader (验证日期范围无文件匹配) — 跳过避免 ZeroDivisionError / ValueError.
            if self.is_main:
                self.log.warning(f"Epoch {epoch}/{epochs}: val_loader is empty, skipping validation.")
            return {"val/loss": float("inf")}
        self.model.eval()
        total_loss, total_mse, total_obserr_square = 0, 0, 0
        pbar = tqdm(loader, desc=f"Validating epoch {epoch}/{epochs}", leave=False, disable=not self.is_main)

        # 兼容 DDP 包装: 取出裸模型的 out_sat_vars 列表 (e.g. ATMS=22, AMSUA=15, ...)
        out_sat_vars = self.model.module.out_sat_vars if hasattr(self.model, "module") else self.model.out_sat_vars

        for batch_idx, batch in enumerate(pbar):
            state, sat, sat_mask, tmbrs_std, tmbrs_vars = batch
            state = state.to(self.device)
            sat = sat.to(self.device)
            sat_mask = sat_mask.to(self.device)
            tmbrs_std = tmbrs_std.to(self.device)

            tgt_sat_var_ids = np.array([tmbrs_vars.index(item) for item in out_sat_vars])
            tgt_sat_var_ids = torch.from_numpy(tgt_sat_var_ids).to(self.device)

            self.optimizer.zero_grad()

            with autocast(self.device_type, dtype=self.precision_type):
                out_sat, log_var, tgt_sat = self.model(state, sat, sat_mask, use_checkpoint=True)
                loss = self.loss_fn(
                    out_sat,
                    log_var,
                    tgt_sat,
                    torch.repeat_interleave(sat_mask, repeats=tgt_sat.shape[1], dim=1)
                )

                total_loss = total_loss + loss.item()

                val_rmse = tmbrs_std[0, tgt_sat_var_ids] * torch.sqrt(torch.sum(sat_mask * (out_sat - tgt_sat) ** 2, dim=(0, -2, -1)) / (sat_mask.sum(dim=(0, -2, -1)) + 1e-6))
                val_rmse = val_rmse.detach()
                val_obserr = tmbrs_std[0, tgt_sat_var_ids] * torch.sqrt(torch.sum(torch.exp(log_var) * sat_mask, dim=(0, -2, -1)) / (sat_mask.sum(dim=(0, -2, -1)) + 1e-6))
                val_obserr = val_obserr.detach()

                total_mse += val_rmse ** 2
                total_obserr_square += val_obserr ** 2

                pbar.set_postfix(
                    {
                        "val_loss": f"{loss.item():.4f}",
                        "val_rmse_ch0": f"{val_rmse[0].item():.4f}",  # 仍展示 ch0 作为指示
                    }
                )

        empty_cache(self.device_type)

        rmse_per_ch = torch.sqrt(total_mse / len(loader))                  # (N_sat_ch,)
        obserr_per_ch = torch.sqrt(total_obserr_square / len(loader))      # (N_sat_ch,)
        metrics = {"val/loss": float(total_loss / len(loader))}
        for ch_idx, ch_name in enumerate(out_sat_vars):
            metrics[f"val/rmse_out_{ch_name}"] = float(rmse_per_ch[ch_idx].item())
            metrics[f"val/obserr_pred_{ch_name}"] = float(obserr_per_ch[ch_idx].item())
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
