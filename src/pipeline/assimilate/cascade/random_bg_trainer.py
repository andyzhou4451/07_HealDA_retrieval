# -*- coding: utf-8 -*-
"""
RandomBgCascadeAssimTrainer - 随机背景场级联资料同化训练器
=========================================================

本模块实现了 XiChen 框架中 ``随机背景场 (Random Background)`` + ``级联 (Cascade)`` 资料同化任务的训练器。
它编排了 4 类组件 (背景场预报 / 观测算子 / 同化分析 / 变分代价) 的实例化、预训练权重加载、分布式包装、AMP 混合精度
以及 DDP 多卡训练循环，对外暴露 ``fit(train_loader, val_loader)`` 主入口。

核心职责：
    1. 按 ``model_training_config`` 实例化 ``Solver`` + ``DA_models`` + ``ObsOp_models`` + ``forecast_model``,
       并按 ``trainable`` 标志冻结或解冻对应参数；
    2. 加载 ``forecast / DA / ObsOp`` 三类预训练权重,支持 ``cross_model_loading`` 跨模型参数共享;
    3. 在外层自回归 (AR) 滚动预报 ``lead_times`` 步后,调用 ``Solver`` 一次性完成 7 类观测的同化分析;
    4. 采用 Kendall 多任务不确定度加权 (CRPS / L1 损失) 进行反向传播,支持 bf16/fp16 AMP 与 ``max_grad_norm`` 梯度裁剪;
    5. 每 epoch 保存 ``last.ckpt`` (断点续训) 和 ``best.ckpt`` (最佳验证损失)。

观测算子 (H_models) 与变分代价 (VarCost_models) 仅在 ``Solver`` 内部作为算子被调用,自身不参与训练,故不在 ``optimizer`` 之内。

典型 YAML 配置 (节选)::

    training:
        model_training_config:
            forecast_model: {trainable: false, pretrained_path: ".../forecast.ckpt"}
            da_models:
                atms: {trainable: true,  pretrained_path: ".../da_atms.ckpt"}
            obsop_models:
                atms: {trainable: false, pretrained_path: ".../obsop_atms.ckpt"}
        obs_list: {atms: {...}, amsua: {...}}
        lr: 1e-4
        epochs: 100
        use_amp: true
        precision: {type: bf16}

支持的观测源: atms, amsua, mhs, hrs4, prepbufr, satwnd, ascat。
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
from src.models.assimilate.utils.varcost import Obs_WeighedL2Norm, Model_Var_Cost, Model_H
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

class RandomBgCascadeAssimTrainer(BaseTrainer):
    """随机背景场级联资料同化训练器。

    负责级联方案 (cascade) 下 7 类观测 (atms/amsua/mhs/hrs4/prepbufr/satwnd/ascat) 的资料同化训练。
    训练过程中 ``Solver`` 会对每个观测源执行 ``AR 预报 → H(x) 观测算子 → VarCost 变分代价 → DA 分析`` 流程,
    最终对所有观测源的分析场取平均得到 ``xa``。

    Attributes:
        config (DictConfig): Hydra 全局配置,含 ``model`` / ``training`` / ``paths`` / ``loss_fn`` 等子段。
        device: 训练设备 (NPU 或 CUDA 张量)。
        local_rank (int): 当前进程本地 rank (DDP)。
        world_size (int): 分布式进程总数 (>1 表示 DDP 模式)。
        is_main (bool): 是否为主进程 (rank 0);控制日志/Profiler/TensorBoard 写入。
        training_config (DictConfig): ``config.training`` 子树,含学习率、调度器、冻结策略等。
        log_dir (str): TensorBoard 日志根目录。
        ckpt_dir (str): Checkpoint 输出目录 (last.ckpt / best.ckpt)。
        log: 名称为 ``xichen.trainer`` 的 DDP-rank-filtered logger。
        solver: 实例化后的 ``src.models.assimilate.fdvarsolver.cascade.Solver``。
        DA_models (nn.ModuleDict): 键为观测源,值为 ``XiChenDA`` 分析网络。
        ObsOp_models (nn.ModuleDict): 键为观测源,值为 ``XiChenObsOp`` 辐射率观测算子。
        forecast_model: 预训练自回归预报模型 (默认冻结)。
        H_models (nn.ModuleDict): 每个观测源对应的线性观测算子 ``Model_H(obs_err)`` (不可训练)。
        VarCost_models (nn.ModuleDict): 每个观测源对应的变分代价 ``Model_Var_Cost`` (不可训练)。
        obs_list (dict): 当前任务启用的观测源字典 (来自 ``training.obs_list``)。
        optimizer: ``torch.optim.AdamW`` 优化器,参数来自所有可训练组件。
        scheduler: 学习率调度器 (``CosineSchedulerWithWarmup`` 或 ``CosineAnnealingLR``)。
        loss_fn: Hydra 实例化的损失函数 (例如 CRPS-Gaussian)。
        scaler: AMP ``GradScaler`` (来自 ``src.utils.device.get_grad_scaler``)。
        precision_type (``torch.dtype``): 实际使用的 AMP 精度 (bf16/fp16/fp32)。
        writer: TensorBoard ``SummaryWriter`` (仅主进程)。
        profiler: torch NPU/GPU profiler (可选,仅主进程)。
        start_epoch (int): 断点续训起始 epoch。
        best_loss (float): 历史最佳验证损失。
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
        """初始化 cascade DA trainer — 全部由基类 __init__ 模板驱动."""
        super().__init__(cfg, device, local_rank, world_size, is_main, **kwargs)

    def _build_models(self) -> None:
        """一次性装配 6 子模型 (solver / DA / ObsOp / forecast / H / VarCost) + 预训练权重.

        调用顺序严格: 6 个 _init_* → _setup_model_training_states → _load_pretrained_models → _log_model_statistics.
        """
        # 获取模型训练配置
        model_training_config = self.training_config.get("model_training_config", {})

        # 实例化所有模型
        self._init_solver()
        self._init_da_models(model_training_config)
        self._init_obsop_models(model_training_config)
        self._init_forecast_model(model_training_config)
        self._init_observation_models()

        # 关键决策点:先设置 requires_grad 标志再 load_state_dict,
        # 以避免预训练权重在加载时被 strict=False 静默丢弃与冻结策略不一致的参数。
        self._setup_model_training_states(model_training_config)

        # 加载预训练权重(必须在 requires_grad 标志确定之后)
        self._load_pretrained_models(model_training_config)

        # 统计参数
        self._log_model_statistics()

    def _init_solver(self):
        """实例化 ``Solver`` (级联变分求解器) 并迁移到目标设备。

        ``Solver`` 是整个训练循环的核心算子,内含 AR 预报循环 + per-obs VarCost 代价 + DA 修正逻辑。
        实例化后立即 ``.to(self.device)``,避免后续逐参数迁移。
        """
        # 实例化模型
        self.solver = hydra.utils.instantiate(self.config.model.Solver, _recursive_=False).to(self.device)

    def _init_da_models(self, model_training_config):
        """实例化每个观测源对应的 ``XiChenDA`` 分析网络。

        对 ``config.model.DA_models`` 的每个条目 (键为观测源名,如 ``atms``) 依次实例化并放入
        ``self.DA_models`` (``nn.ModuleDict``)。若 ``model_training_config.da_models`` 中缺失该观测源,
        自动补齐 ``{"trainable": False, "pretrained_path": None}`` 默认配置,确保后续冻结/权重加载逻辑
        不会因 KeyError 中断。

        Args:
            model_training_config (dict): ``training.model_training_config.da_models`` 子树,会被原地补齐。
        """
        # 实例化模型
        da_config = model_training_config.get("da_models", {})
        self.DA_models = nn.ModuleDict()

        for name, model_config in self.config.model.DA_models.items():
            # 实例化模型
            model = hydra.utils.instantiate(model_config, _recursive_=False).to(self.device)
            self.DA_models[name] = model

            # 记录模型配置
            if name not in da_config:
                da_config[name] = {"trainable": False, "pretrained_path": None}

    def _init_obsop_models(self, model_training_config):
        """实例化每个观测源对应的 ``XiChenObsOp`` 辐射率观测算子。

        ObsOp_models 在 ``Solver`` 内部被 ``H(x)`` 调用以将模型状态空间投影到观测空间 (亮温 Tb)。
        它们默认冻结 (由 ``model_training_config`` 决定),仅作为静态算子使用。

        Args:
            model_training_config (dict): ``training.model_training_config.obsop_models`` 子树。
        """
        obsop_config = model_training_config.get("obsop_models", {})
        self.ObsOp_models = nn.ModuleDict()

        for name, model_config in self.config.model.ObsOp_models.items():
            # 实例化模型
            model = hydra.utils.instantiate(model_config, _recursive_=False).to(self.device)
            self.ObsOp_models[name] = model

            # 记录模型配置
            if name not in obsop_config:
                obsop_config[name] = {"trainable": False, "pretrained_path": None}

    def _init_forecast_model(self, model_training_config):
        """实例化自回归背景场预报模型 (``XiChenForecast``)。

        默认冻结 — 在 DA 训练中预报模型作为静态先验,只在 ``Solver`` 的 AR 循环里被调用;
        若 ``model_training_config.forecast_model.trainable=True`` 则会随主训练一起微调。

        Args:
            model_training_config (dict): ``training.model_training_config.forecast_model`` 子树。
        """
        forecast_config = model_training_config.get("forecast_model", {})
        self.forecast_model = hydra.utils.instantiate(
            self.config.model.forecast_model, _recursive_=False
        ).to(self.device)
        # 记录模型配置
        if not forecast_config:
            forecast_config.update({"trainable": False, "pretrained_path": None})

    def _init_observation_models(self):
        """实例化 H 观测算子和 VarCost 变分代价(全部冻结,仅在 ``Solver`` 内部被调用)。

        流程:
            * 读取 ``training.obs_list`` (本次任务启用的观测源字典);
            * 对每个观测源调用 ``_load_observation_error`` 加载其对应观测误差 npz;
            * ``H_models[obs] = Model_H(obs_err)`` — 将状态映射到观测空间;
            * ``VarCost_models[obs] = Model_Var_Cost(Obs_WeighedL2Norm(obs_err))`` — R⁻¹ 加权 L2 代价。

        这些模块不需要梯度,故不会进入 optimizer 也不参与 DDP 包装 (DDP wrap 阶段会跳过)。
        """
        H_models = {}
        VarCost_models = {}
        self.obs_list = self.training_config.get("obs_list", {})
        obs_dir = self.config.paths.get("obs_dir", "data/obs")

        for obs_name in self.obs_list.keys():
            # 加载观测误差数据
            obs_err = self._load_observation_error(obs_name, obs_dir)

            # 创建观测模型
            H_models[obs_name] = Model_H(obs_err)
            m_NormObs = Obs_WeighedL2Norm(obs_err)
            VarCost_models[obs_name] = Model_Var_Cost(m_NormObs)

        # 保存观测模型
        self.H_models = nn.ModuleDict({name: model for name, model in H_models.items()})
        self.VarCost_models = nn.ModuleDict({name: model for name, model in VarCost_models.items()})

    def _load_observation_error(self, obs_name, obs_dir):
        """根据观测源类型加载对应的观测误差 std (用于 ``Obs_WeighedL2Norm`` 与 ``Model_H``)。

        不同观测源的 npz 路径与键名存在差异,故按来源分派:
            * 微波辐射率 (atms/amsua/mhs/hrs4) — 1b<sat>_merged_npy_1.0deg/avg_obs_error.npz;
            * prepbufr (常规探空) — GDAS_prepbufr_merged_npy_1.0deg/obs_sigma.npz;
            * satwnd (卫星风) — satwnd_merged_npy_1.0deg/obs_sigma.npz;
            * ascat (散射计海面风) — ascat_b_merged_npy_1.0deg/obs_sigma.npz。

        Args:
            obs_name (str): 观测源名 (atms/amsua/mhs/hrs4/prepbufr/satwnd/ascat)。
            obs_dir (str): ``config.paths.obs_dir`` 路径。

        Returns:
            torch.Tensor: 形状 ``(C_obs,)`` 的 obs σ 张量, ``requires_grad=False``。
        """
        if obs_name in ["atms", "amsua", "mhs", "hrs4"]:
            obs_err = np.load(os.path.join(obs_dir, f"1b{obs_name}_merged_npy_1.0deg", "avg_obs_error.npz"))
        elif obs_name == "prepbufr":
            obs_err = np.load(os.path.join(obs_dir, "GDAS_prepbufr_merged_npy_1.0deg/obs_sigma.npz"))
        elif obs_name == "satwnd":
            obs_err = np.load(os.path.join(obs_dir, "satwnd_merged_npy_1.0deg/obs_sigma.npz"))
        elif obs_name == "ascat":
            obs_err = np.load(os.path.join(obs_dir, "ascat_b_merged_npy_1.0deg/obs_sigma.npz"))

        obs_err_list = []
        for i, key in enumerate(obs_err.keys()):
            obs_err_list.append(torch.tensor(obs_err[key], dtype=torch.float32, requires_grad=False))

        return torch.stack(obs_err_list, dim=0)

    def _setup_model_training_states(self, model_training_config):
        """按 ``model_training_config`` 中各组件的 ``trainable`` 标志统一设置 ``requires_grad``。

        关键决策点 — 此方法 **必须** 在 ``_load_pretrained_models`` 之前调用。
        因为 ``load_state_dict(strict=False)`` 不会改变参数的 ``requires_grad`` 标志;
        若先加载预训练权重,某些被冻结的子模块可能会保留其原始梯度状态,与配置不一致。

        Args:
            model_training_config (dict): ``training.model_training_config`` 子树。
        """
        forecast_config = model_training_config.get("forecast_model", {})
        da_config = model_training_config.get("da_models", {})
        obsop_config = model_training_config.get("obsop_models", {})

        # 设置预报模型训练状态
        forecast_trainable = forecast_config.get("trainable", False)
        for param in self.forecast_model.parameters():
            param.requires_grad = forecast_trainable

        # 设置DA模型训练状态
        for name, model in self.DA_models.items():
            is_trainable = da_config.get(name, {}).get("trainable", False)
            for param in model.parameters():
                param.requires_grad = is_trainable

        # 设置ObsOp模型训练状态
        for name, model in self.ObsOp_models.items():
            is_trainable = obsop_config.get(name, {}).get("trainable", False)
            for param in model.parameters():
                param.requires_grad = is_trainable

    def _load_pretrained_models(self, model_training_config):
        """按顺序加载 ``forecast → ObsOp → DA`` 三类预训练权重。

        支持 ``cross_model_loading`` 跨模型参数共享:某 DA 观测源可用另一个观测源 DA 的
        检查点进行初始化(常用于无该卫星训练数据时的迁移)。

        加载顺序约定:先 ``forecast_model``,再 ``ObsOp_models``,最后 ``DA_models``,
        便于在日志中按依赖链追踪权重来源。

        Args:
            model_training_config (dict): ``training.model_training_config`` 子树。
        """
        da_config = model_training_config.get("da_models", {})
        obsop_config = model_training_config.get("obsop_models", {})
        forecast_config = model_training_config.get("forecast_model", {})

        # 加载预报模型预训练权重
        if "pretrained_path" in forecast_config and forecast_config["pretrained_path"]:
            self._load_model_checkpoint(
                self.forecast_model,
                forecast_config["pretrained_path"],
                forecast_config.get("trainable", False),
                strict=forecast_config.get("strict", False)
            )

        # 加载ObsOp模型预训练权重
        for name, config in obsop_config.items():
            if "pretrained_path" in config and config["pretrained_path"]:
                self._load_model_checkpoint(
                    self.ObsOp_models[name],
                    config["pretrained_path"],
                    config.get("trainable", False),
                    strict=config.get("strict", False),
                    model_name=name
                )

        # 加载DA模型预训练权重
        for name, config in da_config.items():
            if "pretrained_path" in config and config["pretrained_path"]:
                # 检查是否有跨模型加载配置
                cross_model_config = config.get("cross_model_loading", {})
                if cross_model_config and cross_model_config.get("enabled", False):
                    # 跨模型加载：使用其他模型的检查点
                    source_model = cross_model_config.get("source_model")
                    self._load_model_checkpoint(
                        self.DA_models[name],
                        da_config[source_model]["pretrained_path"],  # 使用源模型的路径
                        config.get("trainable", False),
                        strict=config.get("strict", False),
                        model_name=source_model  # 指定源模型名称
                    )
                else:
                    # 常规加载：使用自己的检查点
                    self._load_model_checkpoint(
                        self.DA_models[name],
                        config["pretrained_path"],
                        config.get("trainable", False),
                        strict=config.get("strict", False),
                        model_name=name  # 指定模型名称
                    )

    def _load_model_checkpoint(self, model, checkpoint_path, is_trainable, strict=True, model_name=None):
        """加载单个模型检查点(支持多种 ckpt 格式 + 跨模型加载),加载后强制刷新 ``requires_grad``。

        支持的 ckpt 格式(按优先级检查):
            1. ``{"model_state_dict": ...}`` — 标准训练器保存格式;
            2. ``{"state_dict": ...}`` — Lightning / DDP 通用格式;
            3. ``{"da_model_state_dict": {"atms": ..., "amsua": ...}}`` — 多观测源 DA 打包格式;
               此时按 ``model_name`` 选择对应子字典,若未指定则取第一个;
            4. 直接 ``torch.save(model.state_dict())`` — 裸 state_dict。

        加载完成后无论 ``strict`` 设置如何,均按 ``is_trainable`` 强制刷新所有参数的
        ``requires_grad`` 标志,确保与 ``model_training_config`` 一致。

        Args:
            model (``nn.Module``): 目标模型。
            checkpoint_path (str): ``.ckpt`` 路径;若不存在则仅 ``log.warning`` 并跳过。
            is_trainable (bool): 加载后是否允许梯度。
            strict (bool): ``load_state_dict`` 的 ``strict`` 参数;默认 ``True``。
            model_name (str, optional): 跨模型加载时,``da_model_state_dict`` 中要取哪一份参数。
        """
        if not os.path.exists(checkpoint_path):
            if self.is_main:
                self.log.warning(f"Pretrained checkpoint not found: {checkpoint_path}")
            return

        if self.is_main:
            self.log.info(f"Loading pretrained model from: {checkpoint_path}")

        try:
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            state_dict = None

            # 支持多种检查点格式
            if "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]
            elif "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            elif "da_model_state_dict" in ckpt:
                # 如果提供了模型名称，尝试加载特定模型的参数
                if model_name and model_name in ckpt["da_model_state_dict"]:
                    state_dict = ckpt["da_model_state_dict"][model_name]
                # 如果没有指定模型名称，尝试加载第一个可用的模型参数
                elif ckpt["da_model_state_dict"]:
                    first_model_name = next(iter(ckpt["da_model_state_dict"].keys()))
                    state_dict = ckpt["da_model_state_dict"][first_model_name]
                    if self.is_main:
                        self.log.info(f"Using parameters from model: {first_model_name}")
                else:
                    raise ValueError("No DA model state dict found in checkpoint")
            else:
                state_dict = ckpt

            # 如果state_dict仍然为空，抛出异常
            if state_dict is None:
                raise ValueError("Could not extract state dictionary from checkpoint")

            # 加载权重
            model.load_state_dict(state_dict, strict=strict)

            # 设置梯度状态
            for param in model.parameters():
                param.requires_grad = is_trainable

            if self.is_main:
                self.log.info(f"Successfully loaded pretrained model from {checkpoint_path}")

        except Exception as e:
            if self.is_main:
                self.log.error(f"Failed to load pretrained model from {checkpoint_path}: {e}")


    def _log_model_statistics(self):
        """打印每个组件的 (总参数, 可训练参数, 比例) 统计信息 (仅主进程)。

        用途:帮助确认 ``model_training_config`` 冻结策略是否按预期生效,以及全局可训练参数量级。
        注意: ``H_models`` (obs operator) 与 ``VarCost_models`` 是 forward-time 算子,
        不参与 autograd 反向,故不计入 ``all_models`` 字典;若需全量审计请单独统计。
        """
        if self.is_main:
            # 统计所有模型参数
            all_models = {
                "forecast_model": self.forecast_model,
                "DA_models": self.DA_models,
                "ObsOp_models": self.ObsOp_models,
            }

            total_params = 0
            trainable_params = 0

            for model_group_name, model_group in all_models.items():
                if isinstance(model_group, nn.ModuleDict):
                    for name, model in model_group.items():
                        model_total = sum(p.numel() for p in model.parameters())
                        model_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                        total_params += model_total
                        trainable_params += model_trainable

                        self.log.info(
                            f"{model_group_name}.{name}: "
                            f"Total={format_number(model_total)}, "
                            f"Trainable={format_number(model_trainable)}, "
                            f"Ratio={model_trainable/model_total*100:.2f}%"
                        )
                else:
                    model_total = sum(p.numel() for p in model_group.parameters())
                    model_trainable = sum(p.numel() for p in model_group.parameters() if p.requires_grad)
                    total_params += model_total
                    trainable_params += model_trainable

                    self.log.info(
                        f"{model_group_name}: "
                        f"Total={format_number(model_total)}, "
                        f"Trainable={format_number(model_trainable)}, "
                        f"Ratio={model_trainable/model_total*100:.2f}%"
                    )

            self.log.info(
                f"Overall - Total: {format_number(total_params)}, "
                f"Trainable: {format_number(trainable_params)}, "
                f"Ratio: {trainable_params/total_params*100:.2f}%"
            )

    def _setup_optimizer_scheduler(self):
        """收集所有 ``requires_grad=True`` 的参数,构建 ``AdamW`` 优化器和 LR 调度器。

        参数来源(按收集顺序): ``solver`` → ``forecast_model`` → ``ObsOp_models`` → ``DA_models``。
        不可训练组件(冻结的)不会进入 optimizer,也就不会收到梯度更新。

        调度器选择:
            * ``cosine_warmup`` (默认) — ``CosineSchedulerWithWarmup`` (带 warmup 的余弦退火);
            * ``cosine`` — ``CosineAnnealingLR`` (无 warmup);
            * 其他 — 不用调度器,LR 保持常量。

        Raises:
            ValueError: 没有任何可训练参数时 (说明 ``model_training_config`` 配置错误)。
        """
        if self.is_main:
            self.log.info("Initializing optimizer...")

        # 收集所有需要训练的参数
        train_params = []

        # 收集solver模型中需要训练的参数
        if hasattr(self, 'solver') and any(param.requires_grad for param in self.solver.parameters()):
            train_params.extend(
                [param for param in self.solver.parameters() if param.requires_grad]
            )
            if self.is_main:
                self.log.info("Adding solver model parameters to optimizer")

        # 收集预报模型中需要训练的参数
        if hasattr(self, 'forecast_model') and any(param.requires_grad for param in self.forecast_model.parameters()):
            train_params.extend(
                [param for param in self.forecast_model.parameters() if param.requires_grad]
            )
            if self.is_main:
                self.log.info("Adding forecast model parameters to optimizer")

        # 收集ObsOp模型中需要训练的参数
        if hasattr(self, 'ObsOp_models'):
            for name, model in self.ObsOp_models.items():
                if any(param.requires_grad for param in model.parameters()):
                    train_params.extend(
                        [param for param in model.parameters() if param.requires_grad]
                    )
                    if self.is_main:
                        self.log.info(f"Adding ObsOp model '{name}' parameters to optimizer")

        # 收集DA模型中需要训练的参数
        if hasattr(self, 'DA_models'):
            for name, model in self.DA_models.items():
                if any(param.requires_grad for param in model.parameters()):
                    train_params.extend(
                        [param for param in model.parameters() if param.requires_grad]
                    )
                    if self.is_main:
                        self.log.info(f"Adding DA model '{name}' parameters to optimizer")

        if len(train_params) == 0:
            raise ValueError("No trainable parameters found. Check trainable_models configuration.")

        # 优化器
        self.optimizer = torch.optim.AdamW(
            params=train_params,
            lr=self.training_config.get("lr", 1e-4),
            betas=self.training_config.get("betas", [0.9, 0.95]),
            weight_decay=self.training_config.get("weight_decay", 5e-5),
        )

        if self.is_main:
            self.log.info("Initializing scheduler...")
        # 调度器
        self.scheduler = self._build_scheduler(self.optimizer)

    def _load_checkpoint(self):
        """断点续训:从 ``last.ckpt`` 恢复 solver / DA / ObsOp / forecast / H / VarCost / 优化器 / 调度器 / AMP 状态。

        关键决策点 — 此方法 **必须** 在 ``_wrap_ddp`` 之前调用。原因:DDP 包装后,
        ``state_dict`` 键名前会加 ``module.`` 前缀,直接 ``load_state_dict`` 会因键不匹配而失败。

        Raises:
            Exception: ckpt 加载或 state_dict 恢复过程中任何异常都会被记录并重新抛出,
                以便上游决定是否中止训练。
        """
        ckpt_path = os.path.join(self.ckpt_dir, "last.ckpt")
        if not os.path.exists(ckpt_path):
            if self.is_main:
                self.log.info("No checkpoint found, training from scratch.")
            return

        if self.is_main:
            self.log.info(f"Resuming training from: {ckpt_path}")

        try:
            ckpt = torch.load(ckpt_path, map_location=self.device)

            # 加载所有模型权重
            if "solver_state_dict" in ckpt:
                self.solver.load_state_dict(ckpt["solver_state_dict"])
            if "forecast_state_dict" in ckpt:
                self.forecast_model.load_state_dict(ckpt["forecast_state_dict"])
            if "obsop_state_dict" in ckpt:
                self.ObsOp_models.load_state_dict(ckpt["obsop_state_dict"])
            if "da_model_state_dict" in ckpt:
                self.DA_models.load_state_dict(ckpt["da_model_state_dict"])
            if "h_models_state_dict" in ckpt:
                self.H_models.load_state_dict(ckpt["h_models_state_dict"])
            if "varcost_models_state_dict" in ckpt:
                self.VarCost_models.load_state_dict(ckpt["varcost_models_state_dict"])

            # 加载优化器状态
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

            # 加载调度器状态（如果存在且不为None）
            if self.scheduler and "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"] is not None:
                self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])

            # 加载AMP标量（如果存在且不为None）
            if self.scaler and "scaler_state_dict" in ckpt and ckpt["scaler_state_dict"] is not None:
                self.scaler.load_state_dict(ckpt["scaler_state_dict"])

            # 恢复标量状态
            self.start_epoch = ckpt.get("epoch", -1) + 1  # 兼容无 epoch 字段的旧 ckpt
            self.best_loss = ckpt.get("best_loss", float("inf"))  # 兼容无 best_loss 字段的旧 ckpt

            # 重新设置模型梯度
            self._setup_model_training_states(self.training_config.get("model_training_config", {}))

            if self.is_main:
                self.log.info(f"Resumed successfully. Start Epoch: {self.start_epoch}, Best Loss: {self.best_loss:.4f}")

        except Exception as e:
            if self.is_main:
                self.log.error(f"Failed to load checkpoint: {e}")
            raise

    def _wrap_ddp(self):
        """为所有可训练组件执行 DDP 包装(world_size > 1 时)。

        关键决策点:
            * 仅对 ``requires_grad=True`` 的组件包装 — 冻结的子模块不必同步梯度,可显著降低通信开销;
            * H_models / VarCost_models 不参与前向梯度 (它们是算子),也跳过包装;
            * ``find_unused_parameters`` 通过 ``training_config.find_unused_parameters``
              控制（默认 ``True``） — 适配 DDP 中部分参数在某 step 缺席前向图的情况
              (例如冻结的 DA 模型在 cascade 单 obs 任务里不会全部出现在计算图中)。
        """
        if self.world_size > 1:
            # 只包装需要训练的模型
            models_to_wrap = []

            # 检查solver模型是否需要包装（修正拼写错误）
            if hasattr(self, 'solver') and any(param.requires_grad for param in self.solver.parameters()):
                models_to_wrap.append(("solver", self.solver))

            # 检查预报模型是否需要包装
            if hasattr(self, 'forecast_model') and any(param.requires_grad for param in self.forecast_model.parameters()):
                models_to_wrap.append(("forecast_model", self.forecast_model))

            # 检查ObsOp模型是否需要包装
            if hasattr(self, 'ObsOp_models'):
                for name, model in self.ObsOp_models.items():
                    if any(param.requires_grad for param in model.parameters()):
                        models_to_wrap.append((name, model))

            # 检查DA模型是否需要包装
            if hasattr(self, 'DA_models'):
                for name, model in self.DA_models.items():
                    if any(param.requires_grad for param in model.parameters()):
                        models_to_wrap.append((name, model))

            # 检查H_models模型是否需要包装
            if hasattr(self, 'H_models'):
                for name, model in self.H_models.items():
                    if any(param.requires_grad for param in model.parameters()):
                        models_to_wrap.append((name, model))

            # 检查VarCost_models模型是否需要包装
            if hasattr(self, 'VarCost_models'):
                for name, model in self.VarCost_models.items():
                    if any(param.requires_grad for param in model.parameters()):
                        models_to_wrap.append((name, model))

            # 包装所有需要训练的模型
            for name, model in models_to_wrap:
                if not isinstance(model, DDP):
                    setattr(self, name, self._wrap_single_ddp(model))

    def train_epoch(self, loader, epoch, epochs):
        """训练一个 epoch:solver.train / 其他模块均 eval,完成 AR 预报 → 同化 → 损失 → 反向传播。

        关键决策点 — 模块模式:
            * ``solver.train()`` — Solver 内的 DA 网络需要接收梯度;
            * ``forecast_model.eval()`` / ``ObsOp_models[*].eval()`` / ``DA_models[*].eval()`` —
              这些组件在此 batch 不接收梯度,使用 BN/dropout 的 eval 路径更稳定;
            * 外层 AR 循环手动 ``detach()`` 截断 ``inps`` 的计算图,避免 AR 滚动过程中显存爆炸;
            * 损失在 ``autocast`` 区域计算,经 ``GradScaler`` 缩放后反传;
            * 多个 ``solver.parameters()`` 列表聚合后做一次 ``clip_grad_norm_`` (max_norm=1.0 默认)。

        Args:
            loader: 训练 ``DataLoader``。
            epoch (int): 当前 epoch 索引 (0-indexed)。
            epochs (int): 总 epoch 数,用于进度条显示。

        Returns:
            float: 该 epoch 的平均训练损失。
        """
        # 设置所有模型为训练模式
        if hasattr(self, 'solver'):
            self.solver.train()
        if hasattr(self, 'forecast_model'):
            self.forecast_model.eval()
        if hasattr(self, 'ObsOp_models'):
            for model in self.ObsOp_models.values():
                model.eval()
        if hasattr(self, 'DA_models'):
            for model in self.DA_models.values():
                model.eval()
        total_loss = 0
        pbar = tqdm(loader, desc=f"Training epoch {epoch}/{epochs}", leave=False, disable=not self.is_main)

        for batch_idx, batch in enumerate(pbar):
            inps, obs_list, obs_data, obs_mask, tgt, lead_times, variables, obs_dict, \
            init_time, tgt_time, era5_transforms, microwave_transforms, conventional_transforms = batch
            inps = inps.to(self.device)
            for obs_name in obs_list:
                obs_data[obs_name] = obs_data[obs_name].to(self.device)
                obs_mask[obs_name] = obs_mask[obs_name].to(self.device)
            tgt = tgt.to(self.device)
            lead_times = lead_times.to(self.device)
            std_dict = {}
            std_dict["era5"] = torch.from_numpy(era5_transforms["std"]).to(self.device)
            for name in obs_list:
                if name in microwave_transforms.keys():
                    std_dict[name] = torch.from_numpy(microwave_transforms[name]["std"]).to(self.device)
                elif name in conventional_transforms.keys():
                    std_dict[name] = torch.from_numpy(conventional_transforms[name]["std"]).to(self.device)

            for iter in range(lead_times.shape[-1]):
                inps, log_var = self.forecast_model(
                    inps,
                    lead_times[:, iter:iter+1],
                    variables,
                    use_checkpoint=True
                )
                inps, log_var = inps.detach(), log_var.detach()

            self.optimizer.zero_grad()

            with autocast(self.device_type, dtype=self.precision_type):
                xa, log_var = self.solver(
                    self.forecast_model,
                    self.ObsOp_models,
                    self.DA_models,
                    self.H_models,
                    self.VarCost_models,
                    self.obs_list,
                    inps,
                    obs_data,
                    obs_mask,
                    obs_dict,
                    std_dict,
                    variables
                )

                loss = self.loss_fn(
                    xa,
                    log_var,
                    tgt,
                    torch.ones_like(log_var).to(self.device, dtype=log_var.dtype)
                )

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)

            all_params = []
            if hasattr(self, 'solver'):
                all_params.extend(self.solver.parameters())
            if hasattr(self, 'forecast_model'):
                all_params.extend(self.forecast_model.parameters())
            if hasattr(self, 'ObsOp_models'):
                for model in self.ObsOp_models.values():
                    all_params.extend(model.parameters())
            if hasattr(self, 'DA_models'):
                for model in self.DA_models.values():
                    all_params.extend(model.parameters())

            torch.nn.utils.clip_grad_norm_(
                all_params,
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
        """验证一个 epoch:全部模块 ``eval()``,无梯度回传;按变量累计 MSE,最终返回平均 RMSE。

        关键决策点:
            * ``xa.detach(), log_var.detach()`` — 验证阶段截断计算图,降低显存;
            * 损失在 ``autocast`` 区域计算;
            * ``val_rmse`` 按变量分通道乘以 std 还原到物理单位 (与训练损失同口径);
            * 进度条同步显示 ``val_loss`` 与 ``val_rmse`` 字符串;
            * TensorBoard 写入放在主进程 + writer 存在的前提下,避免 DDP 重复写。

        Args:
            loader: 验证 ``DataLoader``。
            epoch (int): 当前 epoch 索引。
            epochs (int): 总 epoch 数。

        Returns:
            dict: ``{"val/loss": float, "val/rmse_<var_name>": float, ...}``
            - ``val/loss``: 本 epoch 平均 DA 验证损失
            - ``val/rmse_<var_name>``: 每个变量反归一化加权 RMSE (per-var
              累计),共 ``len(variables)`` 个 key
            空 loader 时返回 ``{"val/loss": float("inf")}``。
        """
        if len(loader) == 0:
            # 空 dataloader (验证日期范围无文件匹配) — 跳过避免 ZeroDivisionError.
            if self.is_main:
                self.log.warning(f"Epoch {epoch}/{epochs}: val_loader is empty, skipping validation.")
            return {"val/loss": float("inf")}
        # 设置所有模型为训练模式
        if hasattr(self, 'solver'):
            self.solver.eval()
        if hasattr(self, 'forecast_model'):
            self.forecast_model.eval()
        if hasattr(self, 'DA_models'):
            for model in self.DA_models.values():
                model.eval()
        if hasattr(self, 'ObsOp_models'):
            for model in self.ObsOp_models.values():
                model.eval()

        total_loss = 0
        total_mse_dict = {}
        pbar = tqdm(loader, desc=f"Validating epoch {epoch}/{epochs}", leave=False, disable=not self.is_main)

        for batch_idx, batch in enumerate(pbar):
            inps, obs_list, obs_data, obs_mask, tgt, lead_times, variables, obs_dict, \
            init_time, tgt_time, era5_transforms, microwave_transforms, conventional_transforms = batch
            inps = inps.to(self.device)
            for obs_name in obs_list:
                obs_data[obs_name] = obs_data[obs_name].to(self.device)
                obs_mask[obs_name] = obs_mask[obs_name].to(self.device)
            tgt = tgt.to(self.device)
            lead_times = lead_times.to(self.device)
            std_dict = {}
            std_dict["era5"] = torch.from_numpy(era5_transforms["std"]).to(self.device)
            for name in obs_list:
                if name in microwave_transforms.keys():
                    std_dict[name] = torch.from_numpy(microwave_transforms[name]["std"]).to(self.device)
                elif name in conventional_transforms.keys():
                    std_dict[name] = torch.from_numpy(conventional_transforms[name]["std"]).to(self.device)

            for iter in range(lead_times.shape[-1]):
                inps, log_var = self.forecast_model(
                    inps,
                    lead_times[:, iter:iter+1],
                    variables,
                    use_checkpoint=True
                )
                inps, log_var = inps.detach(), log_var.detach()

            with autocast(self.device_type, dtype=self.precision_type):
                xa, log_var = self.solver(
                    self.forecast_model,
                    self.ObsOp_models,
                    self.DA_models,
                    self.H_models,
                    self.VarCost_models,
                    self.obs_list,
                    inps,
                    obs_data,
                    obs_mask,
                    obs_dict,
                    std_dict,
                    variables
                )

                xa, log_var = xa.detach(), log_var.detach()

                loss = self.loss_fn(
                    xa,
                    log_var,
                    tgt,
                    torch.ones_like(log_var).to(self.device, dtype=log_var.dtype)
                )

                total_loss = total_loss + loss.item()

                # 计算所有变量的RMSE
                val_rmse = std_dict["era5"].to(self.device, dtype=xa.dtype) * weighted_rmse_torch(xa, tgt)

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
                        "val_loss": f"{(loss.item()):.4f}",
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
        """保存一份完整 Checkpoint(``last.ckpt`` / ``best.ckpt``)以支持断点续训与推理。

        关键决策点 — DDP 安全:
            * 每个 ``nn.Module`` 都要先 ``module.state_dict() if isinstance(m, DDP) else m.state_dict()``
              取出底层模型的 ``state_dict``(避免键名带 ``module.`` 前缀,影响推理侧 ``strict=False`` 加载);
            * 同步保存 ``optimizer`` / ``scheduler`` / ``GradScaler`` 状态,保证续训时优化动量一致;
            * ``config`` 也一并保存,便于推理时还原训练配置;
            * ``is_best`` 参数目前仅用于日志区分,不改变保存路径。

        Args:
            ckpt_dir (str): Checkpoint 输出目录。
            filename (str): 文件名(例如 ``"last.ckpt"`` / ``"best.ckpt"``)。
            epoch (int): 当前 epoch 索引。
            val_loss (float): 当前验证损失。
            is_best (bool): 是否为历史最佳。
        """
        # 获取所有模型的状态（处理DDP情况）
        solver_to_save = self.solver.module if isinstance(self.solver, DDP) else self.solver
        forecast_model_to_save = self.forecast_model.module if isinstance(self.forecast_model, DDP) else self.forecast_model
        obsop_model_to_save = self.ObsOp_models.module if isinstance(self.ObsOp_models, DDP) else self.ObsOp_models
        da_model_to_save = self.DA_models.module if isinstance(self.DA_models, DDP) else self.DA_models
        h_models_to_save = self.H_models.module if isinstance(self.H_models, DDP) else self.H_models
        varcost_models_to_save = self.VarCost_models.module if isinstance(self.VarCost_models, DDP) else self.VarCost_models

        ckpt_dict = {
            "epoch": epoch,
            "solver_state_dict": solver_to_save.state_dict(),
            "forecast_state_dict": forecast_model_to_save.state_dict(),
            "obsop_state_dict": obsop_model_to_save.state_dict(),
            "da_model_state_dict": da_model_to_save.state_dict(),
            "h_models_state_dict": h_models_to_save.state_dict(),
            "varcost_models_state_dict": varcost_models_to_save.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
            "best_loss": self.best_loss,
            "config": self.config,  # 新增：保存配置以便恢复
        }

        save_path = os.path.join(ckpt_dir, filename)
        torch.save(ckpt_dict, save_path)
        if self.is_main:
            self.log.info(f"Checkpoint saved to {save_path}")
