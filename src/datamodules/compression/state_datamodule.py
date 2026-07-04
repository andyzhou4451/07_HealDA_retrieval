# -*- coding: utf-8 -*-
"""
天气压缩任务 (auto-encoder) 的状态场 DataModule。

本模块实现 ``StateCompressionDataModule``，为 :class:`src.models.compression.arch.XiChenAutoEncoder`
提供 train / val / test 三段 ERA5 状态场数据。

与 :mod:`src.datamodules.forecast.state_datamodule` 的差异：
    - 不需要 ``max_lead_time`` / ``iter_num`` —— 压缩任务只重建"当前时刻"，不预测未来；
    - 内部 Dataset 是 :mod:`src.datamodules.compression.state_dataset` (更简单)；
    - collate 函数复用 forecast 的 ``collate_fn_forecast`` (字段定义相同)。
"""
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.transforms import Normalize

from src.datamodules.compression.state_dataset import NpyDataset, collate_fn_forecast

class StateCompressionDataModule:
    """天气压缩任务的 DataModule (纯 torch 实现)。

    Attributes:
        root_dir (str): 数据根目录。
        variables (list): 气象变量名列表。
        batch_size (int): DataLoader 批次大小。
        num_workers (int): DataLoader 子进程数。
        shuffle (bool): 是否打乱样本。
        pin_memory (bool): 是否使用锁页内存。
        prefetch_factor (int): 每个 worker 预取批次数。
        distributed (bool): 是否启用 DDP 分布式采样。
        num_replicas (int): DDP world size。
        rank (int): 当前进程的 local rank。
        seed (int): 随机种子。
        debug (bool): 是否仅加载 1 月数据 (调试模式)。
        start_train_year / start_val_year / start_test_year / end_year (int): 切片年份。
        mean (np.ndarray): 归一化均值。
        std (np.ndarray): 归一化标准差。
        transforms (Normalize): ``torchvision.transforms.Normalize`` 实例。
        train_data / val_data / test_data (Optional[NpyDataset]): 三个分段的数据集。
    """

    def __init__(
        self,
        root_dir: str,
        variables: list,
        start_train_year: int = 2010,
        start_val_year: int = 2022,
        start_test_year: int = 2023,
        end_year: int = 2024,
        seed: int = 1024,
        batch_size: int = 32,
        num_workers: int = 4,
        shuffle: bool = True,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
        distributed: bool = False,
        num_replicas: int = 1,
        rank: int = 0,
        debug: bool = False,
    ):
        """
        Args:
            root_dir: 数据根目录 (包含normalized_mean_std子目录)
            variables: 气象变量列表
            start_train_year: 训练集起始年份
            start_val_year: 验证集起始年份
            start_test_year: 测试集起始年份
            end_year: 数据结束年份
            seed: 随机种子
            batch_size: 批次大小
            num_workers: 数据加载线程数
            shuffle: 是否打乱
            pin_memory: 是否锁页内存
            prefetch_factor: 预取因子
            distributed: 是否分布式训练
            num_replicas: 分布式副本数
            rank: 分布式rank
            debug: debug mode
        """
        self.root_dir = root_dir
        self.variables = variables
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.distributed = distributed
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.debug = debug

        # Year parameters (直接存储，无需属性包装)
        self.start_train_year = start_train_year
        self.start_val_year = start_val_year
        self.start_test_year = start_test_year
        self.end_year = end_year

        # 验证年份顺序
        assert start_val_year > start_train_year
        assert start_test_year > start_val_year
        assert end_year > start_test_year

        # 归一化
        self.mean, self.std = self._load_normalize()
        self.transforms = Normalize(self.mean, self.std)

        # 数据集 (延迟初始化)
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def _load_normalize(self):
        """加载并拼接 ``variables`` 对应的归一化 mean / std。

        从 ``{root_dir}/normalized_mean_std/normalize_mean.npz`` 和 ``normalize_std.npz`` 中
        按 ``self.variables`` 顺序读取每个变量的均值与标准差，并沿第 0 维拼接为一维数组。
        对 ``tp`` (总降水量) 强制使用 ``[0.0]`` 作为均值。

        Returns:
            tuple[np.ndarray, np.ndarray]: (mean, std)，shape 均为 ``[len(variables)]``。
        """
        normalize_mean = dict(np.load(os.path.join(self.root_dir, "normalized_mean_std", "normalize_mean.npz")))
        mean = []
        for var in self.variables:
            if var != "tp":
                mean.append(normalize_mean[var])
            else:
                mean.append(np.array([0.0]))
        normalize_mean = np.concatenate(mean)

        normalize_std = dict(np.load(os.path.join(self.root_dir, "normalized_mean_std", "normalize_std.npz")))
        normalize_std = np.concatenate([normalize_std[var] for var in self.variables])

        return normalize_mean, normalize_std

    def _worker_init_fn(self, worker_id):
        """DataLoader worker 初始化回调。

        每个 worker 子进程启动时被调用一次，向 numpy 注入
        ``seed + worker_id`` 作为随机种子，确保多进程下随机增强可复现。

        Args:
            worker_id (int): 由 DataLoader 传入的 worker 编号。
        """
        np.random.seed(self.seed + worker_id)

    def _get_sampler(self, dataset):
        """根据是否启用 DDP 选择采样器。

        启用 DDP (``distributed=True`` 且 ``num_replicas>1``) 时返回
        :class:`DistributedSampler`，保证各 rank 看到不重叠的样本切片；
        否则返回 ``None``，DataLoader 会以默认顺序采样。

        Args:
            dataset (Dataset): 已构造好的 :class:`NpyDataset` 实例。

        Returns:
            Optional[DistributedSampler]: 分布式采样器或 ``None``。
        """
        if self.distributed and self.num_replicas > 1:
            return DistributedSampler(
                dataset,
                num_replicas=self.num_replicas,
                rank=self.rank,
                shuffle=self.shuffle,
            )
        return None

    def setup(self):
        """按 ``start_*_year`` / ``end_year`` 实例化 train / val / test 三个 :class:`NpyDataset`。

        三个数据集分别覆盖：
            - train: ``[start_train_year, start_val_year)``
            - val:   ``[start_val_year, start_test_year)``
            - test:  ``[start_test_year, end_year)``

        用 ``if self.xxx_data is None`` 守卫，确保可重复调用而不会重建 Dataset。
        由 ``train/val/test_dataloader`` 内部隐式调用，亦可被 trainer 显式调用。
        """
        if self.train_data is None:
            self.train_data = NpyDataset(
                root_dir=self.root_dir,
                mode="train",
                variables=self.variables,
                start_year=self.start_train_year,
                end_year=self.start_val_year,
                transforms=self.transforms,
                std=self.std,
                debug=self.debug,
            )

        if self.val_data is None:
            self.val_data = NpyDataset(
                root_dir=self.root_dir,
                mode="val",
                variables=self.variables,
                start_year=self.start_val_year,
                end_year=self.start_test_year,
                transforms=self.transforms,
                std=self.std,
                debug=self.debug,
            )

        if self.test_data is None:
            self.test_data = NpyDataset(
                root_dir=self.root_dir,
                mode="test",
                variables=self.variables,
                start_year=self.start_test_year,
                end_year=self.end_year,
                transforms=self.transforms,
                std=self.std,
                debug=self.debug,
            )

    def train_dataloader(self):
        """构造并返回训练 DataLoader。

        与 ``val/test_dataloader`` 的区别：
            - ``drop_last=True``：丢弃最后不完整的 batch，避免分布式下 rank 间数据量不一致；
            - 默认 ``shuffle=True`` (在 DDP 中由 ``DistributedSampler`` 控制)；
            - 使用 ``persistent_workers=True``，避免每个 epoch 重建 worker 进程。

        Returns:
            DataLoader: 配置好的训练 DataLoader。
        """
        self.setup()
        sampler = self._get_sampler(self.train_data)
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=collate_fn_forecast,
            drop_last=True,
            pin_memory=self.pin_memory,
            worker_init_fn=self._worker_init_fn,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=True,
        )

    def val_dataloader(self):
        """构造并返回验证 DataLoader。

        ``drop_last=False`` 保证验证阶段不丢样本 (与训练阶段不同)，
        以便最终的 loss / metric 反映全部 val 集。

        Returns:
            DataLoader: 配置好的验证 DataLoader。
        """
        self.setup()
        sampler = self._get_sampler(self.val_data)
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=collate_fn_forecast,
            drop_last=False,
            pin_memory=self.pin_memory,
            worker_init_fn=self._worker_init_fn,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=True,
        )

    def test_dataloader(self):
        """构造并返回测试 DataLoader。

        ``drop_last=False`` 保证所有测试样本都被评估。

        Returns:
            DataLoader: 配置好的测试 DataLoader。
        """
        self.setup()
        sampler = self._get_sampler(self.test_data)
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=collate_fn_forecast,
            drop_last=False,
            pin_memory=self.pin_memory,
            worker_init_fn=self._worker_init_fn,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=True,
        )