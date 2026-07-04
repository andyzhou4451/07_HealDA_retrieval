# -*- coding: utf-8 -*-
"""MHS 卫星观测算子数据模块 (DataModule)。

本模块为 :class:`MHSDataModule`，对应 MHS (Microwave Humidity Sounder) 任务的
纯 PyTorch 数据模块实现，无 PyTorch-Lightning 依赖。

任务特点:

- MHS 是搭载在 NOAA-18/19、MetOp 系列卫星上的 5 通道微波湿度探测仪 (89~190 GHz)，
  本模块对应 1.0° 分辨率的预处理输出。
- 辅助场为 ``fovn`` / ``lsql`` / ``saza`` / ``soza`` / ``hols`` / ``hmsl`` /
  ``solazi`` / ``bearaz`` 共 8 类，与 AMSU-A 共用同一组 ``hols/hmsl`` scaler。
- ``fovn`` 在 MHS 中除以 90 (区别于 AMSU-A 的 30)。

数据布局::

    root_dir/<yyyy>/<yyyy-mm-dd>/<hh:mm:ss>.npy                         # ERA5 状态
    obs_dir/1bmhs_merged_npy_1.0deg/<yyyy>/<yyyy-mm-dd>/                 # MHS 观测
        <hh:mm:ss>-auxiliary_value.npy
        <hh:mm:ss>-tmbrs_value.npy
        <hh:mm:ss>-mask.npy
"""
import os
import numpy as np
import json
import torch
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.transforms import Normalize

from src.datamodules.obsoperator.mhs.npydataset import NpyDataset, collate_fn

class MHSDataModule:
    """MHS 观测算子数据模块 (纯 torch 实现，无 PyTorch-Lightning 依赖)。

    负责 ERA5 状态与 MHS 卫星观测 (亮温 + 辅助场) 的训练/验证/测试加载，按
    ``start_train_year / start_val_year / start_test_year / end_year`` 切分时间窗。
    """

    def __init__(
        self,
        root_dir: str,
        obs_dir: str,
        variables: list,
        tmbrs_vars: list,
        auxiliary_vars: list,
        start_train_year: int = 2010,
        start_val_year: int = 2022,
        start_test_year: int = 2023,
        end_year: int = 2024,
        dt: int = 3,
        num_lat: int = 181,
        num_lon: int = 360,
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
        """初始化 MHS 数据模块。

        加载 ERA5 / 亮温归一化参数、``hols_scaler`` / ``hmsl_scaler`` 以及
        ``mhs_1.0deg_schema.json`` 元数据。

        Args:
            root_dir (str): ERA5 等模式数据根目录。
            obs_dir (str): 卫星观测根目录。
            variables (list[str]): ERA5 状态变量列表。
            tmbrs_vars (list[str]): MHS 亮温变量列表 (典型 5 通道)。
            auxiliary_vars (list[str]): 卫星辅助场变量列表。
            start_train_year (int): 训练集起始年份。
            start_val_year (int): 验证集起始年份。
            start_test_year (int): 测试集起始年份。
            end_year (int): 结束年份。
            dt (int): 数据采样间隔 (小时)，默认 3。
            num_lat (int): 纬度格点数。
            num_lon (int): 经度格点数。
            seed (int): 随机种子。
            batch_size (int): 批大小。
            num_workers (int): DataLoader 工作线程数。
            shuffle (bool): 是否打乱样本。
            pin_memory (bool): 是否锁页内存。
            prefetch_factor (int): DataLoader 预取因子。
            distributed (bool): 是否分布式训练。
            num_replicas (int): 分布式副本数。
            rank (int): 当前分布式 rank。
            debug (bool): 调试模式。
        """
        self.root_dir = root_dir
        self.obs_dir = obs_dir
        self.variables = variables
        self.tmbrs_vars = tmbrs_vars
        self.auxiliary_vars = auxiliary_vars
        self.dt = dt
        self.num_lat = num_lat
        self.num_lon = num_lon
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
        self.era5_mean, self.era5_std = self._load_normalize(f"{self.root_dir}/normalized_mean_std", self.variables)
        self.tmbrs_mean, self.tmbrs_std = self._load_normalize(f"{self.obs_dir}/1bmhs_merged_npy_1.0deg", self.tmbrs_vars)
        
        self.era5_transforms = Normalize(self.era5_mean, self.era5_std)
        self.tmbrs_transforms = Normalize(self.tmbrs_mean, self.tmbrs_std)

        self.hols_scaler = dict(np.load(os.path.join(f"{self.obs_dir}/1bmhs_merged_npy_1.0deg", "hols_scaler.npz")))
        self.hmsl_scaler = dict(np.load(os.path.join(f"{self.obs_dir}/1bmhs_merged_npy_1.0deg", "hmsl_scaler.npz")))
        with open(f"{self.obs_dir}/1bmhs_merged_npy_1.0deg/mhs_1.0deg_schema.json", "r", encoding="utf-8") as f:
            self.meta_data = json.load(f)

        # 数据集 (延迟初始化)
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def _load_normalize(self, scale_dir, variables):
        """从 ``normalize_mean.npz`` / ``normalize_std.npz`` 中按 ``variables`` 顺序加载归一化参数。

        Args:
            scale_dir (str): 归一化参数所在目录。
            variables (list[str]): 变量名列表。

        Returns:
            tuple[np.ndarray, np.ndarray]: ``(mean, std)``。
        """
        normalize_mean = dict(np.load(os.path.join(scale_dir, "normalize_mean.npz")))
        mean = []
        for var in variables:
            if var != "tp":
                mean.append(normalize_mean[var].reshape(1))
            else:
                mean.append(np.array([0.0]).reshape(1))
        normalize_mean = np.concatenate(mean)

        normalize_std = dict(np.load(os.path.join(scale_dir, "normalize_std.npz")))
        normalize_std = np.concatenate([normalize_std[var].reshape(1) for var in variables])

        return normalize_mean, normalize_std

    def _worker_init_fn(self, worker_id):
        """DataLoader worker 初始化函数。"""
        np.random.seed(self.seed + worker_id)

    def _get_sampler(self, dataset):
        """根据 ``distributed`` / ``num_replicas`` 返回 :class:`DistributedSampler` 或 ``None``。"""
        if self.distributed and self.num_replicas > 1:
            return DistributedSampler(
                dataset,
                num_replicas=self.num_replicas,
                rank=self.rank,
                shuffle=self.shuffle,
            )
        return None

    def setup(self):
        """懒初始化 train / val / test :class:`NpyDataset` 实例。"""
        if self.train_data is None:
            self.train_data = NpyDataset(
                root_dir=self.root_dir,
                obs_dir=self.obs_dir,
                mode="train",
                variables=self.variables,
                tmbrs_vars=self.tmbrs_vars,
                auxiliary_vars=self.auxiliary_vars,
                start_year=self.start_train_year,
                end_year=self.start_val_year,
                dt=self.dt,
                era5_transforms=self.era5_transforms,
                tmbrs_transforms=self.tmbrs_transforms,
                num_lat=self.num_lat,
                num_lon=self.num_lon,
                hols_scaler=self.hols_scaler,
                hmsl_scaler=self.hmsl_scaler,
                meta_data=self.meta_data,
                tmbrs_std=self.tmbrs_std,
                debug=self.debug,
            )

        if self.val_data is None:
            self.val_data = NpyDataset(
                root_dir=self.root_dir,
                obs_dir=self.obs_dir,
                mode="val",
                variables=self.variables,
                tmbrs_vars=self.tmbrs_vars,
                auxiliary_vars=self.auxiliary_vars,
                start_year=self.start_val_year,
                end_year=self.start_test_year,
                dt=self.dt,
                era5_transforms=self.era5_transforms,
                tmbrs_transforms=self.tmbrs_transforms,
                num_lat=self.num_lat,
                num_lon=self.num_lon,
                hols_scaler=self.hols_scaler,
                hmsl_scaler=self.hmsl_scaler,
                meta_data=self.meta_data,
                tmbrs_std=self.tmbrs_std,
                debug=self.debug,
            )

        if self.test_data is None:
            self.test_data = NpyDataset(
                root_dir=self.root_dir,
                obs_dir=self.obs_dir,
                mode="test",
                variables=self.variables,
                tmbrs_vars=self.tmbrs_vars,
                auxiliary_vars=self.auxiliary_vars,
                start_year=self.start_test_year,
                end_year=self.end_year,
                dt=self.dt,
                era5_transforms=self.era5_transforms,
                tmbrs_transforms=self.tmbrs_transforms,
                num_lat=self.num_lat,
                num_lon=self.num_lon,
                hols_scaler=self.hols_scaler,
                hmsl_scaler=self.hmsl_scaler,
                meta_data=self.meta_data,
                tmbrs_std=self.tmbrs_std,
                debug=self.debug,
            )

    def train_dataloader(self):
        """构造训练集 :class:`DataLoader`。

        Returns:
            DataLoader: 训练集加载器，``drop_last=True``。
        """
        self.setup()
        sampler = self._get_sampler(self.train_data)
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            drop_last=True,
            pin_memory=self.pin_memory,
            worker_init_fn=self._worker_init_fn,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=True,
        )

    def val_dataloader(self):
        """构造验证集 :class:`DataLoader`。

        Returns:
            DataLoader: 验证集加载器，``drop_last=False``。
        """
        self.setup()
        sampler = self._get_sampler(self.val_data)
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            drop_last=False,
            pin_memory=self.pin_memory,
            worker_init_fn=self._worker_init_fn,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=True,
        )

    def test_dataloader(self):
        """构造测试集 :class:`DataLoader`。

        Returns:
            DataLoader: 测试集加载器，``drop_last=False``。
        """
        self.setup()
        sampler = self._get_sampler(self.test_data)
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            drop_last=False,
            pin_memory=self.pin_memory,
            worker_init_fn=self._worker_init_fn,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=True,
        )