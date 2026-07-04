# -*- coding: utf-8 -*-
"""ATMS 卫星观测算子数据模块 (DataModule)。

本模块为 :class:`ATMSDataModule`，是 ``src.datamodules.obsoperator`` 中 ATMS
(Advanced Technology Microwave Sounder) 任务的纯 PyTorch 数据模块实现，
无 PyTorch-Lightning 依赖，可直接被 ``src/pipeline/obsoperator/trainer.py`` 实例化。

任务特点:

- ATMS 是搭载在 Suomi-NPP / JPSS 系列卫星上的微波探测仪，**22 个通道**，
  横跨 23~183 GHz，本模块对应 1.0° 分辨率的预处理输出。
- 辅助场包含 12 类标量：``scanline`` / ``fov`` / ``orbit_number`` /
  ``satellite_zenith_angle`` / ``satellite_azimuth_angle`` / ``solar_zenith_angle`` /
  ``solar_azimuth_angle`` / ``satellite_height`` / ``geolocation_quality_flags`` /
  ``scan_quality_flags`` / ``granule_quality_flags`` 以及数据有效性 ``mask``。

数据布局::

    root_dir/<yyyy>/<yyyy-mm-dd>/<hh:mm:ss>.npy                           # ERA5 状态
    obs_dir/1batms_merged_npy_1.0deg/<yyyy>/<yyyy-mm-dd>/                  # 卫星观测
        <hh:mm:ss>-auxiliary_value.npy
        <hh:mm:ss>-brightness_temperature_value.npy
        <hh:mm:ss>-mask.npy

归一化参数从 ``root_dir/normalized_mean_std`` (ERA5) 和
``obs_dir/1batms_merged_npy_1.0deg`` (亮温) 加载。
"""
import os
import numpy as np
import json
import torch
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.transforms import Normalize

from src.datamodules.obsoperator.atms.npydataset import NpyDataset, collate_fn

class ATMSDataModule:
    """ATMS 观测算子数据模块 (纯 torch 实现，无 PyTorch-Lightning 依赖)。

    负责 ERA5 状态与 ATMS 卫星观测 (亮温 + 辅助场) 的训练/验证/测试加载，按
    ``start_train_year / start_val_year / start_test_year / end_year`` 切分时间窗，
    并将 1.0° 分辨率下的 22 通道亮温与 12 类辅助场打包为模型可直接消费的 batch。
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
        """初始化 ATMS 数据模块。

        加载 ERA5 / 亮温归一化参数、各类辅助场 ``scaler`` (``satellite_height`` /
        ``scan_quality_flags`` / ``geolocation_quality_flags`` /
        ``granule_quality_flags``) 以及 ``atms_1deg_allvars_schema.json`` 元数据，
        并按 train/val/test 三个时间窗预留 ``NpyDataset`` 实例 (懒初始化)。

        Args:
            root_dir (str): ERA5 等模式数据根目录 (含 ``normalized_mean_std`` 子目录)。
            obs_dir (str): 卫星观测根目录 (含 ``1batms_merged_npy_1.0deg`` 子目录)。
            variables (list[str]): ERA5 状态变量列表。
            tmbrs_vars (list[str]): ATMS 亮温变量列表 (典型 22 通道)。
            auxiliary_vars (list[str]): 卫星辅助场变量列表。
            start_train_year (int): 训练集起始年份。
            start_val_year (int): 验证集起始年份。
            start_test_year (int): 测试集起始年份。
            end_year (int): 结束年份 (test 集的上界)。
            dt (int): 数据采样间隔 (小时)，默认 3。
            num_lat (int): 纬度格点数 (默认 181，对应 1.0°)。
            num_lon (int): 经度格点数 (默认 360，对应 1.0°)。
            seed (int): 随机种子。
            batch_size (int): 批大小。
            num_workers (int): DataLoader 工作线程数。
            shuffle (bool): 是否打乱样本。
            pin_memory (bool): 是否锁页内存。
            prefetch_factor (int): DataLoader 预取因子。
            distributed (bool): 是否分布式训练。
            num_replicas (int): 分布式副本数 (world size)。
            rank (int): 当前分布式 rank。
            debug (bool): 调试模式 (限制数据时间窗)。
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
        self.tmbrs_mean, self.tmbrs_std = self._load_normalize(f"{obs_dir}/1batms_merged_npy_1.0deg", self.tmbrs_vars)
        
        self.era5_transforms = Normalize(self.era5_mean, self.era5_std)
        self.tmbrs_transforms = Normalize(self.tmbrs_mean, self.tmbrs_std)

        self.satellite_height_scaler = dict(np.load(os.path.join(f"{obs_dir}/1batms_merged_npy_1.0deg", "satellite_height_scaler.npz")))
        self.scan_quality_flags_scaler = dict(np.load(os.path.join(f"{obs_dir}/1batms_merged_npy_1.0deg", "scan_quality_flags_scaler.npz")))
        self.geolocation_quality_flags_scaler = dict(np.load(os.path.join(f"{obs_dir}/1batms_merged_npy_1.0deg", "geolocation_quality_flags_scaler.npz")))
        self.granule_quality_flags_scaler = dict(np.load(os.path.join(f"{obs_dir}/1batms_merged_npy_1.0deg", "granule_quality_flags_scaler.npz")))
        with open(f"{obs_dir}/1batms_merged_npy_1.0deg/atms_1deg_allvars_schema.json", "r", encoding="utf-8") as f:
            self.meta_data = json.load(f)

        # 数据集 (延迟初始化)
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def _load_normalize(self, scale_dir, variables):
        """从 ``normalize_mean.npz`` / ``normalize_std.npz`` 中按 ``variables`` 顺序加载归一化参数。

        ``tp`` (总降水量) 的均值用 0 占位，避免对数化的边界效应。

        Args:
            scale_dir (str): 归一化参数所在目录。
            variables (list[str]): 变量名列表，决定输出维度顺序。

        Returns:
            tuple[np.ndarray, np.ndarray]: ``(mean, std)``，shape 均为 ``[len(variables)]``。
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
        """DataLoader worker 初始化函数。

        给每个 worker 注入 ``self.seed + worker_id`` 的随机种子，保证多 worker 间
        数据增强 / 抽样的可复现性。

        Args:
            worker_id (int): DataLoader worker 索引。
        """
        np.random.seed(self.seed + worker_id)

    def _get_sampler(self, dataset):
        """根据 ``distributed`` / ``num_replicas`` 决定是否返回 :class:`DistributedSampler`。

        Args:
            dataset (Dataset): 目标 ``NpyDataset`` 实例。

        Returns:
            DistributedSampler | None: 分布式训练时返回 sampler，否则返回 ``None``
            (DataLoader 退化为非分布式顺序采样)。
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
        """懒初始化 train / val / test 三个 :class:`NpyDataset` 实例。

        若对应属性为 ``None``，则按训练 / 验证 / 测试三个时间窗分别构造数据集并缓存，
        多次调用 ``setup`` 不会重复构造。
        """
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
                satellite_height_scaler=self.satellite_height_scaler,
                scan_quality_flags_scaler=self.scan_quality_flags_scaler,
                geolocation_quality_flags_scaler=self.geolocation_quality_flags_scaler,
                granule_quality_flags_scaler=self.granule_quality_flags_scaler,
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
                satellite_height_scaler=self.satellite_height_scaler,
                scan_quality_flags_scaler=self.scan_quality_flags_scaler,
                geolocation_quality_flags_scaler=self.geolocation_quality_flags_scaler,
                granule_quality_flags_scaler=self.granule_quality_flags_scaler,
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
                satellite_height_scaler=self.satellite_height_scaler,
                scan_quality_flags_scaler=self.scan_quality_flags_scaler,
                geolocation_quality_flags_scaler=self.geolocation_quality_flags_scaler,
                granule_quality_flags_scaler=self.granule_quality_flags_scaler,
                meta_data=self.meta_data,
                tmbrs_std=self.tmbrs_std,
                debug=self.debug,
            )

    def train_dataloader(self):
        """构造训练集 :class:`DataLoader`。

        Returns:
            DataLoader: 训练集加载器，``drop_last=True`` 以保证 DDP 同步。
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
            DataLoader: 验证集加载器，``drop_last=False`` 以保留全部样本。
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