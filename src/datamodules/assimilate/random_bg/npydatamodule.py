# -*- coding: utf-8 -*-
"""
随机背景场 (random background) 同化任务的 DataModule。

本模块为 ``src.pipeline.assimilate.{cascade,multimodal}.random_bg_trainer`` 提供训练数据：
训练 DA (Data Assimilation) 模型时，"真实背景场" (``bg``) 从 ERA5 真值里随机抽取，
DA 模型在此基础上叠加"随机扰动"得到 ``xb``，再由 ``H(x)`` 把 ``xb`` 映射到观测空间，
通过最小化 ``VarCost`` 让 ``DA(xb) → xa`` 逼近 ``bg``。

本文件核心职责：
    1. 加载 ERA5 归一化参数 + 多种观测 (微波、常规) 的归一化参数；
    2. 解析微波观测的 schema JSON (``*_1.0deg_schema.json``) 与各通道的 scaler；
    3. 实例化 :class:`NpyDataset` (按 train/val/test 切片年份)；
    4. DDP 采样器 / persistent_workers 等工程配置。

对应的 Dataset 见 :mod:`src.datamodules.assimilate.random_bg.npydataset`。
"""
import os
import numpy as np
import json
import torch
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.transforms import transforms

from src.datamodules.assimilate.random_bg.npydataset import NpyDataset, collate_fn

class RandomBgAssimDataModule:
    """随机背景场同化任务的 DataModule (纯 torch 实现)。

    Attributes:
        era5_dir (str): ERA5 再分析数据根目录。
        scale_dir (str): 数据缩放 / 归一化参数目录。
        obs_dir (str): 观测数据根目录。
        era5_vars (list): 需要加载的 ERA5 变量名列表。
        obs_list (list): 观测变量 / 通道列表 (如 ["atms", "amsua", "mhs", "hrs4", "prepbufr", "satwnd", "ascat"])。
        obs_dict (dict): 观测变量配置映射字典 (如路径、掩码规则等)。
        max_lead_time (int): 最大预报时效 (小时)。
        iter_num (int): 背景场预报步长。
        daw (int): 数据同化窗口长度 (Data Assimilation Window，单位通常为小时)。
        dt_obs (int): 观测数据采样时间间隔。
        dt_data (int): 训练数据采样时间间隔。
        num_lat (int): 纬度格点数。
        num_lon (int): 经度格点数。
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
        era5_transforms (dict): ERA5 归一化参数 / 算子。
        conventional_transforms (dict): 常规观测 (prepbufr / satwnd / ascat) 的归一化算子。
        microwave_transforms (dict): 微波观测 (atms / amsua / mhs / hrs4) 的归一化算子。
        microwave_meta_data (dict): 微波观测的 schema JSON (字段顺序等)。
        microwave_scaler (dict): 微波观测各通道的 min-max scaler。
        train_data / val_data / test_data (Optional[NpyDataset]): 三个分段的数据集。
    """

    def __init__(
        self,
        era5_dir: str,
        scale_dir: str,
        obs_dir: str,
        era5_vars: list,
        obs_list: list,
        obs_dict: dict,
        start_train_year: int,
        start_val_year: int,
        start_test_year: int,
        end_year: int,
        max_lead_time: int,
        iter_num: int,
        daw: int,
        dt_obs: int,
        dt_data: int,
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
        """
        Args:
            era5_dir (str): ERA5再分析数据根目录
            scale_dir (str): 数据缩放/归一化参数目录
            obs_dir (str): 观测数据根目录
            era5_vars (list): 需要加载的ERA5变量名称列表
            obs_list (list): 观测数据变量/通道列表
            obs_dict (dict): 观测变量配置映射字典（如路径、掩码规则等）
            start_train_year (int): 训练集起始年份
            start_val_year (int): 验证集起始年份
            start_test_year (int): 测试集起始年份
            end_year (int): 数据集结束年份
            max_lead_time (int): 最大预报时效（Lead Time，通常以步数或天数为单位）
            iter_num (int): 背景场预报步长
            daw (int): 数据同化窗口长度（Data Assimilation Window，单位通常为小时）
            dt_obs (int): 观测数据采样时间间隔
            dt_data (int): 训练数据采样时间间隔
            num_lat (int): 纬度格点数
            num_lon (int): 经度格点数
            seed (int): 随机种子，用于可复现性
            batch_size (int): DataLoader批次大小
            num_workers (int): 数据加载子进程/线程数
            shuffle (bool): 每个epoch是否打乱数据顺序
            pin_memory (bool): 是否锁定内存以加速CPU到GPU的传输
            prefetch_factor (int): 每个worker预取的样本批次数量
            distributed (bool): 是否启用分布式采样
            num_replicas (int): 分布式训练节点/副本总数
            rank (int): 当前进程的rank编号
            debug (bool): 是否开启调试模式（如限制数据量、打印详细日志等）
        """
        self.era5_dir = era5_dir
        self.scale_dir = scale_dir
        self.obs_dir = obs_dir
        self.era5_vars = era5_vars
        self.obs_list = obs_list
        self.obs_dict = obs_dict
        self.max_lead_time = max_lead_time
        self.iter_num = iter_num
        self.daw = daw
        self.dt_obs = dt_obs
        self.dt_data = dt_data
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
        self._prepare_transforms()

        # 数据集 (延迟初始化)
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def _prepare_transforms(self):
        """准备 ERA5 + 各观测源的归一化参数、schema JSON、scaler 字典。

        流程：
            1. 调用 :func:`_load_normalize` 加载 ``era5_transforms``；
            2. 对每个微波卫星 (atms / amsua / mhs / hrs4) 加载归一化参数、schema JSON、
               以及各 scaler (satellite_height / *_flags)；
            3. 对每个常规观测 (prepbufr / satwnd / ascat) 加载归一化参数。

        所有结果均写入 self 的同名属性，供 :class:`NpyDataset` 使用。
        """
        self.era5_transforms = self._load_normalize(self.scale_dir, self.era5_vars)
        self.conventional_transforms = {}
        self.microwave_transforms = {}
        self.microwave_meta_data = {}
        self.microwave_scaler = {}
        for name in self.obs_dict["microwave"]:
            self.microwave_transforms[name] = self._load_normalize(
                f"{self.obs_dir}/1b{name}_merged_npy_1.0deg", 
                self.obs_dict["microwave"][name]["tmbrs_vars"]
            )
            with open(f"{self.obs_dir}/1b{name}_merged_npy_1.0deg/{name}_1.0deg_schema.json", "r", encoding="utf-8") as f:
                self.microwave_meta_data[name] = json.load(f)
            self.microwave_scaler[name] = {}
            for scaler_name in self.obs_dict["microwave"][name]["scaler_vars"]:
                self.microwave_scaler[name][scaler_name] = dict(
                    np.load(
                        os.path.join(
                            f"{self.obs_dir}/1b{name}_merged_npy_1.0deg", 
                            f"{scaler_name}.npz"
                        )
                    )
                )

        for name in self.obs_dict["conventional"]:
            self.conventional_transforms[name] = self._load_normalize(
                self.scale_dir, 
                self.obs_dict["conventional"][name]["vars"]
            )

    def _load_normalize(self, scale_dir, variables):
        """加载并拼接 ``variables`` 对应的归一化 mean / std，并打包为 dict。

        Args:
            scale_dir (str): 归一化参数所在目录 (含 ``normalize_mean.npz``、``normalize_std.npz``)。
            variables (list[str]): 需要加载的变量名列表。

        Returns:
            dict: 包含 ``mean`` / ``std`` (np.ndarray) 和 ``transforms``
            (torch ``Normalize`` 算子) 的字典。
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
        data_transforms = transforms.Normalize(normalize_mean, normalize_std)
        out = {
            "mean": normalize_mean,
            "std": normalize_std,
            "transforms": data_transforms
        }
        return out

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

        每个 NpyDataset 会传入全部 transforms / meta_data / scaler，由 Dataset 内部按 obs 名分派。
        """
        if self.train_data is None:
            self.train_data = NpyDataset(
                era5_dir=self.era5_dir,
                obs_dir=self.obs_dir,
                mode="train",
                era5_vars=self.era5_vars,
                obs_list=self.obs_list,
                obs_dict=self.obs_dict,
                start_year=self.start_train_year,
                end_year=self.start_val_year,
                max_lead_time=self.max_lead_time,
                iter_num=self.iter_num,
                daw=self.daw,
                dt_obs=self.dt_obs,
                dt_data=self.dt_data,
                era5_transforms=self.era5_transforms,
                conventional_transforms=self.conventional_transforms,
                microwave_transforms=self.microwave_transforms,
                microwave_meta_data=self.microwave_meta_data,
                microwave_scaler=self.microwave_scaler,
                num_lat=self.num_lat,
                num_lon=self.num_lon,
                debug=self.debug,
            )

        if self.val_data is None:
            self.val_data = NpyDataset(
                era5_dir=self.era5_dir,
                obs_dir=self.obs_dir,
                mode="val",
                era5_vars=self.era5_vars,
                obs_list=self.obs_list,
                obs_dict=self.obs_dict,
                start_year=self.start_val_year,
                end_year=self.start_test_year,
                max_lead_time=self.max_lead_time,
                iter_num=self.iter_num,
                daw=self.daw,
                dt_obs=self.dt_obs,
                dt_data=self.dt_data,
                era5_transforms=self.era5_transforms,
                conventional_transforms=self.conventional_transforms,
                microwave_transforms=self.microwave_transforms,
                microwave_meta_data=self.microwave_meta_data,
                microwave_scaler=self.microwave_scaler,
                num_lat=self.num_lat,
                num_lon=self.num_lon,
                debug=self.debug,
            )

        if self.test_data is None:
            self.test_data = NpyDataset(
                era5_dir=self.era5_dir,
                obs_dir=self.obs_dir,
                mode="test",
                era5_vars=self.era5_vars,
                obs_list=self.obs_list,
                obs_dict=self.obs_dict,
                start_year=self.start_test_year,
                end_year=self.end_year,
                max_lead_time=self.max_lead_time,
                iter_num=self.iter_num,
                daw=self.daw,
                dt_obs=self.dt_obs,
                dt_data=self.dt_data,
                era5_transforms=self.era5_transforms,
                conventional_transforms=self.conventional_transforms,
                microwave_transforms=self.microwave_transforms,
                microwave_meta_data=self.microwave_meta_data,
                microwave_scaler=self.microwave_scaler,
                num_lat=self.num_lat,
                num_lon=self.num_lon,
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
            collate_fn=collate_fn,
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
            collate_fn=collate_fn,
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
            collate_fn=collate_fn,
            drop_last=False,
            pin_memory=self.pin_memory,
            worker_init_fn=self._worker_init_fn,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=True,
        )