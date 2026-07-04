# -*- coding: utf-8 -*-
"""HRS4 卫星观测算子数据集 (Dataset)。

本模块实现 :class:`NpyDataset`，为 HRS4 (FY-4A GIIRS Hyperspectral Infrared
Sounder) 的观测算子任务提供样本加载逻辑：从 ERA5 状态文件与
``1bhrs4_merged_npy_1.0deg`` 下的卫星辅助场 / 亮温 / 掩码 npy 文件中读取单时间点
样本，并对辅助场做角度 → cos 投影、线性归一化等预处理。

支持 lead_time forecasting (input != target)。
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

class NpyDataset(Dataset):
    """加载 ``.npy`` 格式 ERA5 状态 + HRS4 (GIIRS) 卫星观测的 :class:`Dataset`。

    ``__getitem__`` 返回 5 元组 ``(era5, sat_data, mask, tmbrs_std, tmbrs_vars)``。
    注意：HRS4 在 :meth:`_get_hrs4` 中通过 ``tmbrs_value[:-1]`` 丢弃最末通道，与
    ``tmbrs_vars`` 的长度声明保持一致。
    """

    def __init__(
        self,
        root_dir: str,
        obs_dir: str,
        mode: str,
        variables: list,
        tmbrs_vars: list,
        auxiliary_vars: list,
        start_year: int,
        end_year: int,
        dt: int,
        era5_transforms,
        tmbrs_transforms,
        num_lat: int,
        num_lon: int,
        hols_scaler: dict,
        hmsl_scaler: dict,
        meta_data: dict,
        tmbrs_std: np.ndarray,
        debug: bool,
    ):
        """初始化 HRS4 数据集。

        Args:
            root_dir (str): ERA5 状态文件根目录。
            obs_dir (str): 卫星观测根目录。
            mode (str): ``"train"`` / ``"val"`` / ``"test"``。
            variables (list[str]): ERA5 状态变量列表。
            tmbrs_vars (list[str]): HRS4 亮温变量列表 (末通道已丢弃)。
            auxiliary_vars (list[str]): 卫星辅助场变量列表。
            start_year (int): 起始年份。
            end_year (int): 结束年份。
            dt (int): 采样间隔 (小时)。
            era5_transforms: ERA5 归一化 ``Normalize``。
            tmbrs_transforms: 亮温归一化 ``Normalize``。
            num_lat (int): 纬度格点数。
            num_lon (int): 经度格点数。
            hols_scaler (dict): 轨道高度 ``hols`` 的 ``min/max``。
            hmsl_scaler (dict): 卫星高度 ``hmsl`` 的 ``min/max``。
            meta_data (dict): ``hrs4_1.0deg_schema.json`` 元数据。
            tmbrs_std (np.ndarray): 亮温 ``std``，用于反归一化。
            debug (bool): 调试模式。
        """
        super().__init__()
        self.root_dir = root_dir
        self.obs_dir = obs_dir
        self.mode = mode
        self.dt = dt
        self.era5_transforms = era5_transforms
        self.tmbrs_transforms = tmbrs_transforms
        self.tmbrs_vars = tmbrs_vars
        self.auxiliary_vars = auxiliary_vars
        self.tmbrs_shape = (len(self.tmbrs_vars), num_lat, num_lon)
        self.auxiliary_shape = (len(self.auxiliary_vars), num_lat, num_lon)
        self.hols_scaler = hols_scaler
        self.hmsl_scaler = hmsl_scaler
        self.meta_data = meta_data
        self.tmbrs_std = tmbrs_std

        # 计算时间范围
        if debug:
            self.start_time = datetime(start_year, 1, 1, 0, 0)
            self.end_time = datetime(start_year, 2, 1, 0, 0)
        else:
            self.start_time = datetime(start_year, 1, 1, 0, 0)
            self.end_time = datetime(end_year, 1, 1, 0, 0)
        self.total_hours = int((self.end_time - self.start_time).total_seconds() // 3600)

    def __len__(self):
        """样本数 = 总小时数 / 采样间隔。

        Returns:
            int: 当前 split 下可索引的样本总数。
        """
        # 计算有效样本数
        return self.total_hours // self.dt

    def _get_era5(self, idx: int) -> torch.Tensor:
        """按索引加载 ERA5 单时间点状态文件。

        Args:
            idx (int): 样本索引。

        Returns:
            torch.Tensor: ``[V, H, W]`` 的 ERA5 状态张量，``float32``。
        """
        """加载单个时间点的数据"""
        current_time = self.start_time + timedelta(hours=idx * self.dt)
        file_path = os.path.join(
            self.root_dir,
            f"{current_time.year:04d}",
            f"{current_time.year:04d}-{current_time.month:02d}-{current_time.day:02d}",
            f"{current_time.hour:02d}:{current_time.minute:02d}:{current_time.second:02d}.npy",
        )
        data = np.load(file_path)
        return torch.from_numpy(data).to(dtype=torch.float32)

    def _get_hrs4(self, idx):
        """按索引加载并预处理 HRS4 (GIIRS) 单时间点卫星观测。

        与 AMSU-A / MHS 的关键差异：
        - 读取 ``tmbrs_value`` 后通过 ``[:-1]`` **丢弃最末通道**，与 ``tmbrs_vars``
          长度声明保持一致。
        - ``fovn`` 除以 56 (GIIRS 每条扫描线 56 个视场)。
        - 其余预处理 (``lsql / saza / soza / hols / hmsl / solazi / bearaz``) 与
          AMSU-A / MHS 相同。
        - mask 文件不存在时填充 NaN + 全 0 掩码。

        Args:
            idx (int): 样本索引。

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: ``(tmbrs, auxiliary, mask)``。
        """
        current_time = self.start_time + relativedelta(hours=idx * self.dt)
        auxiliarty_path = os.path.join(
            self.obs_dir,
            "1bhrs4_merged_npy_1.0deg",
            f"{current_time.year:04d}",
            f"{current_time.year:04d}-{current_time.month:02d}-{current_time.day:02d}",
            f"{current_time.hour:02d}:{current_time.minute:02d}:{current_time.second:02d}-auxiliary_value.npy",
        )
        tmbrs_path = os.path.join(
            self.obs_dir,
            "1bhrs4_merged_npy_1.0deg",
            f"{current_time.year:04d}",
            f"{current_time.year:04d}-{current_time.month:02d}-{current_time.day:02d}",
            f"{current_time.hour:02d}:{current_time.minute:02d}:{current_time.second:02d}-tmbrs_value.npy",
        )
        mask_path = os.path.join(
            self.obs_dir,
            "1bhrs4_merged_npy_1.0deg",
            f"{current_time.year:04d}",
            f"{current_time.year:04d}-{current_time.month:02d}-{current_time.day:02d}",
            f"{current_time.hour:02d}:{current_time.minute:02d}:{current_time.second:02d}-mask.npy",
        )
        if os.path.exists(mask_path):
            auxiliary_value = np.load(auxiliarty_path)
            tmbrs_value = np.load(tmbrs_path)[:-1]
            mask = np.load(mask_path)
            np_tmbrs_data = tmbrs_value.astype(np.float32)
            np_auxiliary_data = auxiliary_value.astype(np.float32)
            fovn_idx = self.meta_data["auxiliary_value"]["fields_in_order"].index("fovn")
            np_auxiliary_data[fovn_idx] = np_auxiliary_data[fovn_idx] / 56
            lsql_idx = self.meta_data["auxiliary_value"]["fields_in_order"].index("lsql")
            np_auxiliary_data[lsql_idx] = np_auxiliary_data[lsql_idx] / 2
            saza_idx = self.meta_data["auxiliary_value"]["fields_in_order"].index("saza")
            np_auxiliary_data[saza_idx] = np.cos(np.deg2rad(np_auxiliary_data[saza_idx]))
            soza_idx = self.meta_data["auxiliary_value"]["fields_in_order"].index("soza")
            np_auxiliary_data[soza_idx] = np.cos(np.deg2rad(np_auxiliary_data[soza_idx]))
            hols_idx = self.meta_data["auxiliary_value"]["fields_in_order"].index("hols")
            np_auxiliary_data[hols_idx] = (np_auxiliary_data[hols_idx] - self.hols_scaler["hols_min"]) / (self.hols_scaler["hols_max"] - self.hols_scaler["hols_min"])
            hmsl_idx = self.meta_data["auxiliary_value"]["fields_in_order"].index("hmsl")
            np_auxiliary_data[hmsl_idx] = (np_auxiliary_data[hmsl_idx] - self.hmsl_scaler["hmsl_min"]) / (self.hmsl_scaler["hmsl_max"] - self.hmsl_scaler["hmsl_min"])
            solazi_idx = self.meta_data["auxiliary_value"]["fields_in_order"].index("solazi")
            np_auxiliary_data[solazi_idx] = np.cos(np.deg2rad(np_auxiliary_data[solazi_idx]) / 2) 
            bearaz_idx = self.meta_data["auxiliary_value"]["fields_in_order"].index("bearaz")
            np_auxiliary_data[bearaz_idx] = np.cos(np.deg2rad(np_auxiliary_data[bearaz_idx]) / 2) 
            np_mask = mask.astype(np.int32)
        else:
            np_tmbrs_data = (np.ones(self.tmbrs_shape) * np.nan).astype(np.float32)
            np_auxiliary_data = (np.ones(self.auxiliary_shape) * np.nan).astype(np.float32)
            np_mask = (np.zeros(self.tmbrs_shape[-2:])).astype(np.int32)

        return np_tmbrs_data, np_auxiliary_data, np_mask

    def __getitem__(self, global_idx):
        """按全局索引取一个 HRS4 (GIIRS) 样本。

        Args:
            global_idx (int): 样本索引 (支持负索引)。

        Returns:
            tuple:
            - era5 (torch.Tensor): ``[V, H, W]``，已归一化。
            - sat_data (torch.Tensor): ``[len(aux)+len(tmbrs), H, W]``。
            - mask (torch.Tensor): ``[1, H, W]``。
            - tmbrs_std (torch.Tensor): ``[len(tmbrs_vars)]``。
            - tmbrs_vars (list[str]): 亮温变量名 (末通道已丢弃)。
        """
        if global_idx < 0:
            global_idx += self.__len__()

        inp = self._get_era5(global_idx)

        tmbrs_data, auxiliary_data, mask = self._get_hrs4(global_idx)

        tmbrs_data = torch.from_numpy(np.nan_to_num(tmbrs_data))
        auxiliary_data = torch.from_numpy(np.nan_to_num(auxiliary_data))
        mask = torch.unsqueeze(torch.from_numpy(mask), dim=0)
        tmbrs_data = self.tmbrs_transforms(tmbrs_data) * mask
        auxiliary_data = auxiliary_data * mask

        return self.era5_transforms(inp), \
               torch.concat([auxiliary_data, tmbrs_data], dim=0), \
               mask, torch.from_numpy(self.tmbrs_std).to(inp.dtype), self.tmbrs_vars

def collate_fn(batch):
    """HRS4 DataLoader 的默认 ``collate_fn``。

    Args:
        batch (list): 长度为 ``batch_size`` 的样本列表。

    Returns:
        tuple: ``(era5, sat_data, mask, tmbrs_std, tmbrs_vars)``。
    """
    era5 = torch.stack([batch[i][0] for i in range(len(batch))], dim=0)
    sat_data = torch.stack([batch[i][1] for i in range(len(batch))], dim=0)
    mask = torch.stack([batch[i][2] for i in range(len(batch))], dim=0)
    tmbrs_std = torch.stack([batch[i][3] for i in range(len(batch))], dim=0)
    tmbrs_vars = batch[0][4]
    return (
        era5,
        sat_data,
        mask,
        tmbrs_std,
        tmbrs_vars,
    )