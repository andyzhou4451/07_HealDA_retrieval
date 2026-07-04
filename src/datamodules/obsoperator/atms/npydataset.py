# -*- coding: utf-8 -*-
"""ATMS 卫星观测算子数据集 (Dataset)。

本模块实现 :class:`NpyDataset`，为 ATMS (Advanced Technology Microwave Sounder) 的
观测算子任务提供样本加载逻辑：从 ERA5 状态文件与 ``1batms_merged_npy_1.0deg`` 下的
卫星辅助场 / 亮温 / 掩码 npy 文件中读取单时间点样本，并对辅助场做角度 → cos 投影、
线性归一化等预处理。

支持 lead_time forecasting (input != target)。
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

class NpyDataset(Dataset):
    """加载 ``.npy`` 格式 ERA5 状态 + ATMS 卫星观测的 :class:`Dataset`。

    一次 ``__getitem__`` 返回 5 元组 ``(era5, sat_data, mask, tmbrs_std, tmbrs_vars)``：
    - ``era5``：``[V, H, W]`` 的 ERA5 状态张量，已归一化。
    - ``sat_data``：``[len(auxiliary_vars)+len(tmbrs_vars), H, W]``，辅助场在前、
      亮温在后，已乘 ``mask``。
    - ``mask``：``[1, H, W]`` 有效性掩码。
    - ``tmbrs_std``：归一化所用 ``std`` (用于反归一化)。
    - ``tmbrs_vars``：亮温变量名列表 (从 batch[0] 透传)。
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
        satellite_height_scaler: str,
        scan_quality_flags_scaler: str,
        geolocation_quality_flags_scaler: str,
        granule_quality_flags_scaler: str,
        meta_data: dict,
        tmbrs_std: np.ndarray,
        debug: bool,
    ):
        """初始化 ATMS 数据集。

        Args:
            root_dir (str): ERA5 状态文件根目录。
            obs_dir (str): 卫星观测根目录。
            mode (str): ``"train"`` / ``"val"`` / ``"test"`` 之一。
            variables (list[str]): ERA5 状态变量列表。
            tmbrs_vars (list[str]): ATMS 亮温变量列表。
            auxiliary_vars (list[str]): 卫星辅助场变量列表。
            start_year (int): 起始年份。
            end_year (int): 结束年份。
            dt (int): 采样间隔 (小时)。
            era5_transforms: ERA5 归一化 ``Normalize``。
            tmbrs_transforms: 亮温归一化 ``Normalize``。
            num_lat (int): 纬度格点数。
            num_lon (int): 经度格点数。
            satellite_height_scaler (dict): ``satellite_height`` 的 ``min/max``。
            scan_quality_flags_scaler (dict): ``scan_quality_flags`` 的 ``min/max``。
            geolocation_quality_flags_scaler (dict): ``geolocation_quality_flags`` 的 ``min/max``。
            granule_quality_flags_scaler (dict): ``granule_quality_flags`` 的 ``min/max``。
            meta_data (dict): ``atms_1deg_allvars_schema.json`` 元数据，含字段索引。
            tmbrs_std (np.ndarray): 亮温 ``std``，用于反归一化。
            debug (bool): 调试模式 (仅取 1 月数据)。
        """
        super().__init__()
        self.root_dir = root_dir
        self.obs_dir = obs_dir
        self.mode = mode
        self.dt = dt
        self.era5_transforms = era5_transforms
        self.tmbrs_transforms = tmbrs_transforms
        self.tmbrs_vars = tmbrs_vars
        self.tmbrs_shape = (len(tmbrs_vars), num_lat, num_lon)
        self.auxiliary_shape = (len(auxiliary_vars), num_lat, num_lon)
        self.satellite_height_scaler = satellite_height_scaler
        self.scan_quality_flags_scaler = scan_quality_flags_scaler
        self.geolocation_quality_flags_scaler = geolocation_quality_flags_scaler
        self.granule_quality_flags_scaler = granule_quality_flags_scaler
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
            idx (int): 样本索引，对应 ``start_time + idx * dt`` 小时。

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

    def _get_atms(self, idx):
        """按索引加载并预处理 ATMS 单时间点卫星观测。

        关键预处理：
        - ``scanline / fov``：除以 12 / 96 归一化到 ``[0, 1]``。
        - ``satellite_zenith / solar_zenith``：``cos(deg2rad(.))``。
        - ``satellite_azimuth / solar_azimuth``：``cos(deg2rad(./2))``，消除 0/360 跳变。
        - ``satellite_height / *_quality_flags``：min-max 归一化到 ``[0, 1]``。
        - 当 mask 文件不存在时，填充 NaN + 全 0 掩码 (由 ``__getitem__`` 中的
          ``nan_to_num`` 处理)。

        Args:
            idx (int): 样本索引。

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]:
            ``(tmbrs, auxiliary, mask)`` 三个 ``float32/int32`` 数组。
        """
        current_time = self.start_time + relativedelta(hours=idx * self.dt)
        auxiliarty_path = os.path.join(
            self.obs_dir,
            "1batms_merged_npy_1.0deg",
            f"{current_time.year:04d}",
            f"{current_time.year:04d}-{current_time.month:02d}-{current_time.day:02d}",
            f"{current_time.hour:02d}:{current_time.minute:02d}:{current_time.second:02d}-auxiliary_value.npy",
        )
        tmbrs_path = os.path.join(
            self.obs_dir,
            "1batms_merged_npy_1.0deg",
            f"{current_time.year:04d}",
            f"{current_time.year:04d}-{current_time.month:02d}-{current_time.day:02d}",
            f"{current_time.hour:02d}:{current_time.minute:02d}:{current_time.second:02d}-brightness_temperature_value.npy",
        )
        mask_path = os.path.join(
            self.obs_dir,
            "1batms_merged_npy_1.0deg",
            f"{current_time.year:04d}",
            f"{current_time.year:04d}-{current_time.month:02d}-{current_time.day:02d}",
            f"{current_time.hour:02d}:{current_time.minute:02d}:{current_time.second:02d}-mask.npy",
        )
        if os.path.exists(mask_path):
            auxiliary_value = np.load(auxiliarty_path)
            tmbrs_value = np.load(tmbrs_path)
            mask = np.load(mask_path)
            np_tmbrs_data = tmbrs_value.astype(np.float32)
            np_auxiliary_data = auxiliary_value.astype(np.float32)
            scanline_idx = self.meta_data["auxiliary_value"]["fields_in_order"].index("scanline")
            np_auxiliary_data[scanline_idx] = auxiliary_value[scanline_idx] / 12
            fovn_idx = self.meta_data["auxiliary_value"]["fields_in_order"].index("fov")
            np_auxiliary_data[fovn_idx] = auxiliary_value[fovn_idx] / 96
            orbit_number_idx = self.meta_data["auxiliary_value"]["fields_in_order"].index("orbit_number")
            np_auxiliary_data[orbit_number_idx] = auxiliary_value[orbit_number_idx]
            saza_idx = self.meta_data["auxiliary_value"]["fields_in_order"].index("satellite_zenith_angle")
            np_auxiliary_data[saza_idx] = np.cos(np.deg2rad(auxiliary_value[saza_idx]))
            saaa_idx = self.meta_data["auxiliary_value"]["fields_in_order"].index("satellite_azimuth_angle")
            np_auxiliary_data[saaa_idx] = np.cos(np.deg2rad(auxiliary_value[saaa_idx] / 2))
            soza_idx = self.meta_data["auxiliary_value"]["fields_in_order"].index("solar_zenith_angle")
            np_auxiliary_data[soza_idx] = np.cos(np.deg2rad(auxiliary_value[soza_idx]))
            soaa_idx = self.meta_data["auxiliary_value"]["fields_in_order"].index("solar_azimuth_angle")
            np_auxiliary_data[soaa_idx] = np.cos(np.deg2rad(auxiliary_value[soaa_idx] / 2))
            satellite_height_idx = self.meta_data["auxiliary_value"]["fields_in_order"].index("satellite_height")
            np_auxiliary_data[satellite_height_idx] = (auxiliary_value[satellite_height_idx] - self.satellite_height_scaler["satellite_height_min"]) / (self.satellite_height_scaler["satellite_height_max"] - self.satellite_height_scaler["satellite_height_min"])
            geolocation_quality_flags_idx = self.meta_data["auxiliary_value"]["fields_in_order"].index("geolocation_quality_flags")
            np_auxiliary_data[geolocation_quality_flags_idx] = (auxiliary_value[geolocation_quality_flags_idx] - self.geolocation_quality_flags_scaler["geolocation_quality_flags_min"]) / (self.geolocation_quality_flags_scaler["geolocation_quality_flags_max"] - self.geolocation_quality_flags_scaler["geolocation_quality_flags_min"])
            scan_quality_flags_idx = self.meta_data["auxiliary_value"]["fields_in_order"].index("scan_quality_flags")
            np_auxiliary_data[scan_quality_flags_idx] = (auxiliary_value[scan_quality_flags_idx] - self.scan_quality_flags_scaler["scan_quality_flags_min"]) / (self.scan_quality_flags_scaler["scan_quality_flags_max"] - self.scan_quality_flags_scaler["scan_quality_flags_min"])
            granule_quality_flags_idx = self.meta_data["auxiliary_value"]["fields_in_order"].index("granule_quality_flags")
            np_auxiliary_data[granule_quality_flags_idx] = (auxiliary_value[granule_quality_flags_idx] - self.granule_quality_flags_scaler["granule_quality_flags_min"]) / (self.granule_quality_flags_scaler["granule_quality_flags_max"] - self.granule_quality_flags_scaler["granule_quality_flags_min"])
            np_mask = mask.astype(np.int32)
        else:
            np_tmbrs_data = (np.ones(self.tmbrs_shape) * np.nan).astype(np.float32)
            np_auxiliary_data = (np.ones(self.auxiliary_shape) * np.nan).astype(np.float32)
            np_mask = (np.zeros(self.tmbrs_shape[-2:])).astype(np.int32)

        return np_tmbrs_data, np_auxiliary_data, np_mask

    def __getitem__(self, global_idx):
        """按全局索引取一个样本。

        负索引按 Python 习惯转换为正索引；将 ``_get_atms`` 中的 NaN 占位替换为 0，
        亮温按 ``tmbrs_transforms`` 归一化后乘 ``mask``，辅助场直接乘 ``mask``。

        Args:
            global_idx (int): 样本索引 (支持负索引)。

        Returns:
            tuple:
            - era5 (torch.Tensor): ``[V, H, W]``，已归一化。
            - sat_data (torch.Tensor): ``[len(aux)+len(tmbrs), H, W]``，辅助场 + 亮温。
            - mask (torch.Tensor): ``[1, H, W]``。
            - tmbrs_std (torch.Tensor): ``[len(tmbrs_vars)]``。
            - tmbrs_vars (list[str]): 亮温变量名。
        """
        if global_idx < 0:
            global_idx += self.__len__()

        inp = self._get_era5(global_idx)

        tmbrs_data, auxiliary_data, mask = self._get_atms(global_idx)

        tmbrs_data = torch.from_numpy(np.nan_to_num(tmbrs_data))
        auxiliary_data = torch.from_numpy(np.nan_to_num(auxiliary_data))
        mask = torch.unsqueeze(torch.from_numpy(mask), dim=0)
        tmbrs_data = self.tmbrs_transforms(tmbrs_data) * mask
        auxiliary_data = auxiliary_data * mask

        return self.era5_transforms(inp), \
               torch.concat([auxiliary_data, tmbrs_data], dim=0), \
               mask, torch.from_numpy(self.tmbrs_std).to(inp.dtype), self.tmbrs_vars

def collate_fn(batch):
    """ATMS DataLoader 的默认 ``collate_fn``。

    把 :meth:`NpyDataset.__getitem__` 返回的 5 元组在 batch 维拼接。
    ``tmbrs_vars`` 是 Python list，从 ``batch[0]`` 透传 (假设同 DataLoader 内变量顺序一致)。

    Args:
        batch (list): 长度为 ``batch_size`` 的样本列表。

    Returns:
        tuple: ``(era5, sat_data, mask, tmbrs_std, tmbrs_vars)``，前四项为带 batch 维
        的张量，最后一项为亮温变量名列表。
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