import math
import os
import sys
import glob
from typing import Any, Dict, Optional, Tuple
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import re
import logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format='%(name)s - %(levelname)s - %(message)s')


class NpyDataset(Dataset):
    """随机背景场同化任务的 :class:`NpyDataset`。

    与 forecast / compression 的 NpyDataset 差异：
        1. 每条样本产出 ``(bg, obs_data, obs_mask, tgt, lead_times, ...)``，
           其中 ``bg`` = 随机起点 ERA5 状态场，``tgt`` = 起点 + ``max_lead_time`` 时刻的 ERA5 真值；
        2. 在 :func:`__getitem__` 中按 ``obs_list`` 调度 ``_get_atms`` / ``_get_amsua`` / ``_get_mhs`` /
           ``_get_hrs4`` / ``_get_prepbufr`` / ``_get_satwnd`` / ``_get_ascat`` 加载各种观测；
        3. 微波观测在 DAW (Data Assimilation Window) 内按 ``dt_obs`` 步长多次采样，
           堆叠后与 ERA5 ``bg / tgt`` 一同交给 trainer。

    Attributes:
        era5_dir (str): ERA5 再分析数据根目录。
        obs_dir (str): 观测数据根目录。
        mode (str): ``"train"`` / ``"val"`` / ``"test"``。
        era5_vars (list): ERA5 变量名列表。
        obs_list (list): 观测变量名列表 (e.g. ``["atms", "amsua", "mhs", "hrs4",
            "prepbufr", "satwnd", "ascat"]``)。
        obs_dict (dict): 观测变量配置映射。
        max_lead_time (int): 最大预报时效 (小时)。
        iter_num (int): 背景场预报步长 (训练时随机分配 lead_time)。
        daw (int): DAW 长度 (小时)。
        dt_obs (int): DAW 内观测采样间隔 (小时)。
        dt_data (int): 训练数据采样间隔 (小时)。
        num_lat (int): 纬度格点数。
        num_lon (int): 经度格点数。
        era5_transforms (dict): ERA5 归一化算子 (来自 datamodule)。
        conventional_transforms (dict): 常规观测归一化算子。
        microwave_transforms (dict): 微波观测归一化算子。
        microwave_meta_data (dict): 微波观测 schema JSON。
        microwave_scaler (dict): 微波观测 min-max scaler。
        get_obs (dict): ``name → method`` 的观测调度表。
    """

    def __init__(self, 
            era5_dir,
            obs_dir,
            mode,
            era5_vars,
            obs_list,
            obs_dict,
            start_year,
            end_year,
            max_lead_time,
            iter_num,
            daw,
            dt_obs,
            dt_data,
            era5_transforms,
            conventional_transforms,
            microwave_transforms,
            microwave_meta_data,
            microwave_scaler,
            num_lat,
            num_lon,
            debug,
        ) -> None:
        """初始化随机背景场同化任务的数据集。

        主要流程：
            1. 保存全部配置参数；
            2. 调用 :func:`prepare_microwave_auxiliary_shape` 和 :func:`prepare_conventional_shape`
               确定每个观测源的 ``tmbrs / auxiliary / conventional`` 张量形状 (用于 NaN 缺测回退)；
            3. 构建 ``self.get_obs`` 调度表 (obs_name → ``_get_<name>`` 方法)；
            4. 根据 ``debug`` 标记选择 1 个月或完整时间区间。
        """
        super().__init__()
        self.era5_dir = era5_dir
        self.obs_dir = obs_dir
        self.mode = mode
        self.era5_vars = era5_vars
        self.obs_list = obs_list
        self.obs_dict = obs_dict
        self.max_lead_time = max_lead_time
        self.iter_num = iter_num
        self.daw = daw
        self.dt_obs = dt_obs
        self.dt_data = dt_data
        self.num_lat, self.num_lon = num_lat, num_lon

        self.era5_transforms = era5_transforms
        self.conventional_transforms = conventional_transforms
        self.microwave_transforms = microwave_transforms
        self.microwave_meta_data = microwave_meta_data
        self.microwave_scaler = microwave_scaler
        self.prepare_microwave_auxiliary_shape()
        self.prepare_conventional_shape()
        
        self.get_obs = {}
        for name in self.obs_list:
            if name == "atms":
                self.get_obs["atms"] = self._get_atms
            elif name == "amsua":
                self.get_obs["amsua"] = self._get_amsua
            elif name == "mhs":
                self.get_obs["mhs"] = self._get_mhs
            elif name == "hrs4":
                self.get_obs["hrs4"] = self._get_hrs4
            elif name == "prepbufr":
                self.get_obs["prepbufr"] = self._get_prepbufr
            elif name == "satwnd":
                self.get_obs["satwnd"] = self._get_satwnd
            elif name == "ascat":
                self.get_obs["ascat"] = self._get_ascat

        # 计算时间范围
        if debug:
            self.start_time = datetime(start_year, 1, 1, 0, 0)
            self.end_time = datetime(start_year, 2, 1, 0, 0)
        else:
            self.start_time = datetime(start_year, 1, 1, 0, 0)
            self.end_time = datetime(end_year, 1, 1, 0, 0)
        self.total_hours = int((self.end_time - self.start_time).total_seconds() // 3600)
                
    def prepare_microwave_auxiliary_shape(self):
        """预计算每个微波观测 (atms / amsua / mhs / hrs4) 的 tmbrs / auxiliary 形状。

        用于 :func:`_get_atms` 等方法在 NaN 缺测时构造占位张量。
        形状 ``[channels, num_lat, num_lon]`` 与 ERA5 网格对齐。
        """
        self.microwave_tmbrs_shape = {}
        self.microwave_auxiliary_shape = {}
        for name in self.obs_list:
            if "microwave" in self.obs_dict.keys() and name in self.obs_dict["microwave"].keys():
                self.microwave_tmbrs_shape[name] = (len(self.obs_dict["microwave"][name]["tmbrs_vars"]), self.num_lat, self.num_lon)
                self.microwave_auxiliary_shape[name] = (len(self.obs_dict["microwave"][name]["auxiliary_dict"]), self.num_lat, self.num_lon)

    def prepare_conventional_shape(self):
        """预计算每个常规观测 (prepbufr / satwnd / ascat) 的张量形状。

        形状 ``[vars, num_lat, num_lon]``，与 ERA5 网格对齐。
        """
        self.conventional_shape = {}
        for name in self.obs_list:
            if "conventional" in self.obs_dict.keys() and name in self.obs_dict["conventional"].keys():
                self.conventional_shape[name] = (len(self.obs_dict["conventional"][name]["vars"]), self.num_lat, self.num_lon)

    def __len__(self):
        """计算有效样本数。

        按 ``dt_data`` (训练数据采样间隔) 采样，并扣除 ``max_lead_time + daw``。

        Returns:
            int: 有效样本数量。
        """
        return int(self.total_hours - self.max_lead_time - self.daw) // self.dt_data

    def _get_era5(self, idx):
        """加载单个时间点的 ERA5 状态场 (不做归一化)，并返回 ``current_time``。

        与 forecast 版本不同：本方法返回的是未归一化的 np → torch.float32 张量，
        归一化由 ``__getitem__`` 末尾通过 ``self.era5_transforms["transforms"]`` 完成。

        Args:
            idx (int): 自 ``self.start_time`` 起的小时偏移。

        Returns:
            tuple[torch.Tensor, datetime]: ``(data, current_time)``。
        """
        current_time = self.start_time + relativedelta(hours=idx)
        file_path = os.path.join(
            self.era5_dir,
            f"{current_time.year:04d}",
            f"{current_time.year:04d}-{current_time.month:02d}-{current_time.day:02d}",
            f"{current_time.hour:02d}:{current_time.minute:02d}:{current_time.second:02d}.npy",
        )

        data = np.load(file_path)

        return torch.from_numpy(data).to(dtype=torch.float32), current_time

    def _get_atms(self, idx):
        """加载 ATMS 卫星亮温 + 辅助场 + 掩码。

        DAW 内按 ``dt_obs`` 多次采样后堆叠。对辅助场做 scanline/12、fov/96、
        ``cos(satellite_zenith_angle)`` / ``cos(azimuth/2)`` 等归一化 / 角度编码，
        以及 min-max scaler 对 satellite_height / *_flags 归一化。缺测时刻回退为 NaN。

        Args:
            idx (int): 自 ``self.start_time`` 起的小时偏移。

        Returns:
            tuple[torch.Tensor, torch.Tensor]: ``(sat_data, mask)``，
            其中 ``sat_data`` = ``concat([auxiliary, tmbrs], dim=1)``。
        """
        current_time = self.start_time + relativedelta(hours=idx)
        obs_times = [current_time + relativedelta(hours=i) for i in range(0, self.daw, self.dt_obs)]
        # logging.info(f"Load observations at {obs_times}")
        np_tmbrs_data, np_auxiliary_data, np_mask = [], [], []
        for obs_time in obs_times:
            auxiliarty_path = os.path.join(
                self.obs_dir,
                "1batms_merged_npy_1.0deg",
                f"{obs_time.year:04d}",
                f"{obs_time.year:04d}-{obs_time.month:02d}-{obs_time.day:02d}",
                f"{obs_time.hour:02d}:{obs_time.minute:02d}:{obs_time.second:02d}-auxiliary_value.npy",
            )
            tmbrs_path = os.path.join(
                self.obs_dir,
                "1batms_merged_npy_1.0deg",
                f"{obs_time.year:04d}",
                f"{obs_time.year:04d}-{obs_time.month:02d}-{obs_time.day:02d}",
                f"{obs_time.hour:02d}:{obs_time.minute:02d}:{obs_time.second:02d}-brightness_temperature_value.npy",
            )
            mask_path = os.path.join(
                self.obs_dir,
                "1batms_merged_npy_1.0deg",
                f"{obs_time.year:04d}",
                f"{obs_time.year:04d}-{obs_time.month:02d}-{obs_time.day:02d}",
                f"{obs_time.hour:02d}:{obs_time.minute:02d}:{obs_time.second:02d}-mask.npy",
            )
            if os.path.exists(mask_path):
                auxiliary_value = np.load(auxiliarty_path)
                tmbrs_value = np.load(tmbrs_path)
                mask = np.load(mask_path)
                np_tmbrs_data.append(tmbrs_value.astype(np.float32))
                np_auxiliary_data_ = auxiliary_value.astype(np.float32)
                scanline_idx = self.microwave_meta_data["atms"]["auxiliary_value"]["fields_in_order"].index("scanline")
                np_auxiliary_data_[scanline_idx] = np_auxiliary_data_[scanline_idx] / 12
                fovn_idx = self.microwave_meta_data["atms"]["auxiliary_value"]["fields_in_order"].index("fov")
                np_auxiliary_data_[fovn_idx] = np_auxiliary_data_[fovn_idx] / 96
                orbit_number_idx = self.microwave_meta_data["atms"]["auxiliary_value"]["fields_in_order"].index("orbit_number")
                np_auxiliary_data_[orbit_number_idx] = np_auxiliary_data_[orbit_number_idx]
                saza_idx = self.microwave_meta_data["atms"]["auxiliary_value"]["fields_in_order"].index("satellite_zenith_angle")
                np_auxiliary_data_[saza_idx] = np.cos(np.deg2rad(np_auxiliary_data_[saza_idx]))
                saaa_idx = self.microwave_meta_data["atms"]["auxiliary_value"]["fields_in_order"].index("satellite_azimuth_angle")
                np_auxiliary_data_[saaa_idx] = np.cos(np.deg2rad(np_auxiliary_data_[saaa_idx] / 2))
                soza_idx = self.microwave_meta_data["atms"]["auxiliary_value"]["fields_in_order"].index("solar_zenith_angle")
                np_auxiliary_data_[soza_idx] = np.cos(np.deg2rad(np_auxiliary_data_[soza_idx]))
                soaa_idx = self.microwave_meta_data["atms"]["auxiliary_value"]["fields_in_order"].index("solar_azimuth_angle")
                np_auxiliary_data_[soaa_idx] = np.cos(np.deg2rad(np_auxiliary_data_[soaa_idx] / 2))
                satellite_height_idx = self.microwave_meta_data["atms"]["auxiliary_value"]["fields_in_order"].index("satellite_height")
                np_auxiliary_data_[satellite_height_idx] = (np_auxiliary_data_[satellite_height_idx] - self.microwave_scaler["atms"]["satellite_height_scaler"]["satellite_height_min"]) / (self.microwave_scaler["atms"]["satellite_height_scaler"]["satellite_height_max"] - self.microwave_scaler["atms"]["satellite_height_scaler"]["satellite_height_min"])
                geolocation_quality_flags_idx = self.microwave_meta_data["atms"]["auxiliary_value"]["fields_in_order"].index("geolocation_quality_flags")
                np_auxiliary_data_[geolocation_quality_flags_idx] = (np_auxiliary_data_[geolocation_quality_flags_idx] - self.microwave_scaler["atms"]["geolocation_quality_flags_scaler"]["geolocation_quality_flags_min"]) / (self.microwave_scaler["atms"]["geolocation_quality_flags_scaler"]["geolocation_quality_flags_max"] - self.microwave_scaler["atms"]["geolocation_quality_flags_scaler"]["geolocation_quality_flags_min"])
                scan_quality_flags_idx = self.microwave_meta_data["atms"]["auxiliary_value"]["fields_in_order"].index("scan_quality_flags")
                np_auxiliary_data_[scan_quality_flags_idx] = (np_auxiliary_data_[scan_quality_flags_idx] - self.microwave_scaler["atms"]["scan_quality_flags_scaler"]["scan_quality_flags_min"]) / (self.microwave_scaler["atms"]["scan_quality_flags_scaler"]["scan_quality_flags_max"] - self.microwave_scaler["atms"]["scan_quality_flags_scaler"]["scan_quality_flags_min"])
                granule_quality_flags_idx = self.microwave_meta_data["atms"]["auxiliary_value"]["fields_in_order"].index("granule_quality_flags")
                np_auxiliary_data_[granule_quality_flags_idx] = (np_auxiliary_data_[granule_quality_flags_idx] - self.microwave_scaler["atms"]["granule_quality_flags_scaler"]["granule_quality_flags_min"]) / (self.microwave_scaler["atms"]["granule_quality_flags_scaler"]["granule_quality_flags_max"] - self.microwave_scaler["atms"]["granule_quality_flags_scaler"]["granule_quality_flags_min"])
                np_auxiliary_data.append(np_auxiliary_data_)
                np_mask.append(mask.astype(np.float32))
            else:
                np_tmbrs_data.append((np.ones(self.microwave_tmbrs_shape["atms"]) * np.nan).astype(np.float32))
                np_auxiliary_data.append((np.ones(self.microwave_auxiliary_shape["atms"]) * np.nan).astype(np.float32))
                np_mask.append((np.zeros(self.microwave_tmbrs_shape["atms"][-2:])).astype(np.float32))

        np_tmbrs_data = np.nan_to_num(np.stack(np_tmbrs_data, axis=0), nan=0.0, posinf=0.0, neginf=0.0)
        np_auxiliary_data = np.nan_to_num(np.stack(np_auxiliary_data, axis=0), nan=0.0, posinf=0.0, neginf=0.0)
        np_mask = np.nan_to_num(np.stack(np_mask, axis=0), nan=0.0, posinf=0.0, neginf=0.0)

        tmbrs_tensor = torch.from_numpy(np_tmbrs_data)
        auxiliary_tensor = torch.from_numpy(np_auxiliary_data)
        mask_tensor = torch.unsqueeze(torch.from_numpy(np_mask), dim=1)
        tmbrs_tensor = self.microwave_transforms["atms"]["transforms"](tmbrs_tensor) * mask_tensor
        auxiliary_tensor = auxiliary_tensor * mask_tensor

        return torch.concat([auxiliary_tensor, tmbrs_tensor], dim=1), mask_tensor

    def _get_amsua(self, idx):
        """加载 AMSU-A 卫星亮温 + 辅助场 + 掩码。

        对辅助场做 fovn/30、lsql/2、``cos(saza)`` / ``cos(soza)`` 等归一化，
        hols / hmsl / solazi / bearaz 按 min-max scaler 归一化或 cos 编码。

        Args:
            idx (int): 自 ``self.start_time`` 起的小时偏移。

        Returns:
            tuple[torch.Tensor, torch.Tensor]: ``(sat_data, mask)``。
        """
        current_time = self.start_time + relativedelta(hours=idx)
        obs_times = [current_time + relativedelta(hours=i) for i in range(0, self.daw, self.dt_obs)]
        # logging.info(f"Load observations at {obs_times}")
        np_tmbrs_data, np_auxiliary_data, np_mask = [], [], []
        for obs_time in obs_times:
            auxiliarty_path = os.path.join(
                self.obs_dir,
                "1bamsua_merged_npy_1.0deg",
                f"{obs_time.year:04d}",
                f"{obs_time.year:04d}-{obs_time.month:02d}-{obs_time.day:02d}",
                f"{obs_time.hour:02d}:{obs_time.minute:02d}:{obs_time.second:02d}-auxiliary_value.npy",
            )
            tmbrs_path = os.path.join(
                self.obs_dir,
                "1bamsua_merged_npy_1.0deg",
                f"{obs_time.year:04d}",
                f"{obs_time.year:04d}-{obs_time.month:02d}-{obs_time.day:02d}",
                f"{obs_time.hour:02d}:{obs_time.minute:02d}:{obs_time.second:02d}-tmbrs_value.npy",
            )
            mask_path = os.path.join(
                self.obs_dir,
                "1bamsua_merged_npy_1.0deg",
                f"{obs_time.year:04d}",
                f"{obs_time.year:04d}-{obs_time.month:02d}-{obs_time.day:02d}",
                f"{obs_time.hour:02d}:{obs_time.minute:02d}:{obs_time.second:02d}-mask.npy",
            )
            if os.path.exists(mask_path):
                auxiliary_value = np.load(auxiliarty_path)
                tmbrs_value = np.load(tmbrs_path)
                mask = np.load(mask_path)
                np_tmbrs_data.append(tmbrs_value.astype(np.float32))
                np_auxiliary_data_ = auxiliary_value.astype(np.float32)
                fovn_idx = self.microwave_meta_data["amsua"]["auxiliary_value"]["fields_in_order"].index("fovn")
                np_auxiliary_data_[fovn_idx] = np_auxiliary_data_[fovn_idx] / 30
                lsql_idx = self.microwave_meta_data["amsua"]["auxiliary_value"]["fields_in_order"].index("lsql")
                np_auxiliary_data_[lsql_idx] = np_auxiliary_data_[lsql_idx] / 2
                saza_idx = self.microwave_meta_data["amsua"]["auxiliary_value"]["fields_in_order"].index("saza")
                np_auxiliary_data_[saza_idx] = np.cos(np.deg2rad(np_auxiliary_data_[saza_idx]))
                soza_idx = self.microwave_meta_data["amsua"]["auxiliary_value"]["fields_in_order"].index("soza")
                np_auxiliary_data_[soza_idx] = np.cos(np.deg2rad(np_auxiliary_data_[soza_idx]))
                hols_idx = self.microwave_meta_data["amsua"]["auxiliary_value"]["fields_in_order"].index("hols")
                np_auxiliary_data_[hols_idx] = (np_auxiliary_data_[hols_idx] - self.microwave_scaler["amsua"]["hols_scaler"]["hols_min"]) / (self.microwave_scaler["amsua"]["hols_scaler"]["hols_max"] - self.microwave_scaler["amsua"]["hols_scaler"]["hols_min"])
                hmsl_idx = self.microwave_meta_data["amsua"]["auxiliary_value"]["fields_in_order"].index("hmsl")
                np_auxiliary_data_[hmsl_idx] = (np_auxiliary_data_[hmsl_idx] - self.microwave_scaler["amsua"]["hmsl_scaler"]["hmsl_min"]) / (self.microwave_scaler["amsua"]["hmsl_scaler"]["hmsl_max"] - self.microwave_scaler["amsua"]["hmsl_scaler"]["hmsl_min"])
                solazi_idx = self.microwave_meta_data["amsua"]["auxiliary_value"]["fields_in_order"].index("solazi")
                np_auxiliary_data_[solazi_idx] = np.cos(np.deg2rad(np_auxiliary_data_[solazi_idx]) / 2) 
                bearaz_idx = self.microwave_meta_data["amsua"]["auxiliary_value"]["fields_in_order"].index("bearaz")
                np_auxiliary_data_[bearaz_idx] = np.cos(np.deg2rad(np_auxiliary_data_[bearaz_idx]) / 2) 
                np_auxiliary_data.append(np_auxiliary_data_)
                np_mask.append(mask.astype(np.float32))
            else:
                np_tmbrs_data.append((np.ones(self.microwave_tmbrs_shape["amsua"]) * np.nan).astype(np.float32))
                np_auxiliary_data.append((np.ones(self.microwave_auxiliary_shape["amsua"]) * np.nan).astype(np.float32))
                np_mask.append((np.zeros(self.microwave_tmbrs_shape["amsua"][-2:])).astype(np.float32))

        np_tmbrs_data = np.nan_to_num(np.stack(np_tmbrs_data, axis=0), nan=0.0, posinf=0.0, neginf=0.0)
        np_auxiliary_data = np.nan_to_num(np.stack(np_auxiliary_data, axis=0), nan=0.0, posinf=0.0, neginf=0.0)
        np_mask = np.nan_to_num(np.stack(np_mask, axis=0), nan=0.0, posinf=0.0, neginf=0.0)

        tmbrs_tensor = torch.from_numpy(np_tmbrs_data)
        auxiliary_tensor = torch.from_numpy(np_auxiliary_data)
        mask_tensor = torch.unsqueeze(torch.from_numpy(np_mask), dim=1)
        tmbrs_tensor = self.microwave_transforms["amsua"]["transforms"](tmbrs_tensor) * mask_tensor
        auxiliary_tensor = auxiliary_tensor * mask_tensor

        return torch.concat([auxiliary_tensor, tmbrs_tensor], dim=1), mask_tensor

    def _get_mhs(self, idx):
        """加载 MHS 卫星亮温 + 辅助场 + 掩码。

        对辅助场做 fovn/90、lsql/2、``cos(saza)`` / ``cos(soza)`` 等归一化，
        hols / hmsl / solazi / bearaz 按 min-max scaler 归一化或 cos 编码。

        Args:
            idx (int): 自 ``self.start_time`` 起的小时偏移。

        Returns:
            tuple[torch.Tensor, torch.Tensor]: ``(sat_data, mask)``。
        """
        current_time = self.start_time + relativedelta(hours=idx)
        obs_times = [current_time + relativedelta(hours=i) for i in range(0, self.daw, self.dt_obs)]
        # logging.info(f"Load observations at {obs_times}")
        np_tmbrs_data, np_auxiliary_data, np_mask = [], [], []
        for obs_time in obs_times:
            auxiliarty_path = os.path.join(
                self.obs_dir,
                "1bmhs_merged_npy_1.0deg",
                f"{obs_time.year:04d}",
                f"{obs_time.year:04d}-{obs_time.month:02d}-{obs_time.day:02d}",
                f"{obs_time.hour:02d}:{obs_time.minute:02d}:{obs_time.second:02d}-auxiliary_value.npy",
            )
            tmbrs_path = os.path.join(
                self.obs_dir,
                "1bmhs_merged_npy_1.0deg",
                f"{obs_time.year:04d}",
                f"{obs_time.year:04d}-{obs_time.month:02d}-{obs_time.day:02d}",
                f"{obs_time.hour:02d}:{obs_time.minute:02d}:{obs_time.second:02d}-tmbrs_value.npy",
            )
            mask_path = os.path.join(
                self.obs_dir,
                "1bmhs_merged_npy_1.0deg",
                f"{obs_time.year:04d}",
                f"{obs_time.year:04d}-{obs_time.month:02d}-{obs_time.day:02d}",
                f"{obs_time.hour:02d}:{obs_time.minute:02d}:{obs_time.second:02d}-mask.npy",
            )
            if os.path.exists(mask_path):
                auxiliary_value = np.load(auxiliarty_path)
                tmbrs_value = np.load(tmbrs_path)
                mask = np.load(mask_path)
                np_tmbrs_data.append(tmbrs_value.astype(np.float32))
                np_auxiliary_data_ = auxiliary_value.astype(np.float32)
                fovn_idx = self.microwave_meta_data["mhs"]["auxiliary_value"]["fields_in_order"].index("fovn")
                np_auxiliary_data_[fovn_idx] = np_auxiliary_data_[fovn_idx] / 90
                lsql_idx = self.microwave_meta_data["mhs"]["auxiliary_value"]["fields_in_order"].index("lsql")
                np_auxiliary_data_[lsql_idx] = np_auxiliary_data_[lsql_idx] / 2
                saza_idx = self.microwave_meta_data["mhs"]["auxiliary_value"]["fields_in_order"].index("saza")
                np_auxiliary_data_[saza_idx] = np.cos(np.deg2rad(np_auxiliary_data_[saza_idx]))
                soza_idx = self.microwave_meta_data["mhs"]["auxiliary_value"]["fields_in_order"].index("soza")
                np_auxiliary_data_[soza_idx] = np.cos(np.deg2rad(np_auxiliary_data_[soza_idx]))
                hols_idx = self.microwave_meta_data["mhs"]["auxiliary_value"]["fields_in_order"].index("hols")
                np_auxiliary_data_[hols_idx] = (np_auxiliary_data_[hols_idx] - self.microwave_scaler["mhs"]["hols_scaler"]["hols_min"]) / (self.microwave_scaler["mhs"]["hols_scaler"]["hols_max"] - self.microwave_scaler["mhs"]["hols_scaler"]["hols_min"])
                hmsl_idx = self.microwave_meta_data["mhs"]["auxiliary_value"]["fields_in_order"].index("hmsl")
                np_auxiliary_data_[hmsl_idx] = (np_auxiliary_data_[hmsl_idx] - self.microwave_scaler["mhs"]["hmsl_scaler"]["hmsl_min"]) / (self.microwave_scaler["mhs"]["hmsl_scaler"]["hmsl_max"] - self.microwave_scaler["mhs"]["hmsl_scaler"]["hmsl_min"])
                solazi_idx = self.microwave_meta_data["mhs"]["auxiliary_value"]["fields_in_order"].index("solazi")
                np_auxiliary_data_[solazi_idx] = np.cos(np.deg2rad(np_auxiliary_data_[solazi_idx]) / 2) 
                bearaz_idx = self.microwave_meta_data["mhs"]["auxiliary_value"]["fields_in_order"].index("bearaz")
                np_auxiliary_data_[bearaz_idx] = np.cos(np.deg2rad(np_auxiliary_data_[bearaz_idx]) / 2) 
                np_auxiliary_data.append(np_auxiliary_data_)
                np_mask.append(mask.astype(np.float32))
            else:
                np_tmbrs_data.append((np.ones(self.microwave_tmbrs_shape["mhs"]) * np.nan).astype(np.float32))
                np_auxiliary_data.append((np.ones(self.microwave_auxiliary_shape["mhs"]) * np.nan).astype(np.float32))
                np_mask.append((np.zeros(self.microwave_tmbrs_shape["mhs"][-2:])).astype(np.float32))

        np_tmbrs_data = np.nan_to_num(np.stack(np_tmbrs_data, axis=0), nan=0.0, posinf=0.0, neginf=0.0)
        np_auxiliary_data = np.nan_to_num(np.stack(np_auxiliary_data, axis=0), nan=0.0, posinf=0.0, neginf=0.0)
        np_mask = np.nan_to_num(np.stack(np_mask, axis=0), nan=0.0, posinf=0.0, neginf=0.0)

        tmbrs_tensor = torch.from_numpy(np_tmbrs_data)
        auxiliary_tensor = torch.from_numpy(np_auxiliary_data)
        mask_tensor = torch.unsqueeze(torch.from_numpy(np_mask), dim=1)
        tmbrs_tensor = self.microwave_transforms["mhs"]["transforms"](tmbrs_tensor) * mask_tensor
        auxiliary_tensor = auxiliary_tensor * mask_tensor

        return torch.concat([auxiliary_tensor, tmbrs_tensor], dim=1), mask_tensor

    def _get_hrs4(self, idx):
        """加载 HIRS4 卫星亮温 + 辅助场 + 掩码。

        注意：HIRS4 的 tmbrs 文件比通道数多 1 行末尾冗余，本函数会通过
        ``np.load(tmbrs_path)[:-1]`` 去除。辅助场处理与 AMSU-A 类似
        (fovn/56、lsql/2、``cos(saza)`` / ``cos(soza)`` 等)。

        Args:
            idx (int): 自 ``self.start_time`` 起的小时偏移。

        Returns:
            tuple[torch.Tensor, torch.Tensor]: ``(sat_data, mask)``。
        """
        current_time = self.start_time + relativedelta(hours=idx)
        obs_times = [current_time + relativedelta(hours=i) for i in range(0, self.daw, self.dt_obs)]
        # logging.info(f"Load observations at {obs_times}")
        np_tmbrs_data, np_auxiliary_data, np_mask = [], [], []
        for obs_time in obs_times:
            auxiliarty_path = os.path.join(
                self.obs_dir,
                "1bhrs4_merged_npy_1.0deg",
                f"{obs_time.year:04d}",
                f"{obs_time.year:04d}-{obs_time.month:02d}-{obs_time.day:02d}",
                f"{obs_time.hour:02d}:{obs_time.minute:02d}:{obs_time.second:02d}-auxiliary_value.npy",
            )
            tmbrs_path = os.path.join(
                self.obs_dir,
                "1bhrs4_merged_npy_1.0deg",
                f"{obs_time.year:04d}",
                f"{obs_time.year:04d}-{obs_time.month:02d}-{obs_time.day:02d}",
                f"{obs_time.hour:02d}:{obs_time.minute:02d}:{obs_time.second:02d}-tmbrs_value.npy",
            )
            mask_path = os.path.join(
                self.obs_dir,
                "1bhrs4_merged_npy_1.0deg",
                f"{obs_time.year:04d}",
                f"{obs_time.year:04d}-{obs_time.month:02d}-{obs_time.day:02d}",
                f"{obs_time.hour:02d}:{obs_time.minute:02d}:{obs_time.second:02d}-mask.npy",
            )
            if os.path.exists(mask_path):
                auxiliary_value = np.load(auxiliarty_path)
                tmbrs_value = np.load(tmbrs_path)[:-1]
                mask = np.load(mask_path)
                np_tmbrs_data.append(tmbrs_value.astype(np.float32))
                np_auxiliary_data_ = auxiliary_value.astype(np.float32)
                fovn_idx = self.microwave_meta_data["hrs4"]["auxiliary_value"]["fields_in_order"].index("fovn")
                np_auxiliary_data_[fovn_idx] = np_auxiliary_data_[fovn_idx] / 56
                lsql_idx = self.microwave_meta_data["hrs4"]["auxiliary_value"]["fields_in_order"].index("lsql")
                np_auxiliary_data_[lsql_idx] = np_auxiliary_data_[lsql_idx] / 2
                saza_idx = self.microwave_meta_data["hrs4"]["auxiliary_value"]["fields_in_order"].index("saza")
                np_auxiliary_data_[saza_idx] = np.cos(np.deg2rad(np_auxiliary_data_[saza_idx]))
                soza_idx = self.microwave_meta_data["hrs4"]["auxiliary_value"]["fields_in_order"].index("soza")
                np_auxiliary_data_[soza_idx] = np.cos(np.deg2rad(np_auxiliary_data_[soza_idx]))
                hols_idx = self.microwave_meta_data["hrs4"]["auxiliary_value"]["fields_in_order"].index("hols")
                np_auxiliary_data_[hols_idx] = (np_auxiliary_data_[hols_idx] - self.microwave_scaler["hrs4"]["hols_scaler"]["hols_min"]) / (self.microwave_scaler["hrs4"]["hols_scaler"]["hols_max"] - self.microwave_scaler["hrs4"]["hols_scaler"]["hols_min"])
                hmsl_idx = self.microwave_meta_data["hrs4"]["auxiliary_value"]["fields_in_order"].index("hmsl")
                np_auxiliary_data_[hmsl_idx] = (np_auxiliary_data_[hmsl_idx] - self.microwave_scaler["hrs4"]["hmsl_scaler"]["hmsl_min"]) / (self.microwave_scaler["hrs4"]["hmsl_scaler"]["hmsl_max"] - self.microwave_scaler["hrs4"]["hmsl_scaler"]["hmsl_min"])
                solazi_idx = self.microwave_meta_data["hrs4"]["auxiliary_value"]["fields_in_order"].index("solazi")
                np_auxiliary_data_[solazi_idx] = np.cos(np.deg2rad(np_auxiliary_data_[solazi_idx]) / 2) 
                bearaz_idx = self.microwave_meta_data["hrs4"]["auxiliary_value"]["fields_in_order"].index("bearaz")
                np_auxiliary_data_[bearaz_idx] = np.cos(np.deg2rad(np_auxiliary_data_[bearaz_idx]) / 2) 
                np_auxiliary_data.append(np_auxiliary_data_)
                np_mask.append(mask.astype(np.float32))
            else:
                np_tmbrs_data.append((np.ones(self.microwave_tmbrs_shape["hrs4"]) * np.nan).astype(np.float32))
                np_auxiliary_data.append((np.ones(self.microwave_auxiliary_shape["hrs4"]) * np.nan).astype(np.float32))
                np_mask.append((np.zeros(self.microwave_tmbrs_shape["hrs4"][-2:])).astype(np.float32))

        np_tmbrs_data = np.nan_to_num(np.stack(np_tmbrs_data, axis=0), nan=0.0, posinf=0.0, neginf=0.0)
        np_auxiliary_data = np.nan_to_num(np.stack(np_auxiliary_data, axis=0), nan=0.0, posinf=0.0, neginf=0.0)
        np_mask = np.nan_to_num(np.stack(np_mask, axis=0), nan=0.0, posinf=0.0, neginf=0.0)

        tmbrs_tensor = torch.from_numpy(np_tmbrs_data)
        auxiliary_tensor = torch.from_numpy(np_auxiliary_data)
        mask_tensor = torch.unsqueeze(torch.from_numpy(np_mask), dim=1)
        tmbrs_tensor = self.microwave_transforms["hrs4"]["transforms"](tmbrs_tensor) * mask_tensor
        auxiliary_tensor = auxiliary_tensor * mask_tensor

        return torch.concat([auxiliary_tensor, tmbrs_tensor], dim=1), mask_tensor

    def _get_prepbufr(self, idx):
        """加载 prepbufr (常规探空) 观测值 + 掩码。

        DAW 内按 ``dt_obs`` 多次采样后堆叠，缺测时刻回退为 NaN；
        ``mask`` 用 ``~np.isnan(obs_data) * 1`` 标记有效格点。

        Args:
            idx (int): 自 ``self.start_time`` 起的小时偏移。

        Returns:
            tuple[torch.Tensor, torch.Tensor]: ``(obs_normalized * mask, mask)``。
        """
        current_time = self.start_time + relativedelta(hours=idx)
        obs_times = [current_time + relativedelta(hours=i) for i in range(0, self.daw, self.dt_obs)]
        np_obs_data, np_obs_mask = [], []
        for obs_time in obs_times:
            obs_path = os.path.join(
                self.obs_dir,
                "GDAS_prepbufr_merged_npy_1.0deg",
                f"{obs_time.year:04d}",
                f"{obs_time.year:04d}-{obs_time.month:02d}-{obs_time.day:02d}",
                f"{obs_time.hour:02d}:{obs_time.minute:02d}:{obs_time.second:02d}-obs_value.npy",
            )
            if os.path.exists(obs_path):
                obs_data = np.load(obs_path).squeeze()
                np_obs_data.append(obs_data)
                np_obs_mask.append(~np.isnan(obs_data) * 1)
            else:
                np_obs_data.append(np.ones(self.conventional_shape["prepbufr"]) * np.nan)
                np_obs_mask.append(np.zeros(self.conventional_shape["prepbufr"]))

        np_obs_data = np.nan_to_num(np.stack(np_obs_data, axis=0), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        np_obs_mask = np.nan_to_num(np.stack(np_obs_mask, axis=0), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        prepbufrs = torch.from_numpy(np_obs_data)
        prepbufr_masks = torch.from_numpy(np_obs_mask)

        return self.conventional_transforms["prepbufr"]["transforms"](prepbufrs) * prepbufr_masks, prepbufr_masks

    def _get_satwnd(self, idx):
        """加载 satwnd (卫星风) 观测值 + 掩码。

        DAW 内按 ``dt_obs`` 多次采样后堆叠，缺测时刻回退为 NaN；
        ``mask`` 用 ``~np.isnan(obs_data) * 1`` 标记有效格点。

        Args:
            idx (int): 自 ``self.start_time`` 起的小时偏移。

        Returns:
            tuple[torch.Tensor, torch.Tensor]: ``(obs_normalized * mask, mask)``。
        """
        current_time = self.start_time + relativedelta(hours=idx)
        obs_times = [current_time + relativedelta(hours=i) for i in range(0, self.daw, self.dt_obs)]
        np_obs_data, np_obs_mask = [], []
        for obs_time in obs_times:
            obs_path = os.path.join(
                self.obs_dir,
                "satwnd_merged_npy_1.0deg",
                f"{obs_time.year:04d}",
                f"{obs_time.year:04d}-{obs_time.month:02d}-{obs_time.day:02d}",
                f"{obs_time.hour:02d}:{obs_time.minute:02d}:{obs_time.second:02d}-obs_value.npy",
            )
            if os.path.exists(obs_path):
                obs_data = np.load(obs_path).squeeze()
                np_obs_data.append(obs_data)
                np_obs_mask.append(~np.isnan(obs_data) * 1)
            else:
                np_obs_data.append(np.ones(self.conventional_shape["satwnd"]) * np.nan)
                np_obs_mask.append(np.zeros(self.conventional_shape["satwnd"]))

        np_obs_data = np.nan_to_num(np.stack(np_obs_data, axis=0), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        np_obs_mask = np.nan_to_num(np.stack(np_obs_mask, axis=0), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        satwnds = torch.from_numpy(np_obs_data)
        satwnd_masks = torch.from_numpy(np_obs_mask)

        return self.conventional_transforms["satwnd"]["transforms"](satwnds) * satwnd_masks, satwnd_masks

    def _get_ascat(self, idx):
        """加载 ascat (散射计海面风) 观测值 + 掩码。

        DAW 内按 ``dt_obs`` 多次采样后堆叠，缺测时刻回退为 NaN；
        ``mask`` 用 ``~np.isnan(obs_data) * 1`` 标记有效格点。

        Args:
            idx (int): 自 ``self.start_time`` 起的小时偏移。

        Returns:
            tuple[torch.Tensor, torch.Tensor]: ``(obs_normalized * mask, mask)``。
        """
        current_time = self.start_time + relativedelta(hours=idx)
        obs_times = [current_time + relativedelta(hours=i) for i in range(0, self.daw, self.dt_obs)]
        np_obs_data, np_obs_mask = [], []
        for obs_time in obs_times:
            obs_path = os.path.join(
                self.obs_dir,
                "ascat_b_merged_npy_1.0deg",
                f"{obs_time.year:04d}",
                f"{obs_time.year:04d}-{obs_time.month:02d}-{obs_time.day:02d}",
                f"{obs_time.hour:02d}:{obs_time.minute:02d}:{obs_time.second:02d}-obs_value.npy",
            )
            if os.path.exists(obs_path):
                obs_data = np.load(obs_path).squeeze()
                np_obs_data.append(obs_data)
                np_obs_mask.append(~np.isnan(obs_data) * 1)
            else:
                np_obs_data.append(np.ones(self.conventional_shape["ascat"]) * np.nan)
                np_obs_mask.append(np.zeros(self.conventional_shape["ascat"]))

        np_obs_data = np.nan_to_num(np.stack(np_obs_data, axis=0), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        np_obs_mask = np.nan_to_num(np.stack(np_obs_mask, axis=0), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        ascats = torch.from_numpy(np_obs_data)
        ascat_masks = torch.from_numpy(np_obs_mask)

        return self.conventional_transforms["ascat"]["transforms"](ascats) * ascat_masks, ascat_masks

    def __getitem__(self, global_idx):
        """取一条随机背景场同化样本。

        行为细节：
            - ``global_idx`` 先按 ``dt_data`` 还原成绝对小时偏移；
            - 训练时为每个迭代步独立随机选 lead-time (``[6, 12, 24]``)，
              验证 / 测试固定为 24h；
            - ``bg`` 取自 ``global_idx + max_lead_time - sum(lead_times)`` 时刻，
              ``tgt`` 取自 ``global_idx + max_lead_time`` 时刻；
            - 每个 obs 名称调 ``self.get_obs[name]`` 加载对应的卫星 / 常规观测。

        Args:
            global_idx (int): 样本索引 (可负)。

        Returns:
            tuple: ``(bg, obs_list, obs_data, obs_mask, tgt, lead_times/100,
            era5_vars, obs_dict, bg_time, tgt_time, era5_transforms,
            microwave_transforms, conventional_transforms)``。
        """
        if global_idx < 0:
            global_idx += self.__len__()
        global_idx = int(global_idx * self.dt_data)

        if self.mode == "train":
            lead_times = np.random.choice([6, 12, 24], size=self.iter_num)
        else:
            lead_times = 24 * np.ones(shape=self.iter_num)

        bg, bg_time = self._get_era5(int(global_idx + self.max_lead_time - np.sum(lead_times)))
        tgt, tgt_time = self._get_era5(int(global_idx + self.max_lead_time))

        obs_data, obs_mask = {}, {}
        for name in self.obs_list:
            obs_data[name], obs_mask[name] = self.get_obs[name](int(global_idx + self.max_lead_time))

        return self.era5_transforms["transforms"](bg), \
               self.obs_list, \
               obs_data, \
               obs_mask, \
               self.era5_transforms["transforms"](tgt), \
               torch.from_numpy(lead_times).to(tgt.dtype) / 100, \
               self.era5_vars, \
               self.obs_dict, \
               bg_time, tgt_time, \
               self.era5_transforms, self.microwave_transforms, self.conventional_transforms

def collate_fn(batch):
    """随机背景场同化任务的默认 collate 函数。

    把 batch 内多条样本的 tensor 字段 (bg / tgt / lead_times) 沿第 0 维堆叠；
    对 obs_data / obs_mask 这种 ``{name: tensor}`` 嵌套结构，按 name 分别 stack。

    Args:
        batch (list): ``__getitem__`` 输出的若干条样本。

    Returns:
        tuple: ``(inps, obs_list, obs_data, obs_mask, tgt, lead_times, variables,
        obs_dict, init_time, tgt_time, era5_transforms, microwave_transforms,
        conventional_transforms)``。
    """
    inps = torch.stack([batch[i][0] for i in range(len(batch))], dim=0)
    obs_list = batch[0][1]
    obs_data, obs_mask = {}, {}
    for name in obs_list:
        obs_data[name] = torch.stack([batch[i][2][name] for i in range(len(batch))], dim=0)
        obs_mask[name] = torch.stack([batch[i][3][name] for i in range(len(batch))], dim=0)
    tgt = torch.stack([batch[i][4] for i in range(len(batch))], dim=0)
    lead_times = torch.stack([batch[i][5] for i in range(len(batch))])
    variables = batch[0][6]
    obs_dict = batch[0][7]
    init_time = [batch[i][8] for i in range(len(batch))]
    tgt_time = [batch[i][9] for i in range(len(batch))]
    era5_transforms = batch[0][10]
    microwave_transforms = batch[0][11]
    conventional_transforms = batch[0][12]
    return (
        inps,
        obs_list,
        obs_data,
        obs_mask,
        tgt,
        lead_times,
        [v for v in variables],
        obs_dict,
        init_time,
        tgt_time,
        era5_transforms,
        microwave_transforms,
        conventional_transforms
    )