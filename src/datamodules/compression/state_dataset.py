# -*- coding: utf-8 -*-
"""
天气压缩任务的 :class:`NpyDataset`。

与 :mod:`src.datamodules.forecast.state_dataset` 的差异：
    - 不需要 ``max_lead_time`` / ``iter_num`` —— 压缩任务只重建"当前时刻"；
    - ``__len__`` 直接返回 ``total_hours``；
    - ``__getitem__`` 只产出 ``(input, variables, std)``，没有 lead-time / target；
    - ``collate_fn_forecast`` 复用 forecast 的 collate 函数 (字段对齐)。
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from datetime import datetime, timedelta

class NpyDataset(Dataset):
    """压缩任务的 ERA5 状态场 Dataset。

    每个样本产出：
        - ``inp``：当前时刻 ERA5 状态场 (经归一化)；
        - ``variables``：变量名列表 (str)；
        - ``std``：归一化 std (用于反归一化或 loss 计算)。

    Attributes:
        root_dir (str): 数据根目录。
        mode (str): ``"train"`` / ``"val"`` / ``"test"``。
        variables (list): 气象变量名列表。
        transforms (Normalize): ``torchvision`` 归一化算子。
        std (np.ndarray): 归一化 std。
        start_time / end_time (datetime): 数据集覆盖的时间区间。
        total_hours (int): 时间区间内总小时数。
    """

    def __init__(
        self,
        root_dir: str,
        mode: str,
        variables: list,
        start_year: int,
        end_year: int,
        transforms=None,
        std=None,
        debug=False,
    ):
        """
        Args:
            root_dir: 数据根目录
            mode: "train" 或 "val/test" 模式
            variables: 气象变量列表
            start_year: 起始年份
            end_year: 结束年份
            transforms: 数据归一化
            std: 数据标准差
        """
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode
        self.variables = variables
        self.transforms = transforms
        self.std = std

        # 计算时间范围
        if debug:
            self.start_time = datetime(start_year, 1, 1, 0, 0)
            self.end_time = datetime(start_year, 2, 1, 0, 0)
        else:
            self.start_time = datetime(start_year, 1, 1, 0, 0)
            self.end_time = datetime(end_year, 1, 1, 0, 0)
        self.total_hours = int((self.end_time - self.start_time).total_seconds() // 3600)

    def __len__(self):
        """返回有效样本数 (= ``total_hours``)。压缩任务不需要扣减 lead-time。

        Returns:
            int: 有效样本数量。
        """
        # 计算有效样本数
        return self.total_hours

    def _get_era5(self, idx: int) -> torch.Tensor:
        """加载并归一化单个时间点的 ERA5 状态场。

        路径约定：``{root_dir}/{YYYY}/{YYYY-MM-DD}/{HH:MM:SS}.npy``。

        Args:
            idx (int): 自 ``self.start_time`` 起的小时偏移。

        Returns:
            torch.Tensor: 经 ``self.transforms`` 归一化后的张量。
        """
        current_time = self.start_time + timedelta(hours=idx)
        file_path = os.path.join(
            self.root_dir,
            f"{current_time.year:04d}",
            f"{current_time.year:04d}-{current_time.month:02d}-{current_time.day:02d}",
            f"{current_time.hour:02d}:{current_time.minute:02d}:{current_time.second:02d}.npy",
        )
        data = np.load(file_path)
        return self.transforms(torch.from_numpy(data).to(dtype=torch.float32))

    def __getitem__(self, global_idx: int):
        """取一条压缩样本：当前时刻 ERA5 状态场。

        行为细节：
            - 支持负索引；
            - 输出 ``(inp, variables, std)``，没有 lead-time / target。

        Args:
            global_idx (int): 样本索引。

        Returns:
            tuple: ``(inp, variables, std)``。
        """
        if global_idx < 0:
            global_idx += self.__len__()

        # 输入数据
        inp = self._get_era5(global_idx)

        # 返回: (input, variables, std)
        return (
            inp,
            self.variables,
            torch.from_numpy(self.std).to(inp.dtype),
        )

def collate_fn_forecast(batch):
    """压缩任务的默认 collate 函数。

    把 batch 内多条样本的 ``inp`` 沿第 0 维堆叠，
    非 tensor 字段 (``variables``) 复用第 0 条样本。

    Args:
        batch (list): ``__getitem__`` 输出的若干条样本。

    Returns:
        tuple: 与 ``__getitem__`` 形状一致的 batch。
    """
    inps = torch.stack([b[0] for b in batch], dim=0)
    variables = batch[0][1]
    std = batch[0][2]
    return (
        inps,
        variables,
        std,
    )