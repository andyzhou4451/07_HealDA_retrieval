# -*- coding: utf-8 -*-
"""
天气预报任务的状态场 :class:`NpyDataset`。

ERA5 (或同化背景场) 以 ``YYYY/YYYY-MM-DD/HH:MM:SS.npy`` 三级目录组织，
本模块 :class:`NpyDataset` 负责：
    1. 把连续小时索引映射回 ``datetime``，定位具体 npy 文件；
    2. 在 ``__getitem__`` 中产出 ``(input, targets, lead_times/100, variables, iter_num, std)``，
       其中 ``targets`` 是 ``iter_num`` 个不同 lead-time 的 ERA5 真值堆叠；
    3. ``collate_fn_forecast`` 把多个样本的张量字段 ``torch.stack`` 起来。

差异对照：
    - 与 :mod:`obs_dataset.py` 不同：本类只产 ERA5 状态场，不附加卫星亮温；
    - 与 :mod:`compression.state_dataset` 不同：本类额外产出 ``iter_num`` 个 lead-time target，
      并把 ``lead_times/100`` 作为模型 lead-time 嵌入的输入。
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from datetime import datetime, timedelta


class NpyDataset(Dataset):
    """ERA5 状态场的 forecast Dataset。

    每个样本产出：
        - ``inp``：当前时刻 ERA5 状态场 (经归一化)；
        - ``tgts``：未来 ``iter_num`` 个 lead-time 的 ERA5 真值堆叠；
        - ``lead_times/100``：归一化后的 lead-time (小时 / 100)，便于模型端嵌入；
        - ``variables``：变量名列表 (str)；
        - ``iter_num``：自回归迭代数 (int)；
        - ``std``：归一化 std (用于反归一化或 loss 计算)。

    Attributes:
        root_dir (str): 数据根目录。
        mode (str): ``"train"`` / ``"val"`` / ``"test"``。
        variables (list): 气象变量名列表。
        max_lead_time (int): 最大预报时效 (小时)。
        iter_num (int): 自回归迭代次数。
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
        max_lead_time: int,
        iter_num: int,
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
            max_lead_time: 最大预报时效
            iter_num: 迭代预测次数
            transforms: 数据归一化
            std: 数据标准差
        """
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode
        self.variables = variables
        self.max_lead_time = max_lead_time
        self.iter_num = iter_num
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
        """计算有效样本数。

        自回归 (``iter_num > 1``) 时，按 6 小时步长采样 (与 ERA5 数据频率一致)；
        单步 (``iter_num == 1``) 时，按 1 小时步长，并保留 ``max_lead_time`` 小时给未来 target。

        Returns:
            int: 有效样本数量。
        """
        # 计算有效样本数
        if self.iter_num > 1:
            return (self.total_hours - self.max_lead_time * self.iter_num) // 6
        else:
            return self.total_hours - self.max_lead_time

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
        """取一条 forecast 训练/验证样本。

        行为细节：
            - 支持负索引 (``global_idx < 0`` 时回卷到末尾)；
            - 自回归模式下把 global_idx 还原成"绝对小时偏移" (``* 6``)；
            - 训练时从 ``[1, 3, 6, 12, 24]`` 小时随机抽样 lead-time，
              验证 / 测试固定为 24h；
            - 产出 ``iter_num`` 个 target，分别在 ``global_idx + iter_idx * lead_time`` 处采样。

        Args:
            global_idx (int): 样本索引。

        Returns:
            tuple: ``(input, targets, lead_times/100, variables, iter_num, std)``。
        """
        if global_idx < 0:
            global_idx += self.__len__()

        if self.iter_num > 1:
            global_idx = int(6 * global_idx)

        # train模式随机选择lead_time，val/test模式固定为24h
        if self.mode == "train":
            lead_times = np.random.choice([1, 3, 6, 12, 24], size=1)
        else:
            lead_times = 24 * np.ones(shape=1)

        # 输入数据
        inp = self._get_era5(global_idx)

        # 目标数据 (多个迭代预测)
        tgts = []
        for iter_idx in range(1, self.iter_num + 1):
            output_idx = int(global_idx + iter_idx * lead_times[0])
            tgt = self._get_era5(output_idx)
            tgts.append(tgt)

        tgts = torch.stack(tgts, dim=0)

        # 返回: (input, targets, lead_times/100, variables, iter_num, std)
        return (
            inp,
            tgts,
            torch.from_numpy(lead_times).to(tgts.dtype) / 100,
            self.variables,
            self.iter_num,
            torch.from_numpy(self.std).to(tgts.dtype),
        )

def collate_fn_forecast(batch):
    """Forecast 任务的默认 collate 函数。

    把 batch 内多条样本的 tensor 字段 (input / targets / lead_times / std)
    沿第 0 维堆叠，非 tensor 字段 (``variables``, ``iter_num``) 直接复用第 0 条样本。

    Args:
        batch (list): ``__getitem__`` 输出的若干条样本。

    Returns:
        tuple: 与 ``__getitem__`` 形状一致的 batch。
    """
    inps = torch.stack([b[0] for b in batch], dim=0)
    tgts = torch.stack([b[1] for b in batch], dim=0)
    lead_times = torch.stack([b[2] for b in batch])
    variables = batch[0][3]
    iter_num = batch[0][4]
    std = batch[0][5]
    return (
        inps,
        tgts,
        lead_times,
        variables,
        iter_num,
        std,
    )