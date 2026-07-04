import torch
import torch.nn as nn

"""
模型工具：参数统计与梯度统计

提供三个常用工具函数：

- ``format_number``：把大整数格式化为 ``K / M / B`` 后缀，便于日志可读。
- ``count_parameters_detailed``：区分 total / trainable / frozen 参数量。
- ``get_total_norm``：计算所有可训练参数的 L2 梯度范数（gradient clipping 用）。

上游依赖：被 ``BaseTrainer._save_ckpt`` 之外的训练日志模块调用；
每个 trainer 在 ``on_train_epoch_start`` / ``on_before_optimizer_step``
时调用 ``get_total_norm``。
下游调用：纯函数，无副作用，可安全用于任何 DDP rank。
"""


def format_number(num):
    """把整数格式化为 ``K / M / B`` 后缀，便于日志阅读。

    Args:
        num: 待格式化的非负整数 / 浮点。

    Returns:
        字符串，如 ``"1.23M"``、``"4.56B"``；小于 ``1e3`` 时直接返回原值。
    """
    if num >= 1e9: # 十亿
        return f"{num / 1e9:.2f}B"
    elif num >= 1e6: # 百万
        return f"{num / 1e6:.2f}M"
    elif num >= 1e3: # 千
        return f"{num / 1e3:.2f}K"
    else:
        return f"{num}"


def count_parameters_detailed(model):
    """详细统计模型的参数量。

    Args:
        model: ``nn.Module`` 实例；会遍历 ``model.named_parameters()``。

    Returns:
        ``(total_params, trainable_params, frozen_params)``：

        - ``total_params``：所有参数元素数。
        - ``trainable_params``：``requires_grad=True`` 的参数元素数。
        - ``frozen_params``：``requires_grad=False`` 的参数元素数。
    """
    total_params = 0
    trainable_params = 0
    frozen_params = 0

    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params

        if param.requires_grad:
            trainable_params += num_params
        else:
            frozen_params += num_params

    return total_params, trainable_params, frozen_params


def get_total_norm(model):
    """计算所有参数梯度的 L2 范数。

    仅统计 ``param.grad is not None`` 的参数；当某个参数在 backward 中
    未被使用（如 DDP ``find_unused_parameters=True`` 下）会被自动忽略。

    Args:
        model: ``nn.Module`` 实例。

    Returns:
        ``float``，L2 范数 ``sqrt(Σ ‖grad‖²)``；所有梯度均为 ``None`` 时
        返回 ``0.0``。
    """
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            # 计算每个参数的 L2 范数，再拼成全局范数（避免大数溢出）
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm ** 2
    total_norm = total_norm ** 0.5
    return total_norm