"""
Kendall 多任务不确定性加权（多任务损失自适应）

参考 Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh
Losses for Scene Geometry and Semantics" (CVPR 2018)。为每个 task 维护
一个可学习的 ``log σ²``（"noise parameter"），并把多任务损失聚合为：

    L_total = Σ_i [ 0.5 * exp(-log_var_i) * L_i + 0.5 * log_var_i ]

直观含义：

- ``exp(-log_var)`` 即 1/σ² 的"精度"（precision）；
  loss 项前的 ``0.5 * precision`` 等同于把 loss 当作对数似然
  ``-log N(y | f, σ²)`` 后取负，再乘以 1/2。
- 末项 ``0.5 * log_var`` 是正则项；它让模型倾向于不把 σ² 推向 0
  （否则 loss 项爆炸）。

上游依赖：被 ``src/models/assimilate/xichenda/*`` 的 DA 训练器调用，
用于把"forecast loss / obs cost / analysis loss"加权聚合。
下游调用：仅作为 ``nn.Module`` 被 trainer 实例化。

文件名说明：``uncertanty`` 是历史拼写错误（应为 ``uncertainty``），
被外部 import 直接使用，保留拼写以保证向后兼容。
"""

import torch
import torch.nn as nn


class LearnableUncertaintyWeighting(nn.Module):
    """Kendall 多任务不确定性加权器。

    每个 task 拥有一个可学习标量 ``log σ²``（``log_vars``），初始值由
    ``init_log_var`` 控制（默认 ``0.0``，即 σ² = 1，对所有 task 等权起步）。

    Args:
        num_tasks: 多任务个数；``log_vars`` 形状为 ``(num_tasks,)``。
        init_log_var: ``log_vars`` 的初始值；``0.0`` 表示等权起步。

    Attributes:
        log_vars: ``nn.Parameter``，shape ``(num_tasks,)``，dtype ``float32``。

    Note:
        当 ``init_log_var`` 设得过大（> 5）时，``0.5 * log_var`` 项会主导
        总 loss，模型会"偏好"降低不确定性；通常保持 ``0.0`` 即可。
    """

    def __init__(self, num_tasks: int, init_log_var: float = 0.0):
        super().__init__()
        self.log_vars = nn.Parameter(torch.full((num_tasks,), init_log_var))

    def forward(self, task_losses) -> torch.Tensor:
        """把多个 task 的 loss 加权聚合为总 loss。

        Args:
            task_losses: 长度为 ``num_tasks`` 的可迭代对象（如 list）；
                元素可以是 0-D 标量 tensor 或同形 batch loss。

        Returns:
            ``(total_loss, weighted_losses)``：

            - ``total_loss``：标量 tensor，所有 task 加权 loss 之和。
            - ``weighted_losses``：shape ``(num_tasks,)`` 的 stack，
              便于在日志中分别记录每个 task 的有效贡献。

        Raises:
            AssertionError: 当 ``len(task_losses) != num_tasks`` 时。
        """
        assert len(task_losses) == self.log_vars.size(0), \
            f"任务数不匹配，期望{self.log_vars.size(0)}, 实际{len(task_losses)}"

        weighted_losses = []
        total_loss = 0

        for i, loss in enumerate(task_losses):
            # 1/σ²：precision，数值上等于 exp(-log σ²)
            precision = torch.exp(-self.log_vars[i])
            # Kendall 公式：0.5 * precision * loss + 0.5 * log_var
            #  前项为任务 loss 的高斯 NLL（去掉常数），后项为正则项
            weighted = 0.5 * precision * loss + 0.5 * self.log_vars[i]
            weighted_losses.append(weighted)
            total_loss = total_loss + weighted

        return total_loss, torch.stack(weighted_losses)

    def _get_uncerteinties(self):
        """返回当前各任务的 ``σ = exp(0.5 * log σ²)``，detach 避免影响梯度。

        Returns:
            ``Tensor``，shape ``(num_tasks,)``，值 ≥ 0。
        """
        return torch.exp(0.5 * self.log_vars).detach()

    def _get_precisions(self):
        """返回当前各任务的 ``precision = 1 / σ² = exp(-log σ²)``，detach。

        Returns:
            ``Tensor``，shape ``(num_tasks,)``，值 ≥ 0。
        """
        return torch.exp(-self.log_vars).detach()