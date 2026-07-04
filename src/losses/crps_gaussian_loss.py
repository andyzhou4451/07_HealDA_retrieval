"""
CRPS-Gaussian 损失（掩码版本）

对高斯预测分布 ``N(mu, σ²)`` 计算 Continuous Ranked Probability Score
(CRPS)，并对 ``mask`` 为 ``0`` 的位置做"不计入平均"的掩码。

CRPS-Gaussian 解析式：

    CRPS(N(μ, σ²), y) = σ * [ z * (2 Φ(z) - 1) + 2 φ(z) - 1/√π ]

其中 ``z = (y - μ) / σ``，``Φ`` 是标准正态 CDF，``φ`` 是 PDF。
当 ``μ`` 完美预测（即 ``z → 0``）时，CRPS 趋于 ``σ * (√(2/π) - 1/√π)``，
代表"平均不确定性代价"；当 ``y = μ`` 且 ``σ → 0`` 时，CRPS → 0，
完全命中。

上游依赖：被 ``src/pipeline/*/trainer.py`` 作为 ``loss_fn`` 替换默认 MSE；
对应配置文件 ``configs/loss_fn/crps_gaussian.yaml``（注意 ``guassian``
是历史拼写错误，保留以保证向后兼容）。
下游调用：每个训练 step 都会调用 ``forward``，并把 ``mask`` 与
``target`` 一同传入（``mask`` 通常对应 ERA5 的海陆 / 缺测掩码）。
"""

import torch
import torch.nn as nn
import math


class CRPS_Gaussian_Loss(nn.Module):
    """CRPS-Gaussian 训练损失。

    网络预测 ``(mu, log_var)``，其中 ``log_var`` 通过 ``exp(log_var / 2)``
    转换为 ``σ``，再代入 CRPS-Gaussian 解析式。

    Args:
        eps: 防止 ``σ = 0`` 时的数值下溢；默认 ``1e-6``，同时用作
            ``mask.sum() == 0`` 时的安全分母。

    Attributes:
        eps: ``float``。

    Note:
        ``forward`` 接收 ``mask`` 而非 ``nan`` 掩码；当数据中包含 NaN 时，
        应在 datamodule 层先 ``nan_to_num`` 再传入，避免反向传播时梯度
        出现 NaN。
    """

    def __init__(self, eps=1e-6):
        super(CRPS_Gaussian_Loss, self).__init__()
        self.eps = eps

    def crps_gaussian(self, mu, sigma, target):
        """计算单点 CRPS-Gaussian（不带掩码）。

        公式：``CRPS = σ * ( z * (2 Φ(z) - 1) + 2 φ(z) - 1/√π )``，
        其中 ``z = (target - mu) / σ``。

        Args:
            mu: 预测均值，任意形状。
            sigma: 预测标准差，shape 与 ``mu`` 同；需 > 0（已 clamp）。
            target: 真值，shape 与 ``mu`` 同。

        Returns:
            与 ``mu`` 同形的 CRPS 张量，单位与 ``target`` 一致。
        """
        sigma = torch.clamp(sigma, self.eps)
        z = (target - mu) / sigma

        # 标准正态 PDF 和 CDF（数值稳定的写法）
        phi = torch.exp(-z**2 / 2) / math.sqrt(2 * math.pi)
        Phi = 0.5 *(1 + torch.erf(z / math.sqrt(2)))

        # CRPS 公式：σ * [ z*(2Φ-1) + 2φ - 1/√π ]
        #   - z*(2Φ-1)：对称项；y=μ 时为 0
        #   - 2φ：钟形"中心加权"；y=μ 时为 2/√(2π)
        #   - 1/√π：常数偏移，使 CRPS ≥ 0
        crps = sigma * (z * (2 * Phi - 1) + 2 * phi - 1 / math.sqrt(math.pi))

        return crps

    def forward(self, mu, log_var, target, mask):
        """掩码版 CRPS-Gaussian 损失。

        Args:
            mu: 模型输出的订正结果（预测均值）。
            log_var: 模型预测的 log 方差，即 ``log(σ²)``。
            target: 训练标签。
            mask: 观测掩码，``1`` 表示有观测，``0`` 表示无观测。

        Returns:
            标量损失：``masked_loss.sum() / (mask.sum() + eps)``。
        """
        # 把 log_var 转换为 σ
        sigma = torch.exp(log_var / 2)

        loss = self.crps_gaussian(mu, sigma, target)

        masked_loss = loss * mask

        # 分母加 eps 防 0；若所有点都被 mask 掉，则退化为 NaN-safe 的 0
        return masked_loss.sum() / (mask.sum() + self.eps)