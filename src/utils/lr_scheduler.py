"""
学习率调度器

提供 ``CosineSchedulerWithWarmup``：线性 warmup + 余弦退火，
封装自 ``torch.optim.lr_scheduler.LRScheduler``。

公式：

- 当 ``last_epoch < warmup_epochs``：
      lr = warmup_start_lr + (base_lr - warmup_start_lr) * (epoch + 1) / warmup_epochs
- 否则：
      progress = (last_epoch - warmup_epochs) / (max_epochs - warmup_epochs)
      cosine = 0.5 * (1 + cos(pi * progress))
      lr = min_lr + (base_lr - min_lr) * cosine

上游依赖：业务 trainer 在 ``optimizer`` 创建后实例化本 scheduler；
``configs/training/*`` 中提供 ``warmup_epochs`` / ``max_epochs``。
下游调用：每个 epoch 末尾由 trainer 调用 ``scheduler.step()``。
"""

import math
from torch.optim.lr_scheduler import LRScheduler


class CosineSchedulerWithWarmup(LRScheduler):
    """Cosine annealing with linear warmup.

    Args:
        optimizer: 被调度的优化器。
        warmup_epochs: 线性 warmup 阶段 epoch 数。
        max_epochs: 总训练 epoch 数。
        min_lr: cosine 末尾的最小学习率；默认 ``0``。
        warmup_start_lr: warmup 起始学习率；默认 ``1e-7``。

    Note:
        与标准 ``torch.optim.lr_scheduler.CosineAnnealingLR`` 不同，
        本类把"线性 warmup"和"余弦退火"合成到 ``get_lr``，每个 param
        group 都按相同规则调整。
    """

    def __init__(
        self,
        optimizer,
        warmup_epochs,
        max_epochs,
        min_lr=0,
        warmup_start_lr=1e-7,
    ):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        super().__init__(optimizer)

    def get_lr(self):
        """计算当前 epoch 下每个 param group 的学习率。

        Returns:
            ``list[float]``，长度等于 param group 数。
        """
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup：从 warmup_start_lr 线性插值到 base_lr
            scale = (self.last_epoch + 1) / self.warmup_epochs
            return [self.warmup_start_lr + (base_lr - self.warmup_start_lr) * scale
                    for base_lr in self.base_lrs]
        else:
            # Cosine annealing：从 base_lr 余弦退火到 min_lr
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return [self.min_lr + (base_lr - self.min_lr) * cosine_decay
                    for base_lr in self.base_lrs]