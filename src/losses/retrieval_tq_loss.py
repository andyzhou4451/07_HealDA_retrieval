# -*- coding: utf-8 -*-
"""Loss functions for 13-level T/Q profile retrieval."""

from __future__ import annotations

from typing import Mapping, Sequence

import torch
from torch import nn
import torch.nn.functional as F


class RetrievalTQLoss(nn.Module):
    """Huber/MSE loss for ``[B,26,H,W]`` T/Q retrieval outputs."""

    def __init__(
        self,
        type: str = "huber",
        delta: float = 0.1,
        w_t: float = 1.0,
        w_q: float = 1.0,
        pressure_weights_t: Sequence[float] | None = None,
        pressure_weights_q: Sequence[float] | None = None,
        q_log_transform: bool = False,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if type not in {"huber", "mse"}:
            raise ValueError("RetrievalTQLoss.type must be 'huber' or 'mse'")
        self.loss_type = type
        self.delta = float(delta)
        self.w_t = float(w_t)
        self.w_q = float(w_q)
        self.q_log_transform = bool(q_log_transform)
        self.eps = float(eps)
        pt = torch.tensor(pressure_weights_t if pressure_weights_t is not None else [1.0] * 13, dtype=torch.float32)
        pq = torch.tensor(pressure_weights_q if pressure_weights_q is not None else [1.0] * 13, dtype=torch.float32)
        self.register_buffer("pressure_weights_t", pt.view(1, 13, 1, 1), persistent=False)
        self.register_buffer("pressure_weights_q", pq.view(1, 13, 1, 1), persistent=False)

    def _loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "huber":
            return F.huber_loss(pred, target, delta=self.delta, reduction="none")
        return F.mse_loss(pred, target, reduction="none")

    @staticmethod
    def _masked_mean(loss: torch.Tensor, weight: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weighted = loss * weight * mask.to(dtype=loss.dtype)
        denom = (weight * mask.to(dtype=loss.dtype)).sum().clamp_min(1.0)
        return weighted.sum() / denom

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Mapping[str, torch.Tensor]:
        if pred.ndim == 5:
            pred = pred.reshape(pred.shape[0], 26, pred.shape[-2], pred.shape[-1])
        if target.ndim == 5:
            target = target.reshape(target.shape[0], 26, target.shape[-2], target.shape[-1])
        if pred.shape != target.shape:
            raise ValueError(f"Prediction/target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
        if pred.shape[1] != 26:
            raise ValueError(f"Expected 26 output channels, got {pred.shape[1]}")
        pred_t, pred_q = pred[:, :13], pred[:, 13:]
        tgt_t, tgt_q = target[:, :13], target[:, 13:]
        mask_t = torch.isfinite(pred_t) & torch.isfinite(tgt_t)
        mask_q = torch.isfinite(pred_q) & torch.isfinite(tgt_q)
        pred_t = torch.nan_to_num(pred_t, nan=0.0, posinf=0.0, neginf=0.0)
        tgt_t = torch.nan_to_num(tgt_t, nan=0.0, posinf=0.0, neginf=0.0)
        pred_q = torch.nan_to_num(pred_q, nan=0.0, posinf=0.0, neginf=0.0)
        tgt_q = torch.nan_to_num(tgt_q, nan=0.0, posinf=0.0, neginf=0.0)
        if self.q_log_transform:
            pred_q = torch.log(torch.clamp(pred_q, min=0.0) + self.eps)
            tgt_q = torch.log(torch.clamp(tgt_q, min=0.0) + self.eps)
        wt = self.pressure_weights_t.to(pred.device)
        wq = self.pressure_weights_q.to(pred.device)
        loss_t = self._masked_mean(self._loss(pred_t, tgt_t), wt, mask_t)
        loss_q = self._masked_mean(self._loss(pred_q, tgt_q), wq, mask_q)
        total = self.w_t * loss_t + self.w_q * loss_q
        if not torch.isfinite(total):
            total = torch.zeros((), dtype=pred.dtype, device=pred.device, requires_grad=True)
        return {"total_loss": total, "temperature_loss": loss_t, "humidity_loss": loss_q}
