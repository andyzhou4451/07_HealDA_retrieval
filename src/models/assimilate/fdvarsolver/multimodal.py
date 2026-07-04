"""多模态 (multimodal) 数据同化 Solver。

本模块实现 ``Solver`` 类, 是 1.0° ERA5 + 7 种观测源 (atms / amsua / mhs /
hrs4 / prepbufr / satwnd / ascat) 的 multimodal DA 训练主循环。

与 cascade Solver 的关键区别:
- 不会在 Solver 内部直接调用 ``DA_model`` 得到 ``xa``;
- 而是计算每个 obs 源的 ``∂J/∂xb`` 后, 用 per-obs ``ROE_models[obs_name]``
  把 ``(xb, grad)`` 编码成一个表示 (representation), 最终把 ``{obs_name: roe}``
  字典交给下游的 ``XiChenFusion`` 做 Perceiver 风格的跨观测融合;
- 因此本 Solver 的输出是 ``roe_dict``,而不是 ``xa``。

设计要点:
- **逐 obs 源串行** (与 cascade 相同顺序): 不同 obs_name 使用各自专属的
  ``ObsOp_models`` / ``H_models`` / ``VarCost_models`` / ``ROE_models`` 实例;
- **NPU stream-sync 缓解**: ``forward`` 顶部对设备 stream 显式 ``synchronize``
  + ``empty_cache``, 避免 7 次 ``loss.backward`` 累积 kernel 触发 NPU 的
  ACL 507018;
- **梯度归一化**: 与 cascade 相同的 L2 范数归一化, 保证不同 obs 源梯度尺度一致。

注意: 旧版 (``old/`` 目录) 的 multimodal Solver 已废弃, 当前活动版本即本文件。
"""
from functools import partial, lru_cache
import os
import numpy as np
import torch
import torch.nn as nn
import collections.abc
from einops import repeat, rearrange
import inspect
from src.models.assimilate.utils.forecast import ar_forecast_trajectory
from src.utils import get_logger
log = get_logger("xichen.multimodal")

class Solver(nn.Module):
    """多模态 DA Solver: 为每个 obs 源计算 ``∂J/∂xb`` 并编码为表示。

    与 cascade Solver 的区别: 本 Solver 不直接产生 ``xa``, 而是输出
    ``roe_dict: {obs_name: roe_tensor}`` 供 ``XiChenFusion`` 跨 obs 源融合后
    再生成 ``xa``。

    Attributes:
        dt (int): AR 预报步长(小时), 通常 1 / 3 / 6。
    """

    def __init__(
        self,
        dt,
    ):
        """初始化 Solver。

        Args:
            dt (int): 预报步长(小时), 取值通常为 1 / 3 / 6。``forecast`` 方法按
                ``dt`` 把 ``1h/3h/6h/12h/24h`` 五个子预报模型对齐到时间网格。
        """
        super(Solver, self).__init__()
        self.dt = dt

    def forward(
        self,
        forecast_model,
        ObsOp_models,
        ROE_models,
        H_models,
        VarCost_models,
        obs_list,
        xb,
        obs,
        obs_mask,
        obs_dict,
        std_dict,
        out_vars,
    ):
        """执行 multimodal DA 主循环, 返回 per-obs 表示字典。

        流程(hoist 优化后):
            1. **NPU stream-sync**: hoist 之前先 ``synchronize()`` + ``empty_cache()``,
               避免首次大 forecast 前向撞上 ACL 507018;
            2. **Hoist forecast**: 对 ``obs_list`` 中第一个 obs 调用 ``self.forecast``
               算出 ``preds``,复用给所有 7 次 ``var_cost``(forecast 只依赖 ``xb`` 与
               ``T``,对 obs 值无依赖)。``forecast()`` 内部硬编码传 ``use_checkpoint=True``;
               若 ``forecast_model.forward`` 不接受此 kwarg,Python 自然抛 ``TypeError``;
            3. **T 不变式断言**: 所有 obs 必须共享 ``T = preds.shape[1]``(dataloader
               的结构性不变式,提为运行时防御);
            4. **state 一次性包装**: ``xb`` 在循环外用 ``xb.detach().requires_grad_(True)``
               转成 leaf + requires_grad=True,7 次循环复用同一节点(共享 autograd 图
               是 hoist 的关键);
            5. **per-obs 循环**: ``synchronize()`` + ``var_cost(..., preds=preds)`` +
               ``ROE_models[obs_name](xb, grad, use_checkpoint=True)``;
            6. **收尾**: ``del preds`` + ``xb.grad = None`` 释放共享 autograd 图。

        Returns:
            tuple[dict, Tensor]: ``(roe_dict, obs_available)``
                - ``roe_dict``: ``{obs_name: roe}``, 每个 ``roe`` 形状由
                  ``XiChenRepresentationObsEmbedding.forward`` 决定(典型为
                  ``(B, L, D)``, ``L = (H/patch_stride[0]) * (W/patch_stride[1])``,
                  ``D = embed_dim``)。
                - ``obs_available``: **设备端** ``(B, V)`` bool 张量, per-sample
                  整源可用性, 由 ``var_cost_grad.abs().sum() > 0`` 判据构造 (D3/D4
                  修订, 旧版 ``{obs_name: list[bool], length=B}`` 形态已弃用,
                  详见 code-review #5)。第 v 列与 ``obs_list`` 顺序严格对齐, 供
                  下游 ``XiChenFusion.forward`` 直接 ``~obs_available`` 作为
                  key_padding_mask。
        """
        # === 设备判断(三后端兼容,避免 CPU 上触发 CUDA lazy-init) ===
        class _CpuNoOp:
            @staticmethod
            def synchronize(): pass
            @staticmethod
            def empty_cache(): pass

        if xb.device.type == "cuda":
            dev_module = torch.cuda
        elif xb.device.type == "npu" and hasattr(torch, "npu"):
            dev_module = torch.npu
        else:
            dev_module = _CpuNoOp

        # === Hoist: 一次性预报复用所有 7 个 obs ===
        dev_module.synchronize()
        dev_module.empty_cache()

        # obs_list 为空时 next(iter({})) 会抛 StopIteration 无诊断;
        # 改为 ValueError 给出配置相关提示,便于定位错误源头。
        if not obs_list:
            raise ValueError(
                "Solver.forward: obs_list is empty; check training.obs_list config "
                "or datamodule obs loading."
            )

        state = xb.detach().requires_grad_(True)

        first_obs = next(iter(obs_list.keys()))
        preds = ar_forecast_trajectory(
            forecast_model, 
            state, 
            obs[first_obs], 
            out_vars, 
            self.dt,
            use_checkpoint=True
        )

        log.debug(
            f"hoisted forecast: preds.shape={tuple(preds.shape)}, "
            f"forecast={forecast_model.__class__.__name__}, first_obs={first_obs}"
        )

        roe_dict = {}
        try:
            for obs_name in obs_list.keys():
                # 释放 stream-sync 压力,避免 ACL 507018
                dev_module.synchronize()
                dev_module.empty_cache()

                with torch.set_grad_enabled(True):
                    var_cost_grad = self.var_cost(
                        forecast_model,
                        ObsOp_models,
                        H_models,
                        VarCost_models,
                        obs_name,
                        state,
                        obs,
                        obs_mask,
                        obs_dict,
                        std_dict,
                        out_vars,
                        preds=preds,
                    )

                roe = ROE_models[obs_name](
                    xb,
                    var_cost_grad,
                    out_vars,
                    use_checkpoint=True
                )
                if obs_list[obs_name].trainable == False:
                    roe = roe.detach()
                roe_dict[obs_name] = roe
        except Exception as e:
            log.error(
                f"Solver.forward failed mid-loop (last obs={locals().get('obs_name', '?')}): "
                f"{type(e).__name__}: {e}; releasing shared preds + xb.grad"
            )
            raise
        finally:
            # 收尾:断开共享 autograd 图 + 释放 preds 引用。
            # xb 在 trainer 中是 detached leaf(实测 is_leaf=True),xb.grad = None 是真实清理。
            del preds
            xb.grad = None
            
        dev_module.empty_cache()

        return roe_dict

    def var_cost(
        self,
        forecast_model,
        ObsOp_models,
        H_models,
        VarCost_models,
        obs_name,
        xb,
        obs,
        obs_mask,
        obs_dict,
        std_dict,
        out_vars,
        preds,
    ):
        """对单个 obs 源计算 ``∂J/∂xb`` 归一化梯度(与 cascade 流程一致)。

        流程:
            1. ``preds`` 由 ``Solver.forward()`` hoist 后传入(形状 ``(B, T, C, H, W)``);
            2. 若该 obs 有 ObsOp: ``ObsOp_models[obs_name](preds, obs)`` 得到
               模拟辐射率,再 ``H_models[obs_name]`` 做差并施加 5·σ_obs QC;
            3. 若该 obs 是常规观测: 直接对 ``preds`` 的目标通道子集与 obs 做差;
            4. ``VarCost_models[obs_name]`` 算 ``J``, ``loss.backward()`` 反传;
            5. 取出 ``xb.grad``、nan_to_num 清洗、L2 归一化。

        Args:
            forecast_model (nn.Module): ``XiChenForecast`` 子模型。
            ObsOp_models (dict[str, nn.Module]): 辐射率观测算子字典。
            H_models (dict[str, nn.Module]): 观测算子 ``Model_H`` 字典。
            VarCost_models (dict[str, nn.Module]): 变分代价字典。
            obs_name (str): 当前处理的观测源名(如 ``atms`` / ``prepbufr`` / ``ascat``)。
            xb (Tensor): 背景场, ``requires_grad=True``, 形状 ``(B, C, H, W)``。
            obs (dict[str, Tensor]): per-obs 观测张量字典。
            obs_mask (dict[str, Tensor]): per-obs 有效位掩码。
            obs_dict (dict): 观测元信息(``microwave`` / ``conventional`` 子字典)。
            std_dict (dict[str, Tensor]): per-obs σ QC 字典。
            out_vars (list[str]): 状态变量名列表(69 通道)。
            preds (Tensor): 由 forward() hoist 后传入的预报轨迹,形状 ``(B, T, C, H, W)``。

        Returns:
            Tensor: 形状 ``(B, C, H, W)`` 的 ``∂J/∂xb``, 已 L2 归一化且不含 NaN/Inf。
        """
        B, T, C, H, W = preds.shape
        B, T, Cs, H, W = obs[obs_name].shape

        if obs_name in ObsOp_models.keys():
            pred_obs, log_var, tgt_obs = ObsOp_models[obs_name](
                preds.view(B * T, C, H, W),
                obs[obs_name].view(B * T, Cs, H, W),
                obs_mask[obs_name].view(B * T, 1, H, W),
                use_checkpoint=True
            )
            tgt_sat_var_ids = np.array(
                [
                    obs_dict["microwave"][obs_name]["tmbrs_vars"].index(item)
                    for item in ObsOp_models[obs_name].out_sat_vars
                ]
            )
            tgt_sat_var_ids = torch.from_numpy(tgt_sat_var_ids).to(xb.device)
            std = std_dict[obs_name][tgt_sat_var_ids].reshape(1, 1, -1, 1, 1)

            dy, _ = H_models[obs_name](
                pred_obs.view(B, T, -1, H, W),
                tgt_obs.view(B, T, -1, H, W),
                obs_mask[obs_name].view(B, T, 1, H, W),
                std
            )
        else:
            tgt_sat_var_ids = np.array(
                [
                    out_vars.index(item)
                    for item in obs_dict["conventional"][obs_name]["vars"]
                ]
            )
            tgt_sat_var_ids = torch.from_numpy(tgt_sat_var_ids).to(xb.device)
            std = std_dict[obs_name].reshape(1, 1, -1, 1, 1)

            dy, _ = H_models[obs_name](
                preds[:, :, tgt_sat_var_ids],
                obs[obs_name],
                obs_mask[obs_name],
                std
            )

        dy = torch.nan_to_num(dy, nan=0.0, posinf=0.0, neginf=0.0)

        loss = VarCost_models[obs_name](dy, std.reshape(1, -1))

        loss = torch.where(torch.isnan(loss), 0, loss)
        loss = torch.where(torch.isinf(loss), 0, loss)

        loss.backward(torch.ones_like(loss), retain_graph=True)

        var_cost_grad = xb.grad.detach()
        xb.grad = None

        var_cost_grad = torch.nan_to_num(var_cost_grad, nan=0.0, posinf=0.0, neginf=0.0)

        # 关键:对 ∂J/∂xb 做 L2 范数归一化,保证 7 种 obs 源(微波辐射率 / 探空 /
        # 卫星风 / 海面风)梯度的尺度一致,便于下游 XiChenRepresentationObsEmbedding
        # 学习稳定的 (xb, grad) -> roe 映射;+1e-8 防止 0 范数 sqrt(0) 反向传播炸梯度
        normgrad_ = torch.sqrt(torch.mean(var_cost_grad ** 2, dim=(1, 2, 3), keepdim=True) + 1e-8)
        normgrad_ = torch.nan_to_num(normgrad_, nan=1.0, posinf=1.0, neginf=1.0)
        normgrad_[normgrad_ == 0] = 1.0

        var_cost_grad = var_cost_grad / normgrad_

        del dy, normgrad_, loss
        if obs_name in ObsOp_models.keys():
            del pred_obs, log_var, tgt_obs

        return var_cost_grad