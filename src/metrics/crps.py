"""
CRPS 评估指标（基于集合预报）

本模块提供评估阶段（非训练）的 CRPS 指标计算，仅依赖 NumPy。
输入是 ``forecast`` 与 ``truth`` 的集合预报数组（dim 0 是集合成员）。
输出的两个分量：

- ``crps_skill``：CRPS 的"技巧"项 = ``E[|truth - X|]``，集合均值。
- ``_pointwise_crps_spread``：CRPS 的"离散度"项（spread），
  = ``1/(M-1) * 2 * mean( (2*rank - M - 1) * X )``，
  其中 rank 通过排序得到，对应 Zamo & Naveau 的 eFAIR 公式。

完整的 CRPS = skill - 0.5 * spread；本模块分别暴露 skill 和 spread
两个分量，便于在 TensorBoard 中独立监控"准确度"和"离散度"。

上游依赖：无 PyTorch 训练依赖；可被 ``src/pipeline/*/trainer.py`` 在
验证阶段调用，也可在 ``inference/`` 评估脚本中独立使用。
下游调用：纯 NumPy 函数，``numpy.nanmean`` 自动跳过 NaN（如海洋点）。
"""

import numpy as np


def _get_n_ensemble(
    ds: np.ndarray,
    expect_n_ensemble_at_least: int = 1,
) -> int:
    """读取集合预报的大小并在不足时抛出 ``ValueError``。

    Args:
        ds: 集合数组，dim 0 是集合成员。
        expect_n_ensemble_at_least: 最少集合成员数；默认 ``1``。

    Returns:
        ``int``，集合成员数 ``M``。

    Raises:
        ValueError: 当 ``M < expect_n_ensemble_at_least`` 时。
    """
    n_ensemble = ds.shape[0]
    if n_ensemble < expect_n_ensemble_at_least:
        raise ValueError(
            f"{n_ensemble=} is less than expected size of "
            f"{expect_n_ensemble_at_least}"
        )
    return n_ensemble


def _pointwise_crps_spread(
    forecast: np.ndarray
) -> np.ndarray:
    """CRPS 的 spread（离散度）分量：``λ₂ = 1/(M(M-1)) Σ|X_i - X_j|``。

    算法来自 Zamo & Naveau（2018）：通过排序后用 rank 加权代替
    ``O(M²)`` 的两两差分绝对值求和，把复杂度降到 ``O(M log M)``。

    Args:
        forecast: shape ``(M, ...)`` 的集合预报。

    Returns:
        与 ``forecast[0]`` 同形的 spread 张量。
    """
    n_ensemble = _get_n_ensemble(forecast)

    # one_half_spread is ̂̂λ₂ from Zamo. That is, with n_ensemble = M,
    #   λ₂ = 1 / (2 M (M - 1)) Σ_{i,j=1}^M |Xi - Xj|
    # 通过排序+rank 加权替代 O(M²) 的双求和：rank 加权公式
    #   λ₂ = 1 / (M(M-1)) Σ (2*i - M - 1) X_i  (X 已排序)
    # (2*i - M - 1) 等价于 (Xi 大于的元素数) - (Xi 小于的元素数)。
    # 这里不去排序，而是计算每个元素的 rank、再乘 (2*rank - M - 1)；
    # 复杂度 O(M log M)，内存 O(M)。
    rank = _rank_ds(forecast)
    return (
        2
        * (
            np.nanmean((2 * rank - n_ensemble - 1) * forecast)
        )
        / (n_ensemble - 1)
  )


def _pointwise_crps_skill(
    forecast: np.ndarray, truth: np.ndarray
) -> np.ndarray:
    """CRPS 的 skill 分量：``E[|truth - X|]``，沿集合维求均值。

    Args:
        forecast: shape ``(M, ...)`` 的集合预报。
        truth: shape ``(...)`` 的真值。

    Returns:
        与 ``truth`` 同形的 skill 张量。
    """
    _get_n_ensemble(forecast)  # Will raise if no ensembles.
    return np.nanmean(np.abs(truth - forecast), axis=0)


def _rank_ds(ds: np.ndarray) -> np.ndarray:
    """沿 ``dim=0``（集合维）求每个成员的"秩"，1 表示最小。

    Args:
        ds: shape ``(M, ...)`` 的数组。

    Returns:
        与 ``ds`` 同形的 rank 数组（值在 ``[1, M]``）。
    """

    def _rank_arr(arr: np.ndarray) -> np.ndarray:
        return _rankdata(arr, axis=0)

    # Module docstring promises 仅依赖 NumPy — dispatch on type to honor that contract.
    if isinstance(ds, np.ndarray):
        return _rank_arr(ds)
    # Fallback for xarray DataArray / Dataset (best-effort, not contractual).
    if hasattr(ds, 'values') and hasattr(ds, 'dims'):
        return _rankdata(ds.values, axis=0)
    raise TypeError(f"_rank_ds expects np.ndarray or xarray DataArray, got {type(ds)}")


def _rankdata(x: np.ndarray, axis: int) -> np.ndarray:
    """``scipy.stats.rankdata`` 的简化版（ordinal，无平均处理）。

    Args:
        x: 输入数组。
        axis: 沿该轴求 rank。

    Returns:
        与 ``x`` 同形的 rank 数组。
    """
    x = np.asarray(x)
    x = np.swapaxes(x, axis, -1)
    j = np.argsort(x, axis=-1)
    ordinal_ranks = np.broadcast_to(
        np.arange(1, x.shape[-1] + 1, dtype=int), x.shape
    )
    ordered_ranks = np.empty(j.shape, dtype=ordinal_ranks.dtype)
    np.put_along_axis(ordered_ranks, j, ordinal_ranks, axis=-1)
    return np.swapaxes(ordered_ranks, axis, -1)


def crps_skill(
    forecast: np.ndarray, truth: np.ndarray
) -> np.ndarray:
    """对外暴露的 CRPS skill：``E[|truth - X|]``。

    Args:
        forecast: shape ``(M, ...)`` 的集合预报。
        truth: shape ``(...)`` 的真值。

    Returns:
        与 ``truth`` 同形的 skill 张量。
    """
    return _pointwise_crps_skill(forecast, truth)