"""
Hydra / OmegaConf 配置解析工具

提供 ``resolve_model_config``，递归向下查找包含 ``_target_`` 或
``_component_`` 字段的 DictConfig 子树，以便上游从嵌套 config 中拿到
真正可被 Hydra ``instantiate`` 的模型配置块。

上游依赖：``main.py`` 在解析完 config 后，业务代码若需要拿"嵌套很深的
``_target_`` 配置"会调用本函数。
下游调用：纯 OmegaConf 操作，无副作用。
"""

import hydra
from omegaconf import DictConfig


def resolve_model_config(cfg: DictConfig) -> DictConfig:
    """递归查找包含 ``_target_`` 或 ``_component_`` 的实际模型配置。

    在 DA / multimodal 等复杂 config 中，模型配置常常被多层嵌套，例如：

        cfg.model.backbone.submodule = DictConfig(...)
        cfg.model.head = DictConfig(包含 _target_)

    直接 ``hydra.utils.instantiate(cfg.model)`` 会失败；本函数定位到
    第一个含 ``_target_`` 的子树返回。

    Args:
        cfg: Hydra 解析后的 ``DictConfig``（可能多层嵌套）。

    Returns:
        含 ``_target_`` 或 ``_component_`` 的子 ``DictConfig``；若整棵
        树都没有则原样返回 ``cfg``（让 Hydra 自行抛出错误）。
    """
    if '_target_' in cfg or '_component_' in cfg:
        return cfg
    if isinstance(cfg, DictConfig):
        for v in cfg.values():
            if isinstance(v, DictConfig):
                resolved = resolve_model_config(v)
                # 只要找到包含 _target_ 的配置就返回
                if '_target_' in resolved or '_component_' in resolved:
                    return resolved
    # 未找到则返回原配置，Hydra 会抛出明确的错误提示
    return cfg