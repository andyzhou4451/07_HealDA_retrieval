# -*- coding: utf-8 -*-
"""
统一日志系统 - 轻量级实现
- 控制台输出 (DDP主进程过滤)
- 文件输出 (可选)
- Hydra 集成

与 ``src.utils.__init__`` 中同名函数语义完全一致；本文件是
"可单独 import"的入口，避免业务代码依赖 ``src.utils`` 的副作用。

上游依赖：``main.py`` 通过 ``setup_logger`` 初始化项目根 logger，
并把 ``log_file`` 写到 ``logs/tensorboard/<task_name>.log``。
下游调用：被 ``src/pipeline/*/trainer.py`` 与 ``src/utils/tqdm_logger.py``
配合使用——``tqdm`` 通过 ``patch_tqdm_for_logger`` 把进度条重定向到本 logger。
"""
import logging
import os
import sys
from pathlib import Path
from typing import Optional


class RankFilter(logging.Filter):
    """日志过滤器：只允许指定 rank 的日志通过。

    DDP 多卡训练时挂到 console / file handler 上，仅 ``rank == 0`` 的
    进程会输出；其他进程直接被过滤，避免日志重复刷屏。
    """

    def __init__(self, rank: int = 0):
        super().__init__()
        self.rank = rank

    def filter(self, record) -> bool:
        """仅当 ``self.rank == 0`` 时返回 ``True``。"""
        return self.rank == 0


def setup_logger(
    name: str = "xichen",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    rank: int = 0,
    format_str: str = "%(asctime)s | %(levelname)-8s | %(message)s",
    date_format: str = "%H:%M:%S",
) -> logging.Logger:
    """统一 logger 初始化。

    与 ``src.utils.__init__.setup_logger`` 的差异：多暴露 ``format_str``
    与 ``date_format`` 两个参数，便于日志样式自定义（默认时间精度只到秒）。

    Args:
        name: logger 名称；默认 ``"xichen"``。
        log_file: 日志文件路径；可选，为 ``None`` 时仅输出到控制台。
        level: 日志级别；默认 ``logging.INFO``。
        rank: 当前进程 rank；非 0 的进程输出会被过滤。
        format_str: 日志格式字符串；默认 ``"时间 | 等级 | 消息"``。
        date_format: 日期格式；默认 ``"%H:%M:%S"``（仅到秒）。

    Returns:
        配置好的 ``logging.Logger``。
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # 清空历史 handler，避免重复 setup 时重复打印
    logger.handlers.clear()
    # 关闭向上传播，避免根 logger 重复输出
    logger.propagate = False

    formatter = logging.Formatter(format_str, datefmt=date_format)

    # 控制台 handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    console.addFilter(RankFilter(rank))
    logger.addHandler(console)

    # 文件 handler（仅主进程；RankFilter 兜底过滤）
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(RankFilter(rank))
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "xichen") -> logging.Logger:
    """获取已配置的 logger；不会重复初始化。"""
    return logging.getLogger(name)