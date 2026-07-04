"""
tqdm 与 logger 集成（monkey patch）

训练循环里 ``tqdm(range(N))`` 默认把进度写到 ``sys.stderr``，
与项目统一 logger 容易割裂；本模块把 ``tqdm.__init__`` 替换成
"未指定 ``file`` 时注入一个 ``TqdmLoggerRedirect``"的版本，从而把
进度条输出重定向到给定 logger。

设计要点：

- 用 ``functools.wraps`` 保留原 ``tqdm.__init__`` 的签名 / docstring，
  不破坏 tqdm 的其余行为。
- ``TqdmLoggerRedirect.write`` 跳过空行 / 单换行，避免进度条回车把日志刷花。
- 仅在 ``kwargs['file']`` 未指定时生效；用户显式传入 ``file`` 时
  不做重定向。

上游依赖：被 ``main.py`` 在 ``setup_logger`` 之后立即调用
``patch_tqdm_for_logger``；后续所有 trainer / dataloader 的 tqdm
都会自动走 logger。
下游调用：通过修改 ``tqdm.__init__`` 全局生效，无显式调用方。
"""

from tqdm import tqdm
import logging
from functools import wraps


def patch_tqdm_for_logger(logger: logging.Logger, level: int = logging.INFO):
    """Monkey-patch ``tqdm`` 使其输出到 ``logger``。

    Args:
        logger: 目标 logger；通常就是 ``setup_logger`` 返回的实例。
        level: 进度条消息的日志级别；默认 ``logging.INFO``。

    Note:
        - 该函数会原地修改 ``tqdm.__init__``；如需还原，调用方需保留
          原 ``__init__`` 引用。
        - 因为是 monkey patch，建议在 ``main.py`` 入口处只调用一次。
    """
    original_init = tqdm.__init__

    @wraps(original_init)
    def new_init(self, *args, **kwargs):
        # 创建一个文件类对象，把 tqdm 的输出全部转给 logger.log
        class TqdmLoggerRedirect:
            def write(self, msg):
                if msg and msg != '\n':
                    msg = msg.rstrip()
                    if msg:
                        logger.log(level, msg)

            def flush(self):
                pass

        # 如果没有指定 file 参数，使用重定向对象替换默认 stderr
        if 'file' not in kwargs:
            kwargs['file'] = TqdmLoggerRedirect()

        return original_init(self, *args, **kwargs)

    tqdm.__init__ = new_init


# 使用示例
def example_with_patch():
    """示例：在 ``main`` 入口把 tqdm 输出重定向到 logger。

    典型用法：

        from src.utils.logger import setup_logger
        from src.utils.tqdm_logger import patch_tqdm_for_logger

        logger = setup_logger(name="xichen", log_file="logs/train.log", rank=0)
        patch_tqdm_for_logger(logger)

        # 之后所有 tqdm(...) 都会写到 logger
        for i in tqdm(range(100), desc="Training"):
            ...
    """
    logger = setup_logger()  # 使用你原来的 setup_logger

    # 应用 patch
    patch_tqdm_for_logger(logger)

    # 现在所有 tqdm 输出都会重定向到 logger
    for i in tqdm(range(100), desc="Training"):
        # 正常处理逻辑
        if i % 20 == 0:
            logger.info(f"Epoch {i} completed")