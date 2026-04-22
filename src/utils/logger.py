# 日志记录器设置工具
import logging
import sys
from typing import Optional
from pathlib import Path
from loguru import logger as loguru_logger

from config.settings import LOG_LEVEL, LOG_FILE, LOGS_DIR


class InterceptHandler(logging.Handler):
    """将标准日志消息拦截并重定向到Loguru。"""

    def emit(self, record):
        # 如果存在，获取对应的Loguru级别
        try:
            level = loguru_logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # 查找日志消息来源的调用者
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        loguru_logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logger(name: Optional[str] = None, level: str = LOG_LEVEL) -> loguru_logger:
    """使用一致的配置设置日志记录器。

    Args:
        name: 日志记录器名称。如果为 None，则返回根日志记录器。
        level: 日志级别。

    Returns:
        配置好的日志记录器实例。
    """
    # 移除默认处理器
    loguru_logger.remove()

    # 控制台处理器
    loguru_logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True,
    )

    # 文件处理器
    log_file_path = Path(LOG_FILE)
    if not log_file_path.is_absolute():
        log_file_path = LOGS_DIR / log_file_path

    # 确保日志目录存在
    log_file_path.parent.mkdir(exist_ok=True)

    loguru_logger.add(
        str(log_file_path),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=level,
        rotation="10 MB",
        retention="30 days",
        compression="zip",
    )

    # 拦截标准日志
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    if name:
        return loguru_logger.bind(name=name)
    return loguru_logger


# 创建默认日志记录器
logger = setup_logger()