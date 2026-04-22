# Logger setup utility
import logging
import sys
from typing import Optional
from pathlib import Path
from loguru import logger as loguru_logger

from config.settings import LOG_LEVEL, LOG_FILE, LOGS_DIR


class InterceptHandler(logging.Handler):
    """Intercept standard logging messages toward Loguru."""

    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = loguru_logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        loguru_logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logger(name: Optional[str] = None, level: str = LOG_LEVEL) -> loguru_logger:
    """Setup logger with consistent configuration.

    Args:
        name: Logger name. If None, returns the root logger.
        level: Logging level.

    Returns:
        Configured logger instance.
    """
    # Remove default handler
    loguru_logger.remove()

    # Console handler
    loguru_logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True,
    )

    # File handler
    log_file_path = Path(LOG_FILE)
    if not log_file_path.is_absolute():
        log_file_path = LOGS_DIR / log_file_path

    # Ensure log directory exists
    log_file_path.parent.mkdir(exist_ok=True)

    loguru_logger.add(
        str(log_file_path),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=level,
        rotation="10 MB",
        retention="30 days",
        compression="zip",
    )

    # Intercept standard logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    if name:
        return loguru_logger.bind(name=name)
    return loguru_logger


# Create default logger
logger = setup_logger()