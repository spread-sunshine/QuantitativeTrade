# 重试工具：用于处理瞬时故障
import time
import logging
from typing import Callable, Any, Optional, Union
from functools import wraps

logger = logging.getLogger(__name__)


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Union[type[Exception], tuple[type[Exception], ...]] = Exception,
    logger: Optional[logging.Logger] = None,
):
    """异常时重试函数的装饰器。

    Args:
        max_attempts: 最大尝试次数（包括第一次）。
        delay: 尝试之间的初始延迟（秒）。
        backoff: 每次尝试后延迟的乘数。
        exceptions: 要捕获并重试的异常。
        logger: 用于记录重试的日志记录器实例。

    Returns:
        在指定异常时重试的装饰函数。
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            current_delay = delay
            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    if attempt > 1:
                        logger.info(
                            f"Retry attempt {attempt}/{max_attempts} for {func.__name__}"
                        )
                    return func(*args, **kwargs)

                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts:
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}: {e}"
                        )
                        raise

                    logger.warning(
                        f"Attempt {attempt}/{max_attempts} failed for {func.__name__}: {e}. "
                        f"Retrying in {current_delay:.1f}s..."
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff

            # 由于循环中的raise，这应该永远不会到达
            raise last_exception  # type: ignore

        return wrapper

    return decorator


class RetryConfig:
    """重试行为的配置。"""

    def __init__(
        self,
        max_attempts: int = 3,
        delay: float = 1.0,
        backoff: float = 2.0,
        exceptions: tuple[type[Exception], ...] = (Exception,),
    ):
        self.max_attempts = max_attempts
        self.delay = delay
        self.backoff = backoff
        self.exceptions = exceptions


def retry_with_config(config: RetryConfig):
    """使用配置对象创建重试装饰器。

    Args:
        config: RetryConfig实例。

    Returns:
        重试装饰器。
    """
    return retry(
        max_attempts=config.max_attempts,
        delay=config.delay,
        backoff=config.backoff,
        exceptions=config.exceptions,
    )


# 默认重试配置
DEFAULT_RETRY = RetryConfig(max_attempts=3, delay=1.0, backoff=2.0)
NETWORK_RETRY = RetryConfig(
    max_attempts=5, delay=2.0, backoff=2.0, exceptions=(ConnectionError, TimeoutError)
)
API_RETRY = RetryConfig(
    max_attempts=3, delay=3.0, backoff=1.5, exceptions=(RuntimeError, ValueError)
)