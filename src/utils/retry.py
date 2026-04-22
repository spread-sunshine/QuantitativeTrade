# Retry utilities for handling transient failures
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
    """Decorator for retrying a function on exception.

    Args:
        max_attempts: Maximum number of attempts (including first).
        delay: Initial delay between attempts in seconds.
        backoff: Multiplier for delay after each attempt.
        exceptions: Exception(s) to catch and retry on.
        logger: Logger instance for logging retries.

    Returns:
        Decorated function that retries on specified exceptions.
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

            # This should never be reached due to raise in loop
            raise last_exception  # type: ignore

        return wrapper

    return decorator


class RetryConfig:
    """Configuration for retry behavior."""

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
    """Create retry decorator with configuration object.

    Args:
        config: RetryConfig instance.

    Returns:
        Retry decorator.
    """
    return retry(
        max_attempts=config.max_attempts,
        delay=config.delay,
        backoff=config.backoff,
        exceptions=config.exceptions,
    )


# Default retry configurations
DEFAULT_RETRY = RetryConfig(max_attempts=3, delay=1.0, backoff=2.0)
NETWORK_RETRY = RetryConfig(
    max_attempts=5, delay=2.0, backoff=2.0, exceptions=(ConnectionError, TimeoutError)
)
API_RETRY = RetryConfig(
    max_attempts=3, delay=3.0, backoff=1.5, exceptions=(RuntimeError, ValueError)
)