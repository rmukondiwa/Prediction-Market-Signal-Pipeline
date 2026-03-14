import asyncio
import random
from typing import Callable, Awaitable, TypeVar

from src.utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


async def retry_with_backoff(
    coro_fn: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = 10,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True,
    label: str = "operation",
) -> T:
    """
    Retry an async callable with exponential backoff.

    Args:
        coro_fn:      Zero-argument async callable to attempt.
        max_attempts: Maximum number of attempts before re-raising.
        base_delay:   Initial delay in seconds.
        max_delay:    Cap on computed delay.
        jitter:       If True, adds ±25 % random jitter to each delay.
        label:        Human-readable name used in log messages.

    Returns:
        The return value of a successful coro_fn() call.

    Raises:
        The last exception raised by coro_fn after all attempts are exhausted.
    """
    attempt = 0
    last_exc: Exception | None = None

    while attempt < max_attempts:
        try:
            return await coro_fn()
        except Exception as exc:
            attempt += 1
            last_exc = exc

            if attempt >= max_attempts:
                logger.error(
                    f"{label} failed after {attempt} attempts",
                    extra={"attempt": attempt, "error": str(exc)},
                )
                raise

            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            if jitter:
                delay *= 0.75 + random.random() * 0.5  # ±25 %

            logger.warning(
                f"{label} attempt {attempt} failed, retrying in {delay:.1f}s",
                extra={"attempt": attempt, "delay": round(delay, 2), "error": str(exc)},
            )
            await asyncio.sleep(delay)

    raise RuntimeError("unreachable")  # pragma: no cover
