"""Reusable decorators for the dailynews plugin.

Before:
    try:
        from astrbot.api import logger as astrbot_logger
    except Exception:
        import logging
        astrbot_logger = logging.getLogger(__name__)

    async def fetch():
        try:
            return await do_something()
        except Exception as e:
            astrbot_logger.warning("[dailynews] fetch failed: %s", e, exc_info=True)
            return None

After:
    from ...core.utils import get_logger
    from ...core.decorators import safe_async_call

    astrbot_logger = get_logger(__name__)

    @safe_async_call(logger=astrbot_logger, log_msg="fetch failed")
    async def fetch():
        return await do_something()
"""

from __future__ import annotations

import asyncio
import functools
import time
from typing import Any, Awaitable, Callable, TypeVar

try:
    from astrbot.api import logger as _default_logger
except Exception:  # pragma: no cover
    import logging

    _default_logger = logging.getLogger(__name__)

T = TypeVar("T")


def get_logger(name: str) -> Any:
    """Get a logger that works both inside AstrBot and in standalone tests."""
    _ = name
    try:
        from astrbot.api import logger as astrbot_logger

        # AstrBot attaches record-enrichment filters to its exported logger.
        # Child loggers can bypass those filters while still hitting AstrBot's
        # queue handler, which makes formatters that require plugin_tag fail.
        return astrbot_logger
    except Exception:
        import logging

        return logging.getLogger(name)


# ---------------------------------------------------------------------------
# safe_async_call / safe_sync_call
# ---------------------------------------------------------------------------

def safe_async_call(
    *,
    logger: Any = _default_logger,
    log_msg: str = "",
    log_prefix: str = "[dailynews]",
    default_return: T | None = None,
    level: str = "warning",
    exc_info: bool = True,
    reraise: bool = False,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T | None]]]:
    """Decorator that catches *all* exceptions in an async function, logs them,
    and returns *default_return* instead of propagating.

    Parameters
    ----------
    logger:
        Logger instance (defaults to astrbot.api.logger).
    log_msg:
        Human-readable message fragment, e.g. ``"fetch articles failed"``.
    log_prefix:
        Prefix for the log line, e.g. ``"[dailynews]"``.
    default_return:
        Value to return when the decorated function raises.
    level:
        Log level name: ``"debug"``, ``"info"``, ``"warning"``, ``"error"``.
    exc_info:
        Passed to the logger call.
    reraise:
        If ``True``, still re-raises the exception after logging (useful when you
        want the trace but also want the caller to handle it).
    """

    def decorator(
        func: Callable[..., Awaitable[T]],
    ) -> Callable[..., Awaitable[T | None]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T | None:
            try:
                return await func(*args, **kwargs)
            except Exception as e:  # noqa: BLE001
                msg = f"{log_prefix} {log_msg}: {e}" if log_msg else f"{log_prefix} {func.__qualname__} failed: {e}"
                log_fn = getattr(logger, level, logger.warning)
                log_fn(msg, exc_info=exc_info)
                if reraise:
                    raise
                return default_return

        return wrapper

    return decorator


def safe_sync_call(
    *,
    logger: Any = _default_logger,
    log_msg: str = "",
    log_prefix: str = "[dailynews]",
    default_return: T | None = None,
    level: str = "warning",
    exc_info: bool = True,
    reraise: bool = False,
) -> Callable[[Callable[..., T]], Callable[..., T | None]]:
    """Sync version of :func:`safe_async_call`."""

    def decorator(func: Callable[..., T]) -> Callable[..., T | None]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T | None:
            try:
                return func(*args, **kwargs)
            except Exception as e:  # noqa: BLE001
                msg = f"{log_prefix} {log_msg}: {e}" if log_msg else f"{log_prefix} {func.__qualname__} failed: {e}"
                log_fn = getattr(logger, level, logger.warning)
                log_fn(msg, exc_info=exc_info)
                if reraise:
                    raise
                return default_return

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# timed
# ---------------------------------------------------------------------------

def timed(
    *,
    logger: Any = _default_logger,
    log_prefix: str = "[dailynews]",
    label: str = "",
    level: str = "info",
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Log the execution time (ms) of an async function."""

    def decorator(
        func: Callable[..., Awaitable[T]],
    ) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            start = time.monotonic()
            try:
                return await func(*args, **kwargs)
            finally:
                elapsed_ms = int((time.monotonic() - start) * 1000)
                name = label or func.__qualname__
                msg = f"{log_prefix} {name} cost_ms={elapsed_ms}"
                log_fn = getattr(logger, level, logger.info)
                log_fn(msg)

        return wrapper

    return decorator


def timed_sync(
    *,
    logger: Any = _default_logger,
    log_prefix: str = "[dailynews]",
    label: str = "",
    level: str = "info",
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Log the execution time (ms) of a sync function."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            start = time.monotonic()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed_ms = int((time.monotonic() - start) * 1000)
                name = label or func.__qualname__
                msg = f"{log_prefix} {name} cost_ms={elapsed_ms}"
                log_fn = getattr(logger, level, logger.info)
                log_fn(msg)

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# retry_async
# ---------------------------------------------------------------------------

def retry_async(
    *,
    max_retries: int = 1,
    base_delay_s: float = 2.0,
    max_delay_s: float = 6.0,
    logger: Any = _default_logger,
    log_prefix: str = "[dailynews]",
    log_msg: str = "",
    retry_on: tuple[type[BaseException], ...] = (Exception,),
    on_retry: Callable[[int, BaseException], None] | None = None,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Retry an async function with exponential backoff.

    Parameters
    ----------
    max_retries:
        Number of *extra* attempts after the first failure.
        Total attempts = ``1 + max_retries``.
    base_delay_s:
        Initial delay between retries.
    max_delay_s:
        Cap on the delay.
    retry_on:
        Tuple of exception types that trigger a retry.
    on_retry:
        Optional callback ``fn(attempt_index, exception)`` invoked on each retry.
    """

    def decorator(
        func: Callable[..., Awaitable[T]],
    ) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exc: BaseException | None = None
            for attempt in range(1, max_retries + 2):
                try:
                    return await func(*args, **kwargs)
                except retry_on as e:
                    last_exc = e
                    if attempt < max_retries + 1:
                        delay = min(base_delay_s * attempt, max_delay_s)
                        msg = (
                            f"{log_prefix} {log_msg} retry {attempt}/{max_retries + 1}: {e}"
                            if log_msg
                            else f"{log_prefix} {func.__qualname__} retry {attempt}/{max_retries + 1}: {e}"
                        )
                        logger.warning(msg)
                        if on_retry:
                            on_retry(attempt, e)
                        await asyncio.sleep(delay)
                    else:
                        raise
            # Should never reach here because the last iteration either returns or raises.
            raise RuntimeError(
                f"retry_async exhausted all {max_retries + 1} attempts: "
                f"{type(last_exc).__name__ if last_exc else 'UnknownError'}"
            )

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# timeout
# ---------------------------------------------------------------------------

def timeout(
    seconds: float,
    *,
    logger: Any = _default_logger,
    log_prefix: str = "[dailynews]",
    log_msg: str = "",
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Wrap an async function with ``asyncio.wait_for``.

    If the timeout is exceeded the function raises ``asyncio.TimeoutError``.
    """

    def decorator(
        func: Callable[..., Awaitable[T]],
    ) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            coro = func(*args, **kwargs)
            try:
                return await asyncio.wait_for(coro, timeout=seconds)
            except asyncio.TimeoutError:
                msg = (
                    f"{log_prefix} {log_msg} timeout after {seconds}s"
                    if log_msg
                    else f"{log_prefix} {func.__qualname__} timeout after {seconds}s"
                )
                logger.warning(msg)
                raise

        return wrapper

    return decorator
