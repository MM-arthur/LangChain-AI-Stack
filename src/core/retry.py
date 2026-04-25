# Retry utilities for LLM calls - handles transient API failures

import os
import time
import logging
from functools import wraps
from typing import Callable, Any

logger = logging.getLogger(__name__)

# Default retry config - can be overridden via env
DEFAULT_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
DEFAULT_BASE_DELAY = float(os.getenv("LLM_RETRY_BASE_DELAY", "2.0"))
DEFAULT_MAX_DELAY = float(os.getenv("LLM_RETRY_MAX_DELAY", "30.0"))


def with_retry(max_retries: int = DEFAULT_MAX_RETRIES,
               base_delay: float = DEFAULT_BASE_DELAY,
               max_delay: float = DEFAULT_MAX_DELAY):
    """
    Decorator that retries a function on exception with exponential backoff.

    Retries on:
        - API errors (rate limit, timeout, server error)
        - Network errors
        - Any Exception from LLM call

    Does NOT retry on:
        - Validation errors (invalid request, auth failure)
        - User-provided invalid input

    Usage:
        @with_retry(max_retries=3)
        def call_llm(llm, messages):
            return llm.invoke(messages)
    """
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    error_type = type(e).__name__

                    # Non-retryable: validation/auth errors
                    if error_type in ("ValidationError", "AuthenticationError",
                                      "InvalidRequestError", "NotFoundError"):
                        logger.warning(
                            f"[with_retry] Non-retryable error ({error_type}), "
                            f"giving up after {attempt} attempts: {e}"
                        )
                        raise

                    # Retryable
                    if attempt < max_retries:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        logger.warning(
                            f"[with_retry] Attempt {attempt + 1} failed ({error_type}): {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"[with_retry] All {max_retries + 1} attempts failed. "
                            f"Last error: {e}"
                        )

            # All retries exhausted
            raise last_exception

        return wrapper
    return decorator


def is_retryable_error(exception: Exception) -> bool:
    """
    Classify whether an exception is retryable.
    Used for inline retry logic without the decorator.
    """
    error_type = type(exception).__name__
    non_retryable = {
        "ValidationError", "AuthenticationError", "InvalidRequestError",
        "NotFoundError", "PermissionError", "ValueError", "TypeError"
    }
    return error_type not in non_retryable