from __future__ import annotations

import datetime as dt
import time
from typing import Any

from ..transcription import RateLimitEvent


def backoff_delay(attempt: int) -> float:
    return min(2**attempt, 30)


def rate_limit_delay(response: Any, attempt: int) -> float:
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return max(float(retry_after), 1.0)
            except ValueError:
                pass

        reset = response.headers.get("X-RateLimit-Reset")
        if reset:
            try:
                reset_at = float(reset)
                return max(reset_at - time.time(), 1.0)
            except ValueError:
                pass

    return backoff_delay(attempt)


def rate_limit_event(endpoint: str, delay: float, response: Any) -> RateLimitEvent:
    headers = response.headers if response is not None else {}
    limit = _safe_int(headers.get("X-RateLimit-Limit"))
    remaining = _safe_int(headers.get("X-RateLimit-Remaining"))

    reset_dt: dt.datetime | None = None
    reset = headers.get("X-RateLimit-Reset")
    if reset:
        try:
            reset_dt = dt.datetime.fromtimestamp(float(reset), tz=dt.timezone.utc)
        except ValueError:
            reset_dt = None

    return RateLimitEvent(
        endpoint=endpoint,
        delay=delay,
        limit=limit,
        remaining=remaining,
        reset_at=reset_dt,
    )


def _safe_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
