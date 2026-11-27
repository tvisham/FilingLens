"""Retry policy for SEC requests: exponential backoff with full jitter."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 5
    base_delay: float = 0.5
    max_delay: float = 30.0
    rng: random.Random = field(default_factory=random.Random, compare=False)

    def should_retry_status(self, status: int) -> bool:
        return status in RETRYABLE_STATUS

    def delay(self, attempt: int, retry_after: str | None = None) -> float:
        """Seconds to wait before retry number `attempt` (1-based).

        A numeric `Retry-After` header wins (capped at `max_delay`); otherwise full jitter
        on an exponential schedule.
        """
        if retry_after is not None:
            try:
                return min(self.max_delay, max(0.0, float(retry_after)))
            except ValueError:
                pass  # HTTP-date form: fall back to backoff
        ceiling = min(self.max_delay, self.base_delay * (2 ** (attempt - 1)))
        return self.rng.uniform(0, ceiling)
