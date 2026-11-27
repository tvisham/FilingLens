"""A thread-safe token-bucket rate limiter with an injectable clock (for tests)."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable


class TokenBucket:
    """Allows `rate` requests per second on average, with bursts up to `capacity`.

    `acquire()` blocks until a token is available. Clock and sleep are injectable so
    tests can run instantly and deterministically.
    """

    def __init__(
        self,
        rate: float,
        capacity: float | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if rate <= 0:
            raise ValueError("rate must be positive")
        self.rate = rate
        self.capacity = capacity if capacity is not None else rate
        if self.capacity < 1:
            raise ValueError("capacity must be at least 1")
        self._clock = clock
        self._sleep = sleep
        self._tokens = self.capacity
        self._last = clock()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = self._clock()
        elapsed = max(0.0, now - self._last)
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
        self._last = now

    def acquire(self) -> None:
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= 1:
                    self._tokens -= 1
                    return
                wait = (1 - self._tokens) / self.rate
            self._sleep(wait)
