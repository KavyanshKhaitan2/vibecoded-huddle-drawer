"""Token bucket matching Chime's per-meeting data-message limit.

Measured in ``docs/huddles-peer-messages.md#throughput``: ~200 capacity
refilling at ~150/s, shared by every sender in the meeting, counting *payloads*
rather than bytes or frames. Excess is dropped silently -- receivers never see a
``throttled`` flag -- and sustained 800+/s gets the sender disconnected (1006).

So the client has to police itself: there is no feedback channel to react to.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from .config import CHIME_DATA_MESSAGE_BURST, CHIME_DATA_MESSAGE_RATE


@dataclass
class TokenBucket:
    """Asyncio token bucket.

    ``rate`` tokens are added per second up to ``capacity``. Defaults are
    deliberately a little under Chime's measured limits so keep-alives and
    bursts have headroom, per the doc's advice to "leave headroom".
    """

    rate: float = CHIME_DATA_MESSAGE_RATE * 0.8
    capacity: float = CHIME_DATA_MESSAGE_BURST * 0.75
    _tokens: float = field(init=False, default=0.0)
    _updated: float = field(init=False, default=0.0)
    _lock: asyncio.Lock = field(init=False, default_factory=asyncio.Lock)

    def __post_init__(self) -> None:
        if self.rate <= 0:
            raise ValueError("rate must be positive")
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        self._tokens = self.capacity
        self._updated = time.monotonic()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._updated
        if elapsed > 0:
            self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
            self._updated = now

    @property
    def available(self) -> float:
        """Tokens available right now (refills first)."""
        self._refill()
        return self._tokens

    def try_acquire(self, tokens: float = 1.0) -> bool:
        """Take ``tokens`` if available. Never blocks."""
        self._refill()
        if self._tokens >= tokens:
            self._tokens -= tokens
            return True
        return False

    async def acquire(self, tokens: float = 1.0) -> float:
        """Wait until ``tokens`` are available, then take them.

        Returns the time spent waiting, in seconds.
        """
        if tokens > self.capacity:
            raise ValueError(
                f"cannot acquire {tokens} tokens from a bucket of capacity {self.capacity}"
            )
        waited = 0.0
        async with self._lock:
            while True:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return waited
                deficit = tokens - self._tokens
                delay = deficit / self.rate
                waited += delay
                await asyncio.sleep(delay)
