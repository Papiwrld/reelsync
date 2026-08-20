"""Central request manager: per-provider throttling and cooldowns.

Every external call in the research layer passes through here. Providers
declare a minimum interval between requests and a concurrency cap that
reflect their published usage policies (arXiv: 1 per 3s single connection,
Nominatim: 1 per second single thread, Wikimedia: max 3 concurrent, ...).
HTTP 429 responses push a provider cooldown so subsequent calls space out
beyond the normal interval.
"""

from __future__ import annotations

import threading
import time
from typing import Dict, Optional


class ThrottleToken:
    """Holds a concurrency slot; must be released after the request."""

    def __init__(self, throttle: "ProviderThrottle") -> None:
        self._throttle = throttle
        self._released = False

    def release(self) -> None:
        if not self._released:
            self._released = True
            self._throttle._release_slot()


class ProviderThrottle:
    """Minimum-interval + max-concurrency gate for one provider."""

    def __init__(
        self,
        name: str,
        interval: float = 0.0,
        max_concurrent: int = 1,
        cooldown_factor: float = 4.0,
    ) -> None:
        self.name = name
        self.interval = max(0.0, float(interval))
        self.max_concurrent = max(1, int(max_concurrent))
        self.cooldown_factor = max(1.0, float(cooldown_factor))
        self._lock = threading.Lock()
        self._last_request_at: float = 0.0
        self._cooldown_until: float = 0.0
        self._semaphore = threading.Semaphore(self.max_concurrent)

    def wait_until_ready(self) -> None:
        """Block until the provider is ready for one request."""
        with self._lock:
            now = time.time()
            target = max(self._last_request_at + self.interval, self._cooldown_until)
            delay = target - now
            self._last_request_at = max(now, target)
        if delay > 0:
            time.sleep(delay)
        self._semaphore.acquire()

    def _release_slot(self) -> None:
        self._semaphore.release()

    def mark_rate_limited(self, retry_after: float = 0.0) -> None:
        """Extend the cooldown window after an HTTP 429."""
        with self._lock:
            now = time.time()
            delay = max(self.interval * self.cooldown_factor, float(retry_after or 0))
            self._cooldown_until = now + delay


class RequestManager:
    """Registry of per-provider throttles."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._throttles: Dict[str, ProviderThrottle] = {}
        self._events: Dict[str, int] = {}
        self._concurrent: Dict[str, int] = {}

    def register(self, name: str, interval: float = 0.0, max_concurrent: int = 1) -> ProviderThrottle:
        with self._lock:
            throttle = self._throttles.get(name)
            if throttle is None:
                throttle = ProviderThrottle(name, interval, max_concurrent)
                self._throttles[name] = throttle
            return throttle

    def acquire(self, name: str) -> Optional[ThrottleToken]:
        """Wait for readiness and reserve a concurrency slot."""
        throttle = self.register(name)
        throttle.wait_until_ready()
        with self._lock:
            self._concurrent[name] = self._concurrent.get(name, 0) + 1
            self._events[name] = self._events.get(name, 0) + 1
        return ThrottleToken(throttle)

    def release(self, name: str, token: Optional[ThrottleToken]) -> None:
        if token is not None:
            token.release()
        with self._lock:
            self._concurrent[name] = max(0, self._concurrent.get(name, 0) - 1)

    def mark_rate_limited(self, name: str, retry_after: float = 0.0) -> None:
        throttle = self.register(name)
        throttle.mark_rate_limited(retry_after)

    def snapshot(self) -> Dict[str, Dict]:
        with self._lock:
            return {
                name: {
                    "interval": throttle.interval,
                    "max_concurrent": throttle.max_concurrent,
                    "requests": self._events.get(name, 0),
                    "concurrent_now": self._concurrent.get(name, 0),
                }
                for name, throttle in self._throttles.items()
            }