"""Per-job research budget enforcement.

The budget caps external requests per research run. Once exhausted, providers
still return cached results but never hit the network; blocked attempts are
counted in metrics as avoided requests.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional

DEFAULT_MAX_EXTERNAL_REQUESTS = 20
DEFAULT_MAX_REQUESTS_PER_PROVIDER = 5


class BudgetContext:
    """Thread-safe per-job budget ledger."""

    def __init__(
        self,
        max_external_requests: int = DEFAULT_MAX_EXTERNAL_REQUESTS,
        max_requests_per_provider: int = DEFAULT_MAX_REQUESTS_PER_PROVIDER,
        cache_enabled: bool = True,
        deduplication_enabled: bool = True,
        batching_enabled: bool = True,
    ) -> None:
        self.max_external_requests = int(max_external_requests)
        self.max_requests_per_provider = int(max_requests_per_provider)
        self.cache_enabled = bool(cache_enabled)
        self.deduplication_enabled = bool(deduplication_enabled)
        self.batching_enabled = bool(batching_enabled)
        self._lock = threading.Lock()
        self._used_total = 0
        self._used_per_provider: Dict[str, int] = {}
        self._blocked = 0

    @classmethod
    def from_settings(cls, settings: Optional[Dict[str, Any]] = None) -> "BudgetContext":
        settings = settings or {}
        return cls(
            max_external_requests=int(
                settings.get("max_external_requests", DEFAULT_MAX_EXTERNAL_REQUESTS)
            ),
            max_requests_per_provider=int(
                settings.get(
                    "max_requests_per_provider", DEFAULT_MAX_REQUESTS_PER_PROVIDER
                )
            ),
            cache_enabled=bool(settings.get("cache_enabled", True)),
            deduplication_enabled=bool(settings.get("deduplication_enabled", True)),
            batching_enabled=bool(settings.get("batching_enabled", True)),
        )

    def acquire(self, provider: str) -> bool:
        """Reserve one external request for ``provider``; False when over budget."""
        with self._lock:
            if self._used_total >= self.max_external_requests:
                self._blocked += 1
                return False
            if self._used_per_provider.get(provider, 0) >= self.max_requests_per_provider:
                self._blocked += 1
                return False
            self._used_total += 1
            self._used_per_provider[provider] = (
                self._used_per_provider.get(provider, 0) + 1
            )
            return True

    def remaining_total(self) -> int:
        with self._lock:
            return max(0, self.max_external_requests - self._used_total)

    def remaining_for(self, provider: str) -> int:
        with self._lock:
            return max(
                0,
                self.max_requests_per_provider
                - self._used_per_provider.get(provider, 0),
            )

    def blocked_count(self) -> int:
        with self._lock:
            return self._blocked

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "max_external_requests": self.max_external_requests,
                "max_requests_per_provider": self.max_requests_per_provider,
                "used_total": self._used_total,
                "used_per_provider": dict(self._used_per_provider),
                "blocked": self._blocked,
                "remaining_total": max(
                    0, self.max_external_requests - self._used_total
                ),
            }