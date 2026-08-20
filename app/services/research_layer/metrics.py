"""Research metrics: thread-safe counters exposed via ``snapshot()``.

The headline figure is ``requests_avoided``: external API calls prevented by
caching, deduplication, request coalescing, batching and the research budget.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict

_METRIC_KEYS = (
    "total_requests",
    "cache_hits",
    "cache_misses",
    "coalesced_requests",
    "duplicate_queries_prevented",
    "batch_requests",
    "batched_items_saved",
    "budget_blocked",
    "failed_requests",
    "rate_limit_events",
    "retry_after_seconds",
    "total_response_time",
)


class ResearchMetrics:
    """Process-level research observability counters."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: Dict[str, float] = {key: 0 for key in _METRIC_KEYS}
        self._requests_by_provider: Dict[str, int] = {}
        self._started_at: float = time.time()
        self._last_reset: float = time.time()

    def increment(self, key: str, amount: float = 1.0) -> None:
        with self._lock:
            if key not in self._counters:
                self._counters[key] = 0.0
            self._counters[key] += amount

    def record_provider_request(self, provider: str) -> None:
        with self._lock:
            self._requests_by_provider[provider] = (
                self._requests_by_provider.get(provider, 0) + 1
            )
            self._counters["total_requests"] += 1

    def record_response_time(self, seconds: float) -> None:
        with self._lock:
            self._counters["total_response_time"] += seconds

    def record_rate_limit(self, retry_after: float = 0.0) -> None:
        with self._lock:
            self._counters["rate_limit_events"] += 1
            self._counters["retry_after_seconds"] += retry_after

    def requests_avoided(self) -> int:
        with self._lock:
            return int(
                self._counters["cache_hits"]
                + self._counters["coalesced_requests"]
                + self._counters["duplicate_queries_prevented"]
                + self._counters["batched_items_saved"]
                + self._counters["budget_blocked"]
            )

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            counts = dict(self._counters)
            totals = counts["total_requests"]
            avoided = int(
                counts["cache_hits"]
                + counts["coalesced_requests"]
                + counts["duplicate_queries_prevented"]
                + counts["batched_items_saved"]
                + counts["budget_blocked"]
            )
            return {
                **counts,
                "requests_avoided": avoided,
                "requests_by_provider": dict(self._requests_by_provider),
                "cache_hit_rate": (
                    counts["cache_hits"] / (counts["cache_hits"] + counts["cache_misses"])
                    if counts["cache_hits"] + counts["cache_misses"] > 0
                    else 0.0
                ),
                "average_response_time": (
                    counts["total_response_time"] / totals if totals > 0 else 0.0
                ),
                "research_duration": time.time() - self._started_at,
                "uptime_seconds": time.time() - self._last_reset,
                "research_cost": 0.0,
            }

    def reset(self) -> None:
        with self._lock:
            self._counters = {key: 0 for key in _METRIC_KEYS}
            self._requests_by_provider = {}
            self._last_reset = time.time()


_metrics = ResearchMetrics()


def get_metrics() -> ResearchMetrics:
    """Return the process-level metrics singleton."""
    return _metrics


def reset_metrics() -> ResearchMetrics:
    """Reset counters (test isolation)."""
    _metrics.reset()
    return _metrics