"""Provider base classes for the zero-key research layer.

Each provider exposes the unified interface ``search / fetch /
get_metadata / health_check`` and returns normalized ``ResearchResult``
objects. All network access, throttling, budgeting and caching is handled by
the shared base helpers, so individual providers stay thin and testable.
"""

from __future__ import annotations

import abc
import re
import time
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

from app.services.research_layer.budget import BudgetContext
from app.services.research_layer.cache import CachedSearch
from app.services.research_layer.http import ResearchHttpClient
from app.services.research_layer.metrics import ResearchMetrics
from app.services.research_layer.schema import ResearchResult
from app.services.research_layer.throttle import RequestManager

_YEAR_PATTERN = re.compile(r"(?<!\d)(1[0-9]{3}|2[0-9]{3})(?!\d)")
_PERCENT_PATTERN = re.compile(r"[\d][\d.,]*\s?%")
_MONEY_PATTERN = re.compile(r"(?:\$|USD\s?|€|£)\s?[\d][\d.,]*\s?[bBmMkK]?")


def _result_from_dict(raw: Dict[str, Any]) -> Optional[ResearchResult]:
    """Deserialize a cached dict back into a ResearchResult (best effort)."""
    try:
        return ResearchResult(**raw)
    except TypeError as exc:
        logger.warning(f"malformed cached result dropped: {exc}")
        return None


class ProviderContext:
    """Shared infrastructure handed to every provider."""

    def __init__(
        self,
        http: ResearchHttpClient,
        request_manager: RequestManager,
        metrics: ResearchMetrics,
        budget: Optional[BudgetContext] = None,
        settings: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.http = http
        self.request_manager = request_manager
        self.metrics = metrics
        self.budget = budget
        self.settings = settings or {}

    def setting(self, key: str, default: Any = None) -> Any:
        return self.settings.get(key, default)


class ResearchProvider(abc.ABC):
    """Base class for all zero-key providers."""

    name: str = "base"
    display_name: str = "Base"
    base_url: str = ""
    ttl_seconds: int = 24 * 3600
    authority: float = 0.5
    attribution: str = ""
    requires_auth: bool = False
    min_request_interval: float = 0.0
    max_concurrent: int = 1
    supports_batch_fetch: bool = False
    max_fetch_batch: int = 10

    def __init__(self, context: ProviderContext) -> None:
        self.ctx = context
        self.ctx.request_manager.register(
            self.name, self.min_request_interval, self.max_concurrent
        )

    # -- shared plumbing ----------------------------------------------------

    def _budget_acquire(self) -> bool:
        if self.ctx.budget is None:
            return True
        acquired = self.ctx.budget.acquire(self.name)
        if not acquired:
            self.ctx.metrics.increment("budget_blocked")
        return acquired

    def _http_get(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Optional[Any]:
        """Budget → throttle → HTTP GET. Returns a response object or None."""
        if not self._budget_acquire():
            return None
        token = self.ctx.request_manager.acquire(self.name)
        try:
            response = self.ctx.http.get(self.name, url, params=params, headers=headers)
            if response is not None:
                self.ctx.metrics.record_provider_request(self.name)
            return response
        finally:
            self.ctx.request_manager.release(self.name, token)

    def _cached(
        self,
        operation: str,
        params: Dict[str, Any],
        loader: Callable[[], Optional[tuple]],
        ttl_seconds: Optional[int] = None,
    ) -> tuple:
        """Cache-lookup (with coalescing) then ``loader()`` on a miss.

        ``loader`` must return ``(results: List[ResearchResult], metadata: dict)``
        or ``(None, None)`` on failure. Budget is only charged when the cache
        misses and the network is actually hit.
        """
        ttl = ttl_seconds or self.ttl_seconds

        def _load() -> Optional[tuple]:
            if not self._budget_acquire():
                return None
            payload, metadata = loader()
            if payload is None:
                return None
            return [result.to_dict() for result in payload], metadata

        cached = CachedSearch(
            self.name,
            operation,
            params,
            ttl,
            self.ctx.metrics,
        )
        raw_results, metadata = cached.get(_load)
        results: List[ResearchResult] = [
            result
            for raw in raw_results
            if isinstance(raw, dict)
            for result in [_result_from_dict(raw)]
            if result is not None
        ]
        return results, metadata

    def _cached_raw(
        self,
        operation: str,
        params: Dict[str, Any],
        loader: Callable[[], Optional[tuple]],
        ttl_seconds: Optional[int] = None,
    ) -> tuple:
        """Like ``_cached`` but payloads are plain dicts (no schema enforcement)."""
        ttl = ttl_seconds or self.ttl_seconds

        def _load() -> Optional[tuple]:
            if not self._budget_acquire():
                return None
            payload, metadata = loader()
            if payload is None:
                return None
            return payload, metadata

        cached = CachedSearch(
            self.name,
            operation,
            params,
            ttl,
            self.ctx.metrics,
        )
        return cached.get(_load)

    def _parse_json(self, response) -> Optional[Any]:
        try:
            payload = response.json()
        except (ValueError, AttributeError):
            logger.warning(f"malformed JSON response: provider={self.name}")
            return None
        if not isinstance(payload, (dict, list)):
            logger.warning(f"unexpected response shape: provider={self.name}")
            return None
        return payload

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _extract_years(content: str) -> List[str]:
        return list(dict.fromkeys(_YEAR_PATTERN.findall(content or "")))

    @staticmethod
    def _extract_statistics(content: str) -> List[str]:
        found: List[str] = []
        for match in _MONEY_PATTERN.findall(content or ""):
            if match not in found:
                found.append(match)
        for match in _PERCENT_PATTERN.findall(content or ""):
            if match not in found:
                found.append(match)
        return found

    @staticmethod
    def _sentences(content: str, limit: int = 8) -> List[str]:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", content or "") if s.strip()]
        return sentences[:limit]

    def _make_result(
        self,
        query: str,
        title: str,
        url: str,
        content: str,
        retrieved_at: str,
        entities: Optional[List[str]] = None,
        facts: Optional[List[str]] = None,
        dates: Optional[List[str]] = None,
        locations: Optional[List[str]] = None,
        statistics: Optional[List[str]] = None,
        citations: Optional[List[Dict[str, str]]] = None,
        confidence: float = 0.5,
        raw_metadata: Optional[Dict[str, Any]] = None,
    ) -> ResearchResult:
        metadata = dict(raw_metadata or {})
        metadata.setdefault("provider_display", self.display_name)
        metadata.setdefault("authority", self.authority)
        metadata.setdefault("attribution", self.attribution)
        metadata.setdefault("retrieved_ts", time.time())
        return ResearchResult(
            source=self.name,
            source_url=url,
            title=title,
            retrieved_at=retrieved_at,
            query=query,
            content=content,
            entities=list(entities or []),
            facts=list(facts or []),
            dates=list(dates or self._extract_years(content)),
            locations=list(locations or []),
            statistics=list(statistics or self._extract_statistics(content)),
            citations=list(citations or []),
            confidence=round(max(0.0, min(1.0, confidence)), 3),
            raw_metadata=metadata,
        )

    # -- interface ------------------------------------------------------------

    @abc.abstractmethod
    def search(self, query: str, **kwargs) -> List[ResearchResult]:
        """Return normalized results for ``query``."""

    def fetch(self, identifier: str, **kwargs) -> Optional[ResearchResult]:
        """Fetch one item by identifier (title, DOI, QID, ...)."""
        results = self.fetch_many([identifier], **kwargs)
        return results[0] if results else None

    def fetch_many(self, identifiers: List[str], **kwargs) -> List[ResearchResult]:
        """Fetch several items, batching where the provider supports it."""
        results: List[ResearchResult] = []
        identifiers = [str(item) for item in identifiers if str(item).strip()]
        if not identifiers:
            return results
        if self.supports_batch_fetch and len(identifiers) > 1:
            self.ctx.metrics.increment("batch_requests")
            saved = max(0, len(identifiers) - 1)
            self.ctx.metrics.increment("batched_items_saved", saved)
            results = self._fetch_batch(identifiers, **kwargs)
        else:
            for identifier in identifiers:
                item = self._fetch_one(identifier, **kwargs)
                if item is not None:
                    results.append(item)
        return results

    def _fetch_batch(self, identifiers: List[str], **kwargs) -> List[ResearchResult]:
        return [item for item in (self._fetch_one(item, **kwargs) for item in identifiers) if item]

    @abc.abstractmethod
    def _fetch_one(self, identifier: str, **kwargs) -> Optional[ResearchResult]:
        raise NotImplementedError

    @abc.abstractmethod
    def get_metadata(self, identifier: str) -> Dict[str, Any]:
        """Return provider metadata for one item (never raises)."""

    @abc.abstractmethod
    def health_check(self) -> bool:
        """Return True when the provider answers a minimal request."""

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "base_url": self.base_url,
            "ttl_seconds": self.ttl_seconds,
            "authority": self.authority,
            "attribution": self.attribution,
            "requires_auth": self.requires_auth,
            "supports_batch_fetch": self.supports_batch_fetch,
        }