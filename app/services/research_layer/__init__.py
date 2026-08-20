"""Zero-key research layer public API.

``ResearchClient`` is the single entry point used by the research bridge:
it normalizes and deduplicates queries, routes them to the appropriate
no-key providers, enforces the per-job budget, keeps everything cached,
batched and throttled, and finally corroborates and scores the results.

Providers currently available: Wikipedia, Wikidata, OpenAlex, Crossref,
arXiv, World Bank, NASA, Open-Meteo, Nominatim.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from loguru import logger

from app.services.research_layer.budget import BudgetContext
from app.services.research_layer.dedupe import JobDedupStore, normalize_query
from app.services.research_layer.http import ResearchHttpClient
from app.services.research_layer.metrics import get_metrics
from app.services.research_layer.providers.arxiv import ArxivProvider
from app.services.research_layer.providers.base import ProviderContext, ResearchProvider
from app.services.research_layer.providers.crossref import CrossrefProvider
from app.services.research_layer.providers.nasa import NasaProvider
from app.services.research_layer.providers.nominatim import NominatimProvider
from app.services.research_layer.providers.openalex import OpenAlexProvider
from app.services.research_layer.providers.openmeteo import OpenMeteoProvider
from app.services.research_layer.providers.wikidata import WikidataProvider
from app.services.research_layer.providers.wikipedia import WikipediaProvider
from app.services.research_layer.providers.worldbank import WorldBankProvider
from app.services.research_layer.quality import corroborate_all
from app.services.research_layer.router import ResearchRouter
from app.services.research_layer.schema import ResearchClaim, ResearchPackage, ResearchResult, SourceReference
from app.services.research_layer.throttle import RequestManager

_PROVIDER_CLASSES = [
    WikipediaProvider,
    WikidataProvider,
    OpenAlexProvider,
    CrossrefProvider,
    ArxivProvider,
    WorldBankProvider,
    NasaProvider,
    OpenMeteoProvider,
    NominatimProvider,
]


def get_provider_registry() -> List[Dict[str, Any]]:
    """Static registry of available providers (no network access)."""
    return [{"name": cls.name, "display_name": cls.display_name} for cls in _PROVIDER_CLASSES]


def _aggregate(package: ResearchPackage) -> None:
    """Roll per-result fields up into the package and build claims."""
    for result in package.results:
        for entity in result.entities:
            if entity not in package.entities:
                package.entities.append(entity)
        for date in result.dates:
            if date not in package.dates:
                package.dates.append(date)
        for location in result.locations:
            if location not in package.locations:
                package.locations.append(location)
        for statistic in result.statistics:
            if statistic not in package.statistics:
                package.statistics.append(statistic)
        for citation in result.citations:
            if citation not in package.citations:
                package.citations.append(citation)
    seen: set = set()
    for result in package.results:
        for fact in result.facts:
            key = fact.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            package.claims.append(
                ResearchClaim(
                    statement=fact,
                    sources=[
                        SourceReference(
                            source=result.source,
                            source_url=result.source_url,
                            title=result.title,
                            retrieved_at=result.retrieved_at,
                        )
                    ],
                    confidence=result.confidence,
                )
            )


class ResearchClient:
    """Unified client over all zero-key providers."""

    def __init__(
        self,
        budget: Optional[BudgetContext] = None,
        settings: Optional[Dict[str, Any]] = None,
        metrics=None,
    ) -> None:
        self.settings = dict(settings or {})
        self.metrics = metrics or get_metrics()
        self.budget = budget or BudgetContext.from_settings(self.settings)
        self.router = ResearchRouter()
        self._dedupe = JobDedupStore()
        request_manager = RequestManager()
        http = ResearchHttpClient(
            user_agent=str(self.settings.get("user_agent", "") or "").strip()
            or None,
            contact_email=str(self.settings.get("contact_email", "") or "").strip()
            or None,
            max_retries=int(self.settings.get("max_retries", 3) or 3),
            request_manager=request_manager,
            metrics=self.metrics,
        )
        context = ProviderContext(
            http=http,
            request_manager=request_manager,
            metrics=self.metrics,
            budget=self.budget,
            settings=self.settings,
        )
        self.providers: Dict[str, ResearchProvider] = {}
        for cls in _PROVIDER_CLASSES:
            try:
                self.providers[cls.name] = cls(context)
            except Exception as exc:  # defensive: one bad provider must not block the client
                logger.warning(f"failed to initialize provider {cls.name}: {exc}")

    def _candidates(self, normalized: str) -> List[ResearchProvider]:
        return [
            self.providers[name]
            for name in self.router.select(normalized)
            if name in self.providers
        ]

    def research(self, query: str) -> List[ResearchResult]:
        """Full pipeline: normalize → dedupe → route → fetch → corroborate."""
        normalized = normalize_query(query)
        dedupe = self._dedupe
        if dedupe.find("client", "research", normalized) is not None:
            self.metrics.increment("duplicate_queries_prevented")
            return []
        dedupe.store("client", "research", normalized, {"seen": True})
        candidates = self._candidates(normalized)
        collected: List[ResearchResult] = []
        for provider in candidates:
            try:
                results = provider.search(normalized)
            except Exception as exc:
                logger.warning(f"provider {provider.name} failed during research: {exc}")
                continue
            collected.extend(results)
            if self.budget.remaining_total() == 0:
                break
        authorities = {
            name: provider.authority
            for name, provider in self.providers.items()
            if name in {result.source for result in collected}
        }
        ttl_by_source = {
            name: provider.ttl_seconds
            for name, provider in self.providers.items()
            if name in authorities
        }
        package = ResearchPackage(
            query=query,
            normalized_query=normalized,
            category=self.router.classify(normalized),
            providers=[provider.name for provider in candidates],
            results=collected,
            quality_scores=corroborate_all(normalized, collected, authorities, ttl_by_source),
            metrics=self.metrics.snapshot(),
            requests_avoided=self.metrics.requests_avoided(),
        )
        _aggregate(package)
        self._last_package = package
        return sorted(
            collected,
            key=lambda result: package.quality_scores.get(result.source_url, 0.0),
            reverse=True,
        )

    def last_package(self) -> Optional[ResearchPackage]:
        """The ResearchPackage produced by the most recent research() call."""
        return getattr(self, "_last_package", None)

    def batch_fetch(
        self,
        provider_name: str,
        identifiers: List[str],
        query: str = "",
    ) -> List[ResearchResult]:
        """Fetch several items from one provider (batched where supported)."""
        provider = self.providers.get(provider_name)
        if provider is None:
            return []
        try:
            results = provider.fetch_many(identifiers, query=query)
            authorities = {provider_name: provider.authority}
            ttl_by_source = {provider_name: provider.ttl_seconds}
            corroborate_all(query, results, authorities, ttl_by_source)
            return results
        except Exception as exc:
            logger.warning(f"batch_fetch failed for {provider_name}: {exc}")
            return []

    def health_checks(self) -> Dict[str, bool]:
        """Run one cheap request per provider; never raises."""
        status: Dict[str, bool] = {}
        for name, provider in self.providers.items():
            try:
                status[name] = bool(provider.health_check())
            except Exception:
                status[name] = False
        return status

    def snapshot_metrics(self) -> Dict[str, Any]:
        return self.metrics.snapshot()