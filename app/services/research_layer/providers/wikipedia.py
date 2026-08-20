"""Wikipedia (MediaWiki Action API) provider.

Usage policy: an identifying User-Agent is required (403 otherwise);
unauthenticated bots get ~200 requests/minute with at most 3 concurrent
connections; honor 429 Retry-After. Search uses one call; page content is
fetched as a batched multi-title extract call.
"""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional

from app.services.research_layer.providers.base import ResearchProvider

SEARCH_LIMIT = 5
EXTRACT_CHAR_LIMIT = 2500


class WikipediaProvider(ResearchProvider):
    name = "wikipedia"
    display_name = "Wikipedia"
    base_url = "https://en.wikipedia.org/w/api.php"
    ttl_seconds = 30 * 24 * 3600
    authority = 0.9
    attribution = "Content from Wikipedia (CC BY-SA 4.0)."
    requires_auth = False
    min_request_interval = 0.15
    max_concurrent = 3
    supports_batch_fetch = True
    max_fetch_batch = 10

    # -- internals -----------------------------------------------------------

    def _search_titles(self, query: str) -> List[str]:
        response = self._http_get(
            self.base_url,
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": SEARCH_LIMIT,
                "format": "json",
                "formatversion": 2,
            },
        )
        if response is None:
            return []
        payload = self._parse_json(response)
        if not payload:
            return []
        hits = payload.get("query", {}).get("search", [])
        if not isinstance(hits, list):
            return []
        return [
            str(hit.get("title", "")).strip()
            for hit in hits
            if isinstance(hit, dict) and str(hit.get("title", "")).strip()
        ]

    def _fetch_pages(self, titles: List[str]) -> List[Dict[str, Any]]:
        if not titles:
            return []
        response = self._http_get(
            self.base_url,
            params={
                "action": "query",
                "titles": "\n".join(titles),
                "prop": "extracts|info",
                "exintro": 1,
                "explaintext": 1,
                "redirects": 1,
                "inprop": "url",
                "format": "json",
                "formatversion": 2,
            },
        )
        if response is None:
            return []
        payload = self._parse_json(response)
        if not payload:
            return []
        pages = payload.get("query", {}).get("pages", [])
        return [page for page in pages if isinstance(page, dict)]

    def _to_result(self, query: str, page: Dict[str, Any], retrieved_at: str) -> Optional[Any]:
        title = str(page.get("title", "")).strip()
        extract = str(page.get("extract", "") or "").strip()[:EXTRACT_CHAR_LIMIT]
        if not title and not extract:
            return None
        url = str(page.get("fullurl", f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"))
        facts = self._sentences(extract, limit=4)
        return self._make_result(
            query=query,
            title=title or "Wikipedia article",
            url=url,
            content=extract,
            retrieved_at=retrieved_at,
            entities=[title],
            facts=facts,
            citations=[],
            confidence=0.85,
            raw_metadata={
                "pageid": page.get("pageid"),
                "revision": page.get("lastrevid"),
                "length": page.get("length"),
                "license": "CC BY-SA 4.0",
            },
        )

    # -- interface -----------------------------------------------------------

    def search(self, query: str, **kwargs) -> List[Any]:
        def loader():
            retrieved_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
            titles = self._search_titles(query)
            pages = self._fetch_pages(titles)
            results = [
                result
                for page in pages
                for result in [self._to_result(query, page, retrieved_at)]
                if result is not None
            ]
            return results, {"query": query, "titles": titles}

        results, _ = self._cached("search", {"q": query}, loader)
        return results

    def _fetch_one(self, identifier: str, **kwargs) -> Optional[Any]:
        retrieved_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

        def loader():
            pages = self._fetch_pages([identifier])
            results = [
                result
                for page in pages
                for result in [self._to_result(identifier, page, retrieved_at)]
                if result is not None
            ]
            return results, {"title": identifier}

        results, _ = self._cached("fetch", {"id": identifier}, loader)
        return results[0] if results else None

    def _fetch_batch(self, identifiers: List[str], **kwargs) -> List[Any]:
        retrieved_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        query = str(kwargs.get("query", ""))
        results: List[Any] = []
        for chunk_start in range(0, len(identifiers), self.max_fetch_batch):
            chunk = identifiers[chunk_start : chunk_start + self.max_fetch_batch]

            def loader(titles=chunk):
                pages = self._fetch_pages(titles)
                items = [
                    result
                    for page in pages
                    for result in [self._to_result(query or titles[0], page, retrieved_at)]
                    if result is not None
                ]
                return items, {"titles": titles}

            batch_results, _ = self._cached("fetch_batch", {"ids": chunk}, loader)
            results.extend(batch_results)
        return results

    def get_metadata(self, identifier: str) -> Dict[str, Any]:
        response = self._http_get(
            self.base_url,
            params={
                "action": "query",
                "titles": identifier,
                "prop": "info",
                "format": "json",
                "formatversion": 2,
            },
        )
        if response is None:
            return {}
        payload = self._parse_json(response)
        pages = payload.get("query", {}).get("pages", []) if payload else []
        page = pages[0] if pages else {}
        return {
            "title": page.get("title"),
            "pageid": page.get("pageid"),
            "revision": page.get("lastrevid"),
            "touched": page.get("touched"),
            "source": self.name,
        }

    def health_check(self) -> bool:
        response = self._http_get(
            self.base_url,
            params={"action": "query", "meta": "siteinfo", "format": "json", "formatversion": 2},
        )
        return response is not None and response.status_code == 200