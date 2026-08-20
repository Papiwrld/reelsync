"""OpenAlex provider.

Policy change (Feb 2026): a free API key is now required for real-scale use;
keyless requests get a small demo budget (~100 credits/day). The key is
optional here and read from config; without one the provider still works at
the demo tier and degrades gracefully. 429 responses are retried with
backoff and recorded as rate-limit events.
"""

from __future__ import annotations

import datetime
import re
from typing import Any, Dict, List, Optional

from app.services.research_layer.providers.base import ResearchProvider

SEARCH_LIMIT = 8
_SELECT_FIELDS = "id,doi,title,display_name,publication_year,cited_by_count,type,primary_location"


class OpenAlexProvider(ResearchProvider):
    name = "openalex"
    display_name = "OpenAlex"
    base_url = "https://api.openalex.org"
    ttl_seconds = 14 * 24 * 3600
    authority = 0.9
    attribution = "Data from OpenAlex (CC0)."
    requires_auth = True  # optional key; keyless demo tier still works
    min_request_interval = 0.2
    max_concurrent = 2
    supports_batch_fetch = True
    max_fetch_batch = 100

    def _headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        api_key = str(self.ctx.setting("openalex_api_key", "") or "").strip()
        if api_key:
            headers["X-API-Key"] = api_key
        return headers

    def _work_to_result(self, query: str, work: Dict[str, Any], retrieved_at: str) -> Optional[Any]:
        title = str(work.get("title") or work.get("display_name") or "").strip()
        if not title:
            return None
        work_id = str(work.get("id", "")).strip()
        doi = str(work.get("doi", "") or "").strip()
        url = doi or work_id or f"{self.base_url}/works/{work_id.split('/')[-1]}"
        year = work.get("publication_year")
        cited = work.get("cited_by_count")
        venue = ""
        primary = work.get("primary_location") or {}
        source = primary.get("source") or {}
        if isinstance(source, dict):
            venue = str(source.get("display_name", "")) or ""
        authors = []
        for authorship in (work.get("authorships") or [])[:3]:
            if isinstance(authorship, dict) and authorship.get("author"):
                name = authorship["author"].get("display_name")
                if name:
                    authors.append(str(name))
        content_parts = [title]
        if authors:
            content_parts.append("by " + ", ".join(authors))
        if venue:
            content_parts.append(f"published in {venue}")
        if year:
            content_parts.append(f"in {year}")
        facts = [f"Publication: {title}"]
        if authors:
            facts.append(f"Authors: {', '.join(authors)}")
        if year:
            facts.append(f"Published: {year}")
        if cited is not None:
            facts.append(f"Cited by: {cited}")
        return self._make_result(
            query=query,
            title=title,
            url=url,
            content=". ".join(part for part in content_parts if part),
            retrieved_at=retrieved_at,
            entities=[title] + authors,
            facts=facts,
            dates=[str(year)] if year else [],
            citations=[{"title": title, "url": url}] if doi else [],
            statistics=[f"cited by {cited}"] if cited is not None else [],
            confidence=0.85,
            raw_metadata={
                "work_id": work_id,
                "doi": doi,
                "type": work.get("type"),
                "venue": venue,
                "license": "CC0",
            },
        )

    def _works_to_results(self, query: str, payload: Optional[Dict[str, Any]]) -> List[Any]:
        if not payload:
            return []
        works = payload.get("results", [])
        if not isinstance(works, list):
            return []
        retrieved_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        return [
            result
            for work in works
            if isinstance(work, dict)
            for result in [self._work_to_result(query, work, retrieved_at)]
            if result is not None
        ]

    def _search_works(self, query: str) -> List[Any]:
        response = self._http_get(
            f"{self.base_url}/works",
            params={
                "search": query,
                "per_page": SEARCH_LIMIT,
                "select": _SELECT_FIELDS,
            },
            headers=self._headers(),
        )
        if response is None:
            return []
        return self._works_to_results(query, self._parse_json(response))

    def _fetch_by_id(self, identifier: str) -> List[Any]:
        response = self._http_get(
            f"{self.base_url}/works/{identifier}",
            params={"select": _SELECT_FIELDS},
            headers=self._headers(),
        )
        if response is None:
            return []
        payload = self._parse_json(response)
        if not payload or not isinstance(payload, dict):
            return []
        retrieved_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        result = self._work_to_result(identifier, payload, retrieved_at)
        return [result] if result else []

    def _fetch_by_filter(self, identifiers: List[str]) -> List[Any]:
        response = self._http_get(
            f"{self.base_url}/works",
            params={
                "filter": f"ids.openalex:{'|'.join(identifiers)}",
                "per_page": min(len(identifiers), 100),
                "select": _SELECT_FIELDS,
            },
            headers=self._headers(),
        )
        if response is None:
            return []
        return self._works_to_results(identifiers[0], self._parse_json(response))

    # -- interface -----------------------------------------------------------

    def search(self, query: str, **kwargs) -> List[Any]:
        results, _ = self._cached("search", {"q": query}, lambda: (self._search_works(query), {"query": query}))
        return results

    def _fetch_one(self, identifier: str, **kwargs) -> Optional[Any]:
        cleaned = identifier.strip()
        if not cleaned:
            return None

        def loader():
            if re.match(r"^W\d+$", cleaned):
                return self._fetch_by_id(cleaned), {"id": cleaned}
            return self._search_works(cleaned), {"query": cleaned}

        results, _ = self._cached("fetch", {"id": cleaned}, loader)
        return results[0] if results else None

    def _fetch_batch(self, identifiers: List[str], **kwargs) -> List[Any]:
        ids = [str(item) for item in identifiers if re.match(r"^W\d+$", str(item))]
        if ids:
            return self._fetch_by_filter(ids)
        return [item for item in (self._fetch_one(item) for item in identifiers) if item]

    def get_metadata(self, identifier: str) -> Dict[str, Any]:
        response = self._http_get(
            f"{self.base_url}/works/{identifier}",
            params={"select": _SELECT_FIELDS},
            headers=self._headers(),
        )
        if response is None:
            return {}
        payload = self._parse_json(response)
        if not payload:
            return {}
        return {
            "work_id": payload.get("id"),
            "doi": payload.get("doi"),
            "title": payload.get("title"),
            "publication_year": payload.get("publication_year"),
            "cited_by_count": payload.get("cited_by_count"),
            "source": self.name,
        }

    def health_check(self) -> bool:
        response = self._http_get(
            f"{self.base_url}/works",
            params={"per_page": 1, "select": "id"},
            headers=self._headers(),
        )
        return response is not None and response.status_code == 200