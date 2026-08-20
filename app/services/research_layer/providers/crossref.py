"""Crossref provider.

Public pool: 5 requests/sec, 1 concurrent. Polite pool: 10 req/sec,
3 concurrent — activated by adding a contact email (``mailto`` parameter),
which also improves the response quality. Rate limits are communicated via
``x-rate-limit-*`` headers; a 429 means we should back off.
"""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional

from app.services.research_layer.providers.base import ResearchProvider

SEARCH_LIMIT = 8


class CrossrefProvider(ResearchProvider):
    name = "crossref"
    display_name = "Crossref"
    base_url = "https://api.crossref.org/works"
    ttl_seconds = 14 * 24 * 3600
    authority = 0.9
    attribution = "Bibliographic data from Crossref."
    requires_auth = False
    min_request_interval = 0.2
    max_concurrent = 2
    supports_batch_fetch = True
    max_fetch_batch = 100

    def _common_params(self) -> Dict[str, Any]:
        params: Dict[str, Any] = {"rows": SEARCH_LIMIT, "select": "DOI,title,author,issued,published,citation-count,container-title,type,URL,is-referenced-by-count"}
        email = str(self.ctx.setting("contact_email", "") or "").strip()
        if email:
            params["mailto"] = email
        return params

    def _work_to_result(self, query: str, work: Dict[str, Any], retrieved_at: str) -> Optional[Any]:
        title = ""
        raw_title = work.get("title")
        if isinstance(raw_title, list) and raw_title:
            title = str(raw_title[0])
        elif isinstance(raw_title, str):
            title = raw_title
        title = title.strip()
        if not title:
            return None
        doi = str(work.get("DOI", "") or "").strip()
        url = f"https://doi.org/{doi}" if doi else str(work.get("URL", "") or "").strip()
        authors = []
        for author in work.get("author") or []:
            if not isinstance(author, dict):
                continue
            name = (
                author.get("name")
                or " ".join(
                    part
                    for part in (author.get("given", ""), author.get("family", ""))
                    if part
                )
            )
            if name:
                authors.append(str(name))
        venue = ""
        raw_venue = work.get("container-title")
        if isinstance(raw_venue, list) and raw_venue:
            venue = str(raw_venue[0])
        elif isinstance(raw_venue, str):
            venue = raw_venue
        year = None
        for source in ("published", "issued"):
            date_parts = (work.get(source) or {}).get("date-parts")
            if isinstance(date_parts, list) and date_parts and date_parts[0]:
                year = date_parts[0][0]
                break
        citations = None
        citations_count = work.get("is-referenced-by-count")
        if citations_count is not None:
            try:
                citations = int(citations_count)
            except (TypeError, ValueError):
                citations = None
        facts = [f"Publication: {title}"]
        if authors:
            facts.append(f"Authors: {', '.join(authors)}")
        if venue:
            facts.append(f"Published in: {venue}")
        if year:
            facts.append(f"Year: {year}")
        if citations is not None:
            facts.append(f"Cited by: {citations}")
        return self._make_result(
            query=query,
            title=title,
            url=url,
            content=". ".join(part for part in [title] + ([", ".join(authors[:3])] if authors else []) + ([venue] if venue else []) + ([str(year)] if year else []) if part),
            retrieved_at=retrieved_at,
            entities=[title] + authors,
            facts=facts,
            dates=[str(year)] if year else [],
            citations=[{"title": title, "url": url}] if doi else [],
            statistics=[f"cited by {citations}"] if citations is not None else [],
            confidence=0.85,
            raw_metadata={
                "doi": doi,
                "type": work.get("type"),
                "venue": venue,
                "license": "Crossref metadata is CC0",
            },
        )

    def _items_to_results(self, query: str, payload: Optional[Dict[str, Any]]) -> List[Any]:
        if not payload:
            return []
        items = payload.get("message", {}).get("items")
        if not isinstance(items, list):
            return []
        retrieved_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        return [
            result
            for work in items
            if isinstance(work, dict)
            for result in [self._work_to_result(query, work, retrieved_at)]
            if result is not None
        ]

    def _search_works(self, query: str) -> List[Any]:
        params = self._common_params()
        params["query"] = query
        response = self._http_get(self.base_url, params=params)
        if response is None:
            return []
        return self._items_to_results(query, self._parse_json(response))

    def _fetch_dois(self, identifiers: List[str]) -> List[Any]:
        cleaned = []
        for item in identifiers:
            item = str(item).strip()
            if item.startswith("https://doi.org/"):
                item = item.replace("https://doi.org/", "", 1)
            cleaned.append(item)
        if not cleaned:
            return []
        params = self._common_params()
        params["filter"] = f"doi:{','.join(cleaned)}"
        response = self._http_get(self.base_url, params=params)
        if response is None:
            return []
        return self._items_to_results(cleaned[0], self._parse_json(response))

    # -- interface -----------------------------------------------------------

    def search(self, query: str, **kwargs) -> List[Any]:
        results, _ = self._cached("search", {"q": query}, lambda: (self._search_works(query), {"query": query}))
        return results

    def _fetch_one(self, identifier: str, **kwargs) -> Optional[Any]:
        cleaned = str(identifier).strip()
        if not cleaned:
            return None
        if cleaned.startswith("https://doi.org/"):
            cleaned = cleaned.replace("https://doi.org/", "", 1)

        def loader():
            response = self._http_get(f"{self.base_url}/{cleaned}", params={"select": self._common_params()["select"]})
            if response is None:
                return [], {"id": cleaned}
            payload = self._parse_json(response)
            if not payload:
                return [], {"id": cleaned}
            message = payload.get("message")
            if not isinstance(message, dict):
                return [], {"id": cleaned}
            retrieved_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
            result = self._work_to_result(cleaned, message, retrieved_at)
            return ([result] if result else []), {"id": cleaned}

        results, _ = self._cached("fetch", {"id": cleaned}, loader)
        return results[0] if results else None

    def _fetch_batch(self, identifiers: List[str], **kwargs) -> List[Any]:
        return self._fetch_dois(identifiers)

    def get_metadata(self, identifier: str) -> Dict[str, Any]:
        response = self._http_get(f"{self.base_url}/{identifier}", params={"select": "DOI,title,issued,is-referenced-by-count"})
        if response is None:
            return {}
        payload = self._parse_json(response)
        if not payload:
            return {}
        message = payload.get("message") or {}
        return {
            "doi": message.get("DOI"),
            "title": (message.get("title") or [None])[0] if message.get("title") else None,
            "issued": (message.get("issued") or {}).get("date-parts"),
            "citations": message.get("is-referenced-by-count"),
            "source": self.name,
        }

    def health_check(self) -> bool:
        response = self._http_get(self.base_url, params={"rows": 1, "select": "DOI"})
        return response is not None and response.status_code == 200