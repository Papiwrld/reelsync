"""arXiv provider.

Usage policy: at most 1 request per 3 seconds, single connection, Atom XML
responses. Search results arrive with ``published`` dates; abstracts are
truncated to a readable length. The provider deliberately uses its own
throttle settings regardless of the global request interval.
"""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional
from xml.etree import ElementTree as ET

from app.services.research_layer.providers.base import ResearchProvider

SEARCH_LIMIT = 8
ABSTRACT_CHAR_LIMIT = 1200
_ATOM = "{http://www.w3.org/2005/Atom}"


class ArxivProvider(ResearchProvider):
    name = "arxiv"
    display_name = "arXiv"
    base_url = "https://export.arxiv.org/api/query"
    ttl_seconds = 7 * 24 * 3600
    authority = 0.9
    attribution = "Preprints from arXiv."
    requires_auth = False
    min_request_interval = 3.0  # arXiv: 1 request per 3 seconds
    max_concurrent = 1
    supports_batch_fetch = False

    def _entry_to_result(self, query: str, entry: ET.Element, retrieved_at: str) -> Optional[Any]:
        def _text(tag: str) -> str:
            node = entry.find(f"{_ATOM}{tag}")
            return (node.text or "").strip() if node is not None and node.text else ""

        title = _text("title").replace("\n", " ").strip()
        if not title:
            return None
        summary = _text("summary").replace("\n", " ").strip()[:ABSTRACT_CHAR_LIMIT]
        authors = [
            author.find(f"{_ATOM}name").text.strip()
            for author in entry.findall(f"{_ATOM}author")
            if author.find(f"{_ATOM}name") is not None
            and author.find(f"{_ATOM}name").text
        ]
        published = _text("published")
        year = published[:4] if published else ""
        link = entry.find(f"{_ATOM}link")
        url = link.get("href") if link is not None else f"https://arxiv.org/abs/{title}"
        doi = entry.find(f"{_ATOM}doi")
        identifier = str(entry.find(f"{_ATOM}id").text).strip() if entry.find(f"{_ATOM}id") is not None else url
        facts = [f"Paper: {title}"]
        if authors:
            facts.append(f"Authors: {', '.join(authors[:5])}")
        if year:
            facts.append(f"Published: {year}")
        return self._make_result(
            query=query,
            title=title,
            url=url,
            content=summary,
            retrieved_at=retrieved_at,
            entities=[title] + authors[:3],
            facts=facts,
            dates=[year] if year else [],
            citations=[{"title": title, "url": url, "identifier": identifier}] if doi else [],
            confidence=0.8,
            raw_metadata={
                "arxiv_id": identifier,
                "doi": (doi.text or "").strip() if doi is not None else None,
                "published": published,
                "updated": _text("updated"),
            },
        )

    def _parse_feed(self, query: str, text: str) -> List[Any]:
        try:
            root = ET.fromstring(text)
        except ET.ParseError:
            return []
        retrieved_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        results = []
        for entry in root.findall(f"{_ATOM}entry"):
            result = self._entry_to_result(query, entry, retrieved_at)
            if result is not None:
                results.append(result)
        return results

    def _search_query(self, query: str) -> List[Any]:
        response = self._http_get(
            self.base_url,
            params={
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": SEARCH_LIMIT,
                "sortBy": "relevance",
            },
        )
        if response is None:
            return []
        return self._parse_feed(query, response.text or "")

    # -- interface -----------------------------------------------------------

    def search(self, query: str, **kwargs) -> List[Any]:
        results, _ = self._cached("search", {"q": query}, lambda: (self._search_query(query), {"query": query}))
        return results

    def _fetch_one(self, identifier: str, **kwargs) -> Optional[Any]:
        cleaned = str(identifier).strip()
        if not cleaned:
            return None
        if "arxiv.org" in cleaned:
            cleaned = cleaned.rsplit("/", 1)[-1]

        def loader():
            response = self._http_get(
                self.base_url,
                params={"id_list": cleaned, "max_results": 1},
            )
            if response is None:
                return [], {"id": cleaned}
            return self._parse_feed(cleaned, response.text or ""), {"id": cleaned}

        results, _ = self._cached("fetch", {"id": cleaned}, loader)
        return results[0] if results else None

    def get_metadata(self, identifier: str) -> Dict[str, Any]:
        return {"arxiv_id": str(identifier), "source": self.name}

    def health_check(self) -> bool:
        response = self._http_get(self.base_url, params={"search_query": "all:test", "max_results": 1})
        return response is not None and response.status_code == 200