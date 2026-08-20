"""Nominatim (OpenStreetMap) provider.

Usage policy (strict): max 1 request per second, single thread, meaningful
User-Agent/Referer (generic library UAs are rejected), results MUST be cached,
long-running scripts limited to ~4 requests/minute. Attribution is mandatory:
data © OpenStreetMap contributors (ODbL). The provider can be disabled via
the ``nominatim_enabled`` setting.
"""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional

from app.services.research_layer.providers.base import ResearchProvider

SEARCH_LIMIT = 5
ATTRIBUTION_NOTICE = "© OpenStreetMap contributors"


class NominatimProvider(ResearchProvider):
    name = "nominatim"
    display_name = "Nominatim (OpenStreetMap)"
    base_url = "https://nominatim.openstreetmap.org"
    ttl_seconds = 30 * 24 * 3600
    authority = 0.7
    attribution = f"Geodata: {ATTRIBUTION_NOTICE} (ODbL)."
    requires_auth = False
    min_request_interval = 1.0  # Nominatim: max 1 req/sec
    max_concurrent = 1

    def _headers(self) -> Dict[str, str]:
        referer = str(self.ctx.setting("user_agent", "") or "").strip()
        headers: Dict[str, str] = {"Referer": referer or "ReelSync"}
        return headers

    def _place_to_result(self, query: str, place: Dict[str, Any], retrieved_at: str) -> Optional[Any]:
        display_name = str(place.get("display_name", "") or "").strip()
        if not display_name:
            return None
        place_id = str(place.get("place_id", "") or "")
        osm_type = str(place.get("osm_type", "") or "").strip()
        osm_id = str(place.get("osm_id", "") or "")
        osm_url = f"https://www.openstreetmap.org/{osm_type}/{osm_id}" if osm_type and osm_id else ""
        lat = place.get("lat")
        lon = place.get("lon")
        coordinates = f"{lat}, {lon}" if lat and lon else ""
        category = str(place.get("category", "") or "").strip()
        place_type = str(place.get("type", "") or "").strip()
        facts = [f"Location: {display_name}"]
        if coordinates:
            facts.append(f"Coordinates: {coordinates}")
        if category:
            facts.append(f"Category: {category} ({place_type})")
        return self._make_result(
            query=query,
            title=display_name,
            url=osm_url or f"https://nominatim.openstreetmap.org/details.php?place_id={place_id}",
            content=". ".join(facts),
            retrieved_at=retrieved_at,
            entities=[display_name],
            facts=facts,
            locations=[coordinates] if coordinates else [],
            citations=[{"title": display_name, "url": osm_url}] if osm_url else [],
            confidence=0.8,
            raw_metadata={
                "osm_type": osm_type,
                "osm_id": osm_id,
                "attribution": ATTRIBUTION_NOTICE,
                "license": "ODbL",
            },
        )

    def _search_places(self, query: str) -> List[Any]:
        response = self._http_get(
            self.base_url,
            params={
                "q": query,
                "format": "jsonv2",
                "limit": SEARCH_LIMIT,
                "accept-language": "en",
            },
            headers=self._headers(),
        )
        if response is None:
            return []
        payload = self._parse_json(response)
        if not isinstance(payload, list):
            return []
        retrieved_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        return [
            result
            for place in payload
            if isinstance(place, dict)
            for result in [self._place_to_result(query, place, retrieved_at)]
            if result is not None
        ]

    # -- interface -----------------------------------------------------------

    def search(self, query: str, **kwargs) -> List[Any]:
        if not self.ctx.setting("nominatim_enabled", True):
            return []
        results, _ = self._cached("search", {"q": query}, lambda: (self._search_places(query), {"query": query}))
        return results

    def _fetch_one(self, identifier: str, **kwargs) -> Optional[Any]:
        cleaned = str(identifier).strip()
        if not cleaned:
            return None

        def loader():
            results = []
            if "/" in cleaned:
                parts = cleaned.split("/")
                if len(parts) == 2 and parts[0].lower() in ("n", "w", "r"):
                    response = self._http_get(
                        self.base_url,
                        params={
                            "osm_ids": f"{parts[0]}{parts[1]}",
                            "format": "jsonv2",
                        },
                        headers=self._headers(),
                    )
                    if response is not None:
                        payload = self._parse_json(response)
                        retrieved_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
                        results = [
                            result
                            for place in (payload if isinstance(payload, list) else [])
                            if isinstance(place, dict)
                            for result in [self._place_to_result(cleaned, place, retrieved_at)]
                            if result is not None
                        ]
            return results, {"id": cleaned}

        results, _ = self._cached("fetch", {"id": cleaned}, loader)
        return results[0] if results else None

    def get_metadata(self, identifier: str) -> Dict[str, Any]:
        return {"osm_id": str(identifier), "source": self.name}

    def health_check(self) -> bool:
        response = self._http_get(
            self.base_url,
            params={"q": "test", "format": "jsonv2", "limit": 1},
            headers=self._headers(),
        )
        return response is not None and response.status_code == 200