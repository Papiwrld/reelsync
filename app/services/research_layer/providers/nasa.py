"""NASA Open APIs provider (APOD + NEO).

Keyless usage runs on the shared DEMO_KEY pool (30 requests/hour/IP,
50 requests/day/IP); a registered free key raises these limits. The key is
read from config with ``DEMO_KEY`` as the default.
"""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional

from app.services.research_layer.providers.base import ResearchProvider


class NasaProvider(ResearchProvider):
    name = "nasa"
    display_name = "NASA"
    base_url = "https://api.nasa.gov"
    ttl_seconds = 24 * 3600
    authority = 0.9
    attribution = "Imagery and data from NASA Open APIs."
    requires_auth = False  # DEMO_KEY needs no registration
    min_request_interval = 2.0
    max_concurrent = 1

    def _api_key(self) -> str:
        return str(self.ctx.setting("nasa_api_key", "DEMO_KEY") or "DEMO_KEY").strip()

    def _apod(self, query: str) -> List[Any]:
        response = self._http_get(
            f"{self.base_url}/planetary/apod",
            params={"api_key": self._api_key(), "count": 3},
        )
        if response is None:
            return []
        payload = self._parse_json(response)
        if not payload:
            return []
        items = payload if isinstance(payload, list) else [payload]
        retrieved_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        results = []
        for item in items:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "") or "").strip()
            if not title:
                continue
            date = str(item.get("date", "") or "")
            explanation = str(item.get("explanation", "") or "").strip()[:1000]
            url = str(item.get("hdurl") or item.get("url") or f"{self.base_url}/planetary/apod").strip()
            facts = [f"Astronomy Picture of the Day: {title}"]
            if date:
                facts.append(f"Date: {date}")
            if explanation:
                facts.append(explanation)
            results.append(
                self._make_result(
                    query=query,
                    title=title,
                    url=url,
                    content=". ".join(facts),
                    retrieved_at=retrieved_at,
                    entities=[title],
                    facts=facts,
                    dates=[date] if date else [],
                    citations=[{"title": title, "url": url}],
                    confidence=0.8,
                    raw_metadata={"type": "APOD", "date": date},
                )
            )
        return results

    def _neo(self, query: str) -> List[Any]:
        today = datetime.date.today().isoformat()
        response = self._http_get(
            f"{self.base_url}/neo/rest/v1/feed",
            params={
                "api_key": self._api_key(),
                "start_date": today,
                "end_date": today,
            },
        )
        if response is None:
            return []
        payload = self._parse_json(response)
        if not isinstance(payload, dict):
            return []
        near_earth = payload.get("near_earth_objects") or {}
        retrieved_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        results = []
        count = 0
        for _, objects in near_earth.items():
            for obj in objects:
                if not isinstance(obj, dict):
                    continue
                name = str(obj.get("name", "") or "").strip()
                if not name:
                    continue
                count += 1
                estimated = obj.get("estimated_diameter", {}).get("kilometers", {})
                diameter = None
                for key in ("estimated_diameter_max", "estimated_diameter_min"):
                    value = estimated.get(key)
                    if value:
                        diameter = f"{float(value):.1f} km"
                        break
                velocity = None
                velocity_data = (obj.get("close_approach_data") or [{}])[0].get("relative_velocity", {})
                if velocity_data:
                    velocity = f"{float(velocity_data.get('kilometers_per_hour', 0)):.0f} km/h"
                facts = [f"Near-Earth object: {name}"]
                if diameter:
                    facts.append(f"Estimated diameter: {diameter}")
                if velocity:
                    facts.append(f"Approach velocity: {velocity}")
                results.append(
                    self._make_result(
                        query=query,
                        title=f"NEO: {name}",
                        url=f"https://ssd.jpl.nasa.gov/tools/sbdb_lookup.html#/?sstr={name}",
                        content=". ".join(facts),
                        retrieved_at=retrieved_at,
                        entities=[name],
                        facts=facts,
                        dates=[today],
                        statistics=[diameter, velocity] if (diameter or velocity) else [],
                        confidence=0.8,
                        raw_metadata={"type": "NEO", "date": today},
                    )
                )
                if count >= 5:
                    break
            if count >= 5:
                break
        return results

    # -- interface -----------------------------------------------------------

    def search(self, query: str, **kwargs) -> List[Any]:
        results: List[Any] = []

        def apod_loader():
            return self._apod(query), {"query": query}

        def neo_loader():
            return self._neo(query), {"query": query}

        apod_results, _ = self._cached("apod", {"q": query}, apod_loader)
        neo_results, _ = self._cached("neo", {"q": query}, neo_loader)
        results.extend(apod_results)
        results.extend(neo_results)
        return results

    def _fetch_one(self, identifier: str, **kwargs) -> Optional[Any]:
        if str(identifier).lower().startswith("neo"):
            neo_results = self._neo(str(identifier))
            return neo_results[0] if neo_results else None
        apod_results = self._apod(str(identifier))
        return apod_results[0] if apod_results else None

    def get_metadata(self, identifier: str) -> Dict[str, Any]:
        return {"object": str(identifier), "source": self.name}

    def health_check(self) -> bool:
        response = self._http_get(
            f"{self.base_url}/planetary/apod",
            params={"api_key": self._api_key(), "count": 1},
        )
        return response is not None and response.status_code == 200