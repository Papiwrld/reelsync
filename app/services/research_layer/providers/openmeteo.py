"""Open-Meteo provider (geocoding + forecast).

Free (non-commercial) API, no key: 10,000 requests/day, 5,000/hour,
600/minute. Data is CC BY 4.0 and requires attribution. The forecast TTL is
configurable (``openmeteo_ttl_minutes``) since forecasts go stale quickly.
"""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional

from app.services.research_layer.providers.base import ResearchProvider

GEOCODE_LIMIT = 3


class OpenMeteoProvider(ResearchProvider):
    name = "openmeteo"
    display_name = "Open-Meteo"
    base_url = "https://geocoding-api.open-meteo.com/v1"
    forecast_url = "https://api.open-meteo.com/v1"
    ttl_seconds = 3600
    authority = 0.7
    attribution = "Weather data from Open-Meteo (CC BY 4.0)."
    requires_auth = False
    min_request_interval = 0.3
    max_concurrent = 2

    def _forecast_ttl(self) -> int:
        minutes = self.ctx.setting("openmeteo_ttl_minutes", 60)
        try:
            return max(5, int(minutes)) * 60
        except (TypeError, ValueError):
            return 3600

    def _geocode(self, query: str) -> List[Dict[str, Any]]:
        response = self._http_get(
            f"{self.base_url}/search",
            params={"name": query, "count": GEOCODE_LIMIT, "language": "en", "format": "json"},
        )
        if response is None:
            return []
        payload = self._parse_json(response)
        if not payload:
            return []
        results = payload.get("results")
        return [item for item in results if isinstance(item, dict)] if isinstance(results, list) else []

    def _forecast_for_location(self, query: str, location: Dict[str, Any], retrieved_at: str) -> Optional[Any]:
        latitude = location.get("latitude")
        longitude = location.get("longitude")
        name = str(location.get("name", "") or "").strip()
        if latitude is None or longitude is None:
            return None
        country = str(location.get("country", "") or "").strip()
        label = ", ".join(part for part in (name, country) if part)
        response = self._http_get(
            self.forecast_url,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m,precipitation",
                "forecast_days": 1,
                "timezone": "auto",
            },
        )
        if response is None:
            return None
        payload = self._parse_json(response)
        if not payload:
            return None
        current = payload.get("current") or {}
        units = payload.get("current_units") or {}
        temperature = current.get("temperature_2m")
        weather_code = current.get("weather_code")
        wind = current.get("wind_speed_10m")
        facts = [f"Current weather in {label}"]
        if temperature is not None:
            facts.append(f"Temperature: {temperature}{units.get('temperature_2m', '°C')}")
        if weather_code is not None:
            facts.append(f"Weather code: {weather_code}")
        if wind is not None:
            facts.append(f"Wind: {wind}{units.get('wind_speed_10m', 'km/h')}")
        return self._make_result(
            query=query,
            title=f"Weather in {label}",
            url=f"https://open-meteo.com/en/docs#latitude={latitude}&longitude={longitude}",
            content=". ".join(facts),
            retrieved_at=retrieved_at,
            entities=[label],
            facts=facts,
            dates=[str(current.get("time", ""))[:10]],
            locations=[f"{latitude},{longitude}"],
            statistics=[f"temperature {temperature}" if temperature is not None else None] if temperature is not None else [],
            confidence=0.7,
            raw_metadata={
                "latitude": latitude,
                "longitude": longitude,
                "time": current.get("time"),
            },
        )

    # -- interface -----------------------------------------------------------

    def search(self, query: str, **kwargs) -> List[Any]:
        retrieved_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

        def loader():
            locations = self._geocode(query)
            results = [
                result
                for location in locations
                for result in [self._forecast_for_location(query, location, retrieved_at)]
                if result is not None
            ]
            return results, {"query": query, "locations": len(locations)}

        results, _ = self._cached("search", {"q": query}, loader, ttl_seconds=self._forecast_ttl())
        return results

    def _fetch_one(self, identifier: str, **kwargs) -> Optional[Any]:
        parts = str(identifier).split(",")
        if len(parts) != 2:
            return None
        retrieved_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        location = {"latitude": float(parts[0]), "longitude": float(parts[1]), "name": "coordinates"}
        return self._forecast_for_location(identifier, location, retrieved_at)

    def get_metadata(self, identifier: str) -> Dict[str, Any]:
        return {"location": str(identifier), "source": self.name}

    def health_check(self) -> bool:
        response = self._http_get(
            f"{self.base_url}/search",
            params={"name": "berlin", "count": 1, "format": "json"},
        )
        return response is not None and response.status_code == 200