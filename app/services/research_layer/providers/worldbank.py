"""World Bank Data provider.

V2 indicator API: no key, no auth, no published rate limits; caching is
encouraged. Multiple indicators can be requested in one call using
``;``-separated codes; country codes are resolved through the cached country
list endpoint. Data is CC BY 4.0.
"""

from __future__ import annotations

import datetime
import re
from typing import Any, Dict, List, Optional

from app.services.research_layer.providers.base import ResearchProvider

SEARCH_LIMIT = 5
_INDICATOR_LABELS = {
    "NY.GDP.MKTP.CD": "GDP (current US$)",
    "NY.GDP.MKTP.PP.CD": "GDP (PPP, current intl $)",
    "SP.POP.TOTL": "Population, total",
    "SP.POP.GROW": "Population growth (annual %)",
    "SP.URB.TOTL.IN.ZS": "Urban population (% of total)",
    "SP.DYN.LE00.IN": "Life expectancy at birth (years)",
    "NY.GDP.PCAP.CD": "GDP per capita (current US$)",
    "NY.GDP.MKTP.KD.ZG": "GDP growth (annual %)",
}
_COUNTRY_PATTERN = re.compile(
    r"(?:gdp|population|life expectancy|urban population|gdp per capita|gdp growth)\s+of\s+(?:the\s+)?([a-z][a-z \-']{2,49})$",
    re.IGNORECASE,
)


class WorldBankProvider(ResearchProvider):
    name = "worldbank"
    display_name = "World Bank"
    base_url = "https://api.worldbank.org/v2"
    ttl_seconds = 7 * 24 * 3600
    authority = 0.9
    attribution = "Data from World Bank (CC BY 4.0)."
    requires_auth = False
    min_request_interval = 0.3
    max_concurrent = 2
    supports_batch_fetch = False

    def _parse_response(self, query: str, payload: Any, label: str) -> List[Any]:
        if not isinstance(payload, list) or len(payload) < 2:
            return []
        rows = payload[1]
        if not isinstance(rows, list):
            return []
        retrieved_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        results = []
        for row in rows[:SEARCH_LIMIT]:
            if not isinstance(row, dict):
                continue
            country = str(row.get("country", {}).get("value", "") if isinstance(row.get("country"), dict) else "")
            value = row.get("value")
            if value is None:
                continue
            period = str(row.get("date", ""))
            content = f"{label} for {country or 'selected country'} in {period}: {value}."
            results.append(
                self._make_result(
                    query=query,
                    title=f"{label} ({country or 'country'}, {period})",
                    url=f"{self.base_url}/country/{row.get('countryiso3code', '')}/indicator/{row.get('indicator', {}).get('id', '')}?format=json",
                    content=content,
                    retrieved_at=retrieved_at,
                    entities=[country] if country else [],
                    facts=[content.rstrip(".")],
                    dates=[period],
                    statistics=[str(value)],
                    confidence=0.9,
                    raw_metadata={
                        "indicator": label,
                        "period": period,
                        "country_code": row.get("countryiso3code"),
                        "license": "CC BY 4.0",
                    },
                )
            )
        return results

    def _fetch_indicators(self, country_code: str, indicators: List[str], query: str) -> List[Any]:
        if not country_code or not indicators:
            return []
        url = f"{self.base_url}/country/{country_code}/indicator/{';'.join(indicators)}"
        response = self._http_get(
            url,
            params={"format": "json", "per_page": SEARCH_LIMIT, "date": "2020:2025"},
        )
        if response is None:
            return []
        payload = self._parse_json(response)
        if not payload:
            return []
        label = " / ".join(_INDICATOR_LABELS.get(code, code) for code in indicators)
        return self._parse_response(query, payload, label)

    def _resolve_country_code(self, query: str) -> Optional[str]:
        match = _COUNTRY_PATTERN.match(query)
        if not match:
            return None
        country_name = match.group(1).strip()
        codes = self._country_codes()
        return codes.get(country_name.lower())

    def _country_codes(self) -> Dict[str, str]:
        cached = getattr(self, "_country_codes_cache", None)
        if cached is not None:
            return cached

        def loader():
            response = self._http_get(
                f"{self.base_url}/country",
                params={"format": "json", "per_page": 500},
            )
            if response is None:
                return [], {"error": True}
            payload = self._parse_json(response)
            codes: Dict[str, str] = {}
            if isinstance(payload, list) and len(payload) > 1:
                for row in payload[1]:
                    if isinstance(row, dict):
                        code = str(row.get("id", ""))
                        name = str(row.get("name", ""))
                        if code and name:
                            codes[name.lower()] = code
                            codes[str(row.get("iso2Code", "")).lower()] = code
            return [{"code": code, "name": name} for name, code in codes.items()], {"count": len(codes)}

        raw_codes, _ = self._cached_raw("country_codes", {"all": 1}, loader, ttl_seconds=30 * 24 * 3600)
        codes = {
            str(item.get("name", "")).lower(): str(item.get("code", ""))
            for item in raw_codes
            if isinstance(item, dict)
        }
        self._country_codes_cache = codes
        return codes

    # -- interface -----------------------------------------------------------

    def search(self, query: str, **kwargs) -> List[Any]:
        def loader():
            country_code = self._resolve_country_code(query)
            if not country_code:
                return [], {"query": query, "matched": False}
            indicators = list(_INDICATOR_LABELS.keys())
            results = self._fetch_indicators(country_code, indicators, query)
            return results, {"query": query, "country": country_code}

        results, _ = self._cached("search", {"q": query}, loader)
        return results

    def _fetch_one(self, identifier: str, **kwargs) -> Optional[Any]:
        parts = str(identifier).split("/")
        if len(parts) < 2:
            return None
        country_code, indicator = parts[0].upper(), parts[1]
        results = self._fetch_indicators(country_code, [indicator], identifier)
        return results[0] if results else None

    def get_metadata(self, identifier: str) -> Dict[str, Any]:
        return {"indicator": str(identifier), "source": self.name}

    def health_check(self) -> bool:
        response = self._http_get(
            f"{self.base_url}/country/usa/indicator/SP.POP.TOTL",
            params={"format": "json", "per_page": 1, "date": "2023:2023"},
        )
        return response is not None and response.status_code == 200