"""Research router: pick the minimal useful provider set for a query.

Classification is keyword-driven and deterministic. Only the providers that
add value for the detected topic category are selected — never all providers.
"""

from __future__ import annotations

import re
from typing import Dict, List

# Keyword groups per category. Values are regexes matched against the raw query.
_CATEGORY_PATTERNS: Dict[str, List[str]] = {
    "country_statistics": [
        r"\bgdp\b", r"\bgni\b", r"\binflation\b", r"\bunemployment\b",
        r"\bpopulation\b", r"\blife expectancy\b", r"\bgini\b",
        r"\bpoverty rate\b", r"\beconomic growth\b", r"\btrade deficit\b",
        r"\bexports?\b", r"\bimports?\b", r"\bstatistics of\b",
        r"\bstatistical data\b", r"\bindicators?\b", r"\bcapital of\b",
    ],
    "science": [
        r"\bresearch\b", r"\bscientific\b", r"\bpaper\b", r"\bpublication\b",
        r"\bdoi\b", r"\bjournal\b", r"\bacademic\b", r"\bphysics\b",
        r"\bchemistry\b", r"\bbiology\b", r"\bmathematics\b",
        r"\bexperiment\b", r"\bcitation\b", r"\bstudy\b", r"\btreatise\b",
        r"\btheorem\b", r"\bquantum\b", r"\bgenomics?\b", r"\bmachine learning paper\b",
        r"\bpeer[- ]reviewed\b", r"\bliterature review\b",
    ],
    "weather_climate": [
        r"\bweather\b", r"\bclimate\b", r"\bforecast\b", r"\btemperature\b",
        r"\brainfall\b", r"\bprecipitation\b", r"\bhumidity\b", r"\bwind speed\b",
        r"\bstorm\b", r"\bheatwave\b", r"\bsnowfall\b", r"\bweather forecast\b",
        r"\btropical cyclone\b", r"\buv index\b",
    ],
    "space": [
        r"\bnasa\b", r"\bspace\b", r"\bastronaut\b", r"\bplanet\b",
        r"\basteroid\b", r"\bgalaxy\b", r"\bmars\b", r"\bmoon\b",
        r"\bsolar system\b", r"\bastronomy\b", r"\borbit\b", r"\btelescope\b",
        r"\bcomet\b", r"\bnebula\b", r"\bsatellite\b", r"\bconstellation\b",
        r"\bcosmos\b", r"\binterstellar\b", r"\bbig bang\b",
    ],
    "history": [
        r"\bhistory\b", r"\bancient\b", r"\bcentury\b", r"\bempire\b",
        r"\bera\b", r"\bcivilization\b", r"\bmedieval\b", r"\brenaissance\b",
        r"\bfounded\b", r"\borigins?\b", r"\btimeline\b", r"\bthe .*? war\b",
        r"\brevolution\b", r"\bdynasty\b", r"\bkingdom\b", r"\barcheology\b",
        r"\bbce\b", r"\bad\b \d", r"\bmuseum\b", r"\bartifact\b",
    ],
    "location": [
        r"\bwhere\b", r"\blocated\b", r"\bcoordinates\b", r"\baddress\b",
        r"\bmap\b", r"\bneighborhood\b", r"\bdistrict\b", r"\blandmark\b",
        r"\bgeographic\b", r"\bregion of\b", r"\bcity of\b", r"\btown of\b",
        r"\bcapital city\b", r"\bgeography\b",
    ],
}

_CATEGORY_PROVIDERS: Dict[str, List[str]] = {
    "country_statistics": ["worldbank", "wikidata", "wikipedia"],
    "science": ["arxiv", "openalex", "crossref", "wikipedia"],
    "weather_climate": ["openmeteo", "wikipedia"],
    "space": ["nasa", "wikipedia", "openalex"],
    "history": ["wikipedia", "wikidata"],
    "location": ["nominatim", "wikidata"],
    "general": ["wikipedia", "wikidata"],
}


_CATEGORY_PRIORITY = [
    "country_statistics",
    "science",
    "weather_climate",
    "space",
    "history",
    "location",
    "general",
]


class ResearchRouter:
    """Deterministic query → category → provider selection."""

    def __init__(self) -> None:
        self._compiled = {
            category: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for category, patterns in _CATEGORY_PATTERNS.items()
        }

    def classify(self, query: str) -> str:
        if not query:
            return "general"
        matches: Dict[str, int] = {}
        for category, patterns in self._compiled.items():
            count = sum(1 for pattern in patterns if pattern.search(query))
            if count:
                matches[category] = count
        if not matches:
            return "general"
        # Most specific category by pattern density; ties broken by priority.
        best = max(
            matches,
            key=lambda category: (
                matches[category],
                -_CATEGORY_PRIORITY.index(category),
            ),
        )
        return best

    def select(self, query: str) -> List[str]:
        """Return the ordered provider names for this query (minimum set)."""
        category = self.classify(query)
        return list(_CATEGORY_PROVIDERS[category])


def get_router() -> ResearchRouter:
    return ResearchRouter()