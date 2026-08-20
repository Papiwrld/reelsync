"""Wikidata provider (WB API; optional SPARQL).

Same Wikimedia usage policy as Wikipedia (User-Agent required, ~200 req/min
unauthenticated, <= 3 concurrent). Search uses ``wbsearchentities``; entities
are fetched in batches via ``wbgetentities``. A small set of high-value
property IDs is decoded into plain-language facts; sitelinks become citations.
SPARQL is only used when explicitly enabled and the query matches a strict,
non-injectable pattern.
"""

from __future__ import annotations

import datetime
import re
from typing import Any, Dict, List, Optional

from app.services.research_layer.providers.base import ResearchProvider

SEARCH_LIMIT = 5

# High-value property IDs decoded into facts.
_PROPERTY_LABELS = {
    "P31": "instance of",
    "P569": "date of birth",
    "P570": "date of death",
    "P17": "country",
    "P27": "country of citizenship",
    "P36": "capital",
    "P1082": "population",
    "P625": "coordinates",
    "P112": "founder",
    "P57": "director",
    "P175": "performer",
    "P856": "official website",
    "P106": "occupation",
    "P407": "language of work",
    "P495": "country of origin",
}
_DATE_PROPS = {"P569", "P570", "P571", "P576"}
_POPULATION_PROPS = {"P1082"}

_SPARQL_ALLOWED = re.compile(
    r"^(population of|gdp of|capital of)\s+([a-z][a-z \-']{2,49})$", re.IGNORECASE
)


def _render_value(value: Any) -> str:
    """Render a Wikidata datavalue into a short plain-language string."""
    if isinstance(value, str):
        return value
    if not isinstance(value, dict):
        return ""
    if "time" in value:
        return str(value["time"]).replace("+", "")[:10]
    if "amount" in value:
        amount = str(value["amount"])
        if "." in amount:
            try:
                amount = f"{float(amount):,.0f}"
            except ValueError:
                pass
        elif len(amount) >= 5 and amount.isdigit():
            try:
                amount = f"{int(amount):,}"
            except ValueError:
                pass
        return amount
    if "latitude" in value and "longitude" in value:
        return f"{value['latitude']}, {value['longitude']}"
    if "text" in value:
        return str(value["text"])
    if "id" in value:
        return str(value["id"])
    return ""


class WikidataProvider(ResearchProvider):
    name = "wikidata"
    display_name = "Wikidata"
    base_url = "https://www.wikidata.org/w/api.php"
    sparql_url = "https://query.wikidata.org/sparql"
    ttl_seconds = 30 * 24 * 3600
    authority = 0.85
    attribution = "Structured data from Wikidata (CC0)."
    requires_auth = False
    min_request_interval = 0.15
    max_concurrent = 3
    supports_batch_fetch = True
    max_fetch_batch = 20

    # -- internals -----------------------------------------------------------

    @staticmethod
    def _claim_values(claims: Dict[str, Any]) -> List[Dict[str, str]]:
        """Flatten claims into (property, value) pairs with labels."""
        values: List[Dict[str, str]] = []
        if not isinstance(claims, dict):
            return values
        for prop, statements in claims.items():
            label = _PROPERTY_LABELS.get(prop)
            if label is None:
                continue
            for statement in statements[:3]:
                if not isinstance(statement, dict):
                    continue
                mainsnak = statement.get("mainsnak", {})
                if not isinstance(mainsnak, dict):
                    continue
                datavalue = mainsnak.get("datavalue")
                if not isinstance(datavalue, dict):
                    continue
                value = datavalue.get("value")
                rendered = _render_value(value)
                if rendered:
                    values.append({"property": label, "value": rendered})
        return values

    @staticmethod
    def _sitelinks(entity: Dict[str, Any]) -> List[Dict[str, str]]:
        links: List[Dict[str, str]] = []
        sitelinks = entity.get("sitelinks") or {}
        for project, item in sitelinks.items():
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "")).strip()
            if title:
                links.append(
                    {
                        "title": f"{project}:{title}",
                        "url": f"https://{project.replace('_', '.')}/wiki/{title.replace(' ', '_')}",
                    }
                )
        return links

    def _entity_to_result(self, query: str, entity: Dict[str, Any], retrieved_at: str) -> Optional[Any]:
        entity_id = str(entity.get("id", "")).strip()
        labels = entity.get("labels") or {}
        label = ""
        for lang in ("en", "en-us", "en-gb"):
            item = labels.get(lang)
            if isinstance(item, dict) and item.get("value"):
                label = str(item["value"])
                break
        description = ""
        descriptions = entity.get("descriptions") or {}
        desc = descriptions.get("en")
        if isinstance(desc, dict):
            description = str(desc.get("value", ""))
        if not entity_id:
            return None
        title = label or entity_id
        claim_values = self._claim_values(entity.get("claims"))
        facts = [f"{item['property']}: {item['value']}" for item in claim_values]
        dates = [
            item["value"]
            for item in claim_values
            if item["property"] in _DATE_PROPS
        ]
        statistics = [
            item["value"]
            for item in claim_values
            if item["property"] in _POPULATION_PROPS
        ]
        locations = [
            item["value"]
            for item in claim_values
            if item["property"] in ("country", "coordinates")
        ]
        content = title
        if description:
            content += f". {description}"
        if facts:
            content += ". " + " ".join(facts)
        return self._make_result(
            query=query,
            title=title,
            url=f"https://www.wikidata.org/wiki/{entity_id}",
            content=content,
            retrieved_at=retrieved_at,
            entities=[title, entity_id],
            facts=facts,
            dates=dates,
            locations=locations,
            statistics=statistics,
            citations=self._sitelinks(entity),
            confidence=0.8,
            raw_metadata={"qid": entity_id, "description": description},
        )

    def _search_entities(self, query: str) -> List[Dict[str, Any]]:
        response = self._http_get(
            self.base_url,
            params={
                "action": "wbsearchentities",
                "search": query,
                "language": "en",
                "uselang": "en",
                "limit": SEARCH_LIMIT,
                "format": "json",
                "formatversion": 2,
            },
        )
        if response is None:
            return []
        payload = self._parse_json(response)
        if not payload:
            return []
        return [
            item for item in payload.get("search", []) if isinstance(item, dict)
        ]

    def _fetch_entities(self, entity_ids: List[str]) -> List[Dict[str, Any]]:
        if not entity_ids:
            return []
        response = self._http_get(
            self.base_url,
            params={
                "action": "wbgetentities",
                "ids": "\n".join(entity_ids),
                "props": "labels|descriptions|claims|sitelinks",
                "languages": "en",
                "format": "json",
                "formatversion": 2,
            },
        )
        if response is None:
            return []
        payload = self._parse_json(response)
        if not payload:
            return []
        entities = payload.get("entities") or {}
        return [entity for entity in entities.values() if isinstance(entity, dict)]

    def _search_to_results(self, query: str) -> tuple:
        retrieved_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        hits = self._search_entities(query)
        ids = [str(hit.get("id", "")).strip() for hit in hits if str(hit.get("id", "")).strip()]
        results = [
            result
            for entity in self._fetch_entities(ids)
            for result in [self._entity_to_result(query, entity, retrieved_at)]
            if result is not None
        ]
        return results, {"query": query}

    # -- interface -----------------------------------------------------------

    def search(self, query: str, **kwargs) -> List[Any]:
        results, _ = self._cached("search", {"q": query}, lambda: self._search_to_results(query))
        return results

    def _fetch_one(self, identifier: str, **kwargs) -> Optional[Any]:
        retrieved_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

        def loader():
            results = [
                result
                for entity in self._fetch_entities([identifier])
                for result in [self._entity_to_result(identifier, entity, retrieved_at)]
                if result is not None
            ]
            return results, {"id": identifier}

        results, _ = self._cached("fetch", {"id": identifier}, loader)
        return results[0] if results else None

    def _fetch_batch(self, identifiers: List[str], **kwargs) -> List[Any]:
        retrieved_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        query = str(kwargs.get("query", ""))
        results: List[Any] = []
        for chunk_start in range(0, len(identifiers), self.max_fetch_batch):
            chunk = identifiers[chunk_start : chunk_start + self.max_fetch_batch]
            for entity in self._fetch_entities(chunk):
                result = self._entity_to_result(query, entity, retrieved_at)
                if result is not None:
                    results.append(result)
        return results

    def get_metadata(self, identifier: str) -> Dict[str, Any]:
        if not str(identifier).startswith("Q"):
            return {}
        response = self._http_get(
            self.base_url,
            params={
                "action": "wbgetentities",
                "ids": identifier,
                "props": "labels|descriptions",
                "languages": "en",
                "format": "json",
                "formatversion": 2,
            },
        )
        if response is None:
            return {}
        payload = self._parse_json(response)
        entities = payload.get("entities") if payload else {}
        entity = next(iter(entities.values()), {}) if isinstance(entities, dict) else {}
        labels = entity.get("labels") or {}
        label = ""
        item = labels.get("en")
        if isinstance(item, dict):
            label = str(item.get("value", ""))
        return {"qid": identifier, "label": label, "source": self.name}

    def health_check(self) -> bool:
        response = self._http_get(
            self.base_url,
            params={"action": "wbsearchentities", "search": "test", "language": "en", "format": "json"},
        )
        return response is not None and response.status_code == 200

    # -- SPARQL (opt-in, strictly guarded) ------------------------------------

    def structured_facts(self, query: str) -> List[Any]:
        """Answer simple statistics questions via SPARQL when enabled.

        Only ``population of X`` / ``gdp of X`` / ``capital of X`` patterns are
        accepted; anything else returns an empty list. Gated by the
        ``enable_sparql`` setting (default off).
        """
        if not self.ctx.setting("enable_sparql", False):
            return []
        match = _SPARQL_ALLOWED.match(query)
        if not match:
            return []
        intent, subject = match.group(1).lower(), match.group(2)
        subject = subject.strip().replace("'", "\\'")
        if intent == "capital of":
            sparql = (
                "SELECT ?item ?itemLabel WHERE { "
                "?item wdt:P31 wd:Q515 . "
                "?item wdt:P17 ?country . "
                "?country rdfs:label ?countryLabel . "
                f"FILTER(LANG(?countryLabel) = 'en' && ?countryLabel = '{subject}') "
                "SERVICE wikibase:label { bd:serviceParam wikibase:language 'en' . } "
                "} LIMIT 5"
            )
        else:
            prop = "P2131" if intent == "gdp of" else "P1082"
            sparql = (
                "SELECT ?value WHERE { "
                "?country rdfs:label ?countryLabel . "
                f"FILTER(LANG(?countryLabel) = 'en' && ?countryLabel = '{subject}') "
                f"?country wdt:{prop} ?value . "
                "} ORDER BY DESC(?value) LIMIT 1"
            )
        # Non-injectable: subject only appears inside a quoted literal filter.
        results: List[Any] = []
        response = self._http_get(
            self.sparql_url,
            params={"query": sparql, "format": "json"},
            headers={"Accept": "application/sparql-results+json"},
        )
        if response is None:
            return results
        try:
            payload = response.json()
        except ValueError:
            return results
        for binding in (payload.get("results", {}).get("bindings", []) or []):
            value = (
                binding.get("value", {}).get("value")
                or binding.get("itemLabel", {}).get("value")
            )
            if not value:
                continue
            if intent == "capital of":
                facts = [f"capital of {subject}: {value}"]
                statistics = []
            else:
                facts = [f"{intent}: {value}"]
                statistics = [str(value)]
            results.append(
                self._make_result(
                    query=query,
                    title=f"{intent}: {subject}",
                    url="https://query.wikidata.org/",
                    content=f"{facts[0]}.",
                    retrieved_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    entities=[subject],
                    facts=facts,
                    statistics=statistics,
                    confidence=0.75,
                    raw_metadata={"sparql": True},
                )
            )
        return results