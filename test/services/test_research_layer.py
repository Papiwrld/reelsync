"""Tests for the zero-key research layer.

All tests run fully offline: every network interaction is mocked at the
``requests.get`` boundary (or with stub providers). No real external API is
ever contacted.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

import requests

from app.services.research import (
    ResearchProviderError,
    ResearchStrategy,
    ZeroKeyResearchProvider,
    zero_key_metrics,
)
from app.services.research_layer.budget import BudgetContext
from app.services.research_layer.cache import (
    CachedSearch,
    cache_key,
    cleanup_expired,
    load_cached,
    save_cached,
)
from app.services.research_layer.dedupe import (
    JobDedupStore,
    jaccard_similarity,
    normalize_query,
)
from app.services.research_layer.http import ResearchHttpClient
from app.services.research_layer.metrics import ResearchMetrics, reset_metrics
from app.services.research_layer.quality import (
    corroborate_all,
    corroboration_count,
    relevance_score,
    score_result,
)
from app.services.research_layer.router import ResearchRouter
from app.services.research_layer.schema import ResearchResult
from app.services.research_layer.throttle import ProviderThrottle, RequestManager


def _response(payload=None, text="", status_code=200, headers=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.headers = headers or {}
    resp.json.return_value = payload
    resp.text = text
    return resp


def _make_result(source="wikipedia", title="Title", content="content", **kwargs):
    base = dict(
        source=source,
        source_url=f"https://example.com/{source}",
        title=title,
        retrieved_at="2026-01-01T00:00:00+00:00",
        query="query",
        content=content,
        entities=["entity"],
        facts=["fact one"],
        dates=["2020"],
        locations=["london"],
        statistics=["1 million"],
        citations=[{"title": "t", "url": "u"}],
        confidence=0.8,
        raw_metadata={"attribution": "attributed", "retrieved_ts": time.time()},
    )
    base.update(kwargs)
    return ResearchResult(**base)


class FakeClock:
    def __init__(self, start=1000.0):
        self.now = start
        self.slept = []

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


def _cache_dir_fixture(testcase):
    tmp = tempfile.mkdtemp(prefix="zk-test-")
    patcher = patch("app.services.research_layer.cache._cache_dir", return_value=tmp)
    patcher.start()
    testcase.addCleanup(patcher.stop)
    testcase.addCleanup(lambda: shutil.rmtree(tmp, ignore_errors=True))
    return tmp


class RouterTests(unittest.TestCase):
    def test_classify_categories(self):
        router = ResearchRouter()
        cases = {
            "population of France": "country_statistics",
            "weather in Berlin": "weather_climate",
            "ancient Rome history": "history",
            "nasa asteroid news": "space",
            "machine learning paper": "science",
            "coordinates of a landmark": "location",
            "something totally unrelated": "general",
        }
        for query, expected in cases.items():
            self.assertEqual(router.classify(query), expected, query)

    def test_select_minimal_provider_set(self):
        router = ResearchRouter()
        self.assertEqual(router.select("population of France"), ["worldbank", "wikidata", "wikipedia"])
        self.assertEqual(router.select("weather in Berlin"), ["openmeteo", "wikipedia"])
        self.assertNotIn("nasa", router.select("population of France"))

    def test_select_general_never_all_providers(self):
        router = ResearchRouter()
        selected = router.select("what is the capital")
        self.assertEqual(selected, ["wikipedia", "wikidata"])


class DedupeTests(unittest.TestCase):
    def test_normalize_query(self):
        self.assertEqual(
            normalize_query("  Tell me about Ancient Rome's History! "),
            "ancient rome",
        )

    def test_jaccard(self):
        self.assertEqual(jaccard_similarity("population of france", "population of france"), 1.0)
        self.assertGreater(jaccard_similarity("history of rome", "rome history"), 0.5)
        self.assertEqual(jaccard_similarity("gdp of japan", "weather in berlin"), 0.0)

    def test_store_exact_duplicate(self):
        store = JobDedupStore()
        self.assertIsNone(store.find("client", "research", "population of france"))
        store.store("client", "research", "population of france", {"seen": True})
        self.assertIsNotNone(store.find("client", "research", "population of france"))

    def test_store_fuzzy_duplicate(self):
        store = JobDedupStore()
        store.store("client", "research", "population of france statistics", {"seen": True})
        self.assertIsNotNone(store.find("client", "research", "france population statistics data"))

    def test_store_scoped_by_provider(self):
        store = JobDedupStore()
        store.store("a", "search", "history of rome", {"seen": True})
        self.assertIsNone(store.find("b", "search", "history of rome"))


class BudgetTests(unittest.TestCase):
    def test_total_cap(self):
        budget = BudgetContext(max_external_requests=2, max_requests_per_provider=10)
        self.assertTrue(budget.acquire("wikipedia"))
        self.assertTrue(budget.acquire("wikipedia"))
        self.assertFalse(budget.acquire("wikipedia"))
        self.assertEqual(budget.blocked_count(), 1)
        self.assertEqual(budget.remaining_total(), 0)

    def test_per_provider_cap(self):
        budget = BudgetContext(max_external_requests=100, max_requests_per_provider=2)
        self.assertTrue(budget.acquire("wikipedia"))
        self.assertTrue(budget.acquire("wikipedia"))
        self.assertFalse(budget.acquire("wikipedia"))
        self.assertTrue(budget.acquire("wikidata"))

    def test_from_settings(self):
        budget = BudgetContext.from_settings(
            {"max_external_requests": 7, "max_requests_per_provider": 3, "cache_enabled": False}
        )
        self.assertEqual(budget.max_external_requests, 7)
        self.assertEqual(budget.max_requests_per_provider, 3)
        self.assertFalse(budget.cache_enabled)

    def test_snapshot(self):
        budget = BudgetContext(max_external_requests=5, max_requests_per_provider=5)
        budget.acquire("nasa")
        snapshot = budget.snapshot()
        self.assertEqual(snapshot["used_total"], 1)
        self.assertEqual(snapshot["remaining_total"], 4)


class MetricsTests(unittest.TestCase):
    def setUp(self):
        reset_metrics()

    def test_counts_and_snapshot(self):
        metrics = ResearchMetrics()
        metrics.increment("cache_hits", 3)
        metrics.increment("budget_blocked", 2)
        metrics.record_provider_request("wikipedia")
        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["cache_hits"], 3)
        self.assertEqual(snapshot["total_requests"], 1)
        self.assertEqual(snapshot["requests_by_provider"], {"wikipedia": 1})
        self.assertEqual(snapshot["requests_avoided"], 5)

    def test_requests_avoided_components(self):
        metrics = ResearchMetrics()
        metrics.increment("cache_hits", 4)
        metrics.increment("coalesced_requests", 1)
        metrics.increment("duplicate_queries_prevented", 2)
        metrics.increment("batched_items_saved", 10)
        metrics.increment("budget_blocked", 3)
        self.assertEqual(metrics.requests_avoided(), 20)

    def test_rate_limit_recording(self):
        metrics = ResearchMetrics()
        metrics.record_rate_limit(2.5)
        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["rate_limit_events"], 1)
        self.assertEqual(snapshot["retry_after_seconds"], 2.5)


class CacheTests(unittest.TestCase):
    def setUp(self):
        self._tmp_cache_dir = _cache_dir_fixture(self)
        reset_metrics()

    def test_round_trip(self):
        results = [{"source": "wikipedia", "title": "T"}]
        save_cached("wikipedia", "search", {"q": "x"}, results, {"meta": 1}, 3600)
        entry = load_cached("wikipedia", "search", {"q": "x"})
        self.assertIsNotNone(entry)
        self.assertEqual(entry["results"], results)
        self.assertEqual(entry["metadata"], {"meta": 1})

    def test_expired_entry_removed(self):
        now = 1000.0
        save_cached("wikipedia", "search", {"q": "x"}, [{"a": 1}], {}, 3600, now=now)
        self.assertIsNotNone(load_cached("wikipedia", "search", {"q": "x"}, now=now))
        self.assertIsNone(load_cached("wikipedia", "search", {"q": "x"}, now=now + 3601))

    def test_empty_results_ttl_capped(self):
        now = 1000.0
        save_cached("wikipedia", "search", {"q": "empty"}, [], {}, 30 * 24 * 3600, now=now)
        entry = load_cached("wikipedia", "search", {"q": "empty"}, now=now + 3600)
        self.assertIsNone(entry)

    def test_corrupt_entry_ignored(self):
        key = cache_key("wikipedia", "search", {"q": "x"})
        path = os.path.join(_cache_dir_fixture(self), f"zk-{key}.json")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("{not json")
        self.assertIsNone(load_cached("wikipedia", "search", {"q": "x"}))

    def test_cleanup_expired(self):
        now = time.time()
        save_cached("a", "search", {"q": "old"}, [{"x": 1}], {}, 60, now=now - 100)
        old_path = os.path.join(
            self._tmp_cache_dir, f"zk-{cache_key('a', 'search', {'q': 'old'})}.json"
        )
        if os.path.exists(old_path):
            os.utime(old_path, (now - 60, now - 60))
        save_cached("a", "search", {"q": "new"}, [{"x": 1}], {}, 3600, now=now)
        removed = cleanup_expired(force=True)
        self.assertGreaterEqual(removed, 1)

    def test_coalescing_concurrent_loaders(self):
        metrics = ResearchMetrics()
        hits = threading.Event()
        calls = []

        def loader():
            calls.append(1)
            time.sleep(0.1)
            return [{"source": "wikipedia", "title": "T"}], {}

        def worker():
            cached = CachedSearch("wikipedia", "search", {"q": "same"}, 3600, metrics)
            cached.get(loader)
            hits.set()

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        self.assertEqual(len(calls), 1)
        self.assertGreaterEqual(metrics.snapshot()["coalesced_requests"], 0)

    def test_cached_search_hit_after_miss(self):
        metrics = ResearchMetrics()
        calls = []

        def loader():
            calls.append(1)
            return [{"source": "wikipedia", "title": "T"}], {}

        first = CachedSearch("wikipedia", "search", {"q": "x"}, 3600, metrics)
        first.get(loader)
        second = CachedSearch("wikipedia", "search", {"q": "x"}, 3600, metrics)
        results, _ = second.get(loader)
        self.assertEqual(len(calls), 1)
        self.assertEqual(results, [{"source": "wikipedia", "title": "T"}])
        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["cache_hits"], 1)
        self.assertEqual(snapshot["cache_misses"], 1)


class ThrottleTests(unittest.TestCase):
    def test_minimum_interval(self):
        clock = FakeClock()
        with (
            patch("app.services.research_layer.throttle.time.time", clock),
            patch("app.services.research_layer.throttle.time.sleep", clock.sleep),
        ):
            throttle = ProviderThrottle("nominatim", interval=1.0, max_concurrent=1)
            throttle.wait_until_ready()
            self.assertEqual(clock.slept, [])
            throttle._release_slot()
            throttle.wait_until_ready()
            self.assertGreaterEqual(clock.slept[-1], 1.0)

    def test_max_concurrent(self):
        clock = FakeClock()
        throttle = ProviderThrottle("arxiv", interval=0.0, max_concurrent=1)
        with patch("app.services.research_layer.throttle.time.time", clock):
            throttle.wait_until_ready()
            self.assertEqual(throttle._semaphore._value, 0)
            throttle._release_slot()
            self.assertEqual(throttle._semaphore._value, 1)

    def test_rate_limit_cooldown(self):
        clock = FakeClock()
        with (
            patch("app.services.research_layer.throttle.time.time", clock),
            patch("app.services.research_layer.throttle.time.sleep", clock.sleep),
        ):
            throttle = ProviderThrottle("openalex", interval=0.2, max_concurrent=1)
            throttle.mark_rate_limited(retry_after=5.0)
            throttle.wait_until_ready()
            self.assertGreaterEqual(clock.slept[-1], 5.0)
            throttle._release_slot()

    def test_request_manager_snapshot(self):
        manager = RequestManager()
        manager.register("wikipedia", 0.1, 3)
        snapshot = manager.snapshot()
        self.assertEqual(snapshot["wikipedia"]["interval"], 0.1)
        self.assertEqual(snapshot["wikipedia"]["max_concurrent"], 3)


class HttpTests(unittest.TestCase):
    def setUp(self):
        reset_metrics()

    def test_ua_header_and_mailto(self):
        with patch("app.services.research_layer.http.requests.get", return_value=_response({"ok": True})) as get:
            client = ResearchHttpClient(contact_email="me@example.com")
            client.get("wikipedia", "https://example.com")
        _, kwargs = get.call_args
        self.assertIn("ReelSync", kwargs["headers"]["User-Agent"])
        self.assertIn("me@example.com", kwargs["headers"]["User-Agent"])

    def test_429_retry_after_honored(self):
        clock = FakeClock()
        metrics = ResearchMetrics()
        manager = RequestManager()
        manager.register("openalex", 0.1, 1)
        with (
            patch("app.services.research_layer.http.requests.get", side_effect=[
                _response(status_code=429, headers={"Retry-After": "4"}),
                _response({"ok": True}),
            ]),
            patch("app.services.research_layer.http.time.sleep", clock.sleep),
        ):
            client = ResearchHttpClient(
                max_retries=3, request_manager=manager, metrics=metrics, sleep=clock.sleep
            )
            response = client.get("openalex", "https://example.com")
        self.assertIsNotNone(response)
        self.assertEqual(response.status_code, 200)
        self.assertGreaterEqual(clock.slept[0], 4.0)
        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["rate_limit_events"], 1)
        self.assertEqual(snapshot["retry_after_seconds"], 4.0)

    def test_5xx_retry_then_success(self):
        clock = FakeClock()
        with (
            patch("app.services.research_layer.http.requests.get", side_effect=[
                _response(status_code=503),
                _response({"ok": True}),
            ]),
            patch("app.services.research_layer.http.time.sleep", clock.sleep),
        ):
            client = ResearchHttpClient(max_retries=3, sleep=clock.sleep)
            response = client.get("wikipedia", "https://example.com")
        self.assertIsNotNone(response)
        self.assertGreater(len(clock.slept), 0)

    def test_4xx_returns_none_no_retry(self):
        metrics = ResearchMetrics()
        with patch(
            "app.services.research_layer.http.requests.get",
            side_effect=[_response(status_code=403), _response({"ok": True})],
        ) as get:
            client = ResearchHttpClient(max_retries=3, metrics=metrics)
            response = client.get("wikipedia", "https://example.com")
        self.assertIsNone(response)
        self.assertEqual(get.call_count, 1)
        self.assertEqual(metrics.snapshot()["failed_requests"], 1)

    def test_request_exception_retries_then_none(self):
        clock = FakeClock()
        metrics = ResearchMetrics()
        with (
            patch(
                "app.services.research_layer.http.requests.get",
                side_effect=[requests.ConnectionError("boom"), requests.ConnectionError("boom"), _response({"ok": True})],
            ),
            patch("app.services.research_layer.http.time.sleep", clock.sleep),
        ):
            client = ResearchHttpClient(max_retries=3, metrics=metrics, sleep=clock.sleep)
            response = client.get("wikipedia", "https://example.com")
        self.assertIsNotNone(response)
        self.assertEqual(metrics.snapshot()["failed_requests"], 2)

    def test_exhausted_retries_returns_none(self):
        clock = FakeClock()
        metrics = ResearchMetrics()
        with (
            patch(
                "app.services.research_layer.http.requests.get",
                side_effect=requests.ConnectionError("always down"),
            ),
            patch("app.services.research_layer.http.time.sleep", clock.sleep),
        ):
            client = ResearchHttpClient(max_retries=2, metrics=metrics, sleep=clock.sleep)
            response = client.get("wikipedia", "https://example.com")
        self.assertIsNone(response)
        self.assertEqual(metrics.snapshot()["failed_requests"], 3)


class QualityTests(unittest.TestCase):
    def test_relevance_score(self):
        result = _make_result(title="Population of France", content="population france data")
        self.assertEqual(relevance_score("population of france", result), 1.0)
        unrelated = _make_result(title="Weather in Berlin", content="cloudy rain")
        self.assertLess(relevance_score("population of france", unrelated), 0.3)

    def test_corroboration_count_requires_different_sources(self):
        left = _make_result(source="wikipedia", content=" ".join(f"token{i}" for i in range(20)))
        same = _make_result(source="wikipedia", content=" ".join(f"token{i}" for i in range(20)))
        other = _make_result(source="wikidata", content=" ".join(f"token{i}" for i in range(20)))
        self.assertEqual(corroboration_count(left, [same]), 0)
        self.assertEqual(corroboration_count(left, [other]), 1)

    def test_score_result_weights(self):
        result = _make_result(content="c" * 600)
        low = score_result("query", result, authority=0.3, ttl_seconds=3600, corroborations=0)
        high = score_result("query", result, authority=0.95, ttl_seconds=3600, corroborations=2)
        self.assertGreater(high, low)

    def test_corroborate_all_scores_all(self):
        results = [
            _make_result(source="wikipedia", title="A", content=" ".join(["common"] * 20)),
            _make_result(source="wikidata", title="A2", content=" ".join(["common"] * 20)),
        ]
        scores = corroborate_all("common query", results, {"wikipedia": 0.9, "wikidata": 0.85}, {"wikipedia": 3600, "wikidata": 3600})
        self.assertEqual(set(scores.keys()), {r.source_url for r in results})


class ProviderTests(unittest.TestCase):
    def setUp(self):
        reset_metrics()
        _cache_dir_fixture(self)

    def _make_client(self):
        from app.services.research_layer import ResearchClient

        return ResearchClient(settings={"max_retries": 0, "openalex_api_key": "secret-key"})

    def test_wikipedia_search(self):
        client = self._make_client()
        provider = client.providers["wikipedia"]
        payload = {
            "query": {"search": [{"title": "France"}, {"title": "France (disambiguation)"}]},
            "pages": [],
        }
        side_effects = [
            _response(payload),  # title search
            _response(
                {
                    "query": {
                        "pages": [
                            {
                                "title": "France",
                                "extract": "France is a country in Europe. Its population is large.",
                                "fullurl": "https://en.wikipedia.org/wiki/France",
                                "pageid": 42,
                            }
                        ]
                    }
                }
            ),  # extract batch
        ]
        with patch("app.services.research_layer.http.requests.get", side_effect=side_effects):
            results = provider.search("france")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].source, "wikipedia")
        self.assertEqual(results[0].title, "France")
        self.assertEqual(results[0].source_url, "https://en.wikipedia.org/wiki/France")
        self.assertEqual(results[0].raw_metadata["license"], "CC BY-SA 4.0")
        self.assertIn("France is a country", results[0].content)

    def test_wikipedia_empty_response(self):
        client = self._make_client()
        with patch("app.services.research_layer.http.requests.get", return_value=_response({"query": {"search": []}})):
            results = client.providers["wikipedia"].search("nonexistent topic xyz")
        self.assertEqual(results, [])

    def test_wikipedia_health_check(self):
        client = self._make_client()
        with patch("app.services.research_layer.http.requests.get", return_value=_response({})):
            self.assertTrue(client.providers["wikipedia"].health_check())
        with patch("app.services.research_layer.http.requests.get", return_value=_response(status_code=500)):
            self.assertFalse(client.providers["wikipedia"].health_check())

    def test_wikidata_claims_decoded(self):
        client = self._make_client()
        provider = client.providers["wikidata"]
        entity = {
            "id": "Q142",
            "labels": {"en": {"value": "France"}},
            "descriptions": {"en": {"value": "country in Europe"}},
            "claims": {
                "P1082": [{"mainsnak": {"datavalue": {"value": {"amount": "68000000"}}}}],
                "P17": [{"mainsnak": {"datavalue": {"value": {"id": "Q142"}}}}],
            },
            "sitelinks": {"enwiki": {"title": "France"}},
        }
        with patch(
            "app.services.research_layer.http.requests.get",
            side_effect=[
                _response({"search": [{"id": "Q142", "label": "France"}]}),
                _response({"entities": {"Q142": entity}}),
            ],
        ):
            results = provider.search("france")
        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result.source, "wikidata")
        self.assertEqual(result.source_url, "https://www.wikidata.org/wiki/Q142")
        self.assertIn("population: 68,000,000", result.facts)
        self.assertIn("Q142", result.entities)
        self.assertEqual(len(result.citations), 1)

    def test_wikidata_sparql_guarded(self):
        from app.services.research_layer import ResearchClient

        client = self._make_client()
        provider = client.providers["wikidata"]
        with patch("app.services.research_layer.http.requests.get") as get:
            self.assertEqual(provider.structured_facts("population of France"), [])
            self.assertEqual(provider.structured_facts("drop table users"), [])
            get.assert_not_called()
        client2 = ResearchClient(settings={"enable_sparql": True, "max_retries": 0})
        with patch(
            "app.services.research_layer.http.requests.get",
            return_value=_response(
                {"results": {"bindings": [{"value": {"value": "68000000"}}]}}
            ),
        ) as get:
            results = client2.providers["wikidata"].structured_facts("population of France")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].statistics, ["68000000"])

    def test_openalex_search_and_key_header(self):
        client = self._make_client()
        payload = {
            "results": [
                {
                    "id": "https://openalex.org/W1",
                    "title": "A Study",
                    "doi": "https://doi.org/10.1000/xyz",
                    "publication_year": 2023,
                    "cited_by_count": 12,
                    "authorships": [{"author": {"display_name": "Jane Doe"}}],
                }
            ]
        }
        with patch("app.services.research_layer.http.requests.get", return_value=_response(payload)) as get:
            results = client.providers["openalex"].search("study")
        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result.source, "openalex")
        self.assertEqual(result.title, "A Study")
        self.assertEqual(result.source_url, "https://doi.org/10.1000/xyz")
        self.assertIn("Jane Doe", result.entities)
        _, kwargs = get.call_args
        self.assertEqual(kwargs["headers"].get("X-API-Key"), "secret-key")

    def test_crossref_search_and_mailto(self):
        from app.services.research_layer import ResearchClient

        client = ResearchClient(settings={"contact_email": "me@example.com", "max_retries": 0})
        payload = {
            "message": {
                "items": [
                    {
                        "DOI": "10.1000/abc",
                        "title": ["A Paper"],
                        "author": [{"given": "John", "family": "Smith"}],
                        "container-title": ["Journal X"],
                        "issued": {"date-parts": [[2021]]},
                        "is-referenced-by-count": 5,
                    }
                ]
            }
        }
        with patch("app.services.research_layer.http.requests.get", return_value=_response(payload)) as get:
            results = client.providers["crossref"].search("paper")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].source, "crossref")
        self.assertEqual(results[0].source_url, "https://doi.org/10.1000/abc")
        _, kwargs = get.call_args
        self.assertEqual(kwargs["params"]["mailto"], "me@example.com")

    def test_arxiv_xml_parsing(self):
        client = self._make_client()
        xml_text = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2401.00001v1</id>
    <title>  A Great Paper  </title>
    <author><name>Alice</name></author>
    <author><name>Bob</name></author>
    <published>2024-01-02T00:00:00Z</published>
    <summary>  Abstract content here.  </summary>
    <link href="http://arxiv.org/abs/2401.00001v1"/>
  </entry>
</feed>"""
        with patch("app.services.research_layer.http.requests.get", return_value=_response(text=xml_text)):
            results = client.providers["arxiv"].search("paper")
        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result.source, "arxiv")
        self.assertEqual(result.title, "A Great Paper")
        self.assertEqual(result.entities, ["A Great Paper", "Alice", "Bob"])
        self.assertEqual(result.dates, ["2024"])

    def test_arxiv_malformed_xml(self):
        client = self._make_client()
        with patch("app.services.research_layer.http.requests.get", return_value=_response(text="<not xml")):
            results = client.providers["arxiv"].search("paper")
        self.assertEqual(results, [])

    def test_worldbank_indicators(self):
        client = self._make_client()
        with patch(
            "app.services.research_layer.http.requests.get",
            side_effect=[
                _response([{"total": 1}, [{"id": "FRA", "name": "France", "iso2Code": "FR"}]]),
                _response(
                    [{"total": 1},
                     [{"country": {"value": "France"}, "countryiso3code": "FRA",
                       "indicator": {"id": "SP.POP.TOTL"}, "date": "2023", "value": 68000000}]]
                ),
            ],
        ):
            results = client.providers["worldbank"].search("population of France")
        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result.source, "worldbank")
        self.assertEqual(result.statistics, ["68000000"])
        self.assertIn("CC BY 4.0", result.raw_metadata["license"])

    def test_worldbank_unmatched_query(self):
        client = self._make_client()
        with patch("app.services.research_layer.http.requests.get") as get:
            results = client.providers["worldbank"].search("some random question")
        self.assertEqual(results, [])
        get.assert_not_called()

    def test_nasa_apod(self):
        client = self._make_client()
        with patch(
            "app.services.research_layer.http.requests.get",
            return_value=_response(
                [{"title": "Galaxy", "date": "2026-01-01", "explanation": "A nice galaxy.",
                  "url": "https://apod.nasa.gov/x.jpg"}]
            ),
        ) as get:
            results = client.providers["nasa"].search("space galaxy")
        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result.source, "nasa")
        self.assertEqual(result.title, "Galaxy")
        _, kwargs = get.call_args
        self.assertEqual(kwargs["params"]["api_key"], "DEMO_KEY")

    def test_nasa_configured_key(self):
        from app.services.research_layer import ResearchClient

        client = ResearchClient(settings={"nasa_api_key": "mykey", "max_retries": 0})
        with patch("app.services.research_layer.http.requests.get", return_value=_response([])) as get:
            client.providers["nasa"].search("space")
        _, kwargs = get.call_args
        self.assertEqual(kwargs["params"]["api_key"], "mykey")

    def test_openmeteo_geocode_and_forecast(self):
        client = self._make_client()
        with patch(
            "app.services.research_layer.http.requests.get",
            side_effect=[
                _response({"results": [{"name": "Berlin", "country": "Germany",
                                        "latitude": 52.5, "longitude": 13.4}]}),
                _response({"current": {"temperature_2m": 18.0, "weather_code": 2, "time": "2026-08-18T12:00"},
                           "current_units": {"temperature_2m": "°C"}}),
            ],
        ):
            results = client.providers["openmeteo"].search("weather in Berlin")
        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result.source, "openmeteo")
        self.assertIn("18.0", result.facts[1])
        self.assertEqual(result.raw_metadata["attribution"], "Weather data from Open-Meteo (CC BY 4.0).")

    def test_openmeteo_no_geocode_results(self):
        client = self._make_client()
        with patch("app.services.research_layer.http.requests.get", return_value=_response({})):
            results = client.providers["openmeteo"].search("nowhere town xyz")
        self.assertEqual(results, [])

    def test_nominatim_search(self):
        client = self._make_client()
        with patch(
            "app.services.research_layer.http.requests.get",
            return_value=_response(
                [{"display_name": "Eiffel Tower, Paris, France", "place_id": "1",
                  "osm_type": "w", "osm_id": "123", "lat": "48.85", "lon": "2.29",
                  "category": "tourism", "type": "attraction"}]
            ),
        ) as get:
            results = client.providers["nominatim"].search("Eiffel Tower location")
        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result.source, "nominatim")
        self.assertIn("© OpenStreetMap contributors", result.raw_metadata["attribution"])
        _, kwargs = get.call_args
        self.assertIn("Referer", kwargs["headers"])

    def test_nominatim_disabled(self):
        from app.services.research_layer import ResearchClient

        client = ResearchClient(settings={"nominatim_enabled": False, "max_retries": 0})
        with patch("app.services.research_layer.http.requests.get") as get:
            results = client.providers["nominatim"].search("Eiffel Tower")
        self.assertEqual(results, [])
        get.assert_not_called()

    def test_wikipedia_batch_fetch_single_request(self):
        client = self._make_client()
        provider = client.providers["wikipedia"]
        with patch(
            "app.services.research_layer.http.requests.get",
            return_value=_response(
                {"query": {"pages": [
                    {"title": "A", "extract": "about a", "fullurl": "https://en.wikipedia.org/wiki/A"},
                    {"title": "B", "extract": "about b", "fullurl": "https://en.wikipedia.org/wiki/B"},
                ]}}
            ),
        ) as get:
            results = provider.fetch_many(["A", "B"])
        self.assertEqual(len(results), 2)
        self.assertEqual(get.call_count, 1)
        params = get.call_args.kwargs["params"]
        self.assertEqual(params["titles"], "A\nB")

    def test_health_checks_never_raise(self):
        client = self._make_client()
        with patch("app.services.research_layer.http.requests.get", side_effect=Exception("down")):
            status = client.health_checks()
        self.assertEqual(len(status), len(client.providers))
        self.assertTrue(all(value is False for value in status.values()))


class ClientPipelineTests(unittest.TestCase):
    def setUp(self):
        reset_metrics()

    def test_research_pipeline_with_stub_provider(self):
        from app.services.research_layer import ResearchClient

        client = ResearchClient(settings={"max_retries": 0})

        class StubProvider:
            name = "stub"
            authority = 0.9
            ttl_seconds = 3600

            def search(self, query):
                return [_make_result(source="stub", title="Result A", content="content here")]

        stub = StubProvider()
        with patch.object(client, "_candidates", return_value=[stub]):
            results = client.research("What is the population of France?")
        self.assertEqual(len(results), 1)
        package = client.last_package()
        self.assertIsNotNone(package)
        self.assertEqual(package.normalized_query, "population france")
        self.assertIn("stub", package.providers)
        self.assertEqual(len(package.claims), 1)
        self.assertEqual(package.claims[0].sources[0].source, "stub")

    def test_research_duplicate_query_prevented(self):
        from app.services.research_layer import ResearchClient

        client = ResearchClient(settings={"max_retries": 0})

        class StubProvider:
            name = "stub"
            authority = 0.9
            ttl_seconds = 3600

            def search(self, query):
                return [_make_result(source="stub", title="Result")]

        stub = StubProvider()
        with patch.object(client, "_candidates", return_value=[stub]):
            first = client.research("population of france")
            second = client.research("population of France!")
        self.assertEqual(len(first), 1)
        self.assertEqual(second, [])
        self.assertEqual(client.snapshot_metrics()["duplicate_queries_prevented"], 1)

    def test_research_provider_failure_does_not_kill_run(self):
        from app.services.research_layer import ResearchClient

        client = ResearchClient(settings={"max_retries": 0})

        class FailingProvider:
            name = "fail"
            authority = 0.9
            ttl_seconds = 3600

            def search(self, query):
                raise RuntimeError("boom")

        with patch.object(client, "_candidates", return_value=[FailingProvider()]):
            results = client.research("some topic")
        self.assertEqual(results, [])
        self.assertIsNotNone(client.last_package())

    def test_sorted_by_quality_score(self):
        from app.services.research_layer import ResearchClient

        client = ResearchClient(settings={"max_retries": 0})

        class StubProvider:
            name = "stub"
            authority = 0.9
            ttl_seconds = 3600

            def search(self, query):
                return [
                    _make_result(source="stub", title="Exact match", content="population france data"),
                    _make_result(source="stub", title="Unrelated", content="weather berlin cloudy"),
                ]

        stub = StubProvider()
        with patch.object(client, "_candidates", return_value=[stub]):
            results = client.research("population of france")
        self.assertEqual(results[0].title, "Exact match")

    def test_batch_fetch(self):
        from app.services.research_layer import ResearchClient

        client = ResearchClient(settings={"max_retries": 0})
        with patch(
            "app.services.research_layer.http.requests.get",
            return_value=_response(
                {"query": {"pages": [{"title": "A", "extract": "about a", "fullurl": "u"}]}}
            ),
        ):
            results = client.batch_fetch("wikipedia", ["A"])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "A")

    def test_batch_fetch_unknown_provider(self):
        from app.services.research_layer import ResearchClient

        client = ResearchClient(settings={"max_retries": 0})
        self.assertEqual(client.batch_fetch("nope", ["A"]), [])


class BridgeTests(unittest.TestCase):
    def setUp(self):
        reset_metrics()

    def test_disabled_provider_raises(self):
        provider = ZeroKeyResearchProvider()
        with self.assertRaises(ResearchProviderError):
            provider.discover("topic", ResearchStrategy(), {"zero_key_enabled": False})

    def test_discover_maps_results(self):

        fake_result = _make_result(
            source="wikipedia",
            title="France",
            source_url="https://en.wikipedia.org/wiki/France",
            content="France is a country.",
            raw_metadata={"attribution": "Content from Wikipedia (CC BY-SA 4.0)."},
        )
        fake_client = MagicMock()
        fake_client.research.return_value = [fake_result]
        fake_client.snapshot_metrics.return_value = {"total_requests": 1}
        with patch("app.services.research_layer.ResearchClient", return_value=fake_client):
            provider = ZeroKeyResearchProvider()
            items = provider.discover("france", ResearchStrategy(), {"zero_key_enabled": True})
        self.assertEqual(len(items), 1)
        item = items[0]
        self.assertEqual(item["title"], "France")
        self.assertEqual(item["url"], "https://en.wikipedia.org/wiki/France")
        self.assertEqual(item["tier"], "secondary")
        self.assertIs(item["is_primary"], False)
        self.assertIn("CC BY-SA 4.0", item["note"])
        self.assertEqual(item["provenance"], "zero_key")
        self.assertIsNotNone(zero_key_metrics())

    def test_academic_tier_is_primary(self):

        fake_result = _make_result(
            source="openalex", title="Paper", source_url="https://doi.org/x",
            content="abstract", raw_metadata={},
        )
        fake_client = MagicMock()
        fake_client.research.return_value = [fake_result]
        with patch("app.services.research_layer.ResearchClient", return_value=fake_client):
            provider = ZeroKeyResearchProvider()
            items = provider.discover("paper", ResearchStrategy(), {"zero_key_enabled": True})
        self.assertEqual(items[0]["tier"], "academic")
        self.assertIs(items[0]["is_primary"], True)


if __name__ == "__main__":
    unittest.main()