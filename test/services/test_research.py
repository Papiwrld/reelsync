"""Tests for the Research Orchestrator (Phase 2B)."""

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from app.services import llm, research
from app.services.agent_llm import AgentTracker
from app.services.content_profile import get_content_profile
from app.services.intelligence import ContentIntelligence, ContentRequest
from app.services.research import (
    ClaimStatus,
    ResearchClaim,
    ResearchPacket,
    ResearchProviderError,
    ResearchSource,
    SourceTier,
    WebSearchResearchProvider,
    _deterministic_claim_verdict,
    _parse_user_notes,
    build_research_strategy,
    classify_source_tier,
    research_grounding_context,
    research_summary,
    run_research,
)


class TestClassifySourceTier(unittest.TestCase):
    def test_tiers(self):
        cases = [
            ("https://www.nasa.gov/news", SourceTier.GOVERNMENT),
            ("https://mit.edu/research", SourceTier.ACADEMIC),
            ("https://www.reddit.com/r/x", SourceTier.SOCIAL_MEDIA),
            ("https://en.wikipedia.org/wiki/X", SourceTier.SECONDARY),
            ("https://example.com/forum/post", SourceTier.COMMUNITY),
            ("https://unknown-site.org/anything", SourceTier.UNKNOWN),
        ]
        for url, expected in cases:
            with self.subTest(url=url):
                self.assertEqual(classify_source_tier(url=url), expected)

    def test_press_release_industry_hint(self):
        self.assertEqual(
            classify_source_tier(title="Quarterly report highlights"),
            SourceTier.INDUSTRY,
        )


class TestParseUserNotes(unittest.TestCase):
    def test_parses_urls_and_titles(self):
        sources = _parse_user_notes(
            [
                "https://www.nasa.gov/topic Official NASA page",
                "a note without a url",
            ]
        )
        self.assertEqual(len(sources), 2)
        self.assertEqual(sources[0].url, "https://www.nasa.gov/topic")
        self.assertEqual(sources[0].title, "Official NASA page")
        self.assertEqual(sources[0].tier, SourceTier.GOVERNMENT)
        self.assertEqual(sources[0].provenance, "user_notes")
        self.assertEqual(sources[1].title, "a note without a url")

    def test_deduplicates(self):
        sources = _parse_user_notes(["https://a.example.com/x", "https://a.example.com/x"])
        self.assertEqual(len(sources), 1)

    def test_ignores_blank_lines(self):
        self.assertEqual(_parse_user_notes(["", "   "]), [])


class TestBuildResearchStrategy(unittest.TestCase):
    def test_profile_driven_verification(self):
        business = build_research_strategy(get_content_profile("business"), None)
        self.assertTrue(business.verify_explicitly)
        self.assertEqual(business.fact_check_level, "very_strong")
        gaming = build_research_strategy(get_content_profile("gaming"), None)
        self.assertFalse(gaming.verify_explicitly)
        self.assertEqual(gaming.fact_check_level, "normal")

    def test_intelligence_drives_strategy(self):
        intel = ContentIntelligence(fact_check_level="very_strong", risk_profile="high", research_depth="very_high")
        strategy = build_research_strategy(get_content_profile("gaming"), intel)
        self.assertTrue(strategy.verify_explicitly)
        self.assertEqual(strategy.depth, "very_high")
        self.assertEqual(strategy.risk_profile, "high")

    def test_override_wins(self):
        strategy = build_research_strategy(
            get_content_profile("gaming"), None, override_fact_check="very_strong"
        )
        self.assertTrue(strategy.verify_explicitly)


class TestDeterministicClaimVerdict(unittest.TestCase):
    def _source(self, source_id, tier):
        return ResearchSource(id=source_id, title="t", url="u", tier=tier)

    def test_no_refs_unsupported(self):
        claim = ResearchClaim(statement="x", source_refs=[])
        status, confidence = _deterministic_claim_verdict(claim, [])
        self.assertEqual(status, ClaimStatus.UNSUPPORTED)
        self.assertLess(confidence, 0.2)

    def test_single_low_tier_uncertain(self):
        claim = ResearchClaim(statement="x", source_refs=["s1"])
        status, _ = _deterministic_claim_verdict(
            claim, [self._source("s1", SourceTier.SOCIAL_MEDIA)]
        )
        self.assertEqual(status, ClaimStatus.UNCERTAIN)

    def test_two_credible_sources_verified(self):
        claim = ResearchClaim(statement="x", source_refs=["s1", "s2"])
        status, confidence = _deterministic_claim_verdict(
            claim,
            [
                self._source("s1", SourceTier.GOVERNMENT),
                self._source("s2", SourceTier.ACADEMIC),
            ],
        )
        self.assertEqual(status, ClaimStatus.VERIFIED)
        self.assertAlmostEqual(confidence, 0.9)

    def test_single_credible_source_still_uncertain(self):
        claim = ResearchClaim(statement="x", source_refs=["s1"])
        status, confidence = _deterministic_claim_verdict(
            claim, [self._source("s1", SourceTier.GOVERNMENT)]
        )
        self.assertEqual(status, ClaimStatus.UNCERTAIN)
        self.assertAlmostEqual(confidence, 0.6)


class _FakeLLM:
    """Returns per-agent canned research responses."""

    def __init__(self, sources=None, claims=None, verdicts=None):
        self.sources = sources if sources is not None else []
        self.claims = claims if claims is not None else {"claims": [], "contradictions": [], "uncertainties": [], "summary": ""}
        self.verdicts = verdicts
        self.agents = []

    def __call__(self, prompt, app_config=None):
        if "Research Source Scout" in prompt:
            self.agents.append("research_sources")
            return json.dumps(self.sources)
        if "Research Analyst" in prompt:
            self.agents.append("research_claims")
            return json.dumps(self.claims)
        if "Fact Checker" in prompt:
            self.agents.append("fact_checker")
            return json.dumps(self.verdicts if self.verdicts is not None else [])
        raise AssertionError(f"unexpected prompt: {prompt[:80]}")


class TestRunResearch(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._cache_patch = patch.object(research, "_cache_dir", return_value=Path(self._tmp.name))
        self._cache_patch.start()
        self.addCleanup(self._cache_patch.stop)
        self.addCleanup(self._tmp.cleanup)

    def test_full_flow_verifies_claims(self):
        fake = _FakeLLM(
            sources=[
                {"title": "NASA page", "url": "https://nasa.gov/x", "tier": "government", "is_primary": True, "note": "official"},
                {"title": "Journal", "url": "https://academic.example.edu/a", "tier": "academic", "is_primary": False, "note": "peer reviewed"},
            ],
            claims={
                "claims": [
                    {"statement": "Atlantis was described by Plato", "source_refs": ["src-1", "src-2"], "confidence": 0.8, "note": "documented"},
                    {"statement": "Atlantis is under the sea", "source_refs": ["src-1"], "confidence": 0.5, "note": "thin support"},
                ],
                "contradictions": ["sources disagree on location"],
                "uncertainties": ["exact location unknown"],
                "summary": "Two credible sources support the Plato account.",
            },
            verdicts=[
                {"statement": "Atlantis was described by Plato", "status": "verified", "confidence": 0.9, "note": "multiple credible sources"},
                {"statement": "Atlantis is under the sea", "status": "uncertain", "confidence": 0.4, "note": "thin support"},
            ],
        )
        with patch.object(llm, "_generate_response", side_effect=fake):
            packet = run_research(
                topic="Atlantis",
                profile=get_content_profile("business"),
                intelligence=None,
                context=ContentRequest(automation_level="automatic"),
                app_config=None,
            )
        self.assertEqual(len(packet.sources), 2)
        self.assertEqual(packet.sources[0].tier, SourceTier.GOVERNMENT)
        self.assertEqual(len(packet.claims), 2)
        by_statement = {claim.statement: claim for claim in packet.claims}
        self.assertEqual(by_statement["Atlantis was described by Plato"].status, ClaimStatus.VERIFIED)
        self.assertAlmostEqual(by_statement["Atlantis was described by Plato"].confidence, 0.9)
        self.assertEqual(by_statement["Atlantis is under the sea"].status, ClaimStatus.UNCERTAIN)
        self.assertEqual(packet.contradictions, ["sources disagree on location"])
        self.assertIn("fact_checker", fake.agents)
        self.assertTrue(packet.provenance["model_knowledge"])

    def test_deterministic_fallback_when_llm_fails(self):
        def broken(*_args, **_kwargs):
            raise RuntimeError("provider down")

        with patch.object(llm, "_generate_response", side_effect=broken):
            packet = run_research(
                topic="Atlantis",
                profile=get_content_profile("gaming"),
                intelligence=None,
                context=ContentRequest(automation_level="automatic"),
            )
        self.assertEqual(packet.sources, [])
        self.assertEqual(packet.claims, [])
        self.assertTrue(any("no sources available" in item for item in packet.uncertainties))

    def test_never_raises_with_user_notes_only(self):
        fake = _FakeLLM(claims={"claims": [], "contradictions": [], "uncertainties": [], "summary": ""})
        with patch.object(llm, "_generate_response", side_effect=fake):
            packet = run_research(
                topic="Atlantis",
                profile=get_content_profile("mystery"),
                intelligence=None,
                context=ContentRequest(sources=["https://nasa.gov/atlantis The NASA page"]),
            )
        self.assertEqual(len(packet.sources), 1)
        self.assertEqual(packet.sources[0].provenance, "user_notes")

    def test_cache_hit_skips_llm(self):
        fake = _FakeLLM(
            sources=[{"title": "NASA page", "url": "https://nasa.gov/x", "tier": "government", "is_primary": True, "note": "official"}],
            claims={"claims": [{"statement": "s", "source_refs": ["src-1"], "confidence": 0.5, "note": ""}], "contradictions": [], "uncertainties": [], "summary": "s"},
        )
        with patch.object(llm, "_generate_response", side_effect=fake):
            first = run_research(
                topic="Atlantis", profile=get_content_profile("mystery"), intelligence=None,
                context=ContentRequest(automation_level="automatic"),
            )
            second = run_research(
                topic="Atlantis", profile=get_content_profile("mystery"), intelligence=None,
                context=ContentRequest(automation_level="automatic"),
            )
        self.assertFalse(first.provenance["cached"])
        self.assertTrue(second.provenance["cached"])
        self.assertEqual(fake.agents.count("research_sources"), 1)
        self.assertEqual(len(second.sources), 1)

    def test_cache_expires_after_ttl(self):
        fake = _FakeLLM(
            sources=[{"title": "NASA page", "url": "https://nasa.gov/x", "tier": "government", "is_primary": True, "note": "official"}],
            claims={"claims": [], "contradictions": [], "uncertainties": [], "summary": ""},
        )
        with patch.object(llm, "_generate_response", side_effect=fake), patch.object(
            research, "_cache_ttl_hours", return_value=0.0
        ):
            run_research(
                topic="Atlantis", profile=get_content_profile("mystery"), intelligence=None,
                context=ContentRequest(automation_level="automatic"),
            )
            target = Path(self._tmp.name)
            for entry in target.iterdir():
                os.utime(entry, (time.time() - 7200, time.time() - 7200))
            second = run_research(
                topic="Atlantis", profile=get_content_profile("mystery"), intelligence=None,
                context=ContentRequest(automation_level="automatic"),
            )
        self.assertFalse(second.provenance["cached"])
        self.assertEqual(fake.agents.count("research_sources"), 2)

    def test_tracker_records_research_agents(self):
        fake = _FakeLLM(
            sources=[{"title": "NASA page", "url": "https://nasa.gov/x", "tier": "government", "is_primary": True, "note": "official"}],
            claims={"claims": [], "contradictions": [], "uncertainties": [], "summary": ""},
        )
        tracker = AgentTracker()
        with patch.object(llm, "_generate_response", side_effect=fake):
            run_research(
                topic="Atlantis", profile=get_content_profile("mystery"), intelligence=None,
                context=ContentRequest(automation_level="automatic"),
                tracker=tracker,
            )
        snapshot = tracker.snapshot()
        self.assertEqual(snapshot["statuses"].get("research_sources"), "llm")
        self.assertEqual(snapshot["statuses"].get("research_claims"), "llm")


class TestWebSearchProvider(unittest.TestCase):
    def test_not_configured_by_default(self):
        self.assertFalse(WebSearchResearchProvider.is_configured({}))
        self.assertFalse(WebSearchResearchProvider.is_configured({"provider": "web_search"}))

    def test_configured_with_base_url(self):
        self.assertTrue(
            WebSearchResearchProvider.is_configured(
                {"provider": "web_search", "base_url": "https://search.example.com/"}
            )
        )

    def test_discover_raises_when_not_configured(self):
        with self.assertRaises(ResearchProviderError):
            WebSearchResearchProvider().discover("topic", build_research_strategy(get_content_profile("gaming"), None), {})

    def test_discover_parses_results(self):
        class _Response:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "results": [
                        {"title": "NASA news", "url": "https://www.nasa.gov/news", "snippet": "official statement"},
                        {"title": "Random blog", "url": "https://blog.example.com/p", "snippet": "opinion"},
                    ]
                }

        with patch("requests.get", return_value=_Response()):
            items = WebSearchResearchProvider().discover(
                "topic",
                build_research_strategy(get_content_profile("gaming"), None),
                {"provider": "web_search", "base_url": "https://search.example.com/"},
            )
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]["tier"], "government")
        self.assertEqual(items[1]["tier"], "unknown")

    def test_web_failure_degrades_to_model_knowledge(self):
        fake = _FakeLLM(
            sources=[{"title": "NASA page", "url": "https://nasa.gov/x", "tier": "government", "is_primary": True, "note": "official"}],
            claims={"claims": [], "contradictions": [], "uncertainties": [], "summary": ""},
        )
        tracker = AgentTracker()
        with patch.object(llm, "_generate_response", side_effect=fake), patch(
            "requests.get", side_effect=RuntimeError("endpoint down")
        ), patch.object(research, "_load_cached_packet", return_value=None):
            packet = run_research(
                topic="Atlantis web failure",
                profile=get_content_profile("mystery"),
                intelligence=None,
                context=ContentRequest(automation_level="automatic"),
                app_config={"provider": "web_search", "base_url": "https://search.example.com/"},
                tracker=tracker,
            )
        self.assertEqual(len(packet.sources), 1)
        self.assertEqual(packet.sources[0].provenance, "model_knowledge")
        snapshot = tracker.snapshot()
        self.assertIn("web_search unavailable", snapshot["fallback_reasons"].get("research_sources", ""))


class TestGroundingContext(unittest.TestCase):
    def test_none_returns_empty(self):
        self.assertEqual(research_grounding_context(None), "")

    def test_empty_claims_returns_empty(self):
        packet = ResearchPacket(claims=[])
        self.assertEqual(research_grounding_context(packet), "")

    def test_verified_and_uncertain_claims_rendered(self):
        packet = ResearchPacket(
            summary="sources agree",
            claims=[
                ResearchClaim(statement="fact A", status=ClaimStatus.VERIFIED, confidence=0.9),
                ResearchClaim(statement="fact B", status=ClaimStatus.UNCERTAIN, confidence=0.4),
            ],
        )
        block = research_grounding_context(packet)
        self.assertIn("Verified claims", block)
        self.assertIn("fact A", block)
        self.assertIn("Uncertain claims", block)
        self.assertIn("never invent numbers or quotes", block)

    def test_summary_line(self):
        packet = ResearchPacket(
            claims=[ResearchClaim(statement="x", status=ClaimStatus.VERIFIED)]
        )
        self.assertIn("sources=0", research_summary(packet))
        self.assertIn("verified", research_summary(packet))
        self.assertIn("cached=False", research_summary(packet))


if __name__ == "__main__":
    unittest.main()