"""Tests for Trend Intelligence & Topic Discovery (Phase 2E.1-2E.3)."""

import json
import unittest
from unittest.mock import patch

from app.services import llm
from app.services.content_profile import get_content_profile
from app.services.intelligence import build_content_intelligence
from app.services.trends import (
    TREND_CONTEXT_MODEL_INFERENCE,
    TREND_CONTEXT_RECENT,
    TOPIC_MODE_AUTONOMOUS,
    TOPIC_MODE_EVERGREEN,
    TOPIC_MODE_NEWS,
    TOPIC_MODE_TRENDING,
    TOPIC_MODE_USER,
    ModelInferenceTrendProvider,
    TrendSignal,
    WebSearchTrendProvider,
    collect_trend_signals,
    discover_topics,
    score_topic,
    select_topic_for_automation,
    trends_summary,
)


class TestTrendProviders(unittest.TestCase):
    def test_model_inference_is_always_labeled_model_inference(self):
        payload = [
            {"topic": "AI agents in finance", "direction": "rising", "score": 8.0, "note": "model guess"},
            {"topic": "Gaming on Linux", "direction": "stable", "score": 6.0, "note": "model guess"},
        ]
        profile = get_content_profile("business")
        with patch.object(llm, "_generate_response", return_value=json.dumps(payload)):
            signals = ModelInferenceTrendProvider().fetch(profile, niche="business")
        self.assertEqual(len(signals), 2)
        for signal in signals:
            self.assertEqual(signal.context, TREND_CONTEXT_MODEL_INFERENCE)
            self.assertIn(signal.direction, ("rising", "stable", "declining", "unknown"))

    def test_model_inference_never_claims_current(self):
        # Even if the model says "current", the provider overrides it.
        payload = [{"topic": "x", "direction": "rising", "score": 9.0, "note": "trending"}]
        profile = get_content_profile("gaming")
        with patch.object(llm, "_generate_response", return_value=json.dumps(payload)):
            signals = ModelInferenceTrendProvider().fetch(profile, niche="gaming")
        self.assertEqual(signals[0].context, TREND_CONTEXT_MODEL_INFERENCE)

    def test_model_inference_down_returns_empty(self):
        def broken(*_args, **_kwargs):
            raise RuntimeError("provider down")

        profile = get_content_profile("gaming")
        with patch.object(llm, "_generate_response", side_effect=broken):
            signals = ModelInferenceTrendProvider().fetch(profile, niche="gaming")
        self.assertEqual(signals, [])

    def test_web_search_not_configured_by_default(self):
        self.assertFalse(WebSearchTrendProvider.is_configured({}))
        self.assertFalse(WebSearchTrendProvider.is_configured({"provider": "web_search"}))

    def test_web_search_configured_returns_recent(self):
        class _Response:
            def raise_for_status(self):
                return None

            def json(self):
                return {"results": [{"title": "hot topic", "url": "https://x.example/y", "snippet": "s"}]}

        profile = get_content_profile("gaming")
        with patch("requests.get", return_value=_Response()):
            signals = WebSearchTrendProvider().fetch(
                profile,
                app_config={"provider": "web_search", "base_url": "https://search.example.com/"},
                niche="gaming",
            )
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].context, TREND_CONTEXT_RECENT)


class TestCollectTrends(unittest.TestCase):
    def test_collect_deduplicates_and_sorts(self):
        payload = [
            {"topic": "Duplicate topic", "direction": "rising", "score": 5.0, "note": "a"},
            {"topic": "Duplicate topic", "direction": "rising", "score": 9.0, "note": "b"},
            {"topic": "Unique topic", "direction": "stable", "score": 7.0, "note": "c"},
        ]
        profile = get_content_profile("mystery")
        with patch.object(llm, "_generate_response", return_value=json.dumps(payload)):
            signals = collect_trend_signals(profile, niche="mystery")
        topics = [signal.topic for signal in signals]
        self.assertEqual(len(topics), 2)
        self.assertEqual(topics[0], "Unique topic")  # sorted by score desc

    def test_summary(self):
        signals = [TrendSignal(topic="a", context=TREND_CONTEXT_MODEL_INFERENCE, score=5.0)]
        summary = trends_summary(signals)
        self.assertIn("signals=1", summary)
        self.assertEqual(trends_summary([]), "signals=0, contexts={}")


class TestTopicScoring(unittest.TestCase):
    def test_score_dimensions_are_explainable(self):
        profile = get_content_profile("business")
        intel, _ = build_content_intelligence("topic", profile)
        candidate = score_topic("AI agents in finance", profile, intelligence=intel, mode=TOPIC_MODE_TRENDING)
        self.assertTrue(candidate.scores)
        self.assertTrue(candidate.rationales)
        for dimension in ("trend_strength", "audience_relevance", "competition", "evergreen_value", "story_potential"):
            self.assertIn(dimension, candidate.scores)
            self.assertIn(dimension, candidate.rationales)
        self.assertGreaterEqual(candidate.total, 0.0)
        self.assertLessEqual(candidate.total, 10.0)

    def test_evergreen_mode_boosts_evergreen_value(self):
        profile = get_content_profile("science")
        trending = score_topic("space", profile, mode=TOPIC_MODE_TRENDING)
        evergreen = score_topic("space", profile, mode=TOPIC_MODE_EVERGREEN)
        self.assertGreater(evergreen.scores["evergreen_value"], trending.scores["evergreen_value"])

    def test_story_topics_score_story_potential(self):
        profile = get_content_profile("history")
        story = score_topic("the rise and fall of an empire", profile)
        plain = score_topic("a simple update", profile)
        self.assertGreater(story.scores["story_potential"], plain.scores["story_potential"])


class TestTopicDiscovery(unittest.TestCase):
    def test_user_provided_mode_scores_user_topics(self):
        profile = get_content_profile("gaming")
        candidates = discover_topics(
            profile,
            mode=TOPIC_MODE_USER,
            user_topics=["VR gaming headsets", "indie game funding"],
        )
        self.assertEqual(len(candidates), 2)
        self.assertTrue(all(candidate.scores for candidate in candidates))

    def test_provider_modes_return_candidates(self):
        payload = [
            {"topic": "AI agents", "direction": "rising", "score": 8.0, "note": "a"},
            {"topic": "Space telescopes", "direction": "stable", "score": 6.0, "note": "b"},
        ]
        profile = get_content_profile("science")
        with patch.object(llm, "_generate_response", return_value=json.dumps(payload)):
            candidates = discover_topics(profile, mode=TOPIC_MODE_TRENDING, niche="science")
        self.assertTrue(candidates)
        self.assertTrue(all(candidate.total > 0 for candidate in candidates))

    def test_news_mode_requires_fresh_signal(self):
        # With only model knowledge, news mode still works but must not claim
        # CURRENT anywhere in the signal payload.
        payload = [{"topic": "a news topic", "direction": "rising", "score": 7.0, "note": "n"}]
        profile = get_content_profile("gaming")
        with patch.object(llm, "_generate_response", return_value=json.dumps(payload)):
            candidates = discover_topics(profile, mode=TOPIC_MODE_NEWS, niche="gaming")
        self.assertTrue(candidates)
        for candidate in candidates:
            self.assertEqual(candidate.signal["context"], TREND_CONTEXT_MODEL_INFERENCE)

    def test_autonomous_mode_returns_ranked(self):
        payload = [
            {"topic": "topic one", "direction": "rising", "score": 8.0, "note": "a"},
            {"topic": "topic two", "direction": "stable", "score": 6.0, "note": "b"},
            {"topic": "topic three", "direction": "declining", "score": 4.0, "note": "c"},
        ]
        profile = get_content_profile("mystery")
        with patch.object(llm, "_generate_response", return_value=json.dumps(payload)):
            candidates = discover_topics(profile, mode=TOPIC_MODE_AUTONOMOUS, niche="mystery", max_candidates=2)
        self.assertEqual(len(candidates), 2)
        self.assertGreaterEqual(candidates[0].total, candidates[1].total)

    def test_provider_failure_returns_empty_gracefully(self):
        def broken(*_args, **_kwargs):
            raise RuntimeError("down")

        profile = get_content_profile("gaming")
        with patch.object(llm, "_generate_response", side_effect=broken):
            candidates = discover_topics(profile, mode=TOPIC_MODE_TRENDING, niche="gaming")
        self.assertEqual(candidates, [])

    def test_no_duplicate_topics(self):
        payload = [
            {"topic": "same topic", "direction": "rising", "score": 8.0, "note": "a"},
            {"topic": "same topic", "direction": "stable", "score": 6.0, "note": "b"},
        ]
        profile = get_content_profile("education")
        with patch.object(llm, "_generate_response", return_value=json.dumps(payload)):
            candidates = discover_topics(profile, mode=TOPIC_MODE_TRENDING, niche="education")
        self.assertEqual(len(candidates), 1)

    def test_precollected_signals_skip_provider_call(self):
        """调用方（agentic 编排器）已收集过趋势信号时，discover_topics 必须
        复用而不重复请求 provider —— 否则自动驾驶路径会对同一个 run 调用两次
        趋势 LLM，浪费成本并可能让两次结果不一致。"""
        profile = get_content_profile("gaming")
        signals = [TrendSignal(topic="VR headsets", context=TREND_CONTEXT_MODEL_INFERENCE, score=8.0)]
        with patch.object(llm, "_generate_response", side_effect=AssertionError("must not call llm")):
            candidates = discover_topics(
                profile,
                mode=TOPIC_MODE_TRENDING,
                signals=signals,
            )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].topic, "VR headsets")


class TestAutonomousSelection(unittest.TestCase):
    def test_manual_returns_none(self):
        self.assertIsNone(select_topic_for_automation([object()], "manual"))

    def test_assisted_and_automatic_select_best(self):
        profile = get_content_profile("gaming")
        candidate = score_topic("VR gaming", profile, mode=TOPIC_MODE_TRENDING)
        for level in ("assisted", "automatic", "autopilot"):
            with self.subTest(level=level):
                selected = select_topic_for_automation([candidate], level)
                self.assertIsNotNone(selected)
                self.assertEqual(selected.topic, "VR gaming")

    def test_empty_candidates_returns_none(self):
        self.assertIsNone(select_topic_for_automation([], "automatic"))


if __name__ == "__main__":
    unittest.main()
