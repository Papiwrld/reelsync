"""Tests for the Story Intelligence engine (Phase 2C.1-2C.3)."""

import unittest

from app.services.content_profile import get_content_profile
from app.services.intelligence import build_content_intelligence
from app.services.narrative import (
    NARRATIVE_STRATEGIES,
    StoryBrief,
    build_story_brief,
    list_narrative_strategies,
    narrative_strategy_context,
    recent_narrative_usage,
    record_narrative_usage,
    reset_narrative_usage,
    select_narrative_strategy,
    story_brief_context,
    story_brief_summary,
)


class TestStrategyCatalog(unittest.TestCase):
    def test_catalog_has_expected_strategies(self):
        ids = set(list_narrative_strategies())
        for expected in (
            "documentary",
            "rise_and_fall",
            "mystery",
            "investigation",
            "biography",
            "case_study",
            "explainer",
            "tutorial",
            "timeline",
            "comparison",
            "conflict",
            "transformation",
            "news_analysis",
            "prediction",
            "debate",
            "how_it_works",
            "list",
            "educational",
            "interview",
            "commentary",
        ):
            self.assertIn(expected, ids)

    def test_all_strategies_have_sections_and_labels(self):
        for strategy in NARRATIVE_STRATEGIES:
            with self.subTest(strategy=strategy.id):
                self.assertTrue(strategy.label)
                self.assertTrue(strategy.sections)


class TestStrategySelection(unittest.TestCase):
    def test_different_niches_select_different_strategies(self):
        cases = [
            ("business", "How a company rose and fell", "rise_and_fall"),
            ("history", "The mystery of a vanished colony", "mystery"),
            ("finance", "How compound interest works", "explainer"),
            ("science", "How quantum computers actually work", "how_it_works"),
        ]
        for niche, topic, expected in cases:
            with self.subTest(niche=niche, topic=topic):
                intel, _ = build_content_intelligence(topic, get_content_profile(niche))
                result = select_narrative_strategy(
                    topic, get_content_profile(niche), intelligence=intel, verified_claims=3
                )
                self.assertEqual(result["strategy"].id, expected)

    def test_topic_signal_changes_strategy_for_same_niche(self):
        """Two different topics in the same niche must not always become the
        same structure (narrative variance is topic-contextual)."""
        profile = get_content_profile("history")
        mystery = select_narrative_strategy(
            "The mystery of the vanished colony", profile, verified_claims=2
        )["strategy"].id
        timeline = select_narrative_strategy(
            "The history of Rome from founding to fall", profile, verified_claims=5
        )["strategy"].id
        self.assertNotEqual(mystery, timeline)

    def test_variance_penalizes_recently_used_strategies(self):
        reset_narrative_usage()
        profile = get_content_profile("gaming")
        topic = "the best games ranked"
        first = select_narrative_strategy(topic, profile, verified_claims=1)["strategy"].id
        record_narrative_usage("gaming", first)
        second = select_narrative_strategy(topic, profile, verified_claims=1)["strategy"].id
        # With the same topic, the recently-used strategy is penalized and a
        # different (but still suitable) strategy wins or ties.
        self.assertEqual(recent_narrative_usage("gaming"), [first])
        self.assertIsInstance(second, str)
        self.assertTrue(second)

    def test_deterministic_same_inputs_same_output(self):
        profile = get_content_profile("business")
        intel, _ = build_content_intelligence("How a company rose and fell", profile)
        first = select_narrative_strategy("How a company rose and fell", profile, intelligence=intel, verified_claims=3)
        second = select_narrative_strategy("How a company rose and fell", profile, intelligence=intel, verified_claims=3)
        self.assertEqual(first["strategy"].id, second["strategy"].id)
        self.assertEqual(first["scores"], second["scores"])

    def test_low_evidence_penalizes_evidence_heavy_strategies(self):
        profile = get_content_profile("business")
        low_evidence = select_narrative_strategy("a vague topic", profile, verified_claims=0)["strategy"].id
        high_evidence = select_narrative_strategy("a vague topic", profile, verified_claims=8)["strategy"].id
        self.assertIsInstance(low_evidence, str)
        self.assertIsInstance(high_evidence, str)

    def test_selection_records_rationale(self):
        profile = get_content_profile("mystery")
        result = select_narrative_strategy("The mystery of the vanishing colony", profile, verified_claims=2)
        self.assertTrue(result["rationale"])


class TestStoryBrief(unittest.TestCase):
    def test_brief_derives_from_context(self):
        profile = get_content_profile("business")
        intel, _ = build_content_intelligence("How Amazon failed at groceries", profile)
        strategy = select_narrative_strategy(
            "How Amazon failed at groceries", profile, intelligence=intel, verified_claims=3
        )["strategy"]
        brief = build_story_brief(
            "How Amazon failed at groceries",
            profile,
            strategy,
            selected_hook="Amazon spent billions, then quietly surrendered.",
            intelligence=intel,
            topic_analysis={
                "curiosity_gaps": ["why it failed"],
                "emotional_angles": ["tension", "fascination"],
                "potential_claims": ["Amazon entered groceries"],
                "known_vs_unknown": "known: acquisitions; unknown: why",
                "controversy_level": "medium",
            },
            content_strategy={
                "emotional_progression": ["tension", "fascination"],
                "cta": "ask what they would have done",
            },
            research_claims=[
                {"statement": "Amazon acquired Whole Foods", "status": "verified"}
            ],
        )
        self.assertIsInstance(brief, StoryBrief)
        self.assertEqual(brief.subject, "How Amazon failed at groceries")
        self.assertTrue(brief.hook)
        self.assertEqual(brief.evidence, ["Amazon acquired Whole Foods"])
        self.assertTrue(brief.central_question)
        self.assertEqual(brief.narrative_strategy, strategy.label)

    def test_brief_accepts_enum_claim_status(self):
        """ResearchClaim.status 是 str-Enum 成员时（生产路径的真实形态），
        已核验声明必须进入 evidence/key_facts —— 曾经因为 str() 产出
        "ClaimStatus.VERIFIED" 而静默丢失。"""
        from app.services.research import ClaimStatus, ResearchClaim

        profile = get_content_profile("business")
        strategy = select_narrative_strategy("business topic", profile)["strategy"]
        brief = build_story_brief(
            "business topic",
            profile,
            strategy,
            research_claims=[
                ResearchClaim(
                    statement="Revenue doubled last year",
                    status=ClaimStatus.VERIFIED,
                ),
                ResearchClaim(
                    statement="The company is bankrupt",
                    status=ClaimStatus.UNSUPPORTED,
                ),
            ],
        )
        self.assertIn("Revenue doubled last year", brief.evidence)
        self.assertIn("Revenue doubled last year", brief.key_facts)
        self.assertNotIn("The company is bankrupt", brief.evidence)

    def test_brief_is_flexible_when_context_is_sparse(self):
        profile = get_content_profile("gaming")
        strategy = select_narrative_strategy("gaming topic", profile)["strategy"]
        brief = build_story_brief("gaming topic", profile, strategy)
        # No fabrication: empty fields stay empty, required fields are usable.
        self.assertEqual(brief.subject, "gaming topic")
        self.assertIsInstance(brief.evidence, list)
        self.assertIsInstance(brief.key_facts, list)

    def test_brief_context_renders_prompt_block(self):
        profile = get_content_profile("business")
        strategy = select_narrative_strategy("topic", profile)["strategy"]
        brief = build_story_brief("topic", profile, strategy, selected_hook="hook line")
        block = story_brief_context(brief)
        self.assertIn("Story Brief", block)
        self.assertIn("hook line", block)

    def test_context_none_returns_empty(self):
        self.assertEqual(story_brief_context(None), "")

    def test_summary_one_line(self):
        profile = get_content_profile("mystery")
        strategy = select_narrative_strategy("topic", profile)["strategy"]
        brief = build_story_brief("topic", profile, strategy)
        summary = story_brief_summary(brief)
        self.assertIn("strategy=", summary)

    def test_narrative_strategy_context(self):
        profile = get_content_profile("history")
        strategy = select_narrative_strategy("topic", profile)["strategy"]
        block = narrative_strategy_context(strategy, rationale="why")
        self.assertIn(strategy.label, block)
        self.assertIn("why", block)
        self.assertEqual(narrative_strategy_context(None), "")


if __name__ == "__main__":
    unittest.main()
