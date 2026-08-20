"""Tests for long-form to short-form repurposing (Phase 2D.3)."""

import unittest

from app.services.repurposing import (
    RepurposePlan,
    _estimate_seconds,
    _moment_score,
    _split_sentences,
    plan_repurposing,
    repurpose_summary,
)


class TestMomentDetection(unittest.TestCase):
    def test_surprising_numbers_score_high(self):
        scored = _moment_score("Revenue jumped 300 percent last year.")
        self.assertGreater(scored, _moment_score("The company grew over time."))

    def test_emotion_and_insight_boost_score(self):
        emotional = _moment_score("The founder was betrayed and lost everything.")
        plain = _moment_score("The founder made some decisions.")
        self.assertGreater(emotional, plain)

    def test_question_sentences_score_higher(self):
        self.assertGreater(
            _moment_score("Why did nobody see it coming?"),
            _moment_score("It happened quickly."),
        )

    def test_split_sentences(self):
        sentences = _split_sentences("One. Two! Three? Four.")
        self.assertEqual(len(sentences), 4)


class TestRepurposingPlan(unittest.TestCase):
    def setUp(self):
        self.script = (
            "Amazon spent billions conquering groceries. Then it quietly surrendered. "
            "Revenue jumped 300 percent. That shocked analysts. "
            "Investor confidence collapsed. The reason: the strategy never fit the market. "
            "In the end, even giants can lose a war they started."
        )

    def test_plan_selects_narrative_value_not_time_slices(self):
        plan = plan_repurposing(self.script, topic="Amazon groceries", max_shorts=3)
        self.assertIsInstance(plan, RepurposePlan)
        self.assertEqual(len(plan.shorts), 3)
        # Moments are the highest-value sentences, not arbitrary intervals.
        self.assertTrue(all(short.score >= 1.0 for short in plan.shorts))

    def test_each_short_has_hook_context_payoff_duration_title(self):
        plan = plan_repurposing(self.script, topic="Amazon groceries", max_shorts=2)
        for short in plan.shorts:
            with self.subTest(index=short.index):
                self.assertTrue(short.hook)
                self.assertTrue(short.payoff)
                self.assertTrue(short.title)
                self.assertTrue(short.caption)
                self.assertGreaterEqual(short.estimated_seconds, 20.0)
                self.assertLessEqual(short.estimated_seconds, 60.0)

    def test_moment_type_is_detected(self):
        plan = plan_repurposing(self.script, topic="Amazon groceries", max_shorts=3)
        types = {short.moment_type for short in plan.shorts}
        self.assertTrue(types & {"surprise", "insight", "conclusion", "curiosity", "conflict", "emotion", "hook"})

    def test_empty_script_returns_empty_plan(self):
        plan = plan_repurposing("", topic="x")
        self.assertEqual(plan.shorts, [])
        self.assertTrue(plan.rationale)

    def test_short_script_returns_fewer_shorts(self):
        plan = plan_repurposing("Just one sentence here.", topic="x", max_shorts=3)
        self.assertLessEqual(len(plan.shorts), 3)

    def test_summary(self):
        plan = plan_repurposing(self.script, topic="Amazon groceries", max_shorts=2)
        summary = repurpose_summary(plan)
        self.assertIn("shorts=", summary)
        self.assertEqual(repurpose_summary(None), "none")

    def test_duration_estimate(self):
        self.assertAlmostEqual(_estimate_seconds("one two three four five"), 2.0, places=1)


if __name__ == "__main__":
    unittest.main()
