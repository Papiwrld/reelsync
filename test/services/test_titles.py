"""Tests for Title & Thumbnail Intelligence (Phase 2D.1-2D.2)."""

import json
import unittest
from unittest.mock import patch

from app.services import llm
from app.services.content_profile import get_content_profile
from app.services.titles import (
    TitleCandidate,
    _fallback_title_candidates,
    compose_thumbnail_concept,
    generate_title_candidates,
    score_title_candidate,
    select_best_title,
    thumbnail_concept_summary,
    title_summary,
)
from app.services.visual_director import plan_scenes


class TestTitleScoring(unittest.TestCase):
    def setUp(self):
        self.profile = get_content_profile("business")
        self.script = (
            "Amazon acquired Whole Foods in 2017. "
            "Revenue grew 300 percent. "
            "The company lost the grocery war."
        )
        self.key_facts = ["Amazon acquired Whole Foods in 2017"]

    def test_accuracy_beats_clickbait(self):
        clickbait = "The Company That Destroyed the Entire World"
        accurate = "Why Amazon lost the grocery war after spending billions"
        clickbait_scores = score_title_candidate(clickbait, self.script, self.profile, self.key_facts)
        accurate_scores = score_title_candidate(accurate, self.script, self.profile, self.key_facts)
        self.assertGreater(accurate_scores["accuracy"], clickbait_scores["accuracy"])

    def test_title_claiming_unsupported_number_is_penalized(self):
        # "300" is in the script; "900" is not.
        supported = score_title_candidate(
            "Revenue grew 300 percent", self.script, self.profile, self.key_facts
        )
        unsupported = score_title_candidate(
            "Revenue grew 900 percent", self.script, self.profile, self.key_facts
        )
        self.assertGreater(supported["accuracy"], unsupported["accuracy"])

    def test_title_cannot_overstate_magnitude(self):
        """脚本只说“营收明显下滑”时，标题不能声称“一夜之间失去一切”。
        绝对程度词（everything/overnight/completely）只有在脚本同样
        支持时才被允许。"""
        mild_script = "Revenue declined significantly last year. The company restructured."
        overstated = score_title_candidate(
            "The Company Lost Everything Overnight", mild_script, self.profile, []
        )
        honest = score_title_candidate(
            "Revenue declined last year", mild_script, self.profile, []
        )
        self.assertLess(overstated["accuracy"], 6.0)  # 低于 QA 拒绝阈值
        self.assertGreater(honest["accuracy"], overstated["accuracy"])

        # 脚本确实支持绝对程度时，标题不被误伤。
        strong_script = "The company lost everything overnight when the market crashed completely."
        supported = score_title_candidate(
            "The Company Lost Everything Overnight", strong_script, self.profile, []
        )
        self.assertGreaterEqual(supported["accuracy"], 6.0)

    def test_all_dimensions_are_bounded(self):
        scores = score_title_candidate("Some title", self.script, self.profile, self.key_facts, "tiktok")
        for dimension, value in scores.items():
            with self.subTest(dimension=dimension):
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 10.0)

    def test_platform_fit_prefers_short_titles_on_shorts(self):
        short_title = "Why Amazon lost"
        long_title = "The complete and definitive story of why Amazon lost the grocery war in 2017"
        short_scores = score_title_candidate(short_title, self.script, self.profile, self.key_facts, "tiktok")
        long_scores = score_title_candidate(long_title, self.script, self.profile, self.key_facts, "tiktok")
        self.assertGreater(short_scores["platform_fit"], long_scores["platform_fit"])

    def test_question_titles_get_curiosity_bonus(self):
        question = score_title_candidate("Why did Amazon lose?", self.script, self.profile, self.key_facts)
        statement = score_title_candidate("Amazon lost", self.script, self.profile, self.key_facts)
        self.assertGreater(question["curiosity"], statement["curiosity"])


class TestTitleGeneration(unittest.TestCase):
    def setUp(self):
        self.profile = get_content_profile("business")
        self.script = "Amazon acquired Whole Foods in 2017 and lost the grocery war."
        self.key_facts = ["Amazon acquired Whole Foods in 2017"]

    def test_llm_candidates_are_scored_and_ranked(self):
        payload = [
            {"text": "Why Amazon lost the grocery war", "style": "question", "rationale": "accurate"},
            {"text": "The Company That Destroyed the World", "style": "shock", "rationale": "clickbait"},
            {"text": "Amazon spent billions, then lost", "style": "direct", "rationale": "accurate"},
        ]
        with patch.object(llm, "_generate_response", return_value=json.dumps(payload)):
            candidates = generate_title_candidates(
                "Amazon groceries", self.script, self.profile, key_facts=self.key_facts
            )
        self.assertEqual(len(candidates), 3)
        for candidate in candidates:
            self.assertTrue(candidate.scores)
            self.assertGreater(candidate.overall, 0.0)
        # Ranked best first.
        self.assertGreaterEqual(candidates[0].overall, candidates[-1].overall)
        # The clickbait candidate must not be first.
        self.assertNotEqual(candidates[0].text, "The Company That Destroyed the World")

    def test_llm_down_falls_back_to_deterministic_templates(self):
        def broken(*_args, **_kwargs):
            raise RuntimeError("provider down")

        with patch.object(llm, "_generate_response", side_effect=broken):
            candidates = generate_title_candidates(
                "Amazon groceries", self.script, self.profile, key_facts=self.key_facts
            )
        self.assertTrue(candidates)
        for candidate in candidates:
            self.assertIsInstance(candidate, TitleCandidate)
            self.assertTrue(candidate.text)
            self.assertTrue(candidate.scores)

    def test_llm_garbage_falls_back(self):
        with patch.object(llm, "_generate_response", return_value="not json"):
            candidates = generate_title_candidates(
                "Amazon groceries", self.script, self.profile, key_facts=self.key_facts
            )
        self.assertTrue(candidates)

    def test_fallback_candidates_are_grounded(self):
        texts = _fallback_title_candidates(
            "Why Amazon lost", "Amazon lost the war", ["Amazon acquired Whole Foods in 2017"], self.profile
        )
        self.assertTrue(texts)
        # Every candidate references the topic or a key fact.
        for text in texts:
            self.assertTrue(
                "amazon" in text.lower() or "whole foods" in text.lower() or "2017" in text
            )

    def test_select_best_title_returns_top(self):
        candidates = [
            TitleCandidate(text="a", overall=8.0, scores={"accuracy": 8.0}),
            TitleCandidate(text="b", overall=5.0, scores={"accuracy": 5.0}),
        ]
        self.assertEqual(select_best_title(candidates).text, "a")

    def test_select_best_title_empty_raises(self):
        with self.assertRaises(ValueError):
            select_best_title([])

    def test_summary(self):
        candidate = TitleCandidate(text="hello world", overall=7.5, scores={"accuracy": 8.0})
        self.assertIn("hello world", title_summary(candidate))
        self.assertEqual(title_summary(None), "none")


class TestThumbnailConcept(unittest.TestCase):
    def setUp(self):
        self.profile = get_content_profile("business")
        self.script = "Amazon acquired Whole Foods. Revenue grew. Amazon surrendered."
        self.plan = plan_scenes(self.script, self.profile, desired_scene_count=3)

    def test_concept_has_structured_fields(self):
        concept = compose_thumbnail_concept("grocery war", "Why Amazon lost", self.plan, self.profile)
        self.assertTrue(concept.primary_subject)
        self.assertTrue(concept.composition)
        self.assertTrue(concept.focal_point)
        self.assertTrue(concept.visual_contrast)
        self.assertTrue(concept.emotional_cue)
        self.assertTrue(concept.rationale)

    def test_primary_subject_comes_from_scene_plan(self):
        concept = compose_thumbnail_concept("grocery war", "Why Amazon lost", self.plan, self.profile)
        self.assertEqual(concept.primary_subject, "amazon")

    def test_summary(self):
        concept = compose_thumbnail_concept("topic", "title", self.plan, self.profile)
        self.assertIn("subject=", thumbnail_concept_summary(concept))
        self.assertEqual(thumbnail_concept_summary(None), "none")


if __name__ == "__main__":
    unittest.main()
