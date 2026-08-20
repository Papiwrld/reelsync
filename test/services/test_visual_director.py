"""Tests for the Visual Director (Phase 2C.5-2C.6)."""

import unittest

from app.services.content_profile import get_content_profile
from app.services.intelligence import build_content_intelligence
from app.services.visual_director import (
    ALL_MATERIAL_TYPES,
    MaterialType,
    ScenePlan,
    _classify_material_type,
    plan_scenes,
    scene_plan_summary,
)


class TestMaterialClassification(unittest.TestCase):
    def test_numbers_map_to_chart_or_graph(self):
        self.assertEqual(_classify_material_type("Revenue jumped 300 percent"), MaterialType.CHART.value)
        self.assertEqual(_classify_material_type("Sales grew by 40% this year"), MaterialType.CHART.value)
        self.assertEqual(_classify_material_type("The company spent $13.7 billion"), MaterialType.CHART.value)
        self.assertEqual(_classify_material_type("It happened in the 1990s"), MaterialType.GRAPH.value)

    def test_locations_map_to_map(self):
        self.assertEqual(_classify_material_type("The company opened its first store in Seattle"), MaterialType.MAP.value)
        self.assertEqual(_classify_material_type("The city was built on the coast"), MaterialType.MAP.value)

    def test_quotes_map_to_document(self):
        self.assertEqual(_classify_material_type("Analysts said the strategy was doomed"), MaterialType.DOCUMENT.value)
        self.assertEqual(_classify_material_type('The report stated "we are profitable"'), MaterialType.DOCUMENT.value)

    def test_company_names_map_to_product_image(self):
        self.assertEqual(_classify_material_type("Amazon acquired Whole Foods"), MaterialType.PRODUCT_IMAGE.value)
        self.assertEqual(_classify_material_type("Tesla announced a new model"), MaterialType.PRODUCT_IMAGE.value)

    def test_history_maps_to_archival(self):
        self.assertEqual(_classify_material_type("The empire fell in the 5th century"), MaterialType.ARCHIVAL.value)
        self.assertEqual(_classify_material_type("Ancient ruins were uncovered"), MaterialType.ARCHIVAL.value)

    def test_abstract_concepts_map_to_metaphor(self):
        self.assertEqual(_classify_material_type("Investor confidence collapsed"), MaterialType.ABSTRACT_METAPHOR.value)
        self.assertEqual(_classify_material_type("The culture shifted"), MaterialType.ABSTRACT_METAPHOR.value)

    def test_reconstruction_hints_map_to_ai_image(self):
        self.assertEqual(
            _classify_material_type("No photographs survive; we must reconstruct the temple"),
            MaterialType.AI_IMAGE.value,
        )
        self.assertEqual(
            _classify_material_type("No footage exists; imagine the battle"),
            MaterialType.AI_IMAGE.value,
        )

    def test_default_is_stock_video_not_ai_image(self):
        for narration in (
            "The company grew steadily over the years",
            "He walked to work every morning",
            "The market opened quietly",
        ):
            with self.subTest(narration=narration):
                self.assertEqual(_classify_material_type(narration), MaterialType.STOCK_VIDEO.value)


class TestScenePlanning(unittest.TestCase):
    def _mixed_script(self):
        return (
            "Amazon spent a decade and billions trying to conquer groceries. "
            "Then it quietly surrendered. "
            "Revenue jumped 300 percent in the first year. "
            "The company opened its first store in Seattle. "
            "Investor confidence collapsed. "
            "Analysts said the strategy was doomed. "
            "Amazon acquired Whole Foods in 2017 for 13.7 billion dollars. "
            "The lesson: even giants can lose a war they started."
        )

    def test_scene_plan_uses_mixed_material_types(self):
        plan = plan_scenes(self._mixed_script(), get_content_profile("business"), desired_scene_count=6)
        self.assertIsInstance(plan, ScenePlan)
        types = {scene.material_type for scene in plan.scenes}
        # The critical rule: NOT every scene becomes AI_IMAGE.
        self.assertNotIn(MaterialType.AI_IMAGE.value, types) or self.assertLess(
            plan.ai_image_count, len(plan.scenes)
        )
        # Multiple distinct material types prove meaning-based selection.
        self.assertGreaterEqual(len(types), 3)

    def test_not_every_scene_becomes_ai_image_even_when_requested(self):
        """Even when every scene asks for reconstruction, the budget caps AI
        images far below the scene count."""
        script = (
            "No photographs survive of the temple. We must imagine what it looked like. "
            "No footage exists of the battle. Reconstruct the moment from records. "
            "No surviving image of the king remains. Imagine the coronation. "
            "Artist's impression of the lost city. Recreate the shipwreck."
        )
        plan = plan_scenes(script, get_content_profile("history"), desired_scene_count=8, ai_image_budget_ratio=0.2)
        self.assertEqual(len(plan.scenes), 8)
        self.assertLess(plan.ai_image_count, len(plan.scenes))
        self.assertLessEqual(plan.ai_image_count, plan.ai_image_budget)

    def test_ai_image_budget_is_respected(self):
        script = "No footage exists. Imagine it. " * 6
        plan = plan_scenes(script, get_content_profile("mystery"), desired_scene_count=12, ai_image_budget_ratio=0.1)
        self.assertLessEqual(plan.ai_image_count, plan.ai_image_budget)
        self.assertLessEqual(plan.ai_image_count, max(1, round(12 * 0.1)))

    def test_scene_search_terms_are_deduplicated(self):
        script = "Amazon spent billions. Amazon then opened stores. Amazon grew."
        plan = plan_scenes(script, get_content_profile("business"), desired_scene_count=3)
        seen = set()
        for scene in plan.scenes:
            for term in scene.search_terms:
                if term in seen:
                    self.fail(f"search term repeated across scenes: {term}")
                seen.add(term)

    def test_continuity_notes_track_repeated_subjects(self):
        script = "Amazon spent billions. Amazon opened stores. Amazon surrendered."
        plan = plan_scenes(script, get_content_profile("business"), desired_scene_count=3)
        self.assertTrue(any("amazon" in note for note in plan.continuity_notes))

    def test_every_scene_has_rationale_and_intent(self):
        plan = plan_scenes(self._mixed_script(), get_content_profile("business"), desired_scene_count=6)
        for scene in plan.scenes:
            self.assertTrue(scene.rationale)
            self.assertTrue(scene.visual_intent)
            self.assertTrue(scene.search_terms)

    def test_summary_compact(self):
        plan = plan_scenes(self._mixed_script(), get_content_profile("business"), desired_scene_count=6)
        summary = scene_plan_summary(plan)
        self.assertIn("scenes=", summary)
        self.assertIn("ai_images=", summary)
        self.assertEqual(scene_plan_summary(None), "none")

    def test_intelligence_style_flows_into_plan(self):
        intel, _ = build_content_intelligence("topic", get_content_profile("finance"))
        plan = plan_scenes(
            "Revenue rose 5 percent last quarter.",
            get_content_profile("finance"),
            intelligence=intel,
        )
        self.assertTrue(plan.style_language)

    def test_all_material_types_are_valid(self):
        self.assertIn("stock_video", ALL_MATERIAL_TYPES)
        self.assertIn("chart", ALL_MATERIAL_TYPES)
        self.assertIn("archival", ALL_MATERIAL_TYPES)
        self.assertIn("ai_image", ALL_MATERIAL_TYPES)
        self.assertIn("map", ALL_MATERIAL_TYPES)


if __name__ == "__main__":
    unittest.main()
