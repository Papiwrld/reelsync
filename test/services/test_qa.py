"""Tests for the structured QA engine (Phase 2D.4)."""

import unittest

from app.services.content_profile import get_content_profile
from app.services.qa import (
    QaIssue,
    QaReport,
    QaSeverity,
    qa_summary,
    run_quality_assurance,
)
from app.services.research import ClaimStatus, ResearchClaim
from app.services.visual_director import plan_scenes


class _Title:
    def __init__(self, text, accuracy):
        self.text = text
        self.scores = {"accuracy": accuracy}


class TestQaEngine(unittest.TestCase):
    def setUp(self):
        self.profile = get_content_profile("business")

    def test_empty_script_is_critical_and_blocks(self):
        report = run_quality_assurance("", self.profile)
        self.assertTrue(report.publication_blocked)
        self.assertTrue(
            any(issue.severity == QaSeverity.CRITICAL.value for issue in report.issues)
        )

    def test_healthy_input_does_not_block(self):
        script = "Amazon acquired Whole Foods in 2017. Revenue grew. The company lost the grocery war."
        scenes = plan_scenes(script, self.profile, desired_scene_count=3)
        report = run_quality_assurance(
            script,
            self.profile,
            research_claims=[
                ResearchClaim(statement="Amazon acquired Whole Foods in 2017", status=ClaimStatus.VERIFIED)
            ],
            scene_plan=scenes,
            selected_title=_Title("Why Amazon lost the grocery war", 8.0),
            platform="youtube",
        )
        self.assertFalse(report.publication_blocked)
        self.assertIn("ready to publish", report.summary)

    def test_unsupported_research_claims_raise_error(self):
        script = "Revenue grew 300 percent last year."
        report = run_quality_assurance(
            script,
            self.profile,
            research_claims=[
                ResearchClaim(statement="Revenue grew 300 percent", status=ClaimStatus.UNSUPPORTED),
                ResearchClaim(statement="The market shrank", status=ClaimStatus.UNSUPPORTED),
            ],
        )
        self.assertTrue(any(issue.severity == QaSeverity.ERROR.value and issue.category == "research" for issue in report.issues))

    def test_contradictions_are_flagged(self):
        report = run_quality_assurance(
            "The company grew quickly.",
            self.profile,
            research_contradictions=["source A says up, source B says down"],
        )
        self.assertTrue(any("contradict" in issue.message.lower() for issue in report.issues))

    def test_cached_model_knowledge_is_flagged_as_info(self):
        report = run_quality_assurance(
            "The company grew quickly.",
            self.profile,
            research_provenance={"cached": True, "model_knowledge": True, "provider": ""},
        )
        categories = {issue.category for issue in report.issues}
        self.assertIn("research", categories)

    def test_missing_scene_plan_warns(self):
        report = run_quality_assurance("A normal length script sentence.", self.profile, scene_plan=None)
        self.assertTrue(
            any(issue.severity == QaSeverity.WARNING.value and issue.category == "visuals" for issue in report.issues)
        )

    def test_excessive_ai_images_error(self):
        script = "No footage exists. Imagine it. No photo survives. Recreate it. " * 4
        scenes = plan_scenes(script, self.profile, desired_scene_count=16, ai_image_budget_ratio=0.05)
        # Force a plan that violates the budget to verify the QA guard.
        scenes.ai_image_count = len(scenes.scenes)
        report = run_quality_assurance(
            script, self.profile, scene_plan=scenes, ai_image_budget_ratio=0.05
        )
        self.assertTrue(
            any(issue.severity == QaSeverity.ERROR.value and issue.category == "visuals" for issue in report.issues)
        )

    def test_inaccurate_title_is_error(self):
        report = run_quality_assurance(
            "A perfectly normal script sentence.",
            self.profile,
            selected_title=_Title("Completely misleading clickbait title", 3.0),
        )
        self.assertTrue(
            any(issue.severity == QaSeverity.ERROR.value and issue.category == "metadata" for issue in report.issues)
        )

    def test_long_sentences_warn_audio(self):
        long_script = (
            "This is an extremely long sentence that goes on and on and keeps adding "
            "more clauses and more detail and more qualifications until the narrator "
            "would run out of breath completely and the viewer would lose attention. "
            "Another similarly long sentence that never seems to end and keeps piling "
            "up subordinate clauses one after another without ever pausing for breath "
            "and drags itself out well beyond the comfortable narration length limit. "
        )
        report = run_quality_assurance(long_script, self.profile)
        self.assertTrue(any(issue.category == "audio" for issue in report.issues))

    def test_issues_have_category_message_evidence(self):
        report = run_quality_assurance("", self.profile)
        for issue in report.issues:
            self.assertIsInstance(issue, QaIssue)
            self.assertTrue(issue.category)
            self.assertTrue(issue.message)

    def test_summary_counts_severities(self):
        report = run_quality_assurance("", self.profile)
        summary = qa_summary(report)
        self.assertIn("critical", summary)
        self.assertEqual(qa_summary(None), "none")

    def test_report_model(self):
        report = QaReport(issues=[QaIssue(severity="error", category="script", message="x")])
        self.assertEqual(len(report.issues), 1)

    def test_numeric_claims_backed_by_research_no_issue(self):
        report = run_quality_assurance(
            "The market grew 30% last year.",
            self.profile,
            research_claims=[
                ResearchClaim(statement="The market grew 30 percent in 2023", status=ClaimStatus.VERIFIED)
            ],
        )
        self.assertFalse(any("script claims" in issue.message for issue in report.issues))

    def test_unbacked_numeric_claim_warns(self):
        report = run_quality_assurance(
            "The market grew 300% last year.",
            self.profile,
            research_claims=[
                ResearchClaim(statement="Revenue grew steadily over the decade", status=ClaimStatus.VERIFIED)
            ],
        )
        self.assertTrue(
            any(
                issue.severity == QaSeverity.WARNING.value
                and "script claims 300%" in issue.message
                and "do not mention" in issue.message
                for issue in report.issues
            )
        )

    def test_contradicting_numeric_claim_errors(self):
        report = run_quality_assurance(
            "Revenue hit 30% this year.",
            self.profile,
            research_claims=[
                ResearchClaim(statement="Revenue hit 45 percent last year", status=ClaimStatus.VERIFIED)
            ],
        )
        self.assertTrue(
            any(
                issue.severity == QaSeverity.ERROR.value
                and "script claims 30%" in issue.message
                and "conflicting" in issue.message
                for issue in report.issues
            )
        )

    def test_numbers_without_research_are_info(self):
        report = run_quality_assurance("The market grew 300% last year.", self.profile, research_claims=None)
        self.assertTrue(
            any(
                issue.severity == QaSeverity.INFO.value
                and "cannot verify numeric claims" in issue.message
                for issue in report.issues
            )
        )

    def test_years_are_not_flagged_as_numeric_claims(self):
        report = run_quality_assurance(
            "Founded in 2024. The company was restructured in 2021.",
            self.profile,
            research_claims=None,
        )
        self.assertFalse(any("script claims" in issue.message for issue in report.issues))
        self.assertFalse(any("cannot verify numeric claims" in issue.message for issue in report.issues))

    def test_currency_claim_contradiction_errors(self):
        report = run_quality_assurance(
            "The deal was worth $5 billion.",
            self.profile,
            research_claims=[
                ResearchClaim(statement="The deal was valued at $4 billion", status=ClaimStatus.VERIFIED)
            ],
        )
        self.assertTrue(
            any(
                issue.severity == QaSeverity.ERROR.value
                and "script claims $5 billion" in issue.message
                and "conflicting" in issue.message
                for issue in report.issues
            )
        )


if __name__ == "__main__":
    unittest.main()
