"""Tests for the agentic content planning layer (Phase 1)."""

import json
import unittest
from unittest.mock import patch

from app.models.schema import VideoParams
from app.services import agentic, task as tm
from app.services.agentic import (
    AGENTIC_APPROVE_THRESHOLD,
    HookCandidate,
    ScriptReview,
    plan_video_content,
    score_hook,
    score_hook_candidates,
)
from app.services.agent_llm import AgentTracker
from app.services.content_profile import (
    get_content_profile,
    list_content_profiles,
    profile_strategy_context,
)
from app.services.generation_state import GenerationState

_PROFILE_NAMES = {
    "dark_history",
    "mystery",
    "mythology_lore",
    "african_history",
    "technology",
    "ai_news",
    "finance",
    "motivation",
    "education",
    "science",
    "storytelling",
    "business",
    "history",
    "psychology",
    "gaming",
    "custom",
}


class _QueueLLM:
    """Fake LLM returning canned responses in order (JSON or text)."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, prompt, app_config=None):
        self.calls.append(prompt)
        if not self.responses:
            raise AssertionError("llm called more times than canned responses")
        return self.responses.pop(0)


class TestContentProfile(unittest.TestCase):
    def test_registry_contains_all_builtin_profiles(self):
        self.assertEqual(set(list_content_profiles()), _PROFILE_NAMES)

    def test_every_profile_has_required_fields(self):
        for name in list_content_profiles():
            with self.subTest(profile=name):
                profile = get_content_profile(name)
                self.assertEqual(profile.name, name)
                self.assertTrue(profile.description)
                if name == "custom":
                    # custom profile is intentionally blank; users fill it in
                    continue
                self.assertTrue(profile.tone)
                self.assertTrue(profile.cta_strategy)
                self.assertTrue(profile.hook_strategy)
                self.assertTrue(profile.pacing)
                self.assertTrue(profile.media_strategy)

    def test_unknown_profile_falls_back_to_custom(self):
        profile = get_content_profile("no-such-niche")
        self.assertEqual(profile.name, "custom")
        self.assertEqual(get_content_profile(""), "custom" and get_content_profile("custom"))

    def test_strategy_context_only_includes_filled_fields(self):
        profile = get_content_profile("mystery")
        context = profile_strategy_context(profile)
        self.assertIn("Content Profile: mystery", context)
        self.assertIn("Hook strategy:", context)
        self.assertIn("Tone:", context)
        # fields left empty must not be rendered as empty lines
        self.assertNotIn("\n- Description: \n", context)

    def test_custom_profile_respects_custom_instructions(self):
        profile = get_content_profile("custom")
        context = profile_strategy_context(profile)
        self.assertIn("Content Profile: custom", context)


class TestGenerationState(unittest.TestCase):
    def test_round_trip_json_preserves_fields(self):
        state = GenerationState(
            user_input="atlantis",
            profile_name="mystery",
            selected_hook="Why does Atlantis keep disappearing?",
            script="Hello world.",
            revision_count=2,
        )
        restored = GenerationState.from_json(state.to_json())
        self.assertEqual(restored.model_dump(), state.model_dump())

    def test_from_json_tolerates_garbage(self):
        restored = GenerationState.from_json("{not json")
        self.assertEqual(restored.user_input, "")
        self.assertEqual(restored.profile_name, "")

    def test_stage_summary_compact(self):
        state = GenerationState(profile_name="mystery", script="one two three")
        summary = state.stage_summary()
        self.assertIn("profile=mystery", summary)
        self.assertIn("script_words=3", summary)


class TestLlvmJsonHelpers(unittest.TestCase):
    def test_backoff_delay_increases_with_attempts(self):
        with patch.object(agentic.llm.random, "uniform", return_value=1.0):
            delays = [agentic.llm._backoff_delay(i) for i in range(4)]
        self.assertEqual(delays, [1.0, 2.0, 4.0, 8.0])
        for index in range(1, len(delays)):
            self.assertGreaterEqual(delays[index], delays[index - 1])

    def test_backoff_delay_bounded_by_max(self):
        for attempt in range(10):
            delay = agentic.llm._backoff_delay(attempt)
            self.assertLessEqual(delay, 8.0)

    def test_backoff_delay_jittered_around_base(self):
        delays = [agentic.llm._backoff_delay(0) for _ in range(20)]
        # base 1.0s jittered to [0.5, 1.5): values must vary, not be constant.
        self.assertGreater(len(set(delays)), 1)
        for delay in delays:
            self.assertGreaterEqual(delay, 0.5)
            self.assertLessEqual(delay, 1.5)

    def test_extract_json_payload_strips_code_fence(self):
        raw = '```json\n{"a": 1}\n```'
        self.assertEqual(agentic._extract_json_payload(raw), {"a": 1})

    def test_extract_json_payload_tolerates_prose_wrap(self):
        raw = 'Sure! Here is the result: {"sections": ["a", "b"]} Hope it helps.'
        self.assertEqual(
            agentic._extract_json_payload(raw), {"sections": ["a", "b"]}
        )

    def test_extract_json_payload_extracts_arrays(self):
        raw = 'text [{"text": "x", "style": "mystery", "rationale": "y"}] tail'
        payload = agentic._extract_json_payload(raw)
        self.assertEqual(payload[0]["text"], "x")

    def test_extract_json_payload_rejects_malformed(self):
        with self.assertRaises(ValueError):
            agentic._extract_json_payload("no json at all here")

    def test_llm_json_uses_fallback_on_failure(self):
        def broken(*_args, **_kwargs):
            raise RuntimeError("provider down")

        with patch.object(agentic.llm, "_generate_response", side_effect=broken):
            result = agentic._llm_json("prompt", lambda: {"fallback": True})
        self.assertEqual(result, {"fallback": True})

    def test_error_string_trips_circuit_breaker(self):
        """供应商返回的 "Error: ..." 字符串必须被视为失败：熔断器打开后，
        同一 run 的后续 LLM 调用直接短路，不再发起注定失败的请求。"""
        tracker = AgentTracker()
        calls = []

        def failing(*_args, **_kwargs):
            calls.append(1)
            return "Error: 429 quota exhausted"

        with patch.object(agentic.llm, "_generate_response", side_effect=failing):
            with self.assertRaises(agentic.AgenticError):
                agentic._llm_text("prompt", tracker=tracker, agent="a")
        self.assertTrue(tracker.degraded)
        self.assertIn("429", tracker.degrade_reason)
        # 熔断后：后续调用不再触碰供应商
        calls.clear()
        result = agentic._llm_json(
            "prompt2", lambda: {"fallback": True}, tracker=tracker, agent="b"
        )
        self.assertEqual(result, {"fallback": True})
        self.assertEqual(calls, [])
        self.assertEqual(tracker.statuses.get("b"), "fallback")
        self.assertIn("degraded", tracker.reasons.get("b", ""))

    def test_llm_text_skips_call_when_tracker_degraded(self):
        """熔断后 _llm_text 也应直接抛错而不是发起请求。"""
        tracker = AgentTracker()
        tracker.mark_degraded("Error: 429 quota exhausted")
        calls = []

        def side_effect(*_args, **_kwargs):
            calls.append(1)
            return "unused"

        with patch.object(agentic.llm, "_generate_response", side_effect=side_effect):
            with self.assertRaises(agentic.AgenticError):
                agentic._llm_text("prompt", tracker=tracker, agent="c")
        self.assertEqual(calls, [])


class TestHookScoring(unittest.TestCase):
    def setUp(self):
        self.profile = get_content_profile("mystery")

    def test_question_hook_outranks_flat_statement(self):
        question = HookCandidate(
            text="Why does the lost city of Atlantis keep vanishing?",
            style="question",
            rationale="open question",
        )
        flat = HookCandidate(
            text="Atlantis is a topic that people discuss.",
            style="statement",
            rationale="plain",
        )
        scored = score_hook_candidates([flat, question], "lost city of Atlantis", self.profile)
        self.assertGreater(scored[0]["overall"], scored[1]["overall"])
        self.assertEqual(scored[0]["text"], question.text)

    def test_scores_are_bounded(self):
        text = "Nobody knows what really happened in Atlantis. Why?"
        scores = score_hook(text, "Atlantis", self.profile)
        for dimension, value in scores.items():
            with self.subTest(dimension=dimension):
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 10.0)

    def test_select_best_hook_returns_top(self):
        candidates = [
            HookCandidate(text=f"hook {index}", style="mystery", rationale="r")
            for index in range(3)
        ]
        scored = score_hook_candidates(candidates, "topic", self.profile)
        text, record = agentic.select_best_hook(scored)
        self.assertEqual(text, record["text"])
        self.assertEqual(text, scored[0]["text"])

    def test_select_best_hook_empty_raises(self):
        with self.assertRaises(agentic.AgenticError):
            agentic.select_best_hook([])

    def test_stem_relevance_matches_inflected_topic_words(self):
        self.assertTrue(
            agentic._stem_overlap(["colonists"], ["colony"])
        )
        self.assertTrue(
            agentic._stem_overlap(["vanished"], ["vanish"])
        )
        self.assertFalse(
            agentic._stem_overlap(["dancing"], ["colony"])
        )

    def test_topic_words_drop_stopwords(self):
        self.assertEqual(
            agentic._topic_words("The disappearance of the Roanoke Colony"),
            ["disappearance", "roanoke", "colony"],
        )

    def test_ungrounded_clickbait_is_capped_without_judge(self):
        clickbait = HookCandidate(
            text="You won't believe what happened next.",
            style="clickbait",
            rationale="generic",
        )
        grounded = HookCandidate(
            text="Why did 115 settlers vanish from Roanoke Colony in 1590?",
            style="question",
            rationale="specific",
        )
        scored = score_hook_candidates([clickbait, grounded], "Roanoke Colony disappearance", self.profile)
        by_text = {item["text"]: item for item in scored}
        self.assertLessEqual(by_text[clickbait.text]["overall"], 5.0)
        self.assertGreater(by_text[grounded.text]["overall"], by_text[clickbait.text]["overall"])

    def test_judge_outranks_clickbait_semantically(self):
        clickbait = HookCandidate(
            text="You won't believe what happened next.",
            style="clickbait",
            rationale="generic",
        )
        specific = HookCandidate(
            text="In 1590, 115 settlers vanished without a trace. No bodies. No battle.",
            style="specific",
            rationale="concrete",
        )
        candidates = [clickbait, specific]
        judge_payload = [
            {"index": 0, "relevance": 1.0, "quality": 3.0, "why": "generic"},
            {"index": 1, "relevance": 9.0, "quality": 9.0, "why": "specific and credible"},
        ]
        with patch.object(
            agentic.llm, "_generate_response", return_value=json.dumps(judge_payload)
        ):
            judged = agentic.judge_hooks(candidates, "the disappearance of the Roanoke Colony", self.profile)
        self.assertEqual(judged[1], 9.0)
        scored = score_hook_candidates(candidates, "the disappearance of the Roanoke Colony", self.profile)
        for item in scored:
            item["overall"] = round(0.5 * item["overall"] + 0.5 * judged[item["index"]], 1)
        scored.sort(key=lambda item: item["overall"], reverse=True)
        self.assertEqual(scored[0]["text"], specific.text)

    def test_judge_malformed_output_returns_none(self):
        candidates = [HookCandidate(text=f"hook {index}", style="mystery", rationale="r") for index in range(3)]
        with patch.object(agentic.llm, "_generate_response", return_value="not json"):
            judged = agentic.judge_hooks(candidates, "topic", self.profile)
        self.assertIsNone(judged)

    def test_judge_llm_down_returns_none(self):
        def broken(*_args, **_kwargs):
            raise RuntimeError("provider down")

        candidates = [HookCandidate(text="hook", style="mystery", rationale="r")]
        with patch.object(agentic.llm, "_generate_response", side_effect=broken):
            judged = agentic.judge_hooks(candidates, "topic", self.profile)
        self.assertIsNone(judged)

    def test_judge_invalid_scores_are_clamped(self):
        candidates = [HookCandidate(text=f"hook {index}", style="mystery", rationale="r") for index in range(2)]
        judge_payload = [
            {"index": 0, "relevance": 99.0, "quality": -3.0, "why": "bad"},
            {"index": 1, "relevance": 5.0, "quality": 5.0, "why": "ok"},
        ]
        with patch.object(
            agentic.llm, "_generate_response", return_value=json.dumps(judge_payload)
        ):
            judged = agentic.judge_hooks(candidates, "topic", self.profile)
        self.assertLessEqual(judged[0], 10.0)
        self.assertGreaterEqual(judged[0], 0.0)


class TestCritique(unittest.TestCase):
    def test_verdict_approved_above_threshold(self):
        review = ScriptReview(
            scores={"hook": 9.0},
            overall=9.0,
            verdict="APPROVE",
            feedback="strong",
        )
        self.assertEqual(review.verdict, "APPROVE")

    def test_critique_parses_scores_and_decides_verdict(self):
        payload = {
            "hook": 8.0,
            "niche_alignment": 9.0,
            "narrative": 8.0,
            "visual_potential": 9.0,
            "pacing": 8.0,
            "ending": 7.0,
            "cta_quality": 8.0,
            "feedback": "tighten the ending",
        }
        raw = json.dumps(payload)
        with patch.object(agentic.llm, "_generate_response", return_value=raw):
            review = agentic.critique_script(
                "script", "topic", get_content_profile("mystery"), agentic.ContentStrategy(primary_angle="a")
            )
        self.assertEqual(review.overall, 8.1)
        self.assertEqual(review.verdict, "APPROVE")
        self.assertEqual(review.feedback, "tighten the ending")

    def test_critique_rejects_low_score(self):
        payload = {k: 2.0 for k in ("hook", "niche_alignment", "narrative", "visual_potential", "pacing", "ending", "cta_quality")}
        raw = json.dumps(payload)
        with patch.object(agentic.llm, "_generate_response", return_value=raw):
            review = agentic.critique_script(
                "script", "topic", get_content_profile("mystery"), agentic.ContentStrategy(primary_angle="a")
            )
        self.assertLess(review.overall, AGENTIC_APPROVE_THRESHOLD)
        self.assertEqual(review.verdict, "REVISE")

    def test_critique_clamps_out_of_range_scores(self):
        payload = {
            "hook": 99.0,
            "niche_alignment": -5.0,
            "narrative": 8.0,
            "visual_potential": 8.0,
            "pacing": 8.0,
            "ending": 8.0,
            "cta_quality": 8.0,
            "feedback": "ok",
        }
        raw = json.dumps(payload)
        with patch.object(agentic.llm, "_generate_response", return_value=raw):
            review = agentic.critique_script(
                "script", "topic", get_content_profile("mystery"), agentic.ContentStrategy(primary_angle="a")
            )
        self.assertLessEqual(review.overall, 10.0)
        self.assertGreaterEqual(review.overall, 0.0)
        self.assertLessEqual(review.scores["hook"], 10.0)
        self.assertGreaterEqual(review.scores["niche_alignment"], 0.0)

    def test_critique_garbage_scores_fall_back_to_heuristics(self):
        payload = {k: "not a number" for k in ("hook", "niche_alignment", "narrative", "visual_potential", "pacing", "ending", "cta_quality")}
        raw = json.dumps(payload)
        with patch.object(agentic.llm, "_generate_response", return_value=raw):
            review = agentic.critique_script(
                "script", "topic", get_content_profile("mystery"), agentic.ContentStrategy(primary_angle="a")
            )
        self.assertIn(review.verdict, ("APPROVE", "REVISE"))
        for value in review.scores.values():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 10.0)


class TestPlanVideoContent(unittest.TestCase):
    def _canned_planning_responses(self, hook_style="question", critique_overall=8.5):
        hook_texts = [f"hook {index}" for index in range(5)]
        return [
            json.dumps(
                {
                    "topic_type": "historical mystery",
                    "historical_context": "Plato wrote about it.",
                    "potential_claims": ["Atlantis was a real place"],
                    "emotional_angles": ["wonder", "unease"],
                    "curiosity_gaps": ["where exactly it was"],
                    "controversy_level": "high",
                    "known_vs_unknown": "known: Plato's account; unknown: location",
                    "visual_opportunities": ["ocean depths", "ruins", "maps"],
                    "audience_interest": "unsolved mystery",
                    "possible_hooks": ["Why does Atlantis keep vanishing?"],
                    "narrative_options": ["question then evidence"],
                    "research_requirements": ["verify Plato's account"],
                }
            ),
            json.dumps(
                {
                    "primary_angle": "nobody knows whether Atlantis was real",
                    "hook_strategy": "open with uncertainty",
                    "emotional_progression": ["curiosity", "mystery", "wonder"],
                    "pacing": "fast opening, slow reveal",
                    "narrative_structure": ["Hook", "Context", "Evidence", "Open question"],
                    "cta": "ask what they believe",
                }
            ),
            json.dumps(
                [
                    {"text": text, "style": hook_style, "rationale": f"r{index}"}
                    for index, text in enumerate(hook_texts)
                ]
            ),
            json.dumps(
                [
                    {"index": index, "relevance": 7.0, "quality": 7.0, "why": "ok"}
                    for index in range(5)
                ]
            ),
            json.dumps({"sections": ["Hook", "Context", "Evidence", "Open question"]}),
            "This is the script narration for the video. It has enough words to feel real and continues with more content about the topic.",
            json.dumps(
                {
                    "hook": critique_overall,
                    "niche_alignment": critique_overall,
                    "narrative": critique_overall,
                    "visual_potential": critique_overall,
                    "pacing": critique_overall,
                    "ending": critique_overall,
                    "cta_quality": critique_overall,
                    "feedback": "strong script",
                }
            ),
            json.dumps(
                [
                    {
                        "text": f"Accurate title candidate {index} about the topic",
                        "style": "direct",
                        "rationale": "accurate and grounded",
                    }
                    for index in range(5)
                ]
            ),
        ]

    def test_full_flow_produces_state_with_script(self):
        fake = _QueueLLM(self._canned_planning_responses())
        with patch.object(agentic.llm, "_generate_response", side_effect=fake):
            state = plan_video_content(
                subject="The lost city of Atlantis",
                profile_name="mystery",
            )
        self.assertEqual(state.profile_name, "mystery")
        self.assertEqual(state.topic_analysis["topic_type"], "historical mystery")
        self.assertEqual(state.content_strategy["primary_angle"], "nobody knows whether Atlantis was real")
        self.assertIsNotNone(state.selected_hook)
        self.assertEqual(len(state.hook_candidates), 5)
        self.assertEqual(state.narrative_plan["sections"][0], "Hook")
        self.assertIn("script", state.script)
        self.assertEqual(state.script_review["verdict"], "APPROVE")
        self.assertEqual(state.revision_count, 0)

        # Phase 2C-2E produce structured state (previously dead fields).
        self.assertTrue(state.story_brief)
        self.assertTrue(state.scenes)
        self.assertEqual(len(state.scene_plan["scenes"]), len(state.scenes))
        self.assertTrue(state.title_candidates)
        self.assertTrue(state.selected_title)
        self.assertTrue(state.thumbnail_concept)
        self.assertTrue(state.repurposing_plan["shorts"])
        self.assertTrue(state.qa_report["issues"])
        self.assertEqual(state.final_review["verdict"], "approved")
        self.assertEqual(
            state.final_review["summary"], state.qa_report["summary"]
        )

    def test_research_flows_into_brief_script_and_qa(self):
        """链式验证：研究声明 → StoryBrief → 脚本上下文 → QA。

        脚本写入器必须收到 research packet（作为数据），story brief 必须
        记录已核验声明，QA 必须把未支持的声明标记出来——而不是各阶段
        各写各的、互不消费。
        """
        from app.services.research import ClaimStatus, ResearchClaim, ResearchPacket

        research_packet = ResearchPacket(
            topic="Atlantis",
            claims=[
                ResearchClaim(
                    statement="Plato described Atlantis in Timaeus",
                    status=ClaimStatus.VERIFIED,
                    source_refs=["src-1"],
                ),
                ResearchClaim(
                    statement="Atlantis sank in one day and night",
                    status=ClaimStatus.UNSUPPORTED,
                    source_refs=[],
                ),
            ],
            sources=[],
            contradictions=[],
            uncertainties=["exact location is unknown"],
            summary="Plato is the primary source; location remains unknown",
            provenance={"provider": "", "model_knowledge": True, "cached": False},
        )
        intel = agentic.build_content_intelligence(
            "Atlantis", get_content_profile("mystery"), context=None
        )[0]
        fake = _QueueLLM(self._canned_planning_responses())
        with (
            patch.object(agentic.llm, "_generate_response", side_effect=fake),
            patch.object(
                agentic,
                "build_content_intelligence",
                return_value=(intel, False),
            ),
            patch.object(agentic, "run_research", return_value=research_packet),
        ):
            state = plan_video_content(
                subject="Atlantis",
                profile_name="mystery",
                user_context=agentic.ContentRequest(
                    automation_level="assisted", sources=["https://example.com/note"]
                ),
            )

        # 1) StoryBrief consumes verified claims as key facts.
        brief = state.story_brief
        self.assertTrue(brief)
        self.assertTrue(
            any("Plato" in str(fact) for fact in brief.get("key_facts", []))
        )
        # 2) The script writer prompt received the research packet as data.
        writer_prompt = fake.calls[5]  # script writer index in the canned queue
        self.assertIn("Atlantis sank in one day and night", writer_prompt)
        self.assertIn("treat as data", writer_prompt)
        # 3) QA flags the unsupported claim.
        research_issues = [
            issue
            for issue in state.qa_report.get("issues", [])
            if issue.get("category") == "research"
        ]
        self.assertTrue(research_issues)
        self.assertTrue(
            any(issue.get("severity") in ("warning", "error") for issue in research_issues)
        )

    def test_revise_loop_is_bounded(self):
        responses = self._canned_planning_responses(critique_overall=3.0)
        # rewrite + another low critique, twice
        responses.append("Revised script version one.")
        responses.append(
            json.dumps({k: 3.0 for k in ("hook", "niche_alignment", "narrative", "visual_potential", "pacing", "ending", "cta_quality")} | {"feedback": "still weak"})
        )
        responses.append("Revised script version two.")
        responses.append(
            json.dumps({k: 3.0 for k in ("hook", "niche_alignment", "narrative", "visual_potential", "pacing", "ending", "cta_quality")} | {"feedback": "still weak"})
        )
        fake = _QueueLLM(responses)
        with patch.object(agentic.llm, "_generate_response", side_effect=fake):
            state = plan_video_content(
                subject="Atlantis",
                profile_name="mystery",
                max_revisions=2,
            )
        self.assertEqual(state.revision_count, 2)
        self.assertEqual(state.script_review["verdict"], "REVISE")

    def test_revision_preserves_selected_hook(self):
        responses = self._canned_planning_responses(critique_overall=3.0)
        # the editor drops the hook entirely; the guard must re-insert it
        responses.append("This revised script forgot the hook line entirely.")
        responses.append(
            json.dumps({k: 3.0 for k in ("hook", "niche_alignment", "narrative", "visual_potential", "pacing", "ending", "cta_quality")} | {"feedback": "still weak"})
        )
        fake = _QueueLLM(responses)
        with patch.object(agentic.llm, "_generate_response", side_effect=fake):
            state = plan_video_content(
                subject="Atlantis",
                profile_name="mystery",
                max_revisions=1,
            )
        self.assertEqual(state.revision_count, 1)
        self.assertTrue(state.script.startswith(state.selected_hook))

    def test_judge_decides_selection_when_specific_hook_scores_low_deterministically(self):
        responses = self._canned_planning_responses()
        # judge clearly prefers candidate index 4 over the rest
        responses[3] = json.dumps(
            [
                {"index": i, "relevance": 1.0, "quality": 1.0, "why": "generic"}
                for i in range(4)
            ]
            + [{"index": 4, "relevance": 9.0, "quality": 9.0, "why": "specific"}]
        )
        fake = _QueueLLM(responses)
        with patch.object(agentic.llm, "_generate_response", side_effect=fake):
            state = plan_video_content(subject="Atlantis", profile_name="mystery")
        self.assertEqual(state.selected_hook, "hook 4")
        self.assertTrue(state.hook_candidates[0]["judged_by_llm"])

    def test_task_id_flows_into_state(self):
        """任务管线传入的 task_id 必须出现在状态工件中，便于把一次智能规划
        与对应的视频生成任务关联（可观测性）。"""
        with patch.object(
            agentic.llm,
            "_generate_response",
            side_effect=_QueueLLM(self._canned_planning_responses()),
        ):
            state = plan_video_content(
                subject="Atlantis", profile_name="mystery", task_id="task-123"
            )
        self.assertEqual(state.task_id, "task-123")
        # 两次调用分别使用独立的 canned responses，且都必须在 mock 范围内，
        # 避免测试真的去请求远端 LLM（既慢又会消耗配额）。
        with patch.object(
            agentic.llm,
            "_generate_response",
            side_effect=_QueueLLM(self._canned_planning_responses()),
        ):
            empty_id_state = plan_video_content(
                subject="x", profile_name="mystery", task_id=""
            )
        self.assertEqual(empty_id_state.task_id, "")

    def test_full_flow_records_narrative_usage_history(self):
        """The orchestrator must commit each chosen strategy into the variance
        history, otherwise recent-narrative avoidance never engages."""
        from app.services.narrative import recent_narrative_usage, reset_narrative_usage

        reset_narrative_usage()
        try:
            fake = _QueueLLM(self._canned_planning_responses())
            with patch.object(agentic.llm, "_generate_response", side_effect=fake):
                state = plan_video_content(subject="Atlantis", profile_name="mystery")
            strategy_id = state.narrative_strategy["strategy"].get("id")
            self.assertTrue(strategy_id)
            self.assertIn(strategy_id, recent_narrative_usage("mystery"))
        finally:
            reset_narrative_usage()

    def test_autopilot_records_topic_candidates_and_trends(self):
        """自动/自动驾驶模式必须把评分后的主题候选与趋势信号写入状态工件，
        否则“为什么选这个主题”的可解释链路在任务工件中缺失。"""
        from app.services.research import ResearchPacket

        responses = self._canned_planning_responses()
        # 趋势分析师是自动驾驶路径在标题策略之后新增的一次 LLM 调用。
        responses.append(
            json.dumps(
                [
                    {
                        "topic": "Atlantis tourism",
                        "direction": "rising",
                        "score": 8.0,
                        "note": "model inference",
                    }
                ]
            )
        )
        intel = agentic.build_content_intelligence(
            "Atlantis", get_content_profile("mystery"), context=None
        )[0]
        fake = _QueueLLM(responses)
        with (
            patch.object(agentic.llm, "_generate_response", side_effect=fake),
            patch.object(
                agentic,
                "build_content_intelligence",
                return_value=(intel, False),
            ),
            patch.object(
                agentic,
                "run_research",
                return_value=ResearchPacket(topic="Atlantis"),
            ),
        ):
            state = plan_video_content(
                subject="Atlantis",
                profile_name="mystery",
                user_context=agentic.ContentRequest(automation_level="autopilot"),
            )
        self.assertTrue(state.trend_signals)
        self.assertTrue(state.topic_candidates)
        self.assertTrue(
            any(
                entry["stage"] == "topic_discovery"
                for entry in state.decision_log
            )
        )

    def test_manual_mode_skips_topic_discovery(self):
        """manual 模式不委托主题决策，状态工件不应出现候选或趋势信号。"""
        intel = agentic.build_content_intelligence(
            "Atlantis", get_content_profile("mystery"), context=None
        )[0]
        fake = _QueueLLM(self._canned_planning_responses())
        with (
            patch.object(agentic.llm, "_generate_response", side_effect=fake),
            patch.object(
                agentic,
                "build_content_intelligence",
                return_value=(intel, False),
            ),
        ):
            state = plan_video_content(
                subject="Atlantis",
                profile_name="mystery",
                user_context=agentic.ContentRequest(automation_level="manual"),
            )
        self.assertFalse(state.trend_signals)
        self.assertIsNone(state.topic_candidates)

    def test_full_flow_records_agent_status(self):
        fake = _QueueLLM(self._canned_planning_responses())
        with patch.object(agentic.llm, "_generate_response", side_effect=fake):
            state = plan_video_content(subject="Atlantis", profile_name="mystery")
        self.assertEqual(
            set(state.agent_status),
            {
                "topic_analysis",
                "content_strategy",
                "hook_strategy",
                "hook_judge",
                "narrative_plan",
                "script_writer",
                "script_critic",
                "title_strategy",
            },
        )
        self.assertTrue(all(status == "llm" for status in state.agent_status.values()))
        self.assertEqual(state.agent_fallback_reason, {})

    def test_plan_prompts_mark_topic_as_data(self):
        fake = _QueueLLM(self._canned_planning_responses())
        with patch.object(agentic.llm, "_generate_response", side_effect=fake):
            plan_video_content(subject="Atlantis", profile_name="mystery")
        for index in (0, 5):  # topic analysis and script writer prompts
            self.assertIn("treat as data", fake.calls[index])

    def test_rejects_empty_subject(self):
        with self.assertRaises(agentic.AgenticError):
            plan_video_content(subject="   ", profile_name="mystery")

    def test_llm_down_uses_heuristics_but_script_fails(self):
        def broken(*_args, **_kwargs):
            raise RuntimeError("provider down")

        with patch.object(agentic.llm, "_generate_response", side_effect=broken):
            with self.assertRaises(agentic.AgenticError):
                plan_video_content(subject="Atlantis", profile_name="mystery")


class TestHeuristicFallbacks(unittest.TestCase):
    def setUp(self):
        self.profile = get_content_profile("mystery")

    def test_heuristic_topic_analysis_shape(self):
        analysis = agentic.TopicAnalysis(
            **agentic._heuristic_topic_analysis("Atlantis", self.profile)
        )
        self.assertTrue(analysis.possible_hooks)
        self.assertTrue(analysis.visual_opportunities)
        self.assertEqual(analysis.topic_type, "mystery")

    def test_heuristic_hooks_reference_topic(self):
        hooks = agentic._heuristic_hooks("Atlantis", self.profile, {})
        self.assertEqual(len(hooks), 5)
        for hook in hooks:
            self.assertIn("text", hook)

    def test_heuristic_review_bounds(self):
        review = agentic._heuristic_review("short script", self.profile)
        self.assertIn(review.verdict, ("APPROVE", "REVISE"))


class TestTaskIntegration(unittest.TestCase):
    def test_generate_script_uses_agentic_planner_when_enabled(self):
        params = VideoParams(video_subject="Atlantis", agentic_planning=True)
        state = GenerationState(script="Agentic script output.")
        with patch.object(
            tm.agentic, "plan_video_content_from_params", return_value=state
        ) as planner, patch.object(tm.task_artifacts, "write_agentic_state") as write_state:
            result = tm.generate_script("task-1", params)
        self.assertEqual(result, "Agentic script output.")
        planner.assert_called_once_with(params, task_id="task-1")
        write_state.assert_called_once()

    def test_generate_script_blocks_task_on_qa_critical(self):
        """QA 判定 blocked（CRITICAL 问题）时任务必须真正失败，而不是带着
        不合格的脚本继续生成视频——门禁不能只是状态工件里的一条记录。"""
        params = VideoParams(video_subject="Atlantis", agentic_planning=True)
        state = GenerationState(
            script="script that failed QA",
            qa_report={"summary": "QA: 0 info, 0 warnings, 0 errors, 1 critical - PUBLICATION BLOCKED"},
            final_review={"verdict": "blocked", "summary": "QA blocked"},
        )
        def fake_mark_failed(task_id, stage, error):
            # 模拟真实状态存储：QA 失败后任务确实处于 failed 状态，这样
            # generate_script 的兜底检查不会再用泛化的 "script" 覆盖。
            self.marked_stages.append(stage)
            return {"state": "failed"}

        self.marked_stages = []
        with patch.object(
            tm.agentic, "plan_video_content_from_params", return_value=state
        ), patch.object(tm.task_artifacts, "write_agentic_state"), patch.object(
            tm, "_mark_task_failed", side_effect=fake_mark_failed
        ), patch.object(
            tm.sm.state, "get_task", return_value={"state": tm.const.TASK_STATE_FAILED}
        ):
            result = tm.generate_script("task-3", params)
        self.assertIsNone(result)
        self.assertEqual(self.marked_stages, ["qa"])

    def test_generate_script_falls_back_to_linear_on_agentic_failure(self):
        params = VideoParams(video_subject="Atlantis", agentic_planning=True)
        with patch.object(
            tm.agentic,
            "plan_video_content_from_params",
            side_effect=agentic.AgenticError("boom"),
        ), patch.object(
            tm.llm, "generate_script", return_value="Linear fallback script."
        ) as linear, patch.object(tm.task_artifacts, "write_agentic_state") as write_state:
            result = tm.generate_script("task-2", params)
        self.assertEqual(result, "Linear fallback script.")
        linear.assert_called_once()
        write_state.assert_called_once()
        _, payload = write_state.call_args.args
        self.assertEqual(payload["fallback_used"], "linear_script")
        self.assertIn("boom", payload["fallback_reason"])

    def test_generate_script_keeps_linear_when_agentic_disabled(self):
        params = VideoParams(video_subject="Atlantis", agentic_planning=False)
        with patch.object(
            tm.agentic, "plan_video_content_from_params"
        ) as planner, patch.object(
            tm.llm, "generate_script", return_value="Linear script."
        ) as linear:
            result = tm.generate_script("task-3", params)
        self.assertEqual(result, "Linear script.")
        linear.assert_called_once()
        planner.assert_not_called()


class TestProgressCallback(unittest.TestCase):
    """progress_cb 必须按阶段顺序上报（研究/策略/脚本/QA 等），且回调异常
    绝不能影响规划本身（纯观察性的进度汇报）。"""

    def test_reports_agent_stages_in_order(self):
        fake = _QueueLLM(TestPlanVideoContent()._canned_planning_responses())
        stages = []
        with patch.object(agentic.llm, "_generate_response", side_effect=fake):
            state = plan_video_content(
                subject="The lost city of Atlantis",
                profile_name="mystery",
                progress_cb=stages.append,
            )
        self.assertEqual(state.script_review["verdict"], "APPROVE")
        # 经典流程（无 research 上下文）：智能 -> 分析 -> 策略 -> 钩子 ->
        # 叙事 -> 脚本 -> 画面 -> 标题 -> QA。
        self.assertEqual(
            stages,
            [
                "intelligence",
                "analysis",
                "strategy",
                "hooks",
                "narrative",
                "script",
                "visuals",
                "titles",
                "qa",
            ],
        )
        # 关键顺序：脚本必须先于 QA，QA 必须是最后一个阶段。
        self.assertLess(stages.index("script"), stages.index("qa"))
        self.assertEqual(stages[-1], "qa")

    def test_research_stage_reported_when_research_runs(self):
        from app.services.research import (
            ClaimStatus,
            ResearchClaim,
            ResearchPacket,
        )

        research_packet = ResearchPacket(
            topic="Atlantis",
            claims=[
                ResearchClaim(
                    statement="Plato described Atlantis in Timaeus",
                    status=ClaimStatus.VERIFIED,
                    source_refs=["src-1"],
                )
            ],
            sources=[],
            contradictions=[],
            uncertainties=[],
            summary="Plato is the primary source",
            provenance={"provider": "", "model_knowledge": True, "cached": False},
        )
        intel = agentic.build_content_intelligence(
            "Atlantis", get_content_profile("mystery"), context=None
        )[0]
        fake = _QueueLLM(TestPlanVideoContent()._canned_planning_responses())
        stages = []
        with (
            patch.object(agentic.llm, "_generate_response", side_effect=fake),
            patch.object(
                agentic,
                "build_content_intelligence",
                return_value=(intel, False),
            ),
            patch.object(agentic, "run_research", return_value=research_packet),
        ):
            plan_video_content(
                subject="Atlantis",
                profile_name="mystery",
                user_context=agentic.ContentRequest(
                    automation_level="assisted", sources=["https://example.com/note"]
                ),
                progress_cb=stages.append,
            )
        # 研究阶段出现在智能之后、分析之前。
        self.assertIn("research", stages)
        self.assertLess(stages.index("intelligence"), stages.index("research"))
        self.assertLess(stages.index("research"), stages.index("analysis"))

    def test_revision_rounds_reported(self):
        responses = TestPlanVideoContent()._canned_planning_responses(
            critique_overall=3.0
        )
        responses.append("Revised script version one.")
        responses.append(
            json.dumps(
                {
                    k: 3.0
                    for k in (
                        "hook",
                        "niche_alignment",
                        "narrative",
                        "visual_potential",
                        "pacing",
                        "ending",
                        "cta_quality",
                    )
                }
                | {"feedback": "still weak"}
            )
        )
        responses.append("Revised script version two.")
        responses.append(
            json.dumps(
                {
                    k: 3.0
                    for k in (
                        "hook",
                        "niche_alignment",
                        "narrative",
                        "visual_potential",
                        "pacing",
                        "ending",
                        "cta_quality",
                    )
                }
                | {"feedback": "still weak"}
            )
        )
        fake = _QueueLLM(responses)
        stages = []
        with patch.object(agentic.llm, "_generate_response", side_effect=fake):
            state = plan_video_content(
                subject="Atlantis",
                profile_name="mystery",
                max_revisions=2,
                progress_cb=stages.append,
            )
        self.assertEqual(state.revision_count, 2)
        self.assertEqual(stages.count("script_revision"), 2)
        self.assertLess(stages.index("script"), stages.index("script_revision"))

    def test_raising_callback_does_not_break_planning(self):
        fake = _QueueLLM(TestPlanVideoContent()._canned_planning_responses())

        def boom(stage):
            raise RuntimeError(f"reporter exploded at {stage}")

        with patch.object(agentic.llm, "_generate_response", side_effect=fake):
            state = plan_video_content(
                subject="Atlantis",
                profile_name="mystery",
                progress_cb=boom,
            )
        self.assertTrue(state.script)
        self.assertEqual(state.final_review["verdict"], "approved")


if __name__ == "__main__":
    unittest.main()
