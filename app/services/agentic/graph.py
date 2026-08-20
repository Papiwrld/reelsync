"""Agentic planning graph orchestrator.

This module wires the agent implementations together into the full pipeline,
preserving the original API (plan_video_content, plan_video_content_from_params)
and the exact orchestration order the pre-refactor monolithic ``agentic.py``
produced.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from loguru import logger

from app.config import config
from app.services.agent_llm import AgentTracker, AgenticError
from app.services.agentic.compat import (
    analyze_topic,
    build_narrative_plan,
    develop_content_strategy,
    generate_hook_candidates,
    write_and_critique_script,
)
from app.services.agentic.scoring import (
    AGENTIC_DEFAULT_MAX_REVISIONS,
    judge_hooks,
    score_hook_candidates,
    select_best_hook,
)
from app.services.agentic.utils import _strategy_from_choice
from app.services.content_profile import get_content_profile
from app.services.generation_state import GenerationState
from app.services.intelligence import (
    AUTOMATION_AUTOMATIC,
    AUTOMATION_AUTOPILOT,
    ContentIntelligence,
    ContentRequest,
    intelligence_summary,
)
from app.services.narrative import (
    build_story_brief,
    record_narrative_usage,
    select_narrative_strategy,
)
from app.services.qa import QaSeverity, run_quality_assurance
from app.services.repurposing import plan_repurposing
from app.services.research import ResearchPacket, research_summary
from app.services.titles import compose_thumbnail_concept, generate_title_candidates
from app.services.trends import collect_trend_signals, discover_topics, trends_summary
from app.services.visual_director import plan_scenes


def _claim_status_value(claim: Any) -> str:
    """Normalize a ResearchClaim's status to its lowercased string value."""
    raw = (
        claim.get("status")
        if isinstance(claim, dict)
        else getattr(claim, "status", "")
    )
    if raw is None:
        return ""
    return str(getattr(raw, "value", raw)).lower()


def _configured_max_revisions() -> int:
    """Read max script revisions from the ``[agentic]`` config section."""
    try:
        return int(config.agentic.get("max_script_revisions", AGENTIC_DEFAULT_MAX_REVISIONS))
    except (TypeError, ValueError):
        return AGENTIC_DEFAULT_MAX_REVISIONS


def _select_narrative_for_state(
    subject: str,
    profile,
    intelligence: Optional[ContentIntelligence],
    analysis,
    research_packet: Optional[ResearchPacket],
) -> dict:
    """Run narrative strategy selection; keeps the strategy object in the
    payload (``narrative_strategy`` attribute) and serialized copies for the
    state artifact (``strategy``).
    """
    analysis_text = " ".join(
        part
        for part in (
            analysis.topic_type or "",
            " ".join(analysis.curiosity_gaps or []),
            " ".join(analysis.emotional_angles or []),
            analysis.known_vs_unknown or "",
            analysis.historical_context or "",
        )
        if part
    )
    verified_claims = sum(
        1
        for claim in (research_packet.claims if research_packet else [])
        if _claim_status_value(claim) == "verified"
    )
    choice = select_narrative_strategy(
        topic=subject,
        profile=profile,
        intelligence=intelligence,
        topic_analysis_text=analysis_text,
        research_summary=(research_packet.summary if research_packet else ""),
        verified_claims=verified_claims,
    )
    return {
        "narrative_strategy": choice["strategy"],
        "strategy": choice["strategy"].model_dump(),
        "scores": choice["scores"],
        "rationale": choice["rationale"],
    }


def plan_video_content(
    subject: str,
    language: str = "",
    profile_name: str = "",
    paragraph_number: int = 1,
    target_duration_seconds: int = 0,
    app_config: Any = None,
    max_revisions: Optional[int] = None,
    user_context: Optional[ContentRequest] = None,
    task_id: str = "",
    script_style: str = "",
    progress_cb: Optional[Callable[[str], None]] = None,
) -> GenerationState:
    """Run the content intelligence agent graph and return the structured state.

    With no ``user_context`` this is the classic Phase 1 agent graph: the
    ContentIntelligence contract is composed deterministically from profile
    data (zero LLM cost) and research is skipped. When the user supplied the
    content configuration (niche context / automation level / sources), the
    intelligence node refines via one LLM call and research runs per the
    automation level.

    ``progress_cb`` (optional) receives a short stage key ("intelligence",
    "research", "strategy", "script", "qa", ...) before each stage starts so
    the WebUI can show which agent is running inside the status widget. It is
    purely observational: exceptions inside it are swallowed and never affect
    planning.

    Raises AgenticScriptError if the script cannot be produced; every other
    stage degrades to deterministic fallbacks.
    """
    subject = (subject or "").strip()
    if not subject:
        raise AgenticError("plan_video_content requires a non-empty subject")

    profile = get_content_profile(profile_name)
    if profile.name == "custom" and profile_name and profile_name.strip().lower() != "custom":
        logger.warning(f"unknown content profile {profile_name!r}, falling back to 'custom'")

    if max_revisions is None:
        max_revisions = _configured_max_revisions()
    revisions = max(0, min(10, max_revisions))

    task_context = f" task_id={task_id}" if task_id else ""
    logger.info(f"agentic planning started:{task_context} subject={subject!r} profile={profile.name}")
    tracker = AgentTracker()
    state = GenerationState(user_input=subject, profile_name=profile.name, task_id=task_id)
    if user_context:
        state.automation_level = user_context.automation

    def _report(stage: str) -> None:
        # Progress reporting is purely observational: a broken callback must
        # never disturb the planning graph.
        if progress_cb is None:
            return
        try:
            progress_cb(stage)
        except Exception:  # noqa: BLE001
            logger.debug(f"progress callback failed for stage {stage!r}")

    # Resolved dynamically through the facade so tests that patch
    # agentic.build_content_intelligence / agentic.run_research take effect.
    from app.services import agentic as agentic_pkg

    _report("intelligence")
    intelligence, used_llm = agentic_pkg.build_content_intelligence(
        subject,
        profile,
        context=user_context,
        app_config=app_config,
        tracker=tracker,
    )
    state.content_intelligence = intelligence.model_dump()
    state.record_decision(
        "content_intelligence",
        f"fact_check={intelligence.fact_check_level}, risk={intelligence.risk_profile}, "
        f"narrative={intelligence.narrative_strategy[:60]!r}",
        "composed from profile data"
        if not used_llm
        else "refined by the content intelligence agent",
    )
    logger.info(f"content intelligence: {intelligence_summary(intelligence)}")

    research_packet = None
    if user_context and user_context.research_enabled:
        _report("research")
        research_packet = agentic_pkg.run_research(
            topic=subject,
            profile=profile,
            intelligence=intelligence,
            context=user_context,
            app_config=app_config,
            tracker=tracker,
        )
        state.research_packet = research_packet.model_dump()
        state.record_decision(
            "research",
            f"{research_summary(research_packet)}",
            "research runs because automation level delegates decisions or user supplied sources",
        )
        logger.info(f"research: {research_summary(research_packet)}")

    _report("analysis")
    analysis = analyze_topic(
        subject, profile, app_config=app_config, tracker=tracker, intelligence=intelligence
    )
    state.topic_analysis = analysis.model_dump()

    _report("strategy")
    strategy = develop_content_strategy(
        subject, analysis, profile, app_config=app_config, tracker=tracker, intelligence=intelligence
    )
    state.content_strategy = strategy.model_dump()

    # Select the narrative strategy (deterministic, variance-aware).
    narrative_choice = _select_narrative_for_state(
        subject,
        profile,
        intelligence,
        analysis,
        research_packet,
    )
    state.narrative_strategy = {
        "strategy": narrative_choice["strategy"],
        "scores": narrative_choice["scores"],
        "rationale": narrative_choice["rationale"],
    }
    # Persist the selection into the in-memory variance history so the next
    # plan for this profile can avoid repeating the same structure.
    record_narrative_usage(
        profile.name, narrative_choice["strategy"].get("id", "")
    )
    state.record_decision(
        "narrative_strategy",
        narrative_choice["strategy"].get("id", "?"),
        narrative_choice.get("rationale", "") or "selected from the strategy catalog",
    )

    _report("hooks")
    candidates = generate_hook_candidates(
        subject, analysis, strategy, profile, app_config=app_config, tracker=tracker
    )
    scored = score_hook_candidates(candidates, subject, profile)
    judge_scores = judge_hooks(candidates, subject, profile, app_config=app_config, tracker=tracker)
    if judge_scores:
        for item in scored:
            judged = judge_scores.get(item["index"])
            if judged is not None:
                item["overall"] = round(0.5 * item["overall"] + 0.5 * judged, 1)
                item["judged_by_llm"] = True
        scored.sort(key=lambda item: item["overall"], reverse=True)
        logger.info("hook selection: hybrid (deterministic + LLM judge)")
    else:
        logger.info("hook selection: deterministic only (LLM judge unavailable)")
    hook, hook_record = select_best_hook(scored)
    state.hook_candidates = scored
    state.selected_hook = hook
    logger.info(f"hook selected: {hook[:80]!r} (overall={hook_record['overall']})")

    _report("narrative")
    narrative = build_narrative_plan(strategy, profile, app_config=app_config, tracker=tracker)
    state.narrative_plan = narrative.model_dump()

    # Build the structured story brief from the produced context.
    story_brief = build_story_brief(
        topic=subject,
        profile=profile,
        strategy=_strategy_from_choice(narrative_choice),
        selected_hook=hook,
        intelligence=intelligence,
        topic_analysis=state.topic_analysis,
        content_strategy=state.content_strategy,
        research_claims=(research_packet.claims if research_packet else None),
        research_uncertainties=(research_packet.uncertainties if research_packet else None),
        research_contradictions=(research_packet.contradictions if research_packet else None),
    )
    state.story_brief = story_brief.model_dump()
    state.record_decision(
        "story_brief",
        f"strategy={story_brief.narrative_strategy}, question={story_brief.central_question[:48]!r}",
        "composed deterministically from intelligence, research and the narrative strategy",
    )

    _report("script")
    script, review, revision_count = write_and_critique_script(
        topic=subject,
        language=language,
        profile=profile,
        analysis=analysis,
        strategy=strategy,
        hook=hook,
        narrative=narrative,
        app_config=app_config,
        max_revisions=revisions,
        tracker=tracker,
        intelligence=intelligence,
        research=research_packet,
        story_brief=story_brief,
        narrative_strategy=narrative_choice,
        script_style=script_style,
        target_duration_seconds=target_duration_seconds,
        progress_cb=progress_cb,
    )
    state.script = script
    state.script_review = review.model_dump()
    state.revision_count = revision_count

    # Visual Director (scene plan with material strategy).
    _report("visuals")
    scene_plan = plan_scenes(
        script,
        profile,
        intelligence=intelligence,
        platform=(user_context.platform if user_context else ""),
    )
    state.scene_plan = scene_plan.model_dump()
    state.scenes = [scene.model_dump() for scene in scene_plan.scenes]
    state.media_strategy = {
        "style_language": scene_plan.style_language,
        "ai_image_budget": scene_plan.ai_image_budget,
        "ai_image_count": scene_plan.ai_image_count,
        "continuity_notes": scene_plan.continuity_notes,
        "platform": scene_plan.platform,
    }
    state.record_decision(
        "scene_plan",
        f"scenes={len(scene_plan.scenes)}, ai_images={scene_plan.ai_image_count}/{scene_plan.ai_image_budget}",
        "visual director matched material types to narration meaning",
    )

    # Title + thumbnail intelligence.
    _report("titles")
    title_candidates = generate_title_candidates(
        subject,
        script,
        profile,
        hook=hook,
        key_facts=story_brief.key_facts or None,
        platform=(user_context.platform if user_context else ""),
        app_config=app_config,
        tracker=tracker,
    )
    state.title_candidates = [candidate.model_dump() for candidate in title_candidates]
    best_title = title_candidates[0] if title_candidates else None
    state.selected_title = best_title.model_dump() if best_title else None
    state.thumbnail_concept = compose_thumbnail_concept(
        subject, best_title.text if best_title else subject, scene_plan, profile, intelligence
    ).model_dump()
    state.record_decision(
        "title_selection",
        f"{best_title.text[:64]!r} (overall={best_title.overall})" if best_title else "no title produced",
        "deterministic scoring favors accurate titles over clickbait",
    )

    # Repurposing plan.
    repurpose_plan = plan_repurposing(script, topic=subject, profile=profile)
    state.repurposing_plan = repurpose_plan.model_dump()

    # Quality assurance (critical issues block publication).
    _report("qa")
    qa_report = run_quality_assurance(
        script,
        profile,
        intelligence=intelligence,
        research_claims=(research_packet.claims if research_packet else None),
        research_contradictions=(research_packet.contradictions if research_packet else None),
        research_provenance=(research_packet.provenance if research_packet else None),
        scene_plan=scene_plan,
        selected_title=best_title,
        platform=(user_context.platform if user_context else ""),
    )
    state.qa_report = qa_report.model_dump()
    state.record_decision(
        "qa",
        qa_report.summary,
        "deterministic QA checks across research, script, visuals, audio and metadata",
    )
    # The QA gate doubles as the final review: a plan that fails critical
    # checks is marked as blocked for publication.
    state.final_review = {
        "verdict": "blocked" if qa_report.publication_blocked else "approved",
        "summary": qa_report.summary,
        "critical_issues": [
            issue.model_dump()
            for issue in qa_report.issues
            if issue.severity == QaSeverity.CRITICAL.value
        ],
    }

    # Trend signals and scored topic candidates refine the opportunity view
    # when automation delegates decisions (automatic / autopilot).
    if user_context and user_context.automation in (
        AUTOMATION_AUTOMATIC,
        AUTOMATION_AUTOPILOT,
    ):
        signals = collect_trend_signals(
            profile, intelligence, app_config=app_config, tracker=tracker
        )
        state.trend_signals = [signal.model_dump() for signal in signals]
        if signals:
            intelligence.topic_opportunity_score = round(
                0.5 * intelligence.topic_opportunity_score + 0.5 * max(s.score for s in signals), 1
            )
            state.content_intelligence = intelligence.model_dump()
        logger.info(f"trend signals: {trends_summary(signals)}")

        try:
            candidates = discover_topics(
                profile,
                intelligence=intelligence,
                app_config=app_config,
                tracker=tracker,
                niche=subject,
                max_candidates=6,
                signals=signals,
            )
            state.topic_candidates = [c.model_dump() for c in candidates]
            state.record_decision(
                "topic_discovery",
                f"{len(candidates)} candidate(s) scored",
                "deterministic explainable scoring over trend signals",
            )
        except Exception as exc:  # noqa: BLE001 - discovery is advisory
            logger.warning(f"topic discovery failed during planning: {exc}")

    state.agent_status = dict(tracker.statuses)
    state.agent_fallback_reason = dict(tracker.reasons)
    state.agent_retries = dict(tracker.retries)

    logger.success(f"agentic planning complete: {state.stage_summary()}")
    return state