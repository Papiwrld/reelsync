"""Module-level agent functions matching the pre-refactor ``agentic.py`` API.

These thin adapters delegate to the agent classes in ``agents/`` while
preserving the exact public signatures, error contracts and helper behavior
the rest of the codebase (and tests) depend on.
"""

from __future__ import annotations

import re
from typing import Any, Callable, List, Optional, Tuple

from loguru import logger

from app.services.agent_llm import AgentTracker, AgenticError, AgenticScriptError
from app.services.agentic.agents import (
    ContentStrategyAgent,
    HookStrategyAgent,
    NarrativePlanAgent,
    ScriptCriticAgent,
    ScriptReviserAgent,
    ScriptWriterAgent,
    TopicAnalysisAgent,
)
from app.services.agentic.models import (
    ContentStrategy,
    HookCandidate,
    NarrativePlan,
    ScriptReview,
    TopicAnalysis,
)
from app.services.agentic.scoring import (
    AGENTIC_DEFAULT_MAX_REVISIONS,
    _heuristic_review,
)
from app.services.agentic.utils import (
    _heuristic_hooks,
    _heuristic_narrative,
    _heuristic_strategy,
    _heuristic_topic_analysis,
)
from app.services.content_profile import ContentProfile
from app.services.intelligence import ContentIntelligence
from app.services.research import ResearchPacket


# ---------------------------------------------------------------------------
# Phase 1 agent functions
# ---------------------------------------------------------------------------


def analyze_topic(
    topic: str,
    profile: ContentProfile,
    app_config=None,
    tracker: Optional[AgentTracker] = None,
    intelligence: Optional[ContentIntelligence] = None,
) -> TopicAnalysis:
    """Understand what makes this particular topic interesting for this niche."""
    try:
        payload = TopicAnalysisAgent().run(
            None,
            {"topic": topic, "profile": profile, "intelligence": intelligence},
            app_config=app_config,
            tracker=tracker,
        )
        analysis = TopicAnalysis.model_validate(payload)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"topic analysis validation failed: {exc}")
        analysis = TopicAnalysis(**_heuristic_topic_analysis(topic, profile))
    logger.info(
        f"topic analysis complete: type={analysis.topic_type or '?'}, "
        f"hooks={len(analysis.possible_hooks)}"
    )
    return analysis


def develop_content_strategy(
    topic: str,
    analysis: TopicAnalysis,
    profile: ContentProfile,
    app_config=None,
    tracker: Optional[AgentTracker] = None,
    intelligence: Optional[ContentIntelligence] = None,
) -> ContentStrategy:
    """Choose the strongest angle, emotional progression, structure and CTA."""
    try:
        payload = ContentStrategyAgent().run(
            None,
            {"topic": topic, "analysis": analysis, "profile": profile, "intelligence": intelligence},
            app_config=app_config,
            tracker=tracker,
        )
        strategy = ContentStrategy.model_validate(payload)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"content strategy validation failed: {exc}")
        strategy = ContentStrategy(**_heuristic_strategy(topic, profile, analysis.model_dump()))
    logger.info(
        f"content strategy complete: angle={strategy.primary_angle[:80]!r}, "
        f"sections={len(strategy.narrative_structure)}"
    )
    return strategy


def generate_hook_candidates(
    topic: str,
    analysis: TopicAnalysis,
    strategy: ContentStrategy,
    profile: ContentProfile,
    app_config=None,
    tracker: Optional[AgentTracker] = None,
) -> List[HookCandidate]:
    """Produce diverse candidate hooks for later deterministic scoring."""
    try:
        payload = HookStrategyAgent().run(
            None,
            {"topic": topic, "analysis": analysis, "strategy": strategy, "profile": profile},
            app_config=app_config,
            tracker=tracker,
        )
        candidates = [
            HookCandidate.model_validate(item) for item in payload.get("candidates", [])
        ]
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"hook candidate validation failed: {exc}")
        candidates = []
    if not candidates:
        candidates = [
            HookCandidate(**item) for item in _heuristic_hooks(topic, profile, analysis.model_dump())
        ]
    logger.info(f"hook candidates generated: {len(candidates)}")
    return candidates


def build_narrative_plan(
    strategy: ContentStrategy,
    profile: ContentProfile,
    app_config=None,
    tracker: Optional[AgentTracker] = None,
) -> NarrativePlan:
    """Expand the strategy structure into an ordered narrative plan."""
    try:
        payload = NarrativePlanAgent().run(
            None,
            {"strategy": strategy, "profile": profile},
            app_config=app_config,
            tracker=tracker,
        )
        plan = NarrativePlan.model_validate(payload)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"narrative plan validation failed: {exc}")
        plan = NarrativePlan(**_heuristic_narrative(strategy.model_dump(), profile))
    if not plan.sections:
        plan = NarrativePlan(**_heuristic_narrative(strategy.model_dump(), profile))
    logger.info(f"narrative plan complete: sections={len(plan.sections)}")
    return plan


# ---------------------------------------------------------------------------
# Script phase helpers
# ---------------------------------------------------------------------------


def write_script(
    topic: str,
    language: str,
    profile: ContentProfile,
    analysis: TopicAnalysis,
    strategy: ContentStrategy,
    hook: str,
    narrative: NarrativePlan,
    app_config=None,
    tracker: Optional[AgentTracker] = None,
    intelligence: Optional[ContentIntelligence] = None,
    research: Optional[ResearchPacket] = None,
    story_brief: Optional[Any] = None,
    narrative_strategy: Optional[Any] = None,
    script_style: str = "",
    target_duration_seconds: int = 0,
) -> str:
    """Write the spoken script as a consequence of the strategy."""
    try:
        payload = ScriptWriterAgent().run(
            None,
            {
                "topic": topic,
                "language": language,
                "profile": profile,
                "analysis": analysis,
                "strategy": strategy,
                "hook": hook,
                "narrative": narrative,
                "intelligence": intelligence,
                "research": research,
                "story_brief": story_brief,
                "narrative_strategy": narrative_strategy,
                "script_style": script_style,
                "target_duration_seconds": target_duration_seconds,
            },
            app_config=app_config,
            tracker=tracker,
        )
        script = payload["script"]
    except AgenticError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise AgenticScriptError(str(exc)) from exc
    if not script:
        raise AgenticScriptError("script agent returned an empty script")
    logger.info(f"script written: {len(script.split())} words")
    return script


def critique_script(
    script: str,
    topic: str,
    profile: ContentProfile,
    strategy: ContentStrategy,
    app_config=None,
    tracker: Optional[AgentTracker] = None,
) -> ScriptReview:
    """Score the script; verdict is decided locally against the threshold.

    Scores are validated and clamped to [0,10]; any malformed dimension falls
    back to the deterministic review so garbage LLM numbers never approve a
    weak script.
    """
    try:
        payload = ScriptCriticAgent().run(
            None,
            {"script": script, "topic": topic, "profile": profile, "strategy": strategy},
            app_config=app_config,
            tracker=tracker,
        )
        review = ScriptReview.model_validate(payload)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"script critique failed: {exc}")
        review = _heuristic_review(script, profile)
    logger.info(f"script critique: overall={review.overall} verdict={review.verdict}")
    return review


def revise_script(
    script: str,
    review: ScriptReview,
    topic: str,
    language: str,
    profile: ContentProfile,
    strategy: ContentStrategy,
    hook: str,
    app_config=None,
    tracker: Optional[AgentTracker] = None,
    research: Optional[ResearchPacket] = None,
    script_style: str = "",
    target_duration_seconds: int = 0,
) -> str:
    """Rewrite the script addressing the critic's feedback (same hook)."""
    try:
        payload = ScriptReviserAgent().run(
            None,
            {
                "script": script,
                "review": review,
                "topic": topic,
                "language": language,
                "profile": profile,
                "hook": hook,
                "research": research,
                "script_style": script_style,
                "target_duration_seconds": target_duration_seconds,
            },
            app_config=app_config,
            tracker=tracker,
        )
        revised = payload["script"]
    except AgenticError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise AgenticScriptError(str(exc)) from exc
    if not revised:
        raise AgenticScriptError("script editor returned an empty script")
    logger.info(f"script revised: {len(revised.split())} words")
    return revised


def _normalized_prefix(text: str, size: int = 40) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()[:size]


def _ensure_hook_preserved(script: str, hook: str) -> str:
    """Deterministically guarantee the selected hook opens the narration.

    The editor is instructed to keep the hook verbatim, but LLMs drop lines;
    this check re-inserts the hook when it is missing instead of silently
    shipping a video whose hook vanished.
    """
    if not hook or not script:
        return script
    if _normalized_prefix(script).startswith(_normalized_prefix(hook)):
        return script
    logger.warning("revised script dropped the selected hook; re-inserting it")
    return f"{hook.strip()}\n\n{script.strip()}"


def write_and_critique_script(
    topic: str,
    language: str,
    profile: ContentProfile,
    analysis: TopicAnalysis,
    strategy: ContentStrategy,
    hook: str,
    narrative: NarrativePlan,
    app_config=None,
    max_revisions: int = AGENTIC_DEFAULT_MAX_REVISIONS,
    tracker: Optional[AgentTracker] = None,
    intelligence: Optional[ContentIntelligence] = None,
    research: Optional[ResearchPacket] = None,
    story_brief: Optional[Any] = None,
    narrative_strategy: Optional[Any] = None,
    script_style: str = "",
    target_duration_seconds: int = 0,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Tuple[str, ScriptReview, int]:
    """Write the script, then run the critic loop (bounded, no infinite loops).

    ``progress_cb`` (optional) reports "script_revision" before each critic
    revision round (the initial draft is reported by the caller).
    """
    script = write_script(
        topic=topic,
        language=language,
        profile=profile,
        analysis=analysis,
        strategy=strategy,
        hook=hook,
        narrative=narrative,
        app_config=app_config,
        tracker=tracker,
        intelligence=intelligence,
        research=research,
        story_brief=story_brief,
        narrative_strategy=narrative_strategy,
        script_style=script_style,
        target_duration_seconds=target_duration_seconds,
    )
    review = critique_script(script, topic, profile, strategy, app_config=app_config, tracker=tracker)
    revisions = 0
    while review.verdict == "REVISE" and revisions < max_revisions:
        revisions += 1
        logger.info(f"script revise round {revisions}/{max_revisions}")
        if progress_cb is not None:
            try:
                progress_cb("script_revision")
            except Exception:  # noqa: BLE001
                pass
        script = revise_script(
            script=script,
            review=review,
            topic=topic,
            language=language,
            profile=profile,
            strategy=strategy,
            hook=hook,
            app_config=app_config,
            tracker=tracker,
            research=research,
            script_style=script_style,
            target_duration_seconds=target_duration_seconds,
        )
        script = _ensure_hook_preserved(script, hook)
        review = critique_script(script, topic, profile, strategy, app_config=app_config, tracker=tracker)
    return script, review, revisions