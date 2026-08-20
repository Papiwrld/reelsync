"""Agentic planning package.

Public facade for the agentic content-production graph. Re-exports every name
the pre-refactor monolithic ``app/services/agentic.py`` module exposed so
existing callers (WebUI, task pipeline, tests) keep working unchanged.
"""

from __future__ import annotations

from app.services import llm  # noqa: F401 (re-exported for test compatibility)
from app.services.agent_llm import (
    AgenticError,
    AgenticScriptError,
    AgentTracker,
    _clamp_score,
    _extract_json_payload,  # noqa: F401 (re-exported for test compatibility)
    _llm_json,
    _llm_text,
)
from app.services.content_profile import (
    ContentProfile,
    get_content_profile,
    profile_strategy_context,
)
from app.services.generation_state import GenerationState
from app.services.intelligence import (
    AUTOMATION_AUTOMATIC,
    AUTOMATION_AUTOPILOT,
    ContentIntelligence,
    ContentRequest,
    build_content_intelligence,
    intelligence_context,
    intelligence_summary,
)
from app.services.narrative import (
    NarrativeStrategy,
    build_story_brief,
    narrative_strategy_context,
    record_narrative_usage,
    select_narrative_strategy,
    story_brief_context,
)
from app.services.qa import QaSeverity, run_quality_assurance
from app.services.repurposing import plan_repurposing
from app.services.research import (
    ResearchPacket,
    research_grounding_context,
    research_summary,
    run_research,
)
from app.services.titles import compose_thumbnail_concept, generate_title_candidates
from app.services.trends import collect_trend_signals, discover_topics, trends_summary
from app.services.visual_director import plan_scenes

from .models import (
    ContentStrategy,
    HookCandidate,
    NarrativePlan,
    ScriptReview,
    TopicAnalysis,
)
from .scoring import (
    _CRITIC_DIMENSIONS,
    _GROUNDING_CAP,
    _STEM_MIN_PREFIX,
    _STEM_MIN_RATIO,
    _common_prefix_len,
    _heuristic_review,
    _hook_overall,
    _hook_words,
    _is_grounded,
    _stem_overlap,
    _style_match,
    _topic_words,
    AGENTIC_APPROVE_THRESHOLD,
    AGENTIC_DEFAULT_MAX_REVISIONS,
    AGENTIC_HOOK_CANDIDATES,
    judge_hooks,
    score_hook,
    score_hook_candidates,
    select_best_hook,
)
from .utils import (
    _clean_script_response,
    _heuristic_hooks,
    _heuristic_narrative,
    _heuristic_strategy,
    _heuristic_topic_analysis,
    _rationale_of_choice,
    _strategy_from_choice,
    _target_length_hint,
    _topic_analysis_summary,
    topic_analysis_instruction,
)
from .compat import (
    _ensure_hook_preserved,
    _normalized_prefix,
    analyze_topic,
    build_narrative_plan,
    critique_script,
    develop_content_strategy,
    generate_hook_candidates,
    revise_script,
    write_and_critique_script,
    write_script,
)
from .graph import (
    _claim_status_value,
    _configured_max_revisions,
    _select_narrative_for_state,
    plan_video_content,
)
from .plan_from_params import _content_request_from_params, plan_video_content_from_params

__all__ = [
    "llm",
    "AgenticError",
    "AgenticScriptError",
    "AgentTracker",
    "_clamp_score",
    "_extract_json_payload",
    "_llm_json",
    "_llm_text",
    "ContentProfile",
    "get_content_profile",
    "profile_strategy_context",
    "GenerationState",
    "AUTOMATION_AUTOMATIC",
    "AUTOMATION_AUTOPILOT",
    "ContentIntelligence",
    "ContentRequest",
    "build_content_intelligence",
    "intelligence_context",
    "intelligence_summary",
    "NarrativeStrategy",
    "build_story_brief",
    "narrative_strategy_context",
    "record_narrative_usage",
    "select_narrative_strategy",
    "story_brief_context",
    "QaSeverity",
    "run_quality_assurance",
    "plan_repurposing",
    "ResearchPacket",
    "research_grounding_context",
    "research_summary",
    "run_research",
    "compose_thumbnail_concept",
    "generate_title_candidates",
    "collect_trend_signals",
    "discover_topics",
    "trends_summary",
    "plan_scenes",
    "TopicAnalysis",
    "ContentStrategy",
    "HookCandidate",
    "NarrativePlan",
    "ScriptReview",
    "AGENTIC_APPROVE_THRESHOLD",
    "AGENTIC_DEFAULT_MAX_REVISIONS",
    "AGENTIC_HOOK_CANDIDATES",
    "_CRITIC_DIMENSIONS",
    "_GROUNDING_CAP",
    "_STEM_MIN_PREFIX",
    "_STEM_MIN_RATIO",
    "_common_prefix_len",
    "_heuristic_review",
    "_hook_overall",
    "_hook_words",
    "_is_grounded",
    "_stem_overlap",
    "_style_match",
    "_topic_words",
    "judge_hooks",
    "score_hook",
    "score_hook_candidates",
    "select_best_hook",
    "topic_analysis_instruction",
    "_topic_analysis_summary",
    "_heuristic_topic_analysis",
    "_heuristic_strategy",
    "_heuristic_hooks",
    "_heuristic_narrative",
    "_clean_script_response",
    "_target_length_hint",
    "_strategy_from_choice",
    "_rationale_of_choice",
    "analyze_topic",
    "develop_content_strategy",
    "generate_hook_candidates",
    "build_narrative_plan",
    "write_script",
    "critique_script",
    "revise_script",
    "write_and_critique_script",
    "_normalized_prefix",
    "_ensure_hook_preserved",
    "_claim_status_value",
    "_configured_max_revisions",
    "_select_narrative_for_state",
    "_content_request_from_params",
    "plan_video_content",
    "plan_video_content_from_params",
]