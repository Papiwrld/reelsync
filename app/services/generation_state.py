"""Explicit pipeline state for the agentic content-production graph.

Each agent consumes and updates well-defined fields instead of passing huge
unstructured prompts between nodes. Phase 1 fills the strategy fields
(analysis, strategy, hooks, narrative, script, review); later phases fill
scenes, media and review fields. ``extra="allow"`` keeps the model forward
compatible with future phases.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class GenerationState(BaseModel):
    """Structured state produced and consumed by the agent graph."""

    model_config = ConfigDict(extra="allow")

    user_input: str = ""
    profile_name: str = ""
    automation_level: str = ""
    # Optional traceability: the video-generation task this planning run
    # belongs to (set by the task pipeline; empty for standalone runs).
    task_id: str = ""
    content_intelligence: Optional[Dict[str, Any]] = None
    research_packet: Optional[Dict[str, Any]] = None
    topic_analysis: Optional[Dict[str, Any]] = None
    research: Optional[Dict[str, Any]] = None
    content_strategy: Optional[Dict[str, Any]] = None
    hook_candidates: Optional[List[Dict[str, Any]]] = None
    selected_hook: Optional[str] = None
    narrative_plan: Optional[Dict[str, Any]] = None
    script: Optional[str] = None
    script_review: Optional[Dict[str, Any]] = None
    scenes: Optional[List[Dict[str, Any]]] = None
    media_strategy: Optional[Dict[str, Any]] = None
    selected_media: Optional[List[Dict[str, Any]]] = None
    media_reviews: Optional[List[Dict[str, Any]]] = None
    voice_plan: Optional[Dict[str, Any]] = None
    subtitle_plan: Optional[Dict[str, Any]] = None
    render_config: Optional[Dict[str, Any]] = None
    final_review: Optional[Dict[str, Any]] = None
    output: Optional[str] = None
    revision_count: int = 0

    # Phase 2C: Story Intelligence
    narrative_strategy: Optional[Dict[str, Any]] = None
    story_brief: Optional[Dict[str, Any]] = None
    # Phase 2C: Visual Director
    scene_plan: Optional[Dict[str, Any]] = None
    # Phase 2D: packaging + QA + repurposing
    title_candidates: Optional[List[Dict[str, Any]]] = None
    selected_title: Optional[Dict[str, Any]] = None
    thumbnail_concept: Optional[Dict[str, Any]] = None
    qa_report: Optional[Dict[str, Any]] = None
    repurposing_plan: Optional[Dict[str, Any]] = None
    # Phase 2E: trends + topic discovery
    trend_signals: Optional[List[Dict[str, Any]]] = None
    topic_candidates: Optional[List[Dict[str, Any]]] = None

    # Observability: which agents ran on the LLM vs degraded to fallbacks,
    # why, and how many attempts each took. Lets operators answer "why did
    # this video get produced" without exposing chain-of-thought.
    schema_version: int = 2
    agent_status: Dict[str, str] = Field(default_factory=dict)
    agent_fallback_reason: Dict[str, str] = Field(default_factory=dict)
    agent_retries: Dict[str, int] = Field(default_factory=dict)
    # Structured decision metadata: stage, decision, rationale (no CoT).
    decision_log: List[Dict[str, Any]] = Field(default_factory=list)

    def record_decision(self, stage: str, decision: str, rationale: str) -> None:
        """Append a concise, inspectable decision record (never CoT)."""
        self.decision_log.append(
            {
                "stage": stage,
                "decision": decision,
                "rationale": rationale,
            }
        )

    def to_json(self) -> str:
        """Serialize to a JSON string (used for task artifact persistence)."""
        return self.model_dump_json(indent=4)

    @classmethod
    def from_json(cls, raw: str) -> "GenerationState":
        """Deserialize from ``to_json()`` output; invalid input yields an empty state."""
        try:
            payload = json.loads(raw)
            return cls.model_validate(payload)
        except (ValueError, TypeError):
            return cls()

    def stage_summary(self) -> str:
        """One-line human-readable summary for logs/UI (no chain-of-thought)."""
        parts = [f"profile={self.profile_name or 'none'}"]
        if self.content_intelligence:
            parts.append(f"intel={self.content_intelligence.get('fact_check_level', '')}")
        if self.topic_analysis:
            parts.append(f"topic_type={self.topic_analysis.get('topic_type', '')}")
        if self.selected_hook:
            parts.append(f"hook={self.selected_hook[:60]!r}")
        if self.script:
            parts.append(f"script_words={len(self.script.split())}")
        if self.script_review:
            parts.append(f"review={self.script_review.get('overall', '')}")
        if self.revision_count:
            parts.append(f"revisions={self.revision_count}")
        fallback_agents = sorted(
            agent for agent, status in self.agent_status.items() if status != "llm"
        )
        if fallback_agents:
            parts.append(f"fallback_agents={','.join(fallback_agents)}")
        return "; ".join(parts)