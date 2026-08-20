"""Agentic output models (pure Pydantic, no dependencies)."""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, ConfigDict, Field


class TopicAnalysis(BaseModel):
    """What makes THIS topic interesting, for THIS niche."""

    model_config = ConfigDict(extra="ignore")

    topic_type: str = ""
    historical_context: str = ""
    potential_claims: List[str] = []
    emotional_angles: List[str] = []
    curiosity_gaps: List[str] = []
    controversy_level: str = ""
    known_vs_unknown: str = ""
    visual_opportunities: List[str] = []
    audience_interest: str = ""
    possible_hooks: List[str] = []
    narrative_options: List[str] = []
    research_requirements: List[str] = []


class ContentStrategy(BaseModel):
    """The plan the script must follow."""

    model_config = ConfigDict(extra="ignore")

    primary_angle: str = ""
    hook_strategy: str = ""
    emotional_progression: List[str] = []
    pacing: str = ""
    narrative_structure: List[str] = []
    cta: str = ""


class HookCandidate(BaseModel):
    """A candidate opening line with its intended style."""

    model_config = ConfigDict(extra="ignore")

    text: str = ""
    style: str = ""
    rationale: str = ""


class NarrativePlan(BaseModel):
    """Ordered sections the script will follow."""

    model_config = ConfigDict(extra="ignore")

    sections: List[str] = []


class ScriptReview(BaseModel):
    """Critic output: dimension scores, overall, verdict, feedback."""

    model_config = ConfigDict(extra="ignore")

    scores: dict[str, float] = Field(default_factory=dict)
    overall: float = 0.0
    verdict: str = "REVISE"
    feedback: str = ""


__all__ = [
    "TopicAnalysis",
    "ContentStrategy",
    "HookCandidate",
    "NarrativePlan",
    "ScriptReview",
]