"""Narrative Plan Agent - expands strategy into ordered sections."""

from __future__ import annotations

from .protocols import AgentInput
from app.services.content_profile import ContentProfile
from app.services.agentic.models import ContentStrategy, NarrativePlan
from app.services.agentic.utils import _heuristic_narrative

from .base import BaseAgent


class NarrativePlanAgent(BaseAgent):
    name = "narrative_plan"
    output_model = NarrativePlan

    def build_prompt(self, input_data: AgentInput) -> str:
        strategy: ContentStrategy = input_data["strategy"]
        profile: ContentProfile = input_data["profile"]

        return f"""
# Role: Narrative Architect

Turn the content strategy into an ordered narrative plan for a spoken
short-form video. Do not write the script.

## Strategy
Primary angle: {strategy.primary_angle}
Emotional progression: {strategy.emotional_progression}
Narrative structure: {strategy.narrative_structure}
CTA: {strategy.cta}

## Niche
{profile.name}: {profile.narrative_style or 'story-driven'}

Return ONLY a JSON object: {{"sections": ["...", "..."]}}
Each section is a short label plus the narrative job it must accomplish,
in the exact order the narration will follow. Keep 4-8 sections.
""".rstrip()

    def build_fallback(self, input_data: AgentInput) -> NarrativePlan:
        strategy: ContentStrategy = input_data["strategy"]
        profile: ContentProfile = input_data["profile"]
        return NarrativePlan(**_heuristic_narrative(strategy.model_dump(), profile))
