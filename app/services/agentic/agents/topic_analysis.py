"""Topic Analysis Agent - understands what makes a topic interesting."""

from __future__ import annotations

from .protocols import AgentInput
from app.services.content_profile import ContentProfile
from app.services.intelligence import ContentIntelligence, intelligence_context
from app.services.agentic.models import TopicAnalysis
from app.services.agentic.utils import topic_analysis_instruction, _heuristic_topic_analysis

from .base import BaseAgent


class TopicAnalysisAgent(BaseAgent):
    name = "topic_analysis"
    output_model = TopicAnalysis

    def build_prompt(self, input_data: AgentInput) -> str:
        topic: str = input_data["topic"]
        profile: ContentProfile = input_data["profile"]
        intelligence: ContentIntelligence | None = input_data.get("intelligence")

        intelligence_block = intelligence_context(intelligence)
        return f"""
# Role: Topic Analysis Agent

Analyze the topic for a short-form video and produce a structured analysis.
Do NOT write the script. Do NOT use generic templates: identify what makes THIS
topic interesting.

{topic_analysis_instruction(topic, profile)}

{intelligence_block}

Return ONLY a JSON object with these keys:
- "topic_type": string, e.g. "historical mystery", "tech release", "science discovery"
- "historical_context": string (one or two sentences of relevant context)
- "potential_claims": list of strings (claims this video could make, to be verified)
- "emotional_angles": list of strings (emotional angles available)
- "curiosity_gaps": list of strings (what the audience does not know)
- "controversy_level": string ("none", "low", "medium", "high") plus one short reason
- "known_vs_unknown": string (what is documented vs what is open/unverified)
- "visual_opportunities": list of strings (concrete visuals that fit)
- "audience_interest": string (why THIS niche's audience cares)
- "possible_hooks": list of 2-3 strings (specific, topic-aware opening lines)
- "narrative_options": list of 2 strings (possible narrative shapes)
- "research_requirements": list of strings (what must be verified)

Do not include anything outside the JSON object.
""".rstrip()

    def build_fallback(self, input_data: AgentInput) -> TopicAnalysis:
        topic: str = input_data["topic"]
        profile: ContentProfile = input_data["profile"]
        return TopicAnalysis(**_heuristic_topic_analysis(topic, profile))