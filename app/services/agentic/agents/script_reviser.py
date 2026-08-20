"""Script Editor Agent - rewrites the script addressing critic feedback.

Name and prompt intentionally mirror the original ``revise_script``
(``agent="script_editor"``) so tracker statuses and prompts stay identical to
the pre-refactor behavior.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from .base import BaseAgent
from .protocols import AgentInput, AgentOutput
from app.services import llm
from app.services.agent_llm import _llm_text
from app.services.agentic.utils import _clean_script_response, _target_length_hint
from app.services.content_profile import ContentProfile, profile_strategy_context
from app.services.research import ResearchPacket, research_grounding_context


logger = logging.getLogger(__name__)


class ScriptReviserAgent(BaseAgent):
    name = "script_editor"
    output_model = None  # Returns raw string

    def build_prompt(self, input_data: AgentInput) -> str:
        script: str = input_data["script"]
        review = input_data["review"]
        language: str = input_data["language"]
        profile: ContentProfile = input_data["profile"]
        hook: str = input_data["hook"]
        research: ResearchPacket | None = input_data.get("research")
        script_style: str = input_data.get("script_style", "")
        target_duration_seconds: int = input_data.get("target_duration_seconds", 0)

        research_block = research_grounding_context(research)

        scores_block = "\n".join(
            f"- {dim.replace('_', ' ').title()}: {review.scores.get(dim, 0)}/10"
            for dim in ["hook", "niche_alignment", "narrative", "visual_potential", "pacing", "ending", "cta_quality"]
        )

        return f"""
# Role: Script Editor

Rewrite this narration fixing the critic's issues. Keep the same opening
hook (verbatim) and the same angle. Do not change the topic.

## Opening hook (keep verbatim as the first line)
\"\"\"{hook}\"\"\"

## Critic feedback
Overall: {review.overall}/10
Dimension scores:
{scores_block}

Critic notes: {review.feedback or 'Improve the weakest scored dimensions first.'}

## Previous script
\"\"\"{script}\"\"\"

{research_block}

## Niche
{profile_strategy_context(profile)}

## Writing style
{llm.resolve_script_style(script_style)}

## Language
{language or 'the same language as the previous script'}.

## Hard constraints
1. Return only the raw narration text — no markdown, no titles.
2. No speaker labels, no emojis, no mention of this prompt.
3. Fix the specific issues in the feedback; keep what already works. Focus improvements on the lowest-scoring dimensions first.
4. Never use dashes, hyphens or bullet markers in the script: join related clauses with commas instead.
5. Target length: {_target_length_hint(target_duration_seconds, profile.preferred_video_length)}.
""".rstrip()

    def build_fallback(self, input_data: AgentInput) -> str:
        # No silent fallback - if revision fails, the pipeline should fail
        raise RuntimeError("Script editor has no fallback")

    def run(
        self,
        state: Any,
        input_data: AgentInput,
        *,
        app_config: Optional[Any] = None,
        tracker: Optional[Any] = None,
    ) -> AgentOutput:
        prompt = self.build_prompt(input_data)
        try:
            text = _llm_text(prompt, app_config=app_config, tracker=tracker, agent=self.name)
            script = _clean_script_response(text)
            if not script:
                raise RuntimeError("script editor returned an empty script")
            logger.info(f"{self.name} complete: {len(script.split())} words")
            return {"script": script}
        except Exception as exc:
            logger.error(f"{self.name} failed: {exc}")
            raise