"""Agent protocols for the agentic planning graph.

This module defines the interface that every agent in the planning pipeline
must implement. It enables:
  - Unit testing with mock dependencies (no real LLM calls)
  - Clear separation between agent logic and orchestration
  - Easy swapping of implementations
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, TypedDict, runtime_checkable

from app.services.agent_llm import AgentTracker


class AgentInput(TypedDict, total=False):
    """Input data for an agent. Keys match the original function signatures."""
    topic: str
    profile: Any  # ContentProfile
    analysis: Any  # TopicAnalysis
    strategy: Any  # ContentStrategy
    narrative: Any  # NarrativePlan
    hook: str
    language: str
    intelligence: Any  # ContentIntelligence
    research: Any  # ResearchPacket
    story_brief: Any
    narrative_strategy: Any
    script_style: str
    target_duration_seconds: int
    app_config: Any
    tracker: Optional[AgentTracker]


class AgentOutput(TypedDict, total=False):
    """Output data from an agent."""
    topic_type: str
    historical_context: str
    potential_claims: list[str]
    emotional_angles: list[str]
    curiosity_gaps: list[str]
    controversy_level: str
    known_vs_unknown: str
    visual_opportunities: list[str]
    audience_interest: str
    possible_hooks: list[str]
    narrative_options: list[str]
    research_requirements: list[str]
    primary_angle: str
    hook_strategy: str
    emotional_progression: list[str]
    pacing: str
    narrative_structure: list[str]
    cta: str
    candidates: list[dict]
    sections: list[str]
    script: str
    scores: dict[str, float]
    overall: float
    verdict: str
    feedback: str
    text: str
    style: str
    rationale: str


@runtime_checkable
class AgentProtocol(Protocol):
    """Protocol that all agents must implement.

    Every agent receives the current GenerationState and an input dict,
    and returns an output dict. The protocol ensures agents are
    swappable and testable in isolation.
    """

    name: str
    """Unique identifier for logging/tracking."""

    def run(
        self,
        state: Any,  # GenerationState
        input_data: AgentInput,
        *,
        app_config: Optional[Any] = None,
        tracker: Optional[AgentTracker] = None,
    ) -> AgentOutput:
        """Execute the agent and return its structured output.

        Args:
            state: The shared GenerationState (read/write for decisions).
            input_data: Agent-specific input (validated by the orchestrator).
            app_config: Optional configuration override.
            tracker: Optional AgentTracker for recording LLM calls.

        Returns:
            AgentOutput dict with the agent's results.
        """
        ...


class AgentResult:
    """Wrapper for agent execution results with metadata."""

    def __init__(
        self,
        output: AgentOutput,
        *,
        fallback_used: bool = False,
        fallback_reason: str = "",
        llm_calls: int = 0,
    ):
        self.output = output
        self.fallback_used = fallback_used
        self.fallback_reason = fallback_reason
        self.llm_calls = llm_calls

    def __bool__(self) -> bool:
        return bool(self.output)