"""Story Intelligence (Phase 2C.1–2C.4).

A generalized narrative-strategy engine. It selects an appropriate narrative
structure for a piece of content from a catalog of strategies, using:

    topic suitability + niche suitability + audience suitability
    + format suitability + research evidence + previous narrative usage

and it builds a structured ``StoryBrief`` that the script agent consumes.

Design rules:
- Niche-agnostic: every signal is derived from profile DATA fields, never
  from ``if niche == ...`` branches. Adding a niche means authoring a
  profile, not editing this module.
- Constrained intelligent variance: selection is deterministic (no random
  noise that damages quality), but recently used strategies for a profile
  are penalized so consecutive videos do not reuse the same template.
- Deterministic core with optional LLM refinement. The pipeline path is
  fully deterministic (zero LLM cost); ``refine_story_brief`` exists for
  callers who opt into one extra LLM call.
- Every decision records a concise rationale (no chain-of-thought).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from app.services.content_profile import ContentProfile
from app.services.intelligence import ContentIntelligence

# ---------------------------------------------------------------------------
# Strategy catalog (data, not code)
# ---------------------------------------------------------------------------


class NarrativeStrategy(BaseModel):
    """A reusable narrative structure with selection metadata."""

    model_config = ConfigDict(extra="ignore")

    id: str
    label: str
    description: str
    # Ordered narrative sections used as a template (overridden by the LLM
    # narrative architect when available).
    sections: List[str] = Field(default_factory=list)
    # How much research evidence this strategy demands (0-10).
    evidence_weight: float = 5.0
    # Content formats this strategy fits naturally (format id -> affinity 0-10).
    format_affinity: Dict[str, float] = Field(default_factory=dict)
    # Content goals it serves (goal id -> affinity 0-10).
    goal_affinity: Dict[str, float] = Field(default_factory=dict)
    # Keywords in the topic/topic-analysis that hint at this strategy.
    topic_signals: List[str] = Field(default_factory=list)
    # Whether this strategy needs a concrete subject (person/company/event).
    needs_subject: bool = False
    # Whether it can carry an unresolved question as its payoff.
    open_ending_ok: bool = False


def _strategy(
    strategy_id: str,
    label: str,
    description: str,
    sections: List[str],
    evidence_weight: float = 5.0,
    format_affinity: Optional[Dict[str, float]] = None,
    goal_affinity: Optional[Dict[str, float]] = None,
    topic_signals: Optional[List[str]] = None,
    needs_subject: bool = False,
    open_ending_ok: bool = False,
) -> NarrativeStrategy:
    return NarrativeStrategy(
        id=strategy_id,
        label=label,
        description=description,
        sections=sections,
        evidence_weight=evidence_weight,
        format_affinity=format_affinity or {},
        goal_affinity=goal_affinity or {},
        topic_signals=topic_signals or [],
        needs_subject=needs_subject,
        open_ending_ok=open_ending_ok,
    )


NARRATIVE_STRATEGIES: List[NarrativeStrategy] = [
    _strategy(
        "documentary",
        "Documentary",
        "A factual, cinematic account that establishes context and consequence.",
        ["Cold open", "Context", "Key events", "Consequence", "Reflection"],
        evidence_weight=8.0,
        format_affinity={"documentary": 10.0, "news_analysis": 6.0, "case_study": 7.0},
        goal_affinity={"education": 8.0, "awareness": 8.0, "engagement": 6.0},
        topic_signals=["history", "documentary", "the story of", "archive"],
    ),
    _strategy(
        "rise_and_fall",
        "Rise and Fall",
        "A rise, a peak, a conflict, then a fall or reinvention.",
        ["Hook on the outcome", "Humble beginning", "Ascent", "Peak", "Conflict", "Fall or reinvention", "Lesson"],
        evidence_weight=7.5,
        format_affinity={"documentary": 8.0, "case_study": 8.0},
        goal_affinity={"engagement": 7.0, "education": 6.0, "growth": 5.0},
        topic_signals=["rise", "fall", "empire", "collapsed", "crashed", "built", "declined", "founded"],
        needs_subject=True,
    ),
    _strategy(
        "mystery",
        "Mystery",
        "Presents evidence and contradictions, ending on the open question.",
        ["Hook on the anomaly", "What is known", "What is unknown", "Evidence and contradictions", "Open question"],
        evidence_weight=5.0,
        format_affinity={"documentary": 7.0, "mystery": 10.0, "storytelling": 8.0},
        goal_affinity={"engagement": 9.0, "awareness": 6.0},
        topic_signals=["mystery", "unsolved", "disappeared", "unknown", "vanished", "why did", "what happened", "unexplained"],
        open_ending_ok=True,
    ),
    _strategy(
        "investigation",
        "Investigation",
        "A structured inquiry: question, evidence gathering, ruling out, conclusion or verdict.",
        ["Hook on the question", "Initial suspicion", "Evidence trail", "Dead ends", "Verdict or open verdict"],
        evidence_weight=8.0,
        format_affinity={"documentary": 8.0, "case_study": 7.0, "mystery": 8.0},
        goal_affinity={"education": 8.0, "engagement": 8.0},
        topic_signals=["investigat", "case", "evidence", "probe", "inquiry", "who really"],
        needs_subject=True,
    ),
    _strategy(
        "biography",
        "Biography",
        "A person's life told through defining moments.",
        ["Hook on the person", "Origins", "Turning points", "Peak or struggle", "Legacy"],
        evidence_weight=8.0,
        format_affinity={"documentary": 8.0, "biography": 10.0},
        goal_affinity={"education": 8.0, "awareness": 7.0},
        topic_signals=["biography", "founder", "ceo", "life of", "who was"],
        needs_subject=True,
    ),
    _strategy(
        "case_study",
        "Case Study",
        "One specific instance examined closely to draw a transferable lesson.",
        ["Hook on the outcome", "The situation", "What was tried", "The result", "The lesson"],
        evidence_weight=8.0,
        format_affinity={"case_study": 10.0, "documentary": 7.0, "news_analysis": 6.0},
        goal_affinity={"education": 9.0, "growth": 6.0, "monetization": 6.0},
        topic_signals=["case study", "how they", "what made", "example of"],
        needs_subject=True,
    ),
    _strategy(
        "explainer",
        "Explainer",
        "Explains how something works or why it is the way it is.",
        ["Hook on the gap", "The mechanism", "Why it matters", "Implication"],
        evidence_weight=6.0,
        format_affinity={"explainer": 10.0, "how_it_works": 9.0, "educational": 9.0},
        goal_affinity={"education": 10.0, "awareness": 7.0},
        topic_signals=["explain", "how it works", "why", "the science", "the reason", "what is", "how do"],
    ),
    _strategy(
        "tutorial",
        "Tutorial",
        "Step-by-step instructions the viewer can follow.",
        ["Hook on the result", "What you need", "Step by step", "Result", "Next step"],
        evidence_weight=4.0,
        format_affinity={"tutorial": 10.0, "how_it_works": 8.0},
        goal_affinity={"education": 9.0, "growth": 7.0, "monetization": 6.0},
        topic_signals=["how to", "tutorial", "step by step", "guide", "walkthrough", "build your own"],
    ),
    _strategy(
        "timeline",
        "Timeline",
        "A chronological sequence where the order itself is the story.",
        ["Hook on the endpoint", "Earliest point", "Key milestones", "Turning point", "Now / endpoint"],
        evidence_weight=7.0,
        format_affinity={"timeline": 10.0, "documentary": 7.0, "history": 8.0},
        goal_affinity={"education": 9.0, "awareness": 7.0},
        topic_signals=["timeline", "history of", "evolution", "from ... to", "the history"],
    ),
    _strategy(
        "comparison",
        "Comparison",
        "Two or more subjects compared on clear dimensions.",
        ["Hook on the contrast", "Subject A", "Subject B", "Head to head", "Which wins / takeaway"],
        evidence_weight=6.0,
        format_affinity={"comparison": 10.0, "list": 6.0, "news_analysis": 6.0},
        goal_affinity={"education": 8.0, "engagement": 7.0, "growth": 6.0},
        topic_signals=["vs", "compared", "versus", "better than", "difference between", "which is"],
    ),
    _strategy(
        "conflict",
        "Conflict",
        "A confrontation between forces, people, or ideas, resolved or unresolved.",
        ["Hook on the clash", "Side A's position", "Side B's position", "Escalation", "Outcome"],
        evidence_weight=6.5,
        format_affinity={"news_analysis": 8.0, "documentary": 7.0, "storytelling": 7.0},
        goal_affinity={"engagement": 9.0, "awareness": 7.0},
        topic_signals=["war", "battle", "feud", "lawsuit", "clash", "fight", "rivalry", "against"],
        needs_subject=True,
    ),
    _strategy(
        "transformation",
        "Transformation",
        "Before → after, centered on the change itself.",
        ["Hook on the change", "Before", "The catalyst", "After", "What the change means"],
        evidence_weight=5.0,
        format_affinity={"storytelling": 8.0, "case_study": 7.0, "motivation": 8.0},
        goal_affinity={"engagement": 8.0, "growth": 8.0, "education": 6.0},
        topic_signals=["transformed", "before and after", "changed", "turned into", "reinvented", "comeback"],
    ),
    _strategy(
        "news_analysis",
        "News Analysis",
        "A recent development explained with context and implication.",
        ["Hook on the news", "What happened", "Context", "Why it matters", "What's next"],
        evidence_weight=8.0,
        format_affinity={"news_analysis": 10.0, "news": 10.0},
        goal_affinity={"awareness": 9.0, "engagement": 7.0, "growth": 6.0},
        topic_signals=["news", "just announced", "released", "launched", "latest", "today"],
    ),
    _strategy(
        "prediction",
        "Prediction",
        "Uses current evidence to project a future outcome.",
        ["Hook on the forecast", "The current signal", "The forces at work", "The projection", "What to watch"],
        evidence_weight=7.0,
        format_affinity={"news_analysis": 7.0, "prediction": 10.0},
        goal_affinity={"growth": 7.0, "engagement": 7.0, "awareness": 6.0},
        topic_signals=["predict", "future of", "next", "will happen", "forecast", "could become"],
    ),
    _strategy(
        "debate",
        "Debate",
        "Both sides of a contested question, argued fairly.",
        ["Hook on the question", "Side A", "Side B", "The tension", "Your call"],
        evidence_weight=7.0,
        format_affinity={"news_analysis": 7.0, "debate": 10.0},
        goal_affinity={"engagement": 9.0, "awareness": 7.0},
        topic_signals=["debate", "controvers", "argue", "both sides", "pros and cons", "is it worth"],
        open_ending_ok=True,
    ),
    _strategy(
        "how_it_works",
        "How It Works",
        "The inner workings of a system, product, or phenomenon.",
        ["Hook on the surprise", "The system", "The key mechanism", "The catch", "Why it matters"],
        evidence_weight=6.0,
        format_affinity={"how_it_works": 10.0, "explainer": 8.0, "tutorial": 6.0},
        goal_affinity={"education": 10.0, "awareness": 7.0},
        topic_signals=["how it works", "how do", "the mechanism", "inside", "how does", "actually work"],
    ),
    _strategy(
        "list",
        "List",
        "A ranked or ordered set of items with a payoff at the end.",
        ["Hook on the count", "Item by item", "Numbered reveals", "The best / final item"],
        evidence_weight=3.0,
        format_affinity={"list": 10.0, "tutorial": 5.0},
        goal_affinity={"engagement": 8.0, "growth": 7.0},
        topic_signals=["list", "top", "best", "ways to", "reasons", "things", "facts", "number", "ranked"],
    ),
    _strategy(
        "educational",
        "Educational",
        "A concept built foundation-first so it clicks.",
        ["Hook on the misconception", "Foundation", "Step-by-step build", "Application", "Summary"],
        evidence_weight=6.0,
        format_affinity={"educational": 10.0, "explainer": 8.0},
        goal_affinity={"education": 10.0, "awareness": 6.0},
        topic_signals=["educational", "learn", "explained", "the concept", "understand"],
    ),
    _strategy(
        "interview",
        "Interview Style",
        "Answers to questions, conversational and direct.",
        ["Hook question", "Question one", "Question two", "Question three", "Final question"],
        evidence_weight=4.0,
        format_affinity={"interview": 10.0, "storytelling": 6.0},
        goal_affinity={"engagement": 8.0, "education": 6.0},
        topic_signals=["interview", "asked", "q&a", "conversation with", "talked to"],
    ),
    _strategy(
        "commentary",
        "Commentary",
        "A strong point of view on a subject, supported by facts.",
        ["Hook on the stance", "The claim", "Supporting evidence", "Counterpoint", "The verdict"],
        evidence_weight=5.0,
        format_affinity={"commentary": 10.0, "news_analysis": 8.0},
        goal_affinity={"engagement": 9.0, "growth": 6.0},
        topic_signals=["my take", "opinion", "commentary", "hot take", "actually", "the truth"],
    ),
]

_STRATEGY_BY_ID: Dict[str, NarrativeStrategy] = {
    strategy.id: strategy for strategy in NARRATIVE_STRATEGIES
}

# Stable catalog order used for deterministic tie-breaking.
NARRATIVE_STRATEGIES_IDS = [strategy.id for strategy in NARRATIVE_STRATEGIES]


def list_narrative_strategies() -> List[str]:
    """Stable list of supported strategy ids (exposed for UI/tests)."""
    return [strategy.id for strategy in NARRATIVE_STRATEGIES]


def get_narrative_strategy(strategy_id: str) -> NarrativeStrategy:
    """Resolve a strategy id; unknown ids fall back to ``explainer``."""
    return _STRATEGY_BY_ID.get((strategy_id or "").strip().lower(), _STRATEGY_BY_ID["explainer"])


# ---------------------------------------------------------------------------
# Narrative usage history (constrained variance)
# ---------------------------------------------------------------------------

_MAX_USAGE_HISTORY = 20
_DEFAULT_AVOID_WINDOW = 3
_VARIANCE_PENALTY = 1.8

_usage_history: Dict[str, List[str]] = {}


def record_narrative_usage(profile_name: str, strategy_id: str) -> None:
    """Record a strategy used for a profile (in-memory variance history)."""
    key = (profile_name or "custom").strip().lower()
    used = _usage_history.setdefault(key, [])
    used.append((strategy_id or "").strip().lower())
    if len(used) > _MAX_USAGE_HISTORY:
        del used[: len(used) - _MAX_USAGE_HISTORY]


def recent_narrative_usage(profile_name: str, window: int = _DEFAULT_AVOID_WINDOW) -> List[str]:
    """Strategies used in the last ``window`` runs for this profile (oldest→newest)."""
    used = _usage_history.get((profile_name or "custom").strip().lower(), [])
    return list(used[-window:]) if used else []


def _variance_penalty_for(strategy_id: str, recent: List[str]) -> float:
    """Deterministic penalty: strategies used more recently are avoided harder."""
    if not recent or not strategy_id:
        return 0.0
    sid = strategy_id.strip().lower()
    if sid not in recent:
        return 0.0
    # Most recent use gets the full penalty; each step back reduces it.
    recency = len(recent) - 1 - recent[::-1].index(sid)
    return _VARIANCE_PENALTY * (1.0 - 0.4 * recency)


def reset_narrative_usage() -> None:
    """Clear the in-memory history (used by tests)."""
    _usage_history.clear()


# ---------------------------------------------------------------------------
# Topic signal detection (deterministic)
# ---------------------------------------------------------------------------

_FUNCTION_WORDS = frozenset(
    (
        "the", "a", "an", "of", "to", "in", "on", "at", "and", "or", "but",
        "for", "with", "about", "this", "that", "these", "those", "it", "its",
        "is", "are", "was", "were", "be", "been", "being", "will", "would",
        "can", "could", "should", "do", "does", "did", "not", "no", "then",
        "than", "so", "just", "only", "every", "one", "two", "you", "your",
        "my", "me", "we", "our", "they", "them", "their", "he", "she", "his",
        "her", "there", "here", "from", "into", "out", "up", "down", "as",
        "by", "if", "how", "why", "what", "who", "when", "where",
    )
)

_EMOTIONAL_LEXICON = (
    "fear", "afraid", "terror", "lost", "alone", "death", "died", "killed",
    "betray", "miracle", "hope", "pain", "heart", "impossible", "shocking",
    "bizarre", "haunted", "curse", "collapse", "destroyed", "crashed",
)


def _significant_words(text: str) -> List[str]:
    import re

    return [
        word
        for word in re.findall(r"[A-Za-z0-9']+", (text or "").lower())
        if len(word) > 2 and word not in _FUNCTION_WORDS
    ]


def _topic_signal_hits(signals: List[str], topic: str, analysis_text: str) -> float:
    """Score how strongly a strategy's topic signals appear in topic + analysis.

    Signals match at word boundaries so a signal like "best" hits inside
    "the best games of 2025" without requiring surrounding spaces.
    """
    import re

    haystack = f"{(topic or '').lower()} {analysis_text.lower()}"
    hits = 0
    for signal in signals or []:
        signal = signal.strip().lower()
        if not signal:
            continue
        if re.search(r"\b" + re.escape(signal) + r"\b", haystack):
            hits += 1
    return float(hits)


def _estimated_target_seconds(profile: ContentProfile, intelligence: Optional[ContentIntelligence]) -> float:
    """Deterministic estimate of the target duration from profile data."""
    text = " ".join(
        part
        for part in (
            (profile.preferred_video_length or ""),
            (intelligence.preferred_length if hasattr(intelligence, "preferred_length") else ""),
        )
        if part
    )
    import re

    numbers = [int(match) for match in re.findall(r"\d+", text)]
    if len(numbers) >= 2:
        return float(sum(numbers) / len(numbers))
    if numbers:
        return float(numbers[0])
    return 60.0


# ---------------------------------------------------------------------------
# Strategy selection
# ---------------------------------------------------------------------------


def _topic_starts_how(topic: str) -> bool:
    """True when the topic opens with 'how' (a mechanism/explanation frame)."""
    return (topic or "").strip().lower().startswith("how ")


def select_narrative_strategy(
    topic: str,
    profile: ContentProfile,
    intelligence: Optional[ContentIntelligence] = None,
    topic_analysis_text: str = "",
    research_summary: str = "",
    verified_claims: int = 0,
    target_duration_seconds: Optional[float] = None,
    recent_strategies: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Select the best narrative strategy for this content.

    Returns ``{"strategy": NarrativeStrategy, "scores": {...}, "rationale": str}``.
    Fully deterministic: the same inputs always produce the same selection.
    """
    # -- Evidence available -------------------------------------------------
    evidence_strength = min(10.0, 2.0 + float(verified_claims) * 1.5)
    if research_summary:
        evidence_strength = min(10.0, evidence_strength + 1.0)
    duration = target_duration_seconds or _estimated_target_seconds(profile, intelligence)
    format_id = (intelligence.format if intelligence else "") or ""
    goal_id = (intelligence.content_goal if intelligence else "") or ""
    recent = recent_strategies if recent_strategies is not None else recent_narrative_usage(profile.name)

    profile_prefs = [
        pattern.strip().lower()
        for pattern in (profile.preferred_narrative_patterns or [])
        if pattern and pattern.strip()
    ]
    narrative_style = (profile.narrative_style or "").lower()

    scored: Dict[str, Dict[str, float]] = {}
    for strategy in NARRATIVE_STRATEGIES:
        format_score = float(strategy.format_affinity.get(format_id, 0.0))
        goal_score = float(strategy.goal_affinity.get(goal_id, 0.0))
        signal_score = min(6.0, _topic_signal_hits(strategy.topic_signals, topic, topic_analysis_text))

        # Profile preference: exact pattern match or narrative-style overlap.
        profile_score = 0.0
        if strategy.id in profile_prefs or strategy.label.lower() in profile_prefs:
            profile_score = 5.0
        elif any(word in narrative_style for word in (strategy.id.replace("_", " ").split())):
            profile_score = 3.0
        elif strategy.id in narrative_style:
            profile_score = 3.0

        # Evidence fit: a strategy demanding high evidence is penalized when
        # research is thin; low-evidence strategies are not rewarded for thin
        # research but get a small bonus for open-ended framing.
        evidence_penalty = max(0.0, strategy.evidence_weight - evidence_strength)
        if strategy.open_ending_ok and evidence_strength < 5.0:
            evidence_penalty = max(0.0, evidence_penalty - 1.5)

        # Short-form preference: tight strategies get a small boost on short
        # formats; documentary-style structures prefer longer durations.
        duration_score = 0.0
        if duration <= 45.0 and strategy.id in ("list", "commentary", "how_it_works", "explainer"):
            duration_score = 2.0
        elif duration >= 90.0 and strategy.id in ("documentary", "biography", "timeline", "rise_and_fall"):
            duration_score = 2.0

        # "How … works" topics are mechanism frames: prefer explanation engines.
        how_start_boost = 0.0
        if _topic_starts_how(topic) and strategy.id in ("explainer", "how_it_works", "educational", "tutorial"):
            how_start_boost = 2.0

        variance_penalty = _variance_penalty_for(strategy.id, recent)

        total = (
            format_score * 1.2
            + goal_score * 0.8
            + signal_score
            + profile_score
            + duration_score
            + how_start_boost
            - evidence_penalty
            - variance_penalty
        )
        scored[strategy.id] = {
            "format": format_score,
            "goal": goal_score,
            "signal": signal_score,
            "profile": profile_score,
            "duration": duration_score,
            "evidence": evidence_penalty,
            "variance": variance_penalty,
            "total": round(total, 2),
        }

    ordered = sorted(
        scored.items(),
        key=lambda item: (-item[1]["total"], NARRATIVE_STRATEGIES_IDS.index(item[0])),
    )
    best_id, best_scores = ordered[0]
    strategy = _STRATEGY_BY_ID[best_id]

    rationale = (
        f"{strategy.label} fits format={format_id or 'unspecified'!r}, "
        f"goal={goal_id or 'unspecified'!r}, evidence_strength={round(evidence_strength, 1)}/10, "
        f"duration~{round(duration)}s"
    )
    if best_scores["variance"] > 0:
        rationale += "; avoided recently used strategies"

    logger.debug(f"narrative strategy selected: {strategy.id} (total={best_scores['total']})")
    return {
        "strategy": strategy,
        "scores": {strategy_id: round(scores["total"], 2) for strategy_id, scores in scored.items()},
        "rationale": rationale,
    }


def narrative_strategy_context(strategy: Optional[NarrativeStrategy], rationale: str = "") -> str:
    """Prompt block embedding the selected narrative strategy downstream."""
    if strategy is None:
        return ""
    lines = [
        "# Narrative Strategy (selected)",
        f"- Strategy: {strategy.label}",
        f"- What it means: {strategy.description}",
        f"- Section template: {' -> '.join(strategy.sections)}",
    ]
    if rationale:
        lines.append(f"- Why selected: {rationale}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Story Brief
# ---------------------------------------------------------------------------


class StoryBrief(BaseModel):
    """Structured story/content brief consumed by the script agent.

    Fields are intentionally flexible (all optional): the model never forces
    fields that are irrelevant to the content type. ``extra="ignore"`` keeps
    forward-compatible payloads from other phases out of the schema.
    """

    model_config = ConfigDict(extra="ignore")

    subject: str = ""
    central_question: str = ""
    hook: str = ""
    protagonist: str = ""
    conflict: str = ""
    stakes: str = ""
    turning_point: str = ""
    evidence: List[str] = Field(default_factory=list)
    emotional_arc: List[str] = Field(default_factory=list)
    narrative_strategy: str = ""
    key_facts: List[str] = Field(default_factory=list)
    payoff: str = ""
    conclusion: str = ""


def _clean_items(items: List[Any], limit: int = 6) -> List[str]:
    cleaned: List[str] = []
    for item in items or []:
        text = str(item or "").strip()
        if text and text not in cleaned:
            cleaned.append(text)
        if len(cleaned) >= limit:
            break
    return cleaned


def build_story_brief(
    topic: str,
    profile: ContentProfile,
    strategy: NarrativeStrategy,
    selected_hook: str = "",
    intelligence: Optional[ContentIntelligence] = None,
    topic_analysis: Optional[Dict[str, Any]] = None,
    content_strategy: Optional[Dict[str, Any]] = None,
    research_claims: Optional[List[Any]] = None,
    research_uncertainties: Optional[List[str]] = None,
    research_contradictions: Optional[List[str]] = None,
) -> StoryBrief:
    """Compose the StoryBrief deterministically from the structured context.

    No LLM call: everything is derived from already-produced structured data.
    Fields that cannot be derived stay empty rather than being fabricated.
    """
    analysis = topic_analysis or {}
    strategy_data = content_strategy or {}
    emotional_arc = _clean_items(strategy_data.get("emotional_progression") or analysis.get("emotional_angles") or [])
    if not emotional_arc:
        emotional_arc = ["Curiosity", "Engagement"]

    def _claim_status(claim: Any) -> str:
        # ``status`` may be a str-Enum member (e.g. ClaimStatus.VERIFIED);
        # str() of an Enum yields the member repr, so unwrap .value first.
        raw = claim.get("status") if isinstance(claim, dict) else getattr(claim, "status", "")
        if raw is None:
            return ""
        return str(getattr(raw, "value", raw)).lower()

    verified = []
    for claim in research_claims or []:
        statement = (
            str(claim.get("statement", "")) if isinstance(claim, dict) else str(getattr(claim, "statement", ""))
        )
        if statement and _claim_status(claim) == "verified":
            verified.append(statement)
    key_facts = verified or _clean_items(analysis.get("potential_claims") or [])

    questions = _clean_items(analysis.get("curiosity_gaps") or [])
    central_question = questions[0] if questions else str(analysis.get("known_vs_unknown") or "")

    conflict = ""
    controversy = str(analysis.get("controversy_level") or "").lower()
    if controversy in ("medium", "high"):
        conflict = f"documented controversy: {controversy} level"
    elif strategy_data.get("primary_angle"):
        conflict = str(strategy_data["primary_angle"])
    if not conflict and research_contradictions:
        conflict = "sources disagree on key details"

    stakes = ""
    known_vs_unknown = str(analysis.get("known_vs_unknown") or "")
    if known_vs_unknown:
        stakes = known_vs_unknown

    uncertainty = ""
    if research_uncertainties:
        uncertainty = str(research_uncertainties[0])

    sections = strategy.sections or []
    turning_point = sections[len(sections) // 2] if sections else ""

    payoff = sections[-1] if sections else ""
    conclusion = str(strategy_data.get("cta") or "") or payoff

    brief = StoryBrief(
        subject=topic,
        central_question=central_question[:200],
        hook=(selected_hook or "").strip(),
        protagonist=_detect_protagonist(topic, analysis),
        conflict=conflict[:200],
        stakes=stakes[:200],
        turning_point=turning_point[:120],
        evidence=_clean_items(verified, limit=6),
        emotional_arc=emotional_arc,
        narrative_strategy=strategy.label,
        key_facts=key_facts,
        payoff=payoff[:120],
        conclusion=conclusion[:200],
    )
    if uncertainty:
        brief.evidence = brief.evidence + [f"unverified: {uncertainty[:120]}"]
    return brief


_PROTAGONIST_HINTS = (
    "founder", "ceo", "president", "king", "queen", "emperor", "general",
    "inventor", "artist", "scientist", "author", "director", "leader",
)


def _detect_protagonist(topic: str, analysis: Dict[str, Any]) -> str:
    """Deterministic protagonist extraction: a named person/entity in context."""
    known_vs_unknown = str(analysis.get("known_vs_unknown") or "")
    text = f"{topic} {known_vs_unknown}"
    lowered = text.lower()
    for hint in _PROTAGONIST_HINTS:
        index = lowered.find(hint)
        if index != -1:
            start = max(0, index - 25)
            window = text[start : index + len(hint) + 15].strip(" .,;:()")
            return window[:120]
    return ""


def story_brief_context(brief: Optional[StoryBrief]) -> str:
    """Prompt block embedding the story brief into the script writer."""
    if brief is None:
        return ""
    lines = ["# Story Brief (follow this structure)"]
    fields = [
        ("Central question", brief.central_question),
        ("Hook (open with this)", brief.hook),
        ("Subject", brief.subject),
        ("Protagonist", brief.protagonist),
        ("Conflict", brief.conflict),
        ("Stakes", brief.stakes),
        ("Turning point", brief.turning_point),
        ("Emotional arc", ", ".join(brief.emotional_arc)),
        ("Narrative strategy", brief.narrative_strategy),
        ("Payoff", brief.payoff),
        ("Conclusion", brief.conclusion),
    ]
    for label, value in fields:
        if value:
            lines.append(f"- {label}: {value}")
    if brief.key_facts:
        lines.append("- Key facts (ground the script in these):")
        lines += [f"  - {fact[:160]}" for fact in brief.key_facts[:5]]
    return "\n".join(lines)


def story_brief_summary(brief: Optional[StoryBrief]) -> str:
    """One-line summary for logs/UI (no chain-of-thought)."""
    if brief is None:
        return "none"
    return (
        f"strategy={brief.narrative_strategy or '?'}, "
        f"question={brief.central_question[:48]!r}, facts={len(brief.key_facts)}"
    )
