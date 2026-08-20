"""Trend Intelligence & Topic Discovery (Phase 2E.1–2E.3).

Provider-agnostic trend layer:
- ``TrendSignal`` carries a context classification (CURRENT / RECENT /
  HISTORICAL / EVERGREEN / MODEL_INFERENCE) so freshness is never faked.
- Providers are swappable (protocol + registry). The built-in providers are
  model-inference (LLM, explicitly labeled) and web-search (reuses the
  research ``web_search`` endpoint when configured). No provider fabricates
  trend data: when no fresh source exists, signals are labeled
  MODEL_INFERENCE, never CURRENT.

Topic Discovery:
- Modes: TRENDING / EVERGREEN / OPPORTUNITY / NEWS / COMPETITOR_INSPIRED /
  CHANNEL_OPTIMIZED / AUTONOMOUS / USER_PROVIDED.
- Candidates are scored on explainable dimensions and ranked; the scoring
  function is deterministic and every dimension records a rationale.
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from app.config import config
from app.services.agent_llm import AgentTracker, _clamp_score, _llm_json
from app.services.content_profile import ContentProfile
from app.services.intelligence import ContentIntelligence

TREND_CONTEXT_CURRENT = "current"
TREND_CONTEXT_RECENT = "recent"
TREND_CONTEXT_HISTORICAL = "historical"
TREND_CONTEXT_EVERGREEN = "evergreen"
TREND_CONTEXT_MODEL_INFERENCE = "model_inference"

TREND_CONTEXTS = (
    TREND_CONTEXT_CURRENT,
    TREND_CONTEXT_RECENT,
    TREND_CONTEXT_HISTORICAL,
    TREND_CONTEXT_EVERGREEN,
    TREND_CONTEXT_MODEL_INFERENCE,
)

TOPIC_MODE_TRENDING = "trending"
TOPIC_MODE_EVERGREEN = "evergreen"
TOPIC_MODE_OPPORTUNITY = "opportunity"
TOPIC_MODE_NEWS = "news"
TOPIC_MODE_COMPETITOR = "competitor_inspired"
TOPIC_MODE_CHANNEL = "channel_optimized"
TOPIC_MODE_AUTONOMOUS = "autonomous"
TOPIC_MODE_USER = "user_provided"

TOPIC_MODES = (
    TOPIC_MODE_TRENDING,
    TOPIC_MODE_EVERGREEN,
    TOPIC_MODE_OPPORTUNITY,
    TOPIC_MODE_NEWS,
    TOPIC_MODE_COMPETITOR,
    TOPIC_MODE_CHANNEL,
    TOPIC_MODE_AUTONOMOUS,
    TOPIC_MODE_USER,
)


class TrendSignal(BaseModel):
    model_config = ConfigDict(extra="ignore")

    topic: str = ""
    context: str = TREND_CONTEXT_MODEL_INFERENCE  # current|recent|historical|evergreen|model_inference
    direction: str = "unknown"  # rising | stable | declining | unknown
    score: float = 5.0  # 0-10 trend strength
    source: str = ""  # provider name
    observed_at: float = 0.0
    note: str = ""


class TrendProvider:
    """Protocol: a provider returns candidate trend signals.

    Providers must never fabricate trend data: they either return real
    observations or raise TrendProviderError, and the context field states
    exactly how fresh the data is.
    """

    name: str = "base"

    def fetch(self, profile: ContentProfile, intelligence: Optional[ContentIntelligence], app_config=None) -> List[TrendSignal]:
        raise NotImplementedError


class TrendProviderError(RuntimeError):
    """Raised when a trend provider cannot serve."""


class ModelInferenceTrendProvider(TrendProvider):
    """LLM-sourced trend candidates, explicitly labeled MODEL_INFERENCE.

    These are hypotheses, not current data: the context is always
    ``model_inference`` unless the model itself is fed fresh evidence.
    """

    name = "model_inference"

    def fetch(
        self,
        profile: ContentProfile,
        intelligence: Optional[ContentIntelligence] = None,
        app_config=None,
        tracker: Optional[AgentTracker] = None,
        niche: str = "",
    ) -> List[TrendSignal]:
        niche = niche or (intelligence.niche if intelligence else "") or profile.name
        prompt = f"""
# Role: Trend Analyst (model knowledge only)

Suggest topics with growth potential in the niche below. You have NO live
trend data: everything you produce is model knowledge. Label every signal
honestly as model_inference; never claim "current" or "recent".

## Niche
{niche} — {profile.description}

## Audience
{profile.audience or 'general'}

## Goal
{profile.content_goals or 'grow the channel'}

Return ONLY a JSON array of up to 5 objects:
[{{"topic": "specific topic", "direction": "rising|stable|declining|unknown", "score": 0-10, "note": "why it might grow (model knowledge, not live data)"}}]
"""

        def fallback() -> List[TrendSignal]:
            return []

        try:
            payload = _llm_json(prompt, fallback, app_config=app_config, tracker=tracker, agent="trend_analyst")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"model-inference trend fetch failed: {exc}")
            return []
        if not isinstance(payload, list):
            return []

        signals = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            topic = str(item.get("topic", "")).strip()
            if not topic:
                continue
            direction = str(item.get("direction", "")).strip().lower()
            if direction not in ("rising", "stable", "declining", "unknown"):
                direction = "unknown"
            try:
                score = round(_clamp_score(item.get("score", 5.0), "trend score"), 1)
            except (TypeError, ValueError):
                score = 5.0
            signals.append(
                TrendSignal(
                    topic=topic[:120],
                    context=TREND_CONTEXT_MODEL_INFERENCE,
                    direction=direction,
                    score=score,
                    source=self.name,
                    observed_at=time.time(),
                    note=str(item.get("note", ""))[:200],
                )
            )
        return signals


class WebSearchTrendProvider(TrendProvider):
    """Live-search trend provider (activates only when configured).

    Reuses the research ``web_search`` configuration: when a search endpoint
    is configured, topics discovered from its results are labeled RECENT
    (they come from a live search, but we do not have engagement metrics to
    claim CURRENT). Failure degrades gracefully to zero signals.
    """

    name = "web_search"

    @staticmethod
    def is_configured(app_config=None) -> bool:
        base_url = _trend_setting(app_config, "base_url")
        return bool(base_url and _trend_setting(app_config, "provider") == "web_search")

    def fetch(
        self,
        profile: ContentProfile,
        intelligence: Optional[ContentIntelligence] = None,
        app_config=None,
        niche: str = "",
    ) -> List[TrendSignal]:
        if not self.is_configured(app_config):
            raise TrendProviderError("web search trend provider is not configured")
        import requests

        query = niche or (intelligence.niche if intelligence else "") or profile.name
        base_url = _trend_setting(app_config, "base_url")
        api_key = _trend_setting(app_config, "api_key")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        response = requests.get(
            base_url,
            params={"q": f"trending topics {query}"},
            headers=headers,
            timeout=(10, 30),
        )
        response.raise_for_status()
        payload = response.json()
        results = payload.get("results") if isinstance(payload, dict) else payload
        if not isinstance(results, list):
            raise TrendProviderError("web search trend provider returned an unexpected payload")

        signals = []
        for item in results:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "")).strip()
            snippet = str(item.get("snippet", "")).strip()
            if not title:
                continue
            signals.append(
                TrendSignal(
                    topic=title[:120],
                    context=TREND_CONTEXT_RECENT,  # live search result, not engagement data
                    direction="unknown",
                    score=6.0,
                    source=self.name,
                    observed_at=time.time(),
                    note=snippet[:200],
                )
            )
        return signals


def _trend_setting(app_config, key: str, default: Any = "") -> Any:
    runtime = app_config or {}
    if key in runtime:
        return runtime[key]
    return config.research.get(key, default)


def collect_trend_signals(
    profile: ContentProfile,
    intelligence: Optional[ContentIntelligence] = None,
    app_config=None,
    tracker: Optional[AgentTracker] = None,
    niche: str = "",
) -> List[TrendSignal]:
    """Run all available trend providers and merge signals (deduplicated)."""
    collected: List[TrendSignal] = []
    seen: set[str] = set()

    if WebSearchTrendProvider.is_configured(app_config):
        try:
            web = WebSearchTrendProvider()
            collected.extend(web.fetch(profile, intelligence, app_config, niche=niche))
        except Exception as exc:  # noqa: BLE001 - trend failure must not propagate
            logger.warning(f"web trend provider unavailable: {exc}")
            if tracker:
                tracker.set_fallback("trend_analyst", f"web_search unavailable: {exc}")

    try:
        model = ModelInferenceTrendProvider()
        collected.extend(
            model.fetch(profile, intelligence, app_config, tracker=tracker, niche=niche)
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"model-inference trend provider unavailable: {exc}")
        if tracker:
            tracker.set_fallback("trend_analyst", f"model_inference unavailable: {exc}")

    deduped = []
    for signal in collected:
        key = signal.topic.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(signal)
    deduped.sort(key=lambda item: item.score, reverse=True)
    return deduped


# ---------------------------------------------------------------------------
# Topic discovery & scoring
# ---------------------------------------------------------------------------

# Deterministic, explainable scoring dimensions.
TOPIC_SCORE_DIMENSIONS = (
    "trend_strength",
    "audience_relevance",
    "competition",
    "novelty",
    "evergreen_value",
    "story_potential",
    "research_availability",
    "visual_potential",
    "channel_fit",
    "risk",
    "monetization",
)


class TopicCandidate(BaseModel):
    model_config = ConfigDict(extra="ignore")

    topic: str = ""
    mode: str = TOPIC_MODE_TRENDING
    scores: Dict[str, float] = Field(default_factory=dict)
    total: float = 0.0
    rationales: Dict[str, str] = Field(default_factory=dict)
    signal: Optional[Dict[str, Any]] = None


def score_topic(
    topic: str,
    profile: ContentProfile,
    intelligence: Optional[ContentIntelligence] = None,
    mode: str = TOPIC_MODE_TRENDING,
    signal: Optional[TrendSignal] = None,
) -> TopicCandidate:
    """Score a topic candidate on explainable dimensions (0-10 each)."""
    text = (topic or "").lower()
    word_count = len(re.findall(r"[A-Za-z0-9']+", topic or ""))
    scores: Dict[str, float] = {}
    rationales: Dict[str, str] = {}

    trend = signal.score if signal is not None else 5.0
    trend_ctx = signal.context if signal is not None else TREND_CONTEXT_MODEL_INFERENCE
    if mode == TOPIC_MODE_EVERGREEN:
        trend = 7.0  # evergreen mode prefers stable interest over spikes
    if mode == TOPIC_MODE_NEWS and signal is not None and trend_ctx == TREND_CONTEXT_RECENT:
        trend = min(10.0, trend + 2.0)
    scores["trend_strength"] = round(min(10.0, trend), 1)
    rationales["trend_strength"] = f"signal={signal.source if signal else 'baseline'}, context={trend_ctx}"

    audience = profile.audience or ""
    audience_hits = 0
    for word in re.findall(r"[a-z]{4,}", audience.lower()):
        if word in text:
            audience_hits += 1
    scores["audience_relevance"] = round(min(10.0, 4.0 + audience_hits * 1.5), 1)
    rationales["audience_relevance"] = f"{audience_hits} audience keyword(s) matched"

    # Competition: unknown baseline stays neutral; a specific niche with few
    # broad competitors scores better (deterministic heuristic).
    scores["competition"] = 5.0 if word_count >= 3 else 6.0
    rationales["competition"] = "specific topic implies narrower competition" if word_count >= 3 else "broad topic may be competitive"

    scores["novelty"] = 6.0 if signal is None or trend_ctx == TREND_CONTEXT_MODEL_INFERENCE else 5.0
    rationales["novelty"] = "model-inferred topics are rarely saturated" if signal is None else "topic seen in live search"

    sensitivity = (intelligence.trend_sensitivity if intelligence else "") or (profile.trend_sensitivity or "")
    evergreen = 8.0 if sensitivity in ("low", "medium") else 5.0
    if mode == TOPIC_MODE_EVERGREEN:
        evergreen = 9.0
    scores["evergreen_value"] = evergreen
    rationales["evergreen_value"] = f"niche trend sensitivity={sensitivity or 'medium'}"

    story_potential = 6.0
    if any(word in text for word in ("history", "story", "rise", "fall", "war", "secret", "mystery", "case")):
        story_potential = 8.0
        rationales["story_potential"] = "topic contains narrative cues"
    else:
        rationales["story_potential"] = "no strong narrative cues; assumed average"
    scores["story_potential"] = story_potential

    scores["research_availability"] = 7.0
    rationales["research_availability"] = "standard web/model sources assumed"

    visual_potential = 6.0
    if any(word in text for word in ("how", "explain", "tutorial", "build", "inside", "tour")):
        visual_potential = 8.0
        rationales["visual_potential"] = "topic is visually demonstrable"
    else:
        rationales["visual_potential"] = "no explicit visual hooks; assumed average"
    scores["visual_potential"] = visual_potential

    scores["channel_fit"] = 7.0 if mode in (TOPIC_MODE_CHANNEL, TOPIC_MODE_AUTONOMOUS) else 6.0
    rationales["channel_fit"] = "topic aligned with the channel strategy" if mode in (TOPIC_MODE_CHANNEL, TOPIC_MODE_AUTONOMOUS) else "generic fit"

    risk = 5.0
    if mode == TOPIC_MODE_NEWS:
        risk = 6.0
        rationales["risk"] = "news topics age quickly; freshness risk"
    elif mode == TOPIC_MODE_TRENDING:
        risk = 6.0
        rationales["risk"] = "trending topics may be volatile"
    else:
        rationales["risk"] = "low volatility expected"
    scores["risk"] = risk

    monetization = 5.0
    if profile.content_goals and "monetiz" in profile.content_goals.lower():
        monetization = 7.0
        rationales["monetization"] = "niche goal is monetization"
    else:
        rationales["monetization"] = "monetization not a stated goal"
    scores["monetization"] = monetization

    total = round(sum(scores.values()) / len(scores), 1)
    return TopicCandidate(
        topic=topic,
        mode=mode,
        scores=scores,
        total=total,
        rationales=rationales,
        signal=signal.model_dump() if signal is not None else None,
    )


def discover_topics(
    profile: ContentProfile,
    intelligence: Optional[ContentIntelligence] = None,
    mode: str = TOPIC_MODE_TRENDING,
    app_config=None,
    tracker: Optional[AgentTracker] = None,
    niche: str = "",
    user_topics: Optional[List[str]] = None,
    max_candidates: int = 6,
    signals: Optional[List[TrendSignal]] = None,
) -> List[TopicCandidate]:
    """Discover and score topic candidates for the given mode.

    Modes:
    - trending / news / competitor_inspired / channel_optimized: provider
      signals (web search when configured, otherwise model inference).
    - evergreen: model-inferred evergreen topics.
    - autonomous: mix of the above, best overall wins.
    - user_provided: score the caller's own topics deterministically.

    ``signals`` lets callers reuse signals they already collected (the
    agentic orchestrator collects once for trend refinement and passes them
    here), avoiding a duplicated LLM provider call.
    """
    candidates: List[TopicCandidate] = []

    if mode == TOPIC_MODE_USER:
        for topic in user_topics or []:
            topic = str(topic or "").strip()
            if topic:
                candidates.append(
                    score_topic(topic, profile, intelligence, mode=mode)
                )
        candidates.sort(key=lambda item: item.total, reverse=True)
        return candidates[:max_candidates]

    if signals is None:
        signals = collect_trend_signals(
            profile, intelligence, app_config=app_config, tracker=tracker, niche=niche
        )
    if mode == TOPIC_MODE_EVERGREEN:
        # For evergreen, drop the "trending" framing: keep model signals but
        # re-score with the evergreen preference.
        for signal in signals:
            candidates.append(score_topic(signal.topic, profile, intelligence, mode=mode, signal=signal))
        candidates.sort(key=lambda item: item.total, reverse=True)
        return candidates[:max_candidates]

    if not signals:
        return []

    if mode in (TOPIC_MODE_TRENDING, TOPIC_MODE_NEWS, TOPIC_MODE_OPPORTUNITY):
        for signal in signals:
            candidates.append(score_topic(signal.topic, profile, intelligence, mode=mode, signal=signal))
    elif mode == TOPIC_MODE_COMPETITOR:
        # Competitor-inspired: keep the highest-scoring unique signals.
        for signal in signals[:max_candidates]:
            candidates.append(score_topic(signal.topic, profile, intelligence, mode=mode, signal=signal))
    elif mode == TOPIC_MODE_CHANNEL:
        # Channel-optimized: rank by channel fit first.
        for signal in signals:
            candidate = score_topic(signal.topic, profile, intelligence, mode=mode, signal=signal)
            candidate.total = round(candidate.scores["channel_fit"] * 0.5 + candidate.total * 0.5, 1)
            candidates.append(candidate)
    elif mode == TOPIC_MODE_AUTONOMOUS:
        for signal in signals:
            candidates.append(score_topic(signal.topic, profile, intelligence, mode=mode, signal=signal))
    else:
        logger.warning(f"unknown topic discovery mode {mode!r}, treating as trending")
        for signal in signals:
            candidates.append(score_topic(signal.topic, profile, intelligence, mode=TOPIC_MODE_TRENDING, signal=signal))

    # Deduplicate by normalized topic.
    deduped: List[TopicCandidate] = []
    seen: set[str] = set()
    for candidate in sorted(candidates, key=lambda item: item.total, reverse=True):
        key = candidate.topic.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped[:max_candidates]


def select_topic_for_automation(
    candidates: List[TopicCandidate],
    automation_level: str,
) -> Optional[TopicCandidate]:
    """Map automation level to topic decision:
    - manual: user controls the topic (no selection here).
    - assisted: recommend the best (caller still decides).
    - automatic / autopilot: select the best candidate.
    """
    level = (automation_level or "").strip().lower()
    if level == "manual":
        return None
    if not candidates:
        return None
    return candidates[0]


def trends_summary(signals: List[TrendSignal]) -> str:
    contexts: Dict[str, int] = {}
    for signal in signals:
        contexts[signal.context] = contexts.get(signal.context, 0) + 1
    return f"signals={len(signals)}, contexts={dict(sorted(contexts.items()))}"
