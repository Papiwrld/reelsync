"""Long-form → short-form repurposing (Phase 2D.3).

Identifies candidate moments in a long-form script based on narrative value
(hooks, surprising facts, emotional moments, useful insights, conclusions,
conflicts, curiosity gaps) — never arbitrary time intervals — and builds a
short-form plan where each short has its own hook, coherent context, payoff,
duration and title/caption.
"""

from __future__ import annotations

import re
from typing import List, Optional

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from app.services.content_profile import ContentProfile

# Nominal speaking rate for duration estimates (words per minute).
DEFAULT_WORDS_PER_MINUTE = 150
DEFAULT_SHORT_MIN_SECONDS = 20
DEFAULT_SHORT_MAX_SECONDS = 60


class ShortMoment(BaseModel):
    """A candidate short derived from one or more script sentences."""

    model_config = ConfigDict(extra="ignore")

    index: int = 0
    source_text: str = ""
    moment_type: str = ""  # hook | surprise | emotion | insight | conclusion | conflict | curiosity
    hook: str = ""
    context: str = ""
    payoff: str = ""
    estimated_seconds: float = 30.0
    title: str = ""
    caption: str = ""
    score: float = 0.0


class RepurposePlan(BaseModel):
    model_config = ConfigDict(extra="ignore")

    shorts: List[ShortMoment] = Field(default_factory=list)
    rationale: str = ""


# ---------------------------------------------------------------------------
# Moment detection (deterministic)
# ---------------------------------------------------------------------------

_SURPRISE_WORDS = (
    "billion", "million", "percent", "unexpected", "surprisingly", "secretly",
    "impossible", "never", "no one", "nobody", "actually", "revealed",
    "shocking", "vanished", "crashed", "collapsed", "destroyed",
)
_EMOTION_WORDS = (
    "fear", "afraid", "terror", "betray", "miracle", "hope", "pain", "heart",
    "lost", "alone", "death", "died", "killed", "bizarre", "haunted", "curse",
    "guilt", "shame", "pride", "rage",
)
_INSIGHT_WORDS = (
    "because", "the reason", "which means", "in other words", "the lesson",
    "what this tells us", "turns out", "the key", "explains why", "therefore",
)
_CONCLUSION_WORDS = ("in the end", "finally", "ultimately", "so what", "the takeaway", "in short")
_CONFLICT_WORDS = ("versus", "vs", "against", "battle", "lawsuit", "feud", "rivalry", "clash", "fight", "war")
_CURIOSITY_WORDS = ("why", "how", "who", "what if", "nobody knows", "unknown", "secret", "mystery", "the truth")


def _split_sentences(script: str) -> List[str]:
    text = (script or "").strip()
    if not text:
        return []
    return [
        part.strip()
        for part in re.split(r"(?<=[.!?。！？])\s+", text)
        if part.strip()
    ]


def _moment_score(sentence: str) -> float:
    """Score a sentence's short-form potential (0-10)."""
    text = sentence.lower()
    score = 1.0
    if re.search(r"\d[\d,]*\.?\d*%?|\$[\d,]+", sentence):
        score += 3.0  # surprising numbers
    score += 0.8 * sum(1 for word in _SURPRISE_WORDS if word in text)
    score += 0.6 * sum(1 for word in _EMOTION_WORDS if word in text)
    score += 0.8 * sum(1 for word in _INSIGHT_WORDS if word in text)
    score += 0.6 * sum(1 for word in _CONCLUSION_WORDS if word in text)
    score += 0.6 * sum(1 for word in _CONFLICT_WORDS if word in text)
    score += 0.5 * sum(1 for word in _CURIOSITY_WORDS if word in text)
    if sentence.rstrip().endswith("?"):
        score += 1.0
    words = len(re.findall(r"[A-Za-z0-9']+", sentence))
    if words < 4:
        score -= 0.5
    if words > 60:
        score -= 1.0
    return round(min(10.0, max(0.0, score)), 2)


def _moment_type(sentence: str) -> str:
    text = sentence.lower()
    if sentence.rstrip().endswith("?") or any(word in text for word in _CURIOSITY_WORDS):
        return "curiosity"
    if re.search(r"\d[\d,]*\.?\d*%?|\$[\d,]+", sentence) or any(word in text for word in _SURPRISE_WORDS):
        return "surprise"
    if any(word in text for word in _EMOTION_WORDS):
        return "emotion"
    if any(word in text for word in _INSIGHT_WORDS):
        return "insight"
    if any(word in text for word in _CONCLUSION_WORDS):
        return "conclusion"
    if any(word in text for word in _CONFLICT_WORDS):
        return "conflict"
    return "hook"


def _estimate_seconds(sentence: str, words_per_minute: int = DEFAULT_WORDS_PER_MINUTE) -> float:
    """Pure duration estimate (no floor); the caller clamps to min/max."""
    words = len(re.findall(r"[A-Za-z0-9']+", sentence))
    return round(words / max(words_per_minute, 1) * 60.0, 1)


def _short_title(sentence: str, topic: str) -> str:
    """Deterministic caption/title from the moment's first clause."""
    text = (sentence or "").strip()
    if len(text) <= 64:
        return text
    return text[:61].rsplit(" ", 1)[0] + "..."


def plan_repurposing(
    script: str,
    topic: str = "",
    profile: Optional[ContentProfile] = None,
    max_shorts: int = 3,
    min_seconds: int = DEFAULT_SHORT_MIN_SECONDS,
    max_seconds: int = DEFAULT_SHORT_MAX_SECONDS,
    words_per_minute: int = DEFAULT_WORDS_PER_MINUTE,
) -> RepurposePlan:
    """Build a short-form plan from the long-form script.

    Deterministic: moments are ranked by narrative value, not by time
    slicing. Each short gets its own hook (the moment), context (the prior
    sentence), payoff (the moment itself or the next sentence), a duration
    estimate, and a title/caption.
    """
    sentences = _split_sentences(script)
    if not sentences:
        return RepurposePlan(shorts=[], rationale="no script content to repurpose")

    scored = [
        (index, sentence, _moment_score(sentence))
        for index, sentence in enumerate(sentences)
    ]
    scored.sort(key=lambda item: item[2], reverse=True)

    selected: List[ShortMoment] = []
    used_indices: set[int] = set()
    for index, sentence, score in scored:
        if len(selected) >= max_shorts:
            break
        if index in used_indices:
            continue
        used_indices.add(index)
        # A short needs a hook AND a payoff; a 1-sentence moment reuses the
        # sentence for both, longer ones pull the following sentence as payoff.
        context = sentences[index - 1] if index > 0 else ""
        payoff_sentence = ""
        for next_index in (index + 1, index + 2):
            if next_index < len(sentences) and next_index not in used_indices:
                payoff_sentence = sentences[next_index]
                used_indices.add(next_index)
                break
        combined = sentence + (" " + payoff_sentence if payoff_sentence else "")
        duration = _estimate_seconds(combined, words_per_minute)
        # A short needs a suitable duration: never below the floor, never
        # above the ceiling. A too-short moment absorbs following sentences
        # as context until it reaches the floor.
        extra_context = []
        while duration < min_seconds and len(extra_context) < 2:
            next_index = index + len(extra_context) + 1
            if next_index >= len(sentences) or next_index in used_indices:
                break
            extra_context.append(sentences[next_index])
            used_indices.add(next_index)
            combined = combined + " " + sentences[next_index]
            duration = _estimate_seconds(combined, words_per_minute)
        duration = min(max(duration, float(min_seconds)), float(max_seconds))

        selected.append(
            ShortMoment(
                index=index,
                source_text=sentence[:400],
                moment_type=_moment_type(sentence),
                hook=sentence[:160],
                context=context[:160],
                payoff=(payoff_sentence or sentence)[:160],
                estimated_seconds=round(duration, 1),
                title=_short_title(sentence, topic)[:80],
                caption=sentence[:160],
                score=score,
            )
        )

    selected.sort(key=lambda item: item.index)
    plan = RepurposePlan(
        shorts=selected,
        rationale=(
            f"{len(selected)} moments selected from {len(sentences)} sentences "
            "by narrative value (numbers, emotion, insight, conflict, curiosity)"
        ),
    )
    logger.debug(f"repurposing plan built: shorts={len(plan.shorts)}")
    return plan


def repurpose_summary(plan: Optional[RepurposePlan]) -> str:
    if plan is None or not plan.shorts:
        return "none"
    types = {}
    for short in plan.shorts:
        types[short.moment_type] = types.get(short.moment_type, 0) + 1
    return (
        f"shorts={len(plan.shorts)}, moments={dict(sorted(types.items()))}, "
        f"durations={[short.estimated_seconds for short in plan.shorts]}"
    )
