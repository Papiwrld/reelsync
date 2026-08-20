"""Structured Quality Assurance (Phase 2D.4).

Evaluates the whole plan — research, script, visuals, audio, metadata —
against deterministic checks and reports issues with severity levels:

    INFO < WARNING < ERROR < CRITICAL

Critical failures are capable of blocking publication (``publication_blocked``).
Everything is deterministic and explainable: each issue carries a category,
a message and the evidence that triggered it.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from app.services.content_profile import ContentProfile
from app.services.intelligence import ContentIntelligence


class QaSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


_SEVERITY_ORDER = {
    QaSeverity.INFO.value: 0,
    QaSeverity.WARNING.value: 1,
    QaSeverity.ERROR.value: 2,
    QaSeverity.CRITICAL.value: 3,
}


class QaIssue(BaseModel):
    model_config = ConfigDict(extra="ignore")

    severity: str = QaSeverity.INFO.value
    category: str = ""  # research | script | visuals | audio | metadata
    message: str = ""
    evidence: str = ""


class QaReport(BaseModel):
    model_config = ConfigDict(extra="ignore")

    issues: List[QaIssue] = Field(default_factory=list)
    publication_blocked: bool = False
    summary: str = ""


def _add(report: QaReport, severity: str, category: str, message: str, evidence: str = "") -> None:
    report.issues.append(
        QaIssue(severity=severity, category=category, message=message, evidence=evidence)
    )


def _sentence_count(script: str) -> int:
    return len([s for s in re.split(r"(?<=[.!?。！？])\s+", (script or "").strip()) if s.strip()])


def _word_count(script: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", script or ""))


# ---------------------------------------------------------------------------
# QA checks
# ---------------------------------------------------------------------------


def _check_research(
    report: QaReport,
    research_claims: Optional[List[Any]],
    research_contradictions: Optional[List[str]],
    research_provenance: Optional[Dict[str, Any]],
) -> None:
    claims = research_claims or []
    # ``status`` may be a str-Enum member (e.g. ClaimStatus.UNSUPPORTED);
    # str() of an Enum yields the member repr, so unwrap .value first.
    def _status(claim: Any) -> str:
        status = getattr(claim, "status", "") or ""
        return str(getattr(status, "value", status)).lower()

    unsupported = [c for c in claims if _status(c) in ("unsupported", "uncertain")]
    if len(unsupported) > max(1, len(claims) // 2):
        _add(
            report,
            QaSeverity.ERROR.value,
            "research",
            "most claims are unverified; the script must not state them as fact",
            evidence=f"{len(unsupported)}/{len(claims)} claims unsupported or uncertain",
        )
    elif unsupported:
        _add(
            report,
            QaSeverity.WARNING.value,
            "research",
            "some claims are not verified; qualify them or remove them",
            evidence="; ".join(str(getattr(c, "statement", ""))[:80] for c in unsupported[:3]),
        )

    contradictions = research_contradictions or []
    if contradictions:
        _add(
            report,
            QaSeverity.WARNING.value,
            "research",
            "sources contradict each other; the script must acknowledge the conflict",
            evidence="; ".join(str(c)[:80] for c in contradictions[:3]),
        )

    provenance = research_provenance or {}
    if provenance.get("cached"):
        _add(
            report,
            QaSeverity.INFO.value,
            "research",
            "research served from cache; confirm freshness for time-sensitive claims",
            evidence=f"cached_at={provenance.get('cached_at', 'unknown')}",
        )
    if provenance.get("model_knowledge") and not provenance.get("provider"):
        _add(
            report,
            QaSeverity.INFO.value,
            "research",
            "research is model knowledge, not live web data; avoid current-event claims",
        )


def _check_script(
    report: QaReport,
    script: str,
    profile: ContentProfile,
    intelligence: Optional[ContentIntelligence],
) -> None:
    words = _word_count(script)
    sentences = _sentence_count(script)
    if sentences == 0:
        _add(report, QaSeverity.CRITICAL.value, "script", "script is empty", evidence="no sentences")
        return

    # Duration: estimate from words at ~150 wpm.
    estimated_seconds = words / 2.5
    length_text = (profile.preferred_video_length or "").lower()
    numbers = [int(m) for m in re.findall(r"\d+", length_text)]
    if numbers:
        lower, upper = (numbers + numbers)[:2]
        if estimated_seconds < lower * 0.6:
            _add(
                report,
                QaSeverity.WARNING.value,
                "script",
                "script is much shorter than the niche target duration",
                evidence=f"~{round(estimated_seconds)}s vs {lower}-{upper}s target",
            )
        elif estimated_seconds > upper * 1.5:
            _add(
                report,
                QaSeverity.WARNING.value,
                "script",
                "script is much longer than the niche target duration",
                evidence=f"~{round(estimated_seconds)}s vs {lower}-{upper}s target",
            )

    # Repetition: repeated significant words across sentences.
    sentences_list = [s for s in re.split(r"(?<=[.!?。！？])\s+", (script or "").strip()) if s.strip()]
    if len(sentences_list) >= 3:
        first_words = [s.split()[0].lower().strip(",. ") for s in sentences_list if s.split()]
        dupes = {w for w in first_words if first_words.count(w) >= 3}
        if dupes:
            _add(
                report,
                QaSeverity.WARNING.value,
                "script",
                "several sentences open with the same word; vary the rhythm",
                evidence=", ".join(sorted(dupes)[:4]),
            )

    # Unsupported superlatives in the script without research backing.
    if intelligence and intelligence.fact_check_level in ("strong", "very_strong"):
        superlatives = re.findall(
            r"\b(?:the most|the biggest|the largest|the first|the only|never|always)\b",
            (script or "").lower(),
        )
        if superlatives:
            _add(
                report,
                QaSeverity.INFO.value,
                "script",
                "superlative claims present; ensure they are source-backed at this fact-check level",
                evidence=", ".join(sorted(set(superlatives))[:5]),
            )


_NUM_PERCENT_RE = re.compile(r"(?<![\w$])(\d{1,3}(?:[.,]\d+)?)\s*%")
_NUM_PERCENT_WORD_RE = re.compile(r"(?<![\w$])(\d{1,3}(?:[.,]\d+)?)\s*percent", re.IGNORECASE)
_NUM_CURRENCY_RE = re.compile(r"\$(\d[\d.,]*)\s*(million|billion|trillion|thousand)?", re.IGNORECASE)
_NUM_WORD_UNIT_RE = re.compile(r"(?<![\w$])(\d[\d.,]*)\s*(million|billion|trillion|thousand)", re.IGNORECASE)
_NUM_BARE_RE = re.compile(r"(?<![\w$%.])\b(\d{1,3}(?:,\d{3})+|\d{4,})\b")
_YEAR_RE = re.compile(r"\b(?:1[0-9]{3}|2[0-9]{3})\b")
_UNIT_SCALE = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12}


def _fmt_num(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(round(value))
    return f"{value:g}"


def _claim_mentions(claim: Any) -> List[str]:
    """Collect all free-text fragments of a claim that may contain numbers."""
    chunks: List[str] = []
    for attr in ("statement", "content", "verdict"):
        value = getattr(claim, attr, None)
        if isinstance(value, str) and value:
            chunks.append(value)
    for attr in ("facts", "statistics"):
        values = getattr(claim, attr, None) or []
        for item in values:
            if isinstance(item, str) and item:
                chunks.append(item)
    return chunks


def _extract_numbers(text: str) -> List[Dict[str, Any]]:
    """Pull numeric claims out of text as (display, value, kind, unit) dicts.

    Kinds: ``percent`` | ``currency`` | ``amount`` (bare or with a magnitude
    word). Values are scaled to the real magnitude so "$5 billion" and
    "5,000,000,000" compare equal. 4-digit years are excluded (dates, not
    claims).
    """
    found: List[Dict[str, Any]] = []
    seen = set()

    def _add(display: str, raw_value: float, kind: str, unit: str) -> None:
        value = raw_value * _UNIT_SCALE.get(unit, 1.0)
        key = (kind, round(value, 6))
        if key in seen:
            return
        seen.add(key)
        found.append({"display": display, "value": value, "kind": kind, "unit": unit})

    for m in _NUM_PERCENT_RE.finditer(text):
        _add(f"{_fmt_num(float(m.group(1)))}%", float(m.group(1)), "percent", "")
    for m in _NUM_PERCENT_WORD_RE.finditer(text):
        _add(f"{_fmt_num(float(m.group(1)))}%", float(m.group(1)), "percent", "")
    for m in _NUM_CURRENCY_RE.finditer(text):
        raw = m.group(1)
        unit = (m.group(2) or "").lower()
        display = f"${raw}" if not unit else f"${raw} {unit}"
        _add(display, float(raw.replace(",", "")), "currency", unit)
    for m in _NUM_WORD_UNIT_RE.finditer(text):
        raw = m.group(1)
        unit = m.group(2).lower()
        _add(f"{raw} {unit}", float(raw.replace(",", "")), "amount", unit)
    for m in _NUM_BARE_RE.finditer(text):
        raw = m.group(1)
        if "," not in raw and len(raw) == 4 and _YEAR_RE.match(raw):
            continue
        _add(raw, float(raw.replace(",", "")), "bare", "")
    return found


def _significant(claim: Dict[str, Any]) -> bool:
    """Filter out trivial numbers (2, 3, small percentages) to avoid over-flagging."""
    if claim["kind"] == "percent":
        return claim["value"] >= 5
    return claim["value"] >= 1000


def _match_variants(claim: Dict[str, Any]) -> set:
    """Normalized spellings that count as backing for a script number."""
    variants = {claim["display"].lower()}
    value = claim["value"]
    unit = claim["unit"]
    iv = round(value)
    variants.update([f"{iv:,}", f"{iv}", f"${iv}", f"${iv:,}"])
    if claim["kind"] == "percent":
        s = _fmt_num(value)
        variants.update([f"{s}%", f"{s} %", f"{s} percent", f"{iv}%", f"{iv:,}%", f"{iv} percent"])
    if unit:
        u = unit
        s = _fmt_num(value)
        variants.update([f"{s} {u}", f"{s}{u}", f"{iv} {u}", f"{iv}{u}", f"{iv:,} {u}", f"${s} {u}", f"${s}{u}"])
        if u == "billion":
            variants.update([f"{s}bn", f"{iv}bn", f"{s}b", f"{iv}b", f"${s}bn"])
        elif u == "million":
            variants.update([f"{s}m", f"{iv}m", f"${s}m"])
        elif u == "thousand":
            variants.update([f"{s}k", f"{iv}k", f"${s}k"])
    return variants


def _contradicting_research(
    claim: Dict[str, Any], research_numbers: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Research figures near the script's number but not equal to it."""
    if claim["kind"] == "percent":
        band = 25.0
        candidates = [r for r in research_numbers if r["kind"] == "percent"]
    else:
        band = 0.25 * max(claim["value"], 1.0)
        candidates = [r for r in research_numbers if r["kind"] != "percent"]
    return [
        r
        for r in candidates
        if r["value"] != claim["value"] and abs(r["value"] - claim["value"]) <= band
    ]


def _check_script_claims_against_research(
    report: QaReport,
    script: str,
    research_claims: Optional[List[Any]],
) -> None:
    """Factuality gate: every significant numeric claim in the script must be
    backed (or at least not contradicted) by a research claim.

    No research -> INFO. Unbacked number -> WARNING. Contradicting number ->
    ERROR. Fully backed -> silent. Years and small numbers are ignored.
    """
    if not script or not re.search(r"\d", script):
        return
    script_numbers = [c for c in _extract_numbers(script) if _significant(c)]
    if not script_numbers:
        return

    claims = research_claims or []
    if not claims:
        _add(
            report,
            QaSeverity.INFO.value,
            "script",
            "cannot verify numeric claims (no research claims provided)",
            evidence="; ".join(c["display"] for c in script_numbers[:5]),
        )
        return

    blob = " ".join(chunk for c in claims for chunk in _claim_mentions(c)).lower()
    research_numbers = _extract_numbers(blob)

    for claim in script_numbers:
        if any(v in blob for v in _match_variants(claim)):
            continue
        conflicts = _contradicting_research(claim, research_numbers)
        if conflicts:
            _add(
                report,
                QaSeverity.ERROR.value,
                "script",
                f"script claims {claim['display']} but research reports a conflicting figure",
                evidence=f"research mentions {conflicts[0]['display']}",
            )
        else:
            _add(
                report,
                QaSeverity.WARNING.value,
                "script",
                f"script claims {claim['display']} but research sources do not mention it",
                evidence="not found in any research claim",
            )


def _check_visuals(
    report: QaReport,
    scene_plan,
    ai_image_budget_ratio: float = 0.2,
) -> None:
    if scene_plan is None or not scene_plan.scenes:
        _add(report, QaSeverity.WARNING.value, "visuals", "no scene plan produced", evidence="scene plan missing")
        return

    total = len(scene_plan.scenes)
    ai_images = scene_plan.ai_image_count
    # The plan enforces its own budget; flag any scenario that still slipped
    # past it (e.g. manually constructed plans).
    if ai_images > max(1, round(total * ai_image_budget_ratio)) or ai_images == total:
        _add(
            report,
            QaSeverity.ERROR.value,
            "visuals",
            "excessive AI-image usage: the video risks looking visually disconnected",
            evidence=f"{ai_images}/{total} scenes use AI imagery",
        )
    if scene_plan.continuity_notes:
        _add(
            report,
            QaSeverity.INFO.value,
            "visuals",
            "visual continuity notes exist; honor them during material selection",
            evidence="; ".join(scene_plan.continuity_notes[:3]),
        )


def _check_audio(report: QaReport, script: str) -> None:
    sentences = [s for s in re.split(r"(?<=[.!?。！？])\s+", (script or "").strip()) if s.strip()]
    long_sentences = [s for s in sentences if len(re.findall(r"[A-Za-z0-9']+", s)) > 32]
    if len(long_sentences) > max(1, len(sentences) // 3):
        _add(
            report,
            QaSeverity.WARNING.value,
            "audio",
            "several sentences are too long for comfortable narration; break them up",
            evidence=f"{len(long_sentences)}/{len(sentences)} sentences over 32 words",
        )


def _check_metadata(
    report: QaReport,
    selected_title: Optional[Any],
    script: str,
    platform: str,
) -> None:
    if selected_title is None or not getattr(selected_title, "text", ""):
        _add(report, QaSeverity.ERROR.value, "metadata", "no title selected", evidence="title missing")
        return

    title = selected_title.text
    if getattr(selected_title, "scores", {}).get("accuracy", 10.0) < 6.0:
        _add(
            report,
            QaSeverity.ERROR.value,
            "metadata",
            "selected title fails the accuracy check; it may overclaim what the video shows",
            evidence=title[:80],
        )

    words = len(re.findall(r"[A-Za-z0-9']+", title))
    if platform in ("tiktok", "youtube_shorts", "instagram_reels", "x") and words > 9:
        _add(
            report,
            QaSeverity.WARNING.value,
            "metadata",
            "title is long for the target short-form platform",
            evidence=f"{words} words",
        )

    # Thumbnail compatibility: a title with a clear number/subject maps well.
    if not re.search(r"\d|\b[A-Z][a-z]+", title):
        _add(
            report,
            QaSeverity.INFO.value,
            "metadata",
            "title has no concrete number or proper noun; thumbnail may lack a focal hook",
        )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_quality_assurance(
    script: str,
    profile: ContentProfile,
    intelligence: Optional[ContentIntelligence] = None,
    research_claims: Optional[List[Any]] = None,
    research_contradictions: Optional[List[str]] = None,
    research_provenance: Optional[Dict[str, Any]] = None,
    scene_plan=None,
    selected_title: Optional[Any] = None,
    platform: str = "",
    ai_image_budget_ratio: float = 0.2,
) -> QaReport:
    """Run all deterministic QA checks and produce a report.

    Never raises: missing inputs degrade to warnings, and the report always
    states exactly what could not be checked.
    """
    report = QaReport()
    _check_research(report, research_claims, research_contradictions, research_provenance)
    _check_script(report, script, profile, intelligence)
    _check_script_claims_against_research(report, script, research_claims)
    _check_visuals(report, scene_plan, ai_image_budget_ratio)
    _check_audio(report, script)
    _check_metadata(report, selected_title, script, platform)

    report.publication_blocked = any(
        issue.severity == QaSeverity.CRITICAL.value for issue in report.issues
    )
    counts: Dict[str, int] = {}
    for issue in report.issues:
        counts[issue.severity] = counts.get(issue.severity, 0) + 1
    blocked_text = "PUBLICATION BLOCKED" if report.publication_blocked else "ready to publish"
    report.summary = (
        f"QA: {counts.get('info', 0)} info, {counts.get('warning', 0)} warnings, "
        f"{counts.get('error', 0)} errors, {counts.get('critical', 0)} critical - {blocked_text}"
    )
    logger.debug(report.summary)
    return report


def qa_summary(report: Optional[QaReport]) -> str:
    if report is None:
        return "none"
    return report.summary
