"""Research quality scoring.

Scores combine relevance (query overlap), authority (provider reputation),
freshness (age relative to TTL), corroboration (same content from independent
providers) and completeness (content depth). The score drives result ordering
and confidence of extracted claims.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

from app.services.research_layer.dedupe import token_set
from app.services.research_layer.schema import ResearchResult

RELEVANCE_WEIGHT = 0.2
AUTHORITY_WEIGHT = 0.35
FRESHNESS_WEIGHT = 0.15
CORROBORATION_WEIGHT = 0.2
COMPLETENESS_WEIGHT = 0.1


def _normalize_text(value: str) -> str:
    return " ".join((value or "").lower().split())


def relevance_score(query: str, result: ResearchResult) -> float:
    query_tokens = token_set(query)
    if not query_tokens:
        return 0.5
    title_tokens = set(_normalize_text(result.title).split())
    content_tokens = set(_normalize_text(result.content[:1000]).split())
    match = (title_tokens | content_tokens) & query_tokens
    if not match:
        return 0.0
    return min(1.0, len(match) / len(query_tokens))


def freshness_score(result: ResearchResult, ttl_seconds: Optional[int] = None) -> float:
    try:
        retrieved = float(result.raw_metadata.get("retrieved_ts", time.time()))
    except (TypeError, ValueError):
        return 0.5
    age = max(0.0, time.time() - retrieved)
    if ttl_seconds is None or ttl_seconds <= 0:
        return max(0.0, 1.0 - age / (30 * 24 * 3600))
    return max(0.0, min(1.0, 1.0 - age / ttl_seconds))


def completeness_score(result: ResearchResult) -> float:
    content_len = len(result.content or "")
    depth = 0.5 + 0.5 * min(1.0, content_len / 500)
    fields = sum(
        1
        for value in (
            result.entities,
            result.facts,
            result.dates,
            result.locations,
            result.statistics,
            result.citations,
        )
        if value
    )
    field_score = 0.3 + 0.7 * min(1.0, fields / 4)
    return min(1.0, 0.6 * depth + 0.4 * field_score)


def corroboration_count(result: ResearchResult, others: List[ResearchResult]) -> int:
    """Count independent results sharing meaningful content with this one."""
    if not result.content:
        return 0
    tokens = set(_normalize_text(result.content[:1500]).split())
    if len(tokens) < 8:
        return 0
    count = 0
    for other in others:
        if other is result or other.source == result.source:
            continue
        other_tokens = set(_normalize_text(other.content[:1500]).split())
        union = tokens | other_tokens
        if not union:
            continue
        if len(tokens & other_tokens) / len(union) >= 0.3:
            count += 1
    return count


def score_result(
    query: str,
    result: ResearchResult,
    authority: float,
    ttl_seconds: Optional[int] = None,
    corroborations: int = 0,
) -> float:
    relevance = relevance_score(query, result)
    freshness = freshness_score(result, ttl_seconds)
    completeness = completeness_score(result)
    corroboration = min(1.0, 0.5 + 0.25 * min(corroborations, 2))
    raw = (
        RELEVANCE_WEIGHT * relevance
        + AUTHORITY_WEIGHT * authority
        + FRESHNESS_WEIGHT * freshness
        + CORROBORATION_WEIGHT * corroboration
        + COMPLETENESS_WEIGHT * completeness
    )
    return round(max(0.0, min(1.0, raw)), 3)


def corroborate_all(
    query: str, results: List[ResearchResult], authorities: Dict[str, float], ttl_by_source: Dict[str, int]
) -> Dict[str, float]:
    """Score every result with per-source authority/TTL and corroboration."""
    scores: Dict[str, float] = {}
    for result in results:
        corroborations = corroboration_count(result, results)
        scores[result.source_url] = score_result(
            query,
            result,
            authorities.get(result.source, 0.5),
            ttl_by_source.get(result.source),
            corroborations,
        )
    return scores