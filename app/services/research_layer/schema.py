"""Normalized research schema for the zero-key research layer.

Every provider returns ``ResearchResult`` objects. Downstream code never sees
provider-specific payloads; provenance (``SourceReference``) is attached to
every fact and preserved end-to-end.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class SourceReference:
    """Provenance for a single factual claim."""

    source: str
    source_url: str
    title: str
    retrieved_at: str
    raw_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ResearchClaim:
    """A factual statement traced back to one or more sources."""

    statement: str
    sources: List[SourceReference] = field(default_factory=list)
    confidence: float = 0.5
    corroborated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ResearchResult:
    """A normalized, provider-agnostic research item."""

    source: str
    source_url: str
    title: str
    retrieved_at: str
    query: str
    content: str
    entities: List[str] = field(default_factory=list)
    facts: List[str] = field(default_factory=list)
    dates: List[str] = field(default_factory=list)
    locations: List[str] = field(default_factory=list)
    statistics: List[str] = field(default_factory=list)
    citations: List[Dict[str, str]] = field(default_factory=list)
    confidence: float = 0.5
    raw_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ResearchPackage:
    """Aggregated output of one research query across selected providers."""

    query: str
    normalized_query: str
    category: str
    providers: List[str]
    results: List[ResearchResult] = field(default_factory=list)
    claims: List[ResearchClaim] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    dates: List[str] = field(default_factory=list)
    locations: List[str] = field(default_factory=list)
    statistics: List[str] = field(default_factory=list)
    citations: List[Dict[str, str]] = field(default_factory=list)
    quality_scores: Dict[str, float] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    requests_avoided: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "normalized_query": self.normalized_query,
            "category": self.category,
            "providers": list(self.providers),
            "results": [result.to_dict() for result in self.results],
            "claims": [claim.to_dict() for claim in self.claims],
            "entities": list(self.entities),
            "dates": list(self.dates),
            "locations": list(self.locations),
            "statistics": list(self.statistics),
            "citations": list(self.citations),
            "quality_scores": dict(self.quality_scores),
            "metrics": dict(self.metrics),
            "requests_avoided": self.requests_avoided,
        }