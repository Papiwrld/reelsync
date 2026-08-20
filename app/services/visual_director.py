"""Visual Director (Phase 2C.5–2C.6).

A semantic planning system that decides the BEST visual representation for
each piece of narration — instead of blindly generating an AI image per scene.

Design rules:
- Deterministic material-type classification from narration meaning: numbers
  → chart/graph, places → map/location, quotes → document/screenshot,
  product/company names → product/logo, abstract concepts → metaphor/B-roll,
  historical events → archival, default → stock video (never AI image).
- An AI-image budget caps generative imagery per video, so the output can
  never degenerate into "every scene is an AI image".
- Visual continuity is enforced: repeated subjects reuse the same material
  family, consecutive scenes avoid duplicate search terms, and the overall
  style follows the profile's visual language.
- ``extra="ignore"`` models keep the scene plan serializable and forward
  compatible; every scene records a concise rationale (no chain-of-thought).
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Dict, List, Optional

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from app.services.content_profile import ContentProfile
from app.services.intelligence import ContentIntelligence

# Default share of scenes allowed to use AI-generated imagery (never 100%).
DEFAULT_AI_IMAGE_BUDGET_RATIO = 0.2
MIN_AI_IMAGE_BUDGET = 1
MAX_AI_IMAGE_BUDGET = 8


class MaterialType(str, Enum):
    ARCHIVAL = "archival"
    STOCK_VIDEO = "stock_video"
    STOCK_IMAGE = "stock_image"
    AI_IMAGE = "ai_image"
    AI_VIDEO = "ai_video"
    SCREENSHOT = "screenshot"
    DOCUMENT = "document"
    CHART = "chart"
    GRAPH = "graph"
    MAP = "map"
    DIAGRAM = "diagram"
    PRODUCT_IMAGE = "product_image"
    LOGO = "logo"
    TEXT = "text"
    B_ROLL = "b_roll"
    ABSTRACT_METAPHOR = "abstract_metaphor"


ALL_MATERIAL_TYPES = tuple(member.value for member in MaterialType)


class SceneVisual(BaseModel):
    """One scene's visual plan."""

    model_config = ConfigDict(extra="ignore")

    index: int = 0
    narration: str = ""
    purpose: str = ""  # narrative job of this scene (hook/context/evidence/...)
    material_type: str = MaterialType.STOCK_VIDEO.value
    visual_intent: str = ""  # what the image must communicate
    search_terms: List[str] = Field(default_factory=list)
    rationale: str = ""
    # Continuity keys: repeated keys across scenes keep a consistent look.
    subject_key: str = ""
    location_key: str = ""
    style_key: str = ""
    source_requirement: str = ""  # e.g. "historical", "map", "document"


class ScenePlan(BaseModel):
    """Ordered visual plan for the whole video."""

    model_config = ConfigDict(extra="ignore")

    scenes: List[SceneVisual] = Field(default_factory=list)
    ai_image_count: int = 0
    ai_image_budget: int = DEFAULT_AI_IMAGE_BUDGET_RATIO
    style_language: str = ""
    continuity_notes: List[str] = Field(default_factory=list)
    platform: str = ""


# ---------------------------------------------------------------------------
# Deterministic narration classifiers
# ---------------------------------------------------------------------------

_NUMBER_PATTERN = re.compile(r"\d[\d,]*\.?\d*%?|\$[\d,]+|percent|percentage|billion|million|thousand")
_CURRENCY_PATTERN = re.compile(r"\$|€|£|¥|usd|eur")
_LOCATION_WORDS = (
    "city", "town", "capital", "coast", "island", "mountain", "river",
    "country", "region", "province", "valley", "border", "state", "district",
    "street", "avenue", "seattle", "london", "paris", "tokyo", "new york",
    "san francisco", "china", "america", "europe", "africa", "asia", "india",
    "russia", "germany", "france", "japan", "brazil", "canada",
)
_QUOTE_PATTERN = re.compile(r"[\"“”]|said|wrote|told|stated|announced|declared|quoted")
# Only SPECIFIC brand names map to product imagery. Generic words like
# "company" or "brand" do not: "the company grew" is stock footage, not a
# product shot, and "stores" is too vague to pin a product image on.
_COMPANY_HINTS = (
    "amazon", "apple", "google", "microsoft", "tesla",
    "meta", "netflix", "nike", "coca-cola", "mcdonald", "sony", "samsung",
    "intel", "nvidia", "spacex", "uber", "airbnb", "spotify", "disney",
)
_HISTORY_WORDS = (
    "history", "historical", "century", "decade", "era", "empire", "kingdom",
    "ancient", "medieval", "war", "battle", "revolution", "archaeolog", "ruins",
    "founder", "founded in", "in 19", "in 18", "in 17", "in 20",
)
_ABSTRACT_WORDS = (
    "confidence", "trust", "fear", "hope", "collapse", "growth", "decline",
    "momentum", "culture", "identity", "power", "influence", "vision",
    "strategy", "mindset", "psycholog", "emotion", "belief", "idea", "concept",
)
_INSTRUCTION_WORDS = (
    "step", "how to", "method", "process", "pipeline", "workflow", "recipe",
    "algorithm", "protocol", "install", "configure", "setup",
)
_MAP_HINTS = ("map", "where is", "located", "geography", "territory", "route")
# Narration that explicitly asks for a reconstruction of something with no
# surviving imagery — the ONLY case where an AI image is the right visual.
_RECONSTRUCTION_HINTS = (
    "no footage", "no photographs", "reconstruct", "recreation", "recreated",
    "imagine", "what it must have looked like", "hypothetical", "artist's impression",
    "no surviving", "never photographed",
)


def _classify_material_type(narration: str) -> str:
    """Deterministic first-pass material classification from narration meaning."""
    text = (narration or "").lower()

    # AI reconstruction is reserved for events with no real imagery; the
    # scene-plan budget still caps how many of these can be used per video.
    if any(hint in text for hint in _RECONSTRUCTION_HINTS):
        return MaterialType.AI_IMAGE.value
    if _CURRENCY_PATTERN.search(text) or re.search(r"\b\d+\s*(%|percent)", text):
        return MaterialType.CHART.value
    # History before generic numbers: "the empire fell in the 5th century" is
    # archival, not a graph; "the 1990s" (no history cue) stays a graph.
    if any(word in text for word in _HISTORY_WORDS):
        return MaterialType.ARCHIVAL.value
    if _NUMBER_PATTERN.search(text):
        return MaterialType.GRAPH.value
    # Named brands before quotes: "Tesla announced..." shows the product, not
    # a headline; a generic statement with no brand still maps to a document.
    if any(word in text for word in _COMPANY_HINTS):
        return MaterialType.PRODUCT_IMAGE.value
    if _QUOTE_PATTERN.search(text):
        return MaterialType.DOCUMENT.value
    if any(word in text for word in _MAP_HINTS):
        return MaterialType.MAP.value
    if any(word in text for word in _LOCATION_WORDS):
        return MaterialType.MAP.value
    if any(word in text for word in _INSTRUCTION_WORDS):
        return MaterialType.DIAGRAM.value
    if any(word in text for word in _ABSTRACT_WORDS):
        return MaterialType.ABSTRACT_METAPHOR.value
    return MaterialType.STOCK_VIDEO.value


def _significant_search_terms(narration: str, limit: int = 3) -> List[str]:
    """Extract meaningful search terms from narration (deterministic)."""
    words = [
        word
        for word in re.findall(r"[A-Za-z][A-Za-z0-9']{2,}", (narration or "").lower())
        if word not in _FUNCTION_WORDS and not word.isdigit()
    ]
    seen: List[str] = []
    for word in words:
        if word not in seen:
            seen.append(word)
        if len(seen) >= limit:
            break
    return seen


_FUNCTION_WORDS = frozenset(
    (
        "the", "a", "an", "of", "to", "in", "on", "at", "and", "or", "but",
        "for", "with", "about", "this", "that", "these", "those", "it", "its",
        "is", "are", "was", "were", "be", "been", "being", "will", "would",
        "can", "could", "should", "do", "does", "did", "not", "no", "then",
        "than", "so", "just", "only", "every", "one", "two", "you", "your",
        "my", "me", "we", "our", "they", "them", "their", "he", "she", "his",
        "her", "there", "here", "from", "into", "out", "up", "down", "as",
        "by", "if", "how", "why", "what", "who", "when", "where", "but",
        "been", "being", "more", "most", "some", "any", "all", "also", "very",
    )
)


def _visual_intent_for(material_type: str, narration: str) -> str:
    """Short statement of what the visual must communicate."""
    text = narration.strip()
    truncated = text[:120] + ("…" if len(text) > 120 else "")
    intent_map = {
        MaterialType.CHART.value: f"data visualization showing the figures in: {truncated}",
        MaterialType.GRAPH.value: f"graphical trend implied by: {truncated}",
        MaterialType.MAP.value: f"location/geography matching: {truncated}",
        MaterialType.ARCHIVAL.value: f"period-appropriate imagery for: {truncated}",
        MaterialType.DOCUMENT.value: f"document or headline supporting: {truncated}",
        MaterialType.DIAGRAM.value: f"diagram of the process in: {truncated}",
        MaterialType.PRODUCT_IMAGE.value: f"the product/company in: {truncated}",
        MaterialType.LOGO.value: f"brand identity for: {truncated}",
        MaterialType.ABSTRACT_METAPHOR.value: f"visual metaphor for the abstract idea in: {truncated}",
        MaterialType.B_ROLL.value: f"contextual b-roll for: {truncated}",
        MaterialType.SCREENSHOT.value: f"screenshot/interface shown in: {truncated}",
        MaterialType.STOCK_IMAGE.value: f"photograph representing: {truncated}",
        MaterialType.AI_IMAGE.value: f"AI reconstruction of the unrecoverable scene: {truncated}",
        MaterialType.AI_VIDEO.value: f"AI-generated motion for: {truncated}",
        MaterialType.TEXT.value: f"on-screen text emphasizing: {truncated}",
    }
    return intent_map.get(material_type, f"visual for: {truncated}")


def _scene_purpose(index: int, total: int) -> str:
    """Assign a narrative job per scene position (hook/context/evidence/payoff)."""
    if index == 0:
        return "hook"
    if index >= total - 1:
        return "payoff"
    if index == total // 2:
        return "turning point"
    return "context/evidence"


# ---------------------------------------------------------------------------
# Scene planning
# ---------------------------------------------------------------------------


def _split_scenes(script: str, target_scenes: int) -> List[str]:
    """Split narration into scene-sized chunks (sentences, grouped)."""
    text = (script or "").strip()
    if not text:
        return []
    sentences = [
        part.strip()
        for part in re.split(r"(?<=[.!?。！？])\s+", text)
        if part.strip()
    ]
    if not sentences:
        return [text]
    if len(sentences) <= target_scenes:
        return sentences
    # Group sentences so we land close to the target scene count.
    per_group = max(1, round(len(sentences) / target_scenes))
    groups: List[str] = []
    for index in range(0, len(sentences), per_group):
        groups.append(" ".join(sentences[index : index + per_group]))
    return groups


def plan_scenes(
    script: str,
    profile: ContentProfile,
    intelligence: Optional[ContentIntelligence] = None,
    ai_image_budget_ratio: float = DEFAULT_AI_IMAGE_BUDGET_RATIO,
    platform: str = "",
    desired_scene_count: Optional[int] = None,
) -> ScenePlan:
    """Build the scene-by-scene visual plan for a script.

    Deterministic. AI-generated imagery is strictly budgeted: the plan can
    never produce an AI image for every scene.
    """
    chunks = _split_scenes(script, desired_scene_count or 8)
    budget = min(
        MAX_AI_IMAGE_BUDGET,
        max(MIN_AI_IMAGE_BUDGET, round(len(chunks) * ai_image_budget_ratio)),
    )
    style_language = (intelligence.visual_language if intelligence else "") or profile.visual_style or "clear, relevant visuals"

    scenes: List[SceneVisual] = []
    ai_used = 0
    subject_keys: Dict[str, int] = {}
    location_keys: Dict[str, int] = {}
    used_search_terms: set[str] = set()
    continuity_notes: List[str] = []

    for index, chunk in enumerate(chunks):
        material_type = _classify_material_type(chunk)
        # Budget enforcement: excess AI images downgrade to B-roll.
        if material_type in (MaterialType.AI_IMAGE.value, MaterialType.AI_VIDEO.value):
            if ai_used < budget:
                ai_used += 1
            else:
                material_type = MaterialType.B_ROLL.value

        terms = _significant_search_terms(chunk)
        # Avoid repeating the exact same search terms across consecutive scenes.
        unique_terms: List[str] = []
        for term in terms:
            if term in used_search_terms:
                continue
            used_search_terms.add(term)
            unique_terms.append(term)
        if not unique_terms:
            unique_terms = terms or ["scene"]

        subject_key = _derive_subject_key(chunk)
        location_key = _derive_location_key(chunk)
        subject_keys[subject_key] = subject_keys.get(subject_key, 0) + 1
        location_keys[location_key] = location_keys.get(location_key, 0) + 1

        scene = SceneVisual(
            index=index,
            narration=chunk[:400],
            purpose=_scene_purpose(index, len(chunks)),
            material_type=material_type,
            visual_intent=_visual_intent_for(material_type, chunk),
            search_terms=unique_terms,
            rationale=_material_rationale(chunk, material_type),
            subject_key=subject_key,
            location_key=location_key,
            style_key=style_language[:40],
            source_requirement=_source_requirement_for(material_type),
        )
        scenes.append(scene)

    # Continuity checks.
    for key, count in subject_keys.items():
        if key and count > 1:
            continuity_notes.append(f"subject '{key}' appears in {count} scenes; keep its look consistent")
    repeated_types: Dict[str, int] = {}
    for scene in scenes:
        repeated_types[scene.material_type] = repeated_types.get(scene.material_type, 0) + 1
    if repeated_types.get(MaterialType.AI_IMAGE.value, 0) == len(scenes):
        continuity_notes.append("plan degenerated into all-AI-image scenes (must not happen)")

    plan = ScenePlan(
        scenes=scenes,
        ai_image_count=ai_used,
        ai_image_budget=budget,
        style_language=style_language,
        continuity_notes=continuity_notes,
        platform=(platform or (intelligence.platform if intelligence else "")) or "",
    )
    logger.debug(f"scene plan built: scenes={len(scenes)}, ai_images={ai_used}/{budget}")
    return plan


def _derive_subject_key(chunk: str) -> str:
    """Deterministic subject key from the narration's most prominent noun."""
    terms = _significant_search_terms(chunk, limit=2)
    return terms[0] if terms else ""


def _derive_location_key(chunk: str) -> str:
    text = (chunk or "").lower()
    for word in _LOCATION_WORDS:
        if re.search(r"\b" + re.escape(word) + r"\b", text):
            return word
    return ""


def _material_rationale(chunk: str, material_type: str) -> str:
    reason_map = {
        MaterialType.CHART.value: "narration contains figures/percentages; a chart communicates data better than an arbitrary image",
        MaterialType.GRAPH.value: "narration implies quantitative change; a graph matches the meaning",
        MaterialType.MAP.value: "narration references a location; a map gives concrete geographic grounding",
        MaterialType.ARCHIVAL.value: "narration concerns history; archival/period imagery is the honest visual",
        MaterialType.DOCUMENT.value: "narration contains a quote or statement; show the document/headline",
        MaterialType.DIAGRAM.value: "narration describes a process; a diagram clarifies structure",
        MaterialType.PRODUCT_IMAGE.value: "narration names a company/product; product imagery is the direct visual",
        MaterialType.LOGO.value: "brand identity is the subject; a logo is the precise visual",
        MaterialType.ABSTRACT_METAPHOR.value: "narration is abstract; a deliberate metaphor avoids a literal mismatch",
        MaterialType.B_ROLL.value: "contextual b-roll; no specific object is named",
        MaterialType.SCREENSHOT.value: "narration references a screen/interface",
        MaterialType.STOCK_IMAGE.value: "a real photograph represents the scene without generation cost",
        MaterialType.AI_IMAGE.value: "no real imagery exists for this event; AI reconstruction is the only visual",
        MaterialType.AI_VIDEO.value: "no real motion footage exists; AI video reconstruction is used",
        MaterialType.TEXT.value: "the point is textual; on-screen text carries it",
        MaterialType.STOCK_VIDEO.value: "general scene; stock video is the default real footage",
    }
    return reason_map.get(material_type, "default visual strategy")


def _source_requirement_for(material_type: str) -> str:
    requirements = {
        MaterialType.ARCHIVAL.value: "historical/archival source preferred",
        MaterialType.MAP.value: "map or geographic imagery preferred",
        MaterialType.CHART.value: "data source for the figures preferred",
        MaterialType.GRAPH.value: "data source for the figures preferred",
        MaterialType.DOCUMENT.value: "the actual document/headline preferred",
        MaterialType.LOGO.value: "official brand assets preferred",
        MaterialType.PRODUCT_IMAGE.value: "official product imagery preferred",
        MaterialType.SCREENSHOT.value: "actual interface screenshot preferred",
    }
    return requirements.get(material_type, "")


def scene_plan_summary(plan: Optional[ScenePlan]) -> str:
    """One-line summary for logs/UI (no chain-of-thought)."""
    if plan is None or not plan.scenes:
        return "none"
    types: Dict[str, int] = {}
    for scene in plan.scenes:
        types[scene.material_type] = types.get(scene.material_type, 0) + 1
    return (
        f"scenes={len(plan.scenes)}, types={dict(sorted(types.items()))}, "
        f"ai_images={plan.ai_image_count}/{plan.ai_image_budget}"
    )
