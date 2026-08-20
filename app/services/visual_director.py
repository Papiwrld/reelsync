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
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from app.services.agent_llm import AgentTracker, _llm_json
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


def _scene_index_from_key(key: Any) -> Optional[int]:
    """Accept numeric keys or ``scene_N`` labels from the LLM payload."""
    if isinstance(key, int):
        return key
    if isinstance(key, str):
        label = key.strip()
        if label.isdigit():
            return int(label)
        if label.startswith("scene_") and label[len("scene_") :].isdigit():
            return int(label[len("scene_") :])
    return None


def _parse_semantic_search_terms(
    payload: Any, scene_count: int, limit: int
) -> Dict[int, List[str]]:
    """Validate an LLM payload into ``{scene_index: [term, ...]}``.

    Drops junk entries (non-list values, unknown scene keys, empty tokens) so a
    noisy model response still degrades to clean per-scene search terms.
    """
    result: Dict[int, List[str]] = {}
    if not isinstance(payload, dict):
        raise ValueError("semantic search terms payload is not a JSON object")
    for key, value in payload.items():
        index = _scene_index_from_key(key)
        if index is None or not (0 <= index < scene_count):
            continue
        if not isinstance(value, list):
            continue
        terms: List[str] = []
        for item in value:
            if not isinstance(item, str):
                continue
            term = item.strip().strip('"').strip()
            if len(term) >= 2 and term.lower() not in _FUNCTION_WORDS:
                terms.append(term)
        result[index] = terms[:limit]
    return result


def _semantic_search_terms_prompt(
    scenes: List[Tuple[int, str, str, str]], limit: int = 3
) -> str:
    """Build one short prompt covering many scenes (one LLM call per batch)."""
    scene_lines = []
    for index, narration, visual_intent, material_type in scenes:
        scene_lines.append(
            f"Scene {index}\n"
            f'  Narration: "{narration[:200]}"\n'
            f"  Visual intent: {visual_intent}\n"
            f"  Material type: {material_type}"
        )
    scene_block = "\n".join(scene_lines)
    return (
        "You turn narration into concrete stock-footage search phrases.\n"
        "For every scene output up to "
        f"{limit} short, concrete search phrases (2-6 words) that Pexels, "
        "Pixabay, or Coverr would return good footage for. Describe what should "
        "appear on screen (e.g. \"growing bar chart\", \"CEO pointing at rising "
        "graph\"), never abstract concepts or generic words. Follow the visual "
        "intent as the guide.\n"
        "Reply with ONLY valid JSON mapping each scene number to a list of "
        f'strings, e.g. {{"0": ["growing bar chart", "CEO pointing at graph"]}}\n'
        f"\nScenes:\n{scene_block}"
    )


def _deterministic_terms_for_scenes(
    scenes: List[Tuple[int, str, str, str]], limit: int = 3
) -> Dict[int, List[str]]:
    """Deterministic fallback search terms for a whole batch of scenes."""
    return {
        index: _significant_search_terms(narration, limit=limit)
        for index, narration, _, _ in scenes
    }


def _semantic_search_terms_for_scenes(
    scenes: List[Tuple[int, str, str, str]],
    app_config=None,
    tracker: Optional[AgentTracker] = None,
    limit: int = 3,
) -> Dict[int, List[str]]:
    """One batched LLM call that returns search terms for ALL scenes.

    A degraded circuit breaker or any LLM failure falls back to deterministic
    terms per scene, so a dead provider never blocks scene planning.
    """
    if not scenes:
        return {}
    prompt = _semantic_search_terms_prompt(scenes, limit=limit)
    try:
        payload = _llm_json(
            prompt,
            fallback=lambda: _deterministic_terms_for_scenes(scenes, limit=limit),
            app_config=app_config,
            tracker=tracker,
            agent="scene_search_terms",
        )
    except Exception:  # noqa: BLE001 - degrade to deterministic on any failure
        logger.warning("semantic search terms llm call failed, using deterministic terms")
        payload = _deterministic_terms_for_scenes(scenes, limit=limit)
    try:
        parsed = _parse_semantic_search_terms(payload, scene_count=len(scenes), limit=limit)
    except ValueError:
        logger.warning("semantic search terms payload invalid, using deterministic terms")
        parsed = _deterministic_terms_for_scenes(scenes, limit=limit)
    # Per-scene fallback: a scene missing from the payload (or emptied by
    # validation) still gets usable deterministic terms.
    for index, narration, _, _ in scenes:
        if not parsed.get(index):
            parsed[index] = _significant_search_terms(narration, limit=limit)
    return parsed


def _generate_semantic_search_terms(
    narration: str,
    visual_intent: str,
    material_type: str,
    limit: int = 3,
    app_config=None,
    tracker: Optional[AgentTracker] = None,
) -> List[str]:
    """LLM-generated footage-searchable terms for a single scene.

    Degraded tracker or any LLM failure falls back to the deterministic
    baseline ``_significant_search_terms``, preserving the ``List[str]``
    contract in every path. Prefer ``_semantic_search_terms_for_scenes`` when
    planning many scenes: one batched LLM call is much cheaper than one here.
    """
    terms = _semantic_search_terms_for_scenes(
        [(0, narration, visual_intent, material_type)],
        app_config=app_config,
        tracker=tracker,
        limit=limit,
    )
    return terms[0] if terms else []


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
    app_config=None,
    tracker: Optional[AgentTracker] = None,
) -> ScenePlan:
    """Build the scene-by-scene visual plan for a script.

    Scene classification is deterministic; search terms are LLM-enhanced when
    an ``app_config`` is in scope (one batched call for all scenes, degraded or
    failed LLM falls back to deterministic term extraction). AI-generated
    imagery is strictly budgeted: the plan can never produce an AI image for
    every scene.
    """
    chunks = _split_scenes(script, desired_scene_count or 8)
    budget = min(
        MAX_AI_IMAGE_BUDGET,
        max(MIN_AI_IMAGE_BUDGET, round(len(chunks) * ai_image_budget_ratio)),
    )
    style_language = (intelligence.visual_language if intelligence else "") or profile.visual_style or "clear, relevant visuals"

    # First pass: classify each chunk's material type (with AI budget caps) and
    # record the visual intent that the search-term stage must honor.
    scene_rows: List[Tuple[int, str, str, str]] = []
    ai_used = 0
    for index, chunk in enumerate(chunks):
        material_type = _classify_material_type(chunk)
        # Budget enforcement: excess AI images downgrade to B-roll.
        if material_type in (MaterialType.AI_IMAGE.value, MaterialType.AI_VIDEO.value):
            if ai_used < budget:
                ai_used += 1
            else:
                material_type = MaterialType.B_ROLL.value
        scene_rows.append(
            (index, chunk, _visual_intent_for(material_type, chunk), material_type)
        )

    # Search terms: one batched semantic call for ALL scenes when an LLM config
    # is in scope; otherwise each scene uses the deterministic extraction.
    if app_config:
        terms_by_scene = _semantic_search_terms_for_scenes(
            scene_rows, app_config=app_config, tracker=tracker, limit=3
        )
    else:
        terms_by_scene = _deterministic_terms_for_scenes(scene_rows, limit=3)

    scenes: List[SceneVisual] = []
    subject_keys: Dict[str, int] = {}
    location_keys: Dict[str, int] = {}
    used_search_terms: set[str] = set()
    continuity_notes: List[str] = []

    for index, chunk, visual_intent, material_type in scene_rows:
        terms = terms_by_scene[index]
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
            visual_intent=visual_intent,
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
