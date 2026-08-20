"""Content Profile / Niche definitions for the agentic planning layer.

A profile carries strategic instructions that influence the whole generation
pipeline (script voice, hook style, pacing, media preferences, subtitle and
music strategy). Profiles are data, not code: the generation logic reads
these fields instead of hard-coding niche behavior, so new niches can be
added without touching the orchestration layer.
"""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict


class ContentProfile(BaseModel):
    """Strategic description of a content niche.

    Fields are intentionally free-form strings/lists so profiles stay simple
    to author and extend. Agent prompts embed the strategy fields directly;
    media/subtitle/music fields are consumed by later pipeline stages.
    """

    model_config = ConfigDict(extra="allow")

    name: str
    description: str = ""
    audience: str = ""
    content_goals: str = ""
    tone: str = ""
    narrative_style: str = ""
    hook_strategy: str = ""
    pacing: str = ""
    preferred_video_length: str = ""
    preferred_scene_duration: str = ""
    vocabulary_style: str = ""
    emotional_style: str = ""
    credibility_level: str = ""
    research_depth: str = ""
    # Content Intelligence (Phase 2A): niche-agnostic strategy data consumed
    # by the intelligence layer. Profiles are data — new niches are added by
    # authoring a profile, never by editing code. Empty fields are derived
    # deterministically from the strategy fields above.
    typical_formats: List[str] = []
    preferred_narrative_patterns: List[str] = []
    fact_check_level: str = ""  # normal | strong | very_strong
    trend_sensitivity: str = ""  # low | medium | high | very_high
    risk_level: str = ""  # low | medium | high
    title_patterns: List[str] = []
    thumbnail_patterns: List[str] = []
    retention_patterns: List[str] = []
    source_preferences: List[str] = []
    media_strategy: str = ""
    visual_style: str = ""
    preferred_media_sources: List[str] = []
    fallback_media_sources: List[str] = []
    subtitle_style: str = ""
    caption_behavior: str = ""
    music_strategy: str = ""
    sound_effect_strategy: str = ""
    cta_strategy: str = ""
    prohibited_patterns: List[str] = []
    custom_instructions: str = ""


_PROFILE_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "dark_history": {
        "description": "Dark, unsettling historical stories, disasters and crimes.",
        "audience": "Viewers who enjoy true crime, dark history and macabre storytelling.",
        "content_goals": "Tell forgotten or disturbing historical stories with atmosphere and tension.",
        "tone": "Dark, grave, dramatic; a respectful but chilling narrator voice.",
        "narrative_style": "Slow-burn storytelling that builds dread, with vivid period detail.",
        "hook_strategy": "Open with a chilling fact, unanswered death, or a sentence that implies something sinister happened.",
        "pacing": "Slow, deliberate opening; accelerating dread toward a grim reveal.",
        "preferred_video_length": "45-75 seconds",
        "preferred_scene_duration": "5-8 seconds",
        "vocabulary_style": "Vivid, sensory, precise period vocabulary; restrained and literate.",
        "emotional_style": "Unease, dread, morbid curiosity, pity.",
        "credibility_level": "High: name dates, places and sources; avoid legend presented as fact.",
        "research_depth": "Deep: verify dates, victims, places, and the historical record.",
        "media_strategy": "Archival footage, historical photographs, maps, old documents, period objects, dark cinematic environments.",
        "visual_style": "Dark, desaturated, aged textures, dramatic shadows.",
        "preferred_media_sources": ["pexels", "web_scrape"],
        "fallback_media_sources": ["pixabay", "coverr"],
        "subtitle_style": "Documentary: clean, high-contrast, bottom-safe-area, short lines.",
        "caption_behavior": "Minimal decoration; no emoji; keywords occasionally emphasized.",
        "music_strategy": "Low, tense ambient or dark drone, swelling at reveals.",
        "sound_effect_strategy": "Rare but impactful: heartbeats, distant sounds, abrupt cuts on reveals.",
        "cta_strategy": "Ask viewers what they would have done, or to share the story.",
        "prohibited_patterns": ["cheerful background music", "comedy tone", "clickbait superlatives without evidence"],
    },
    "mystery": {
        "description": "Unexplained phenomena, unsolved cases and open questions.",
        "audience": "Viewers drawn to unsolved mysteries, conspiracy-adjacent but skeptical content.",
        "content_goals": "Make the audience question what is real and keep them engaged through unresolved tension.",
        "tone": "Intrigued, measured, slightly ominous; never mocking the mystery.",
        "narrative_style": "Present evidence and contradictions; end on the open question.",
        "hook_strategy": "Open with a question, a contradiction, or an event that should not have happened.",
        "pacing": "Fast opening, medium exposition, accelerating evidence, slow final question.",
        "preferred_video_length": "40-70 seconds",
        "preferred_scene_duration": "4-7 seconds",
        "vocabulary_style": "Precise but accessible; words that create uncertainty: unexplained, vanished, no record, allegedly.",
        "emotional_style": "Curiosity, wonder, unease.",
        "credibility_level": "High for facts, explicit about what is unknown or disputed.",
        "research_depth": "Deep: distinguish documented facts from theory.",
        "media_strategy": "Cinematic stock, night scenes, maps, documents, abstract atmosphere, generated images for the unexplained.",
        "visual_style": "Moody, high contrast, deep shadows, slight cool tint.",
        "preferred_media_sources": ["pexels", "custom_api"],
        "fallback_media_sources": ["pixabay", "pollinations"],
        "subtitle_style": "Cinematic: bold short lines, occasional keyword highlight, bottom safe area.",
        "caption_behavior": "Emphasize the central question; no emoji.",
        "music_strategy": "Tension-building ambient, subtle heartbeat or drone under reveals.",
        "sound_effect_strategy": "Subtle risers into evidence beats, silence before the final question.",
        "cta_strategy": "Ask the audience what they believe happened.",
        "prohibited_patterns": ["presenting theories as proven facts", "jump-scare tactics", "hyperbole without evidence"],
    },
    "mythology_lore": {
        "description": "Myths, legends and folklore from any culture.",
        "audience": "Fans of mythology, fantasy-adjacent storytelling and world-building.",
        "content_goals": "Bring myths to life as vivid stories with their cultural weight intact.",
        "tone": "Epic, reverent, dramatic; grand without being theatrical.",
        "narrative_style": "Classic myth structure: origin, conflict, downfall or transformation.",
        "hook_strategy": "Open in the middle of the myth's most dramatic moment, then rewind.",
        "pacing": "Grand opening, measured middle, climactic ending.",
        "preferred_video_length": "45-90 seconds",
        "preferred_scene_duration": "5-8 seconds",
        "vocabulary_style": "Elevated, archaic-flavored but understandable; proper names pronounced clearly.",
        "emotional_style": "Awe, tragedy, wonder, moral weight.",
        "credibility_level": "Present the myth as told by its tradition; note alternate versions.",
        "research_depth": "Medium-deep: respect the source tradition, name the culture.",
        "media_strategy": "Epic landscapes, ancient ruins, temples, statues, fire and storm footage, artworks.",
        "visual_style": "Golden-hour epic, warm light, monument scale.",
        "preferred_media_sources": ["pexels", "pixabay"],
        "fallback_media_sources": ["coverr", "pollinations"],
        "subtitle_style": "Cinematic serif-feel bold lines with generous spacing.",
        "caption_behavior": "Minimal; names of gods/heroes occasionally emphasized.",
        "music_strategy": "Epic orchestral or folk-ambient depending on culture.",
        "sound_effect_strategy": "Drums, thunder, chants at mythic moments.",
        "cta_strategy": "Ask which version of the myth the audience grew up with.",
        "prohibited_patterns": ["modern slang", "disrespectful mockery of a tradition"],
    },
    "african_history": {
        "description": "African history, kingdoms, and folklore told with depth and pride.",
        "audience": "Audiences hungry for African history beyond colonial narratives, diaspora included.",
        "content_goals": "Reveal the sophistication of pre-colonial and modern African civilizations.",
        "tone": "Proud, scholarly, warm, narrative.",
        "narrative_style": "Kingdom-by-kingdom storytelling; center African agency and sources.",
        "hook_strategy": "Open with a specific astonishing achievement or a correction of a common myth.",
        "pacing": "Confident medium pace with ceremonial moments.",
        "preferred_video_length": "50-90 seconds",
        "preferred_scene_duration": "6-8 seconds",
        "vocabulary_style": "Rich, dignified; use authentic names of places, rulers and institutions.",
        "emotional_style": "Pride, wonder, respect, vindication.",
        "credibility_level": "High: cite historians and traditions; name sources like oral histories where relevant.",
        "research_depth": "Deep: cross-check European and African sources.",
        "media_strategy": "Landscapes, architecture, art, textiles, markets, ceremonies, maps of historical kingdoms.",
        "visual_style": "Warm, earthy tones, golden light, textile textures.",
        "preferred_media_sources": ["pexels", "web_scrape"],
        "fallback_media_sources": ["pixabay", "coverr"],
        "subtitle_style": "Documentary: clean, readable, accurate names spelled correctly.",
        "caption_behavior": "Minimal; emphasize authentic names.",
        "music_strategy": "African percussion, kora or djembe textures; respectful and rooted.",
        "sound_effect_strategy": "Ceremonial drums at key moments.",
        "cta_strategy": "Invite viewers to share their heritage or learn more.",
        "prohibited_patterns": ["colonial-framed language", "generic 'Africa' stereotypes", "erasure of specific cultures"],
    },
    "technology": {
        "description": "Tech news, products, AI and engineering explained for the curious.",
        "audience": "Tech enthusiasts and early adopters who want current, accurate information.",
        "content_goals": "Explain what happened, why it matters, and what comes next.",
        "tone": "Sharp, informed, neutral-enthusiastic; no hype without substance.",
        "narrative_style": "Problem -> development -> implication -> what's next.",
        "hook_strategy": "Open with the concrete change or headline-worthy result, not the history.",
        "pacing": "Brisk, information-dense, no dead air.",
        "preferred_video_length": "30-60 seconds",
        "preferred_scene_duration": "3-6 seconds",
        "vocabulary_style": "Technical terms used correctly and briefly explained; precise numbers.",
        "emotional_style": "Excitement, anticipation, skepticism where warranted.",
        "credibility_level": "Very high: attribute claims, date announcements, name companies/models.",
        "research_depth": "Medium-deep: verify release dates, specs, and pricing.",
        "media_strategy": "Product footage, screens, devices, data visualizations, UI clips, tech news footage.",
        "visual_style": "Clean, bright, modern; screens and interfaces prominent.",
        "preferred_media_sources": ["pexels", "web_scrape"],
        "fallback_media_sources": ["pixabay", "coverr"],
        "subtitle_style": "News: tight lines, high readability, numbers kept whole on one line.",
        "caption_behavior": "Clean; highlight key numbers and model names.",
        "music_strategy": "Modern, driving electronic or minimal beat.",
        "sound_effect_strategy": "UI clicks, whooshes on transitions, subtle tech ambience.",
        "cta_strategy": "Ask what feature they want next, or to follow for updates.",
        "prohibited_patterns": ["unverifiable spec claims", "AI-hype clichés without specifics"],
    },
    "ai_news": {
        "description": "Fast-moving AI model and product news.",
        "audience": "AI-curious builders, creators and professionals.",
        "content_goals": "Be first with the meaningful signal in model and product releases.",
        "tone": "Fast, confident, technical but approachable.",
        "narrative_style": "Release -> capability -> benchmark -> implication for creators.",
        "hook_strategy": "Open with the single most important capability or benchmark number.",
        "pacing": "Very fast; cut every 3-4 seconds.",
        "preferred_video_length": "25-50 seconds",
        "preferred_scene_duration": "3-5 seconds",
        "vocabulary_style": "Current model names, benchmark terms, creator-language.",
        "emotional_style": "Urgency, excitement, healthy skepticism.",
        "credibility_level": "Very high: name the model, the paper, the benchmark and the date.",
        "research_depth": "Deep: verify against primary sources (papers, official releases).",
        "media_strategy": "Abstract AI visuals, data visualizations, screenshots of interfaces, generated imagery.",
        "visual_style": "Neon-tech palette, dark mode, glowing accents.",
        "preferred_media_sources": ["custom_api", "pexels"],
        "fallback_media_sources": ["pixabay", "pollinations"],
        "subtitle_style": "Bold kinetic lines, high contrast, numbers emphasized.",
        "caption_behavior": "Emphasize benchmark numbers and model names.",
        "music_strategy": "Fast electronic pulse, subtle risers on releases.",
        "sound_effect_strategy": "Synthetic blips, glitches, whooshes.",
        "cta_strategy": "Ask what they'd build with the new model.",
        "prohibited_patterns": ["outdated model info presented as new", "benchmark cherry-picking without context"],
    },
    "finance": {
        "description": "Money, investing and economic literacy in plain language.",
        "audience": "Beginners to intermediate investors who want practical, honest finance content.",
        "content_goals": "Turn confusing finance into decisions the viewer can act on.",
        "tone": "Trustworthy, direct, calm; never a pump.",
        "narrative_style": "Problem -> principle -> example -> action step.",
        "hook_strategy": "Open with the cost of a mistake or the surprising number people get wrong.",
        "pacing": "Steady, deliberate; slow down at numbers.",
        "preferred_video_length": "40-70 seconds",
        "preferred_scene_duration": "4-7 seconds",
        "vocabulary_style": "Plain-language finance; every jargon term translated once.",
        "emotional_style": "Calm confidence, relief, motivation without hype.",
        "credibility_level": "Very high: no guaranteed returns, disclose risk, cite data.",
        "research_depth": "Deep: check rates, returns, dates, and regulations.",
        "media_strategy": "Charts, graphs, city skylines, working professionals, minimal clean visuals.",
        "visual_style": "Clean, minimal, data-forward; green/blue accents.",
        "preferred_media_sources": ["pexels", "pixabay"],
        "fallback_media_sources": ["coverr", "web_scrape"],
        "subtitle_style": "Classic: very readable, numbers on one line, high contrast.",
        "caption_behavior": "Keep percentages and dollar figures visually intact.",
        "music_strategy": "Light, professional background; quiet under numbers.",
        "sound_effect_strategy": "Minimal; soft ticks for data points.",
        "cta_strategy": "Ask a specific money question, or invite one next step.",
        "prohibited_patterns": ["guaranteed-return language", "get-rich-quick framing", "specific stock pump advice"],
        "typical_formats": ["explainer", "news_analysis", "case_study"],
        "preferred_narrative_patterns": ["problem", "principle", "example", "action step"],
        "fact_check_level": "very_strong",
        "trend_sensitivity": "medium",
        "risk_level": "high",
        "title_patterns": ["mistake-cost framing", "surprising number"],
        "thumbnail_patterns": ["one bold number against a clean background", "chart detail"],
        "retention_patterns": ["money relevance in the first three seconds", "one principle per video", "actionable step payoff"],
        "source_preferences": ["government", "academic", "major_journalism", "official_company"],
    },
    "motivation": {
        "description": "Personal growth, discipline and transformation.",
        "audience": "People seeking momentum, habit change and self-respect.",
        "content_goals": "Create an emotional shift that leads to action today.",
        "tone": "Direct, warm, urgent; second person, speaking TO the viewer.",
        "narrative_style": "Relatable struggle -> turning point -> transformation -> actionable conclusion.",
        "hook_strategy": "Open with the viewer's exact struggle or a blunt truth about it.",
        "pacing": "Rhythmic, punchy; build to a peak, land softly on the action step.",
        "preferred_video_length": "30-60 seconds",
        "preferred_scene_duration": "3-6 seconds",
        "vocabulary_style": "Second-person, plain, visceral; short sentences.",
        "emotional_style": "Resonance, urgency, empowerment, relief.",
        "credibility_level": "Emotional honesty over citations; avoid fake transformation stories.",
        "research_depth": "Light: real psychology/habits when referenced.",
        "media_strategy": "Human activity, training, work, nature, sunrise, crowds, cinematic environments.",
        "visual_style": "High-energy, natural light, human-centered close-ups.",
        "preferred_media_sources": ["pexels", "pixabay"],
        "fallback_media_sources": ["coverr", "custom_api"],
        "subtitle_style": "Bold: large words, word-level emphasis, popping punctuation.",
        "caption_behavior": "Emphasize the action verbs and the turning-point sentence.",
        "music_strategy": "Rising emotional orchestral or cinematic rock, peaking at the turning point.",
        "sound_effect_strategy": "Heartbeat starts, risers at peaks, silence before the action step.",
        "cta_strategy": "Give one specific action for today; ask them to start now.",
        "prohibited_patterns": ["toxic hustle glamorization", "false transformation claims", "guilt-tripping"],
    },
    "education": {
        "description": "Clear explanations of concepts across any subject.",
        "audience": "Lifelong learners who want correct, well-structured explanations.",
        "content_goals": "Make the concept click the first time.",
        "tone": "Friendly, clear, patient; teacher energy without condescension.",
        "narrative_style": "Question -> foundation -> step-by-step build -> application -> summary.",
        "hook_strategy": "Open with the misconception or the 'it actually works this way' surprise.",
        "pacing": "Calm and even; pause at each new idea.",
        "preferred_video_length": "45-90 seconds",
        "preferred_scene_duration": "6-9 seconds",
        "vocabulary_style": "Precise definitions first, then plain-language restatement.",
        "emotional_style": "Clarity, 'aha' moments, gentle encouragement.",
        "credibility_level": "Very high: correct facts, correct names, avoid oversimplification errors.",
        "research_depth": "Deep: verify everything; explain rather than assert.",
        "media_strategy": "Diagrams, real-world examples, laboratory/science footage, clean illustrations, maps.",
        "visual_style": "Bright, clear, organized; diagrams and labels prominent.",
        "preferred_media_sources": ["pexels", "pixabay"],
        "fallback_media_sources": ["coverr", "web_scrape"],
        "subtitle_style": "Classic: full readability, correct punctuation, term emphasis.",
        "caption_behavior": "Emphasize the defined terms; keep equations/labels intact.",
        "music_strategy": "Soft, neutral background or none; clarity over mood.",
        "sound_effect_strategy": "Minimal; soft pings when a definition lands.",
        "cta_strategy": "Ask a comprehension question; invite them to test the idea.",
        "prohibited_patterns": ["teaching myths as facts", "oversimplification that becomes false", "clickbait over substance"],
    },
    "science": {
        "description": "Scientific discoveries and how the universe works.",
        "audience": "Science enthusiasts who respect depth and accuracy.",
        "content_goals": "Convey awe backed by accurate science.",
        "tone": "Wonderful but rigorous; the facts carry the drama.",
        "narrative_style": "Question -> evidence -> mechanism -> implication for us.",
        "hook_strategy": "Open with a result that feels impossible but is documented.",
        "pacing": "Medium; build from accessible to deep, land on the implication.",
        "preferred_video_length": "40-70 seconds",
        "preferred_scene_duration": "4-7 seconds",
        "vocabulary_style": "Scientific terms used correctly with quick glosses.",
        "emotional_style": "Awe, curiosity, 'the universe is stranger than we thought'.",
        "credibility_level": "Very high: name institutions, studies, and what remains unknown.",
        "research_depth": "Deep: check the actual study, not the press release.",
        "media_strategy": "Space, microscopy, laboratory footage, nature, data visualization, simulations.",
        "visual_style": "Deep-space and macro contrast; precise, clean.",
        "preferred_media_sources": ["pexels", "pixabay"],
        "fallback_media_sources": ["custom_api", "coverr"],
        "subtitle_style": "Documentary: precise, clean, technical terms kept intact.",
        "caption_behavior": "Highlight the surprising numbers and the study's conclusion.",
        "music_strategy": "Ambient, spacious, subtle; swell at the implication.",
        "sound_effect_strategy": "Minimal; deep drones for cosmic scale.",
        "cta_strategy": "Ask what they'd want to know next about the topic.",
        "prohibited_patterns": ["pseudoscience framing", "study misrepresentation", "breathless clickbait over accuracy"],
        "typical_formats": ["documentary", "explainer", "how_it_works"],
        "preferred_narrative_patterns": ["question", "hypothesis", "evidence", "discovery", "implications"],
        "fact_check_level": "very_strong",
        "trend_sensitivity": "low",
        "risk_level": "medium",
        "title_patterns": ["evidence-backed surprise", "implication question"],
        "thumbnail_patterns": ["space or macro subject with a one-word label", "before/after evidence"],
        "retention_patterns": ["awe in the first three seconds", "study-backed depth", "implication payoff"],
        "source_preferences": ["academic", "government", "major_journalism"],
    },
    "storytelling": {
        "description": "General narrative content: true stories, retellings, creative narrative.",
        "audience": "Viewers who stay for a good story told well.",
        "content_goals": "Hold attention through narrative craft, regardless of genre.",
        "tone": "Adaptable, but always story-first: scenes, stakes, characters.",
        "narrative_style": "Scene-based storytelling: opening moment, stakes, escalation, resolution.",
        "hook_strategy": "Open in the middle of the action or with the outcome, then flash back.",
        "pacing": "Scene-driven; vary rhythm, never uniform.",
        "preferred_video_length": "45-90 seconds",
        "preferred_scene_duration": "5-8 seconds",
        "vocabulary_style": "Concrete, sensory, character-driven language.",
        "emotional_style": "Matched to the story; always present.",
        "credibility_level": "Medium: 'this is how the story is told' framing for retellings.",
        "research_depth": "Medium: fact-check named events and people.",
        "media_strategy": "Cinematic environments matching each scene's mood.",
        "visual_style": "Cinematic, scene-matched, varied.",
        "preferred_media_sources": ["pexels", "coverr"],
        "fallback_media_sources": ["pixabay", "web_scrape"],
        "subtitle_style": "Cinematic: bold lines, scene-appropriate emphasis.",
        "caption_behavior": "Emphasis on dialogue-like lines and turning points.",
        "music_strategy": "Scene-matched; changes with narrative beats.",
        "sound_effect_strategy": "Diegetic-feeling effects matching scenes.",
        "cta_strategy": "Ask whether they'd have made the same choice.",
        "prohibited_patterns": ["flat summary instead of scenes", "telling instead of showing"],
    },
    "business": {
        "description": "Business documentaries: companies, founders, markets, money, disruption.",
        "audience": "Viewers who want the story behind companies: how they won, how they fell, how markets work.",
        "content_goals": "Turn business history and strategy into a cinematic, research-backed story.",
        "tone": "Cinematic, confident, journalistic; tension without tabloid.",
        "narrative_style": "Story-driven business narrative: rise, peak, conflict, fall or reinvention.",
        "hook_strategy": "Open with the dramatic outcome or the impossible number, then rewind.",
        "pacing": "Hook in the first five seconds, brisk middle, payoff before the CTA.",
        "preferred_video_length": "60-120 seconds",
        "preferred_scene_duration": "5-8 seconds",
        "vocabulary_style": "Precise business vocabulary explained once; concrete numbers.",
        "emotional_style": "Tension, fascination, drama without schadenfreude, satisfaction at the payoff.",
        "credibility_level": "Very high: revenue figures, dates, names, and valuations must be sourced.",
        "research_depth": "Very deep: verify financials, dates, leadership facts, and legal outcomes.",
        "media_strategy": "Archival footage, product shots, charts, buildings, factories, portraits, market visuals.",
        "visual_style": "Cinematic, high-contrast, corporate-modern with a filmic grade.",
        "preferred_media_sources": ["pexels", "web_scrape", "custom_api"],
        "fallback_media_sources": ["pixabay", "coverr"],
        "subtitle_style": "Documentary: bold, clean, numbers kept whole on one line.",
        "caption_behavior": "Emphasize revenue figures, dates, and quotes.",
        "music_strategy": "Cinematic orchestral or dark synth; tension builds with the narrative.",
        "sound_effect_strategy": "Sub drops, tickers for numbers, risers into reveals.",
        "cta_strategy": "Ask what the viewer would have done differently, or to follow for the next story.",
        "prohibited_patterns": ["unverified financial claims", "promotional tone for the featured company", "glorifying fraud"],
        "typical_formats": ["documentary", "case_study", "news_analysis"],
        "preferred_narrative_patterns": ["rise and fall", "founder journey", "corporate conflict", "business model", "disruption", "scandal"],
        "fact_check_level": "very_strong",
        "trend_sensitivity": "medium",
        "risk_level": "high",
        "title_patterns": ["conflict plus curiosity", "outcome-based", "impossible number"],
        "thumbnail_patterns": ["subject portrait with one bold contrasting word", "before/after or rise/fall split", "single strong symbol"],
        "retention_patterns": ["conflict in the first three seconds", "number-driven curiosity gaps", "payoff structure"],
        "source_preferences": ["academic", "major_journalism", "official_company", "government", "industry"],
    },
    "history": {
        "description": "Historical events, eras and figures told with accuracy and narrative drive.",
        "audience": "History viewers who want accuracy with a story, not dates read aloud.",
        "content_goals": "Make history vivid and trustworthy: what happened, why it mattered, what it changed.",
        "tone": "Authoritative, vivid, measured; drama rooted in the record.",
        "narrative_style": "Chronological or mystery-driven narrative with context and consequence.",
        "hook_strategy": "Open with a specific, surprising event or a correction of a common myth.",
        "pacing": "Steady build, scene changes on narrative beats, clear payoff.",
        "preferred_video_length": "50-90 seconds",
        "preferred_scene_duration": "5-8 seconds",
        "vocabulary_style": "Period-appropriate terms explained once; concrete dates and places.",
        "emotional_style": "Wonder, gravity, the stakes of the era.",
        "credibility_level": "High: name dates, places, and sources; distinguish fact from legend.",
        "research_depth": "Deep: cross-check secondary accounts against primary records.",
        "media_strategy": "Archival footage, maps, documents, paintings, period objects, recreated atmosphere.",
        "visual_style": "Aged textures, period palettes, documentary realism.",
        "preferred_media_sources": ["pexels", "web_scrape"],
        "fallback_media_sources": ["pixabay", "coverr"],
        "subtitle_style": "Documentary: clean, high-contrast, correct names.",
        "caption_behavior": "Emphasize dates, places, and named sources.",
        "music_strategy": "Period-flavored or neutral cinematic; restrained at dramatic beats.",
        "sound_effect_strategy": "Period ambience, document handling, subtle drums at turning points.",
        "cta_strategy": "Ask how the era still echoes today, or invite a follow-up topic.",
        "prohibited_patterns": ["legend presented as fact", "anachronistic framing", "presentism without context"],
        "typical_formats": ["documentary", "timeline", "mystery"],
        "preferred_narrative_patterns": ["timeline", "mystery", "rise and fall of civilizations", "conflict", "discovery"],
        "fact_check_level": "strong",
        "trend_sensitivity": "low",
        "risk_level": "medium",
        "title_patterns": ["mystery question", "forgotten event", "myth correction"],
        "thumbnail_patterns": ["period artifact plus one question word", "before/after contrast", "map or document detail"],
        "retention_patterns": ["open tension in the first three seconds", "stakes of the era", "unresolved question until the payoff"],
        "source_preferences": ["academic", "government", "primary", "major_journalism"],
    },
    "psychology": {
        "description": "Psychology and behavior explained through research and real situations.",
        "audience": "Young adults and curious minds who want insight they can apply today.",
        "content_goals": "Turn psychological research into an insight the viewer feels and can use.",
        "tone": "Warm, sharp, non-judgmental; insight without pop-science hype.",
        "narrative_style": "Problem -> explanation -> mechanism -> examples -> practical takeaway.",
        "hook_strategy": "Open with the counterintuitive behavior or the 'this is why you' moment.",
        "pacing": "Fast; one idea per video, built to a usable insight.",
        "preferred_video_length": "45-60 seconds",
        "preferred_scene_duration": "4-7 seconds",
        "vocabulary_style": "Plain language; research terms explained in one breath.",
        "emotional_style": "Recognition, relief, curiosity about oneself.",
        "credibility_level": "High: name the effect or study; avoid pop-psychology certainty.",
        "research_depth": "Deep: verify the actual study and its limits.",
        "media_strategy": "Human close-ups, expressive faces, everyday situations, animated concepts.",
        "visual_style": "Warm, human, slightly editorial.",
        "preferred_media_sources": ["pexels", "pixabay"],
        "fallback_media_sources": ["coverr", "custom_api"],
        "subtitle_style": "Bold, warm, word emphasis on the insight.",
        "caption_behavior": "Emphasize the insight sentence and the mechanism word.",
        "music_strategy": "Soft, warm, modern; neutral under the mechanism.",
        "sound_effect_strategy": "Minimal; soft pops for examples.",
        "cta_strategy": "Ask whether they've caught themselves doing it, or to try the insight today.",
        "prohibited_patterns": ["misdiagnosing viewers", "pop-psychology certainty", "self-help clichés without mechanism"],
        "typical_formats": ["explainer", "case_study", "tutorial"],
        "preferred_narrative_patterns": ["problem", "explanation", "mechanism", "examples", "practical takeaway"],
        "fact_check_level": "strong",
        "trend_sensitivity": "medium",
        "risk_level": "medium",
        "title_patterns": ["question", "myth debunking", "insight promise"],
        "thumbnail_patterns": ["expressive face plus one insight word", "before/after behavior", "single symbol"],
        "retention_patterns": ["self-relevance in the first three seconds", "one insight per video", "practical payoff"],
        "source_preferences": ["academic", "major_journalism"],
    },
    "gaming": {
        "description": "Gaming news, releases, controversies, and the industry behind them.",
        "audience": "Gamers and industry watchers who want fresh, informed takes.",
        "content_goals": "Report what happened in games, why it matters, and what the community thinks.",
        "tone": "Fast, sharp, community-literate; hype with receipts.",
        "narrative_style": "Event -> context -> conflict -> reaction -> analysis.",
        "hook_strategy": "Open with the release, the leak, or the controversy beat itself.",
        "pacing": "Very fast; cut every three to four seconds.",
        "preferred_video_length": "30-60 seconds",
        "preferred_scene_duration": "3-5 seconds",
        "vocabulary_style": "Current game and studio names; community language used accurately.",
        "emotional_style": "Excitement, anticipation, skepticism of spin.",
        "credibility_level": "High for facts: dates, versions, studio statements; label rumor as rumor.",
        "research_depth": "Medium-deep: verify dates, versions, and official statements.",
        "media_strategy": "Gameplay, trailers, screenshots, dev diaries, community reactions, charts.",
        "visual_style": "Saturated, kinetic, game-native.",
        "preferred_media_sources": ["web_scrape", "pexels"],
        "fallback_media_sources": ["pixabay", "coverr"],
        "subtitle_style": "Bold kinetic lines, high contrast, numbers emphasized.",
        "caption_behavior": "Emphasize patch notes numbers and dates.",
        "music_strategy": "Upbeat electronic or game-score energy; stops for big reveals.",
        "sound_effect_strategy": "Menu blips, reveals, impact hits.",
        "cta_strategy": "Ask whether they played it or what they think of the change.",
        "prohibited_patterns": ["rumor presented as fact", "outdated patch info as current", "developer harassment framing"],
        "typical_formats": ["news_analysis", "list", "explainer"],
        "preferred_narrative_patterns": ["release", "controversy", "competition", "gameplay", "community reaction", "industry impact"],
        "fact_check_level": "normal",
        "trend_sensitivity": "very_high",
        "risk_level": "low",
        "title_patterns": ["release-centric", "controversy question", "number-driven"],
        "thumbnail_patterns": ["key art with one bold claim", "comparison split", "reaction face plus game visual"],
        "retention_patterns": ["new information in the first three seconds", "freshness urgency", "community stakes"],
        "source_preferences": ["official_company", "community", "industry", "major_journalism"],
    },
    "custom": {
        "description": "A user-defined niche. Fill in the strategy fields to steer generation.",
        "audience": "",
        "content_goals": "",
        "tone": "",
        "narrative_style": "",
        "hook_strategy": "",
        "pacing": "",
        "preferred_video_length": "30-60 seconds",
        "preferred_scene_duration": "5-8 seconds",
        "vocabulary_style": "",
        "emotional_style": "",
        "credibility_level": "Medium",
        "research_depth": "Medium",
        "media_strategy": "",
        "visual_style": "",
        "preferred_media_sources": [],
        "fallback_media_sources": [],
        "subtitle_style": "",
        "caption_behavior": "",
        "music_strategy": "",
        "sound_effect_strategy": "",
        "cta_strategy": "",
        "prohibited_patterns": [],
        "custom_instructions": "",
    },
}


_PROFILES: Dict[str, ContentProfile] = {
    name: ContentProfile(name=name, **definition)
    for name, definition in _PROFILE_DEFINITIONS.items()
}


def list_content_profiles() -> List[str]:
    """Return available profile names in stable order."""
    return list(_PROFILE_DEFINITIONS.keys())


def get_content_profile(name: str) -> ContentProfile:
    """Resolve a profile by name; unknown names fall back to ``custom``.

    Falling back (rather than raising) keeps the generation pipeline
    resilient against stale UI values or renamed profiles.
    """
    return _PROFILES.get((name or "").strip().lower(), _PROFILES["custom"])


def profile_strategy_context(profile: ContentProfile) -> str:
    """Render the profile's strategy fields as a compact prompt block.

    Only fields that are actually filled in are included, keeping prompts
    short for profiles the user has not customized.
    """
    lines = [f"# Content Profile: {profile.name}", f"- Description: {profile.description}"]
    mapping = [
        ("audience", "Audience"),
        ("content_goals", "Content goals"),
        ("tone", "Tone"),
        ("narrative_style", "Narrative style"),
        ("hook_strategy", "Hook strategy"),
        ("pacing", "Pacing"),
        ("preferred_video_length", "Preferred video length"),
        ("vocabulary_style", "Vocabulary style"),
        ("emotional_style", "Emotional style"),
        ("credibility_level", "Credibility level"),
        ("research_depth", "Research depth"),
        ("media_strategy", "Media strategy"),
        ("visual_style", "Visual style"),
        ("subtitle_style", "Subtitle style"),
        ("caption_behavior", "Caption behavior"),
        ("music_strategy", "Music strategy"),
        ("sound_effect_strategy", "Sound effect strategy"),
        ("cta_strategy", "CTA strategy"),
    ]
    for attribute, label in mapping:
        value = getattr(profile, attribute, "")
        if value:
            lines.append(f"- {label}: {value}")
    if profile.prohibited_patterns:
        lines.append(f"- Prohibited patterns: {', '.join(profile.prohibited_patterns)}")
    if profile.custom_instructions:
        lines.append(f"- Custom instructions: {profile.custom_instructions}")
    return "\n".join(lines)