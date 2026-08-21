# Changelog

All notable changes to ReelSync are documented in this file.

## [1.4.0] - 2026-08-21

### Added

- **Clip Generator** — turn long videos into vertical shorts (ingest, Whisper transcript, ffmpeg `scdet` scene detection, LLM moment selection, face-tracking crop with center-crop fallback); `face_track` option in `ClipRequest`, the `/clips` API and the WebUI.
- **MCP server** (4 tools at `/mcp`), completion webhooks (HMAC-SHA256, `webhook_url` / `webhook_secret`), agent skill (`skills/reelsync/SKILL.md`).
- **Custom (OpenAI-compatible) LLM provider**, with fallback URL/model selection per key.

### Changed

- LLM fallback fields collapsed into an expander; Clip Generator hidden behind an enable toggle.

### Fixed

- Webhook delivery outcome is now recorded in task state (`webhook_state` + `warnings`).

## [1.3.0] - 2026-08-21

### Added

- **Credential Manager storage** — API keys entered in the WebUI Settings are stored in the OS credential manager (Windows Credential Manager / macOS Keychain / Linux libsecret) instead of plaintext `config.toml`; existing plaintext keys are migrated automatically on first start and blanked from the file on the next save. Keys persist across restarts — no re-entry. Env vars still override per run; `REELSYNC_SKIP_SECRET_MIGRATION=1` opts out.
- **Audio loudness normalization** — final mix is normalized to -14 LUFS (`loudnorm` / MoviePy AudioNormalize) by default, preventing multi-track clipping; opt out with `audio_loudnorm = false`.
- **Configurable frame rate** — `video_fps` (24–60, default 30) via config or `VideoParams.video_fps`.
- **Word-timing scene durations** — per-scene material search now uses real Whisper word timestamps instead of character proportions when available, with automatic fallback.
- **Multi-source web video search** — DuckDuckGo HTML video-search fallback finds TikTok/Instagram/Vimeo etc. footage (with per-platform `site:` filters) when YouTube search is empty or a non-YouTube platform is selected; downloads still go through yt-dlp direct URL (1800+ sites).
- **Distributed rate limiting** — the per-IP API rate limiter uses Redis fixed-window counters when `enable_redis` is on (shared across workers/instances), with automatic in-memory fallback and 30 s backoff.
- **Delete All in Task Manager** — two-step-confirm bulk deletion that skips running tasks; toast summary of deleted/kept counts.
- **Custom output folder** — optional `output_dir` copies finished videos to a user-selected location; `keep_intermediate_clips` controls temp-clip retention.

### Changed

- Whisper defaults upgraded to `large-v3-turbo` with CUDA auto-detection (`device = "auto"`) and OOM fallback to `medium`; libass vector subtitle engine available via `subtitle_engine = "ass"` (PIL remains default).
- LLM provider failures now surface as one-line readable errors (invalid key / rate limit / quota / timeout / unreachable) instead of raw tracebacks, translated in all 9 UI languages.
- Task Manager polling caches script/final-video lookups by mtime, making idle 2-second polls nearly free.

### Fixed

- MoviePy 2.x `write_videofile` no longer receives an invalid `crf` kwarg (CRF is routed through ffmpeg params for libx264 only).
- Concat list paths reject newline injection and validate paths stay inside the task directory; cached material writes are atomic (no more truncated duplicates under concurrency).
- Static `/tasks` mount requires the API key when configured; CORS no longer combines wildcard origins with credentials.
- Windows antivirus file locks no longer cause off-by-one cache cleanup counts (bounded retry).

## [1.2.0] - 2026-08-19

### Added

- **Overlay Studio** — scene-layer overlays composited on top of the video: an optional corner overlay image, title cards, fact/stat callout cards (auto-detected from the script: percentages, currency, big numbers, million/billion, CJK 万/亿 units, years, "according to", data/statistics), and text callouts; configurable via `OverlayParams` in the WebUI (Subtitle Settings tab), CLI flags (`--overlay-*`) and the `/videos` API.
- **Auto clip duration** — set `video_clip_duration=0` to let the pipeline size each clip from the target video duration (2–12 s per clip), resolved once and shared across download, image-convert and combine stages.
- **TikTok / CapCut subtitle presets** — two new one-click styles built on the animated-subtitle engine: TikTok Viral (rounded BeVietnamPro-Bold, all-caps, thick outline, yellow active-word highlight, pop-bounce) and CapCut Clean (Roboto-Bold, rounded pill background, yellow highlight, pop-bounce); selectable in the WebUI, CLI and API in all 9 languages.
- **Interactive product simulation** — a self-contained, backend-free browser demo (`demo/index.html`) that walks the full production workflow with mocked data, plus a deterministic `?record=1` mode and `scripts/record_demo.py` to render `docs/reelsync-demo.gif` for the README; rebuilt to match the actual WebUI layout (left sidebar + 3-tab right panel, real controls and labels).

### Fixed

- Subtitle style presets never applied their background color (the `background` key was skipped in the preset resolver); Minimal and CapCut backgrounds now render.
- Subtitle frames with overlapping time ranges (e.g. Whisper segment boundaries or `correct()` merges) could render two subtitles on screen at once; the phrase chunker now clamps each frame's start to the previous frame's end and merges fully-overlapped frames instead of duplicating them.
- Web-scraped video search no longer returns landscape or low-resolution (<720p short side) footage into portrait tasks, which previously caused black-bar letterboxing and blurry upscales; orientation and resolution are now checked at search time.
- Material ranking weights resolution more strongly so HD/4K footage wins over low-res within similar relevance, without letting a poorly matched high-res clip outrank a well-matched one.

## [1.1.0] - 2026-08-17

### Added

- **Agentic Content Planning** — optional strategy graph (topic analysis → content strategy → hook selection → narrative plan → script → critic with revisions) that replaces the single generic prompt; configured per task in the WebUI with the `max_script_revisions` cap in `[agentic]`.
- **Content Profiles** — reusable creator personas (tone, pacing, hooks, media, captions) that steer the agentic pipeline.
- **Content Intelligence contract** — niche, sub-niche, audience, platform, format and goal drive research depth, fact checking, narrative, titles and visuals; automation levels from manual to autopilot/adaptive-autopilot.
- **Research Orchestrator** — risk-adaptive source discovery, claim extraction and fact verification with disk caching. Providers: model knowledge (explicitly labeled), user notes (deterministic), and an optional SSRF-safe generic web-search provider (`[research]`).
- **Import Profile** — paste a TikTok / Instagram / Facebook / X / YouTube profile link to auto-fill niche, audience and tone from public search snippets and (for YouTube/X/Bilibili, behind the Web Video Scraping toggle) yt-dlp public page metadata; the link also flows into research notes.
- **Discover Topics** — scored topic candidates per niche, adoptable as the video subject in one click.
- **Per-stage agent progress** — the script-generation status widget now shows a live, persisted stage log (niche analysis → research → strategy → hooks → narrative → script + critic revisions → visuals → titles → QA) fed by a progress callback threaded through the agentic planning graph; the stage trail survives page reruns in all 9 languages.
- **User Manual** — a built-in 📖 guide in the top bar covering the workflow and every feature; Content Intelligence settings regrouped into Strategy / Audience & Tone / Research sections with a dedicated Niche field.
- **Startup schema guard** — if the WebUI is updated while running, a clear "restart required" message replaces the stale-module `ValueError` traceback.
- **Subtitle supersampling** — subtitles render at 2× and are LANCZOS-downscaled for crisp small text; new auto subtitle position and dynamic sizing options.
- **Resilience hardening** — Gemini transient 503/429/5xx retry with backoff; Pollinations cache writes to the real cache directory; material-search cache tolerates filesystem clock skew; temp render files keep a real `.mp4` extension so partial encodes never look like finished videos.
- **QA gate** — a blocked (CRITICAL) final review now fails the task with the QA reason instead of merely logging it.
- **LLM circuit breaker** — a provider-level failure (quota exhausted, invalid key, network down) trips a per-run breaker: all subsequent LLM calls in that run short-circuit to their deterministic fallbacks instead of firing doomed requests. Error strings from providers are now treated as failures, never as generated content. Legacy retry count reduced 5→2 so a failing provider is not hammered 5×.
- **LLM fallback API keys** — optional extra keys for the same provider (`llm_fallback_api_keys`, configurable in the WebUI LLM panel). When the primary key fails (quota exhausted, invalid key, rate limit), the next key is tried automatically. Because every LLM call shares one entry point, this covers the entire content creation process — script, keywords, agentic planning, research, QA and titles.
- **Script writing style presets** — choose how the AI words the script: simple & humanized (default), field expert, storytelling, persuasive, educational or casual. Applies to both the linear pipeline and the agentic strategy graph (including script revisions), and is exposed in the WebUI, the `/scripts` API and `VideoParams`.

### Changed

- Script settings panel reordered for a "strategy first, then title" workflow.
- README and README.zh updated with the agentic/research documentation.

## [1.0.0] - 2026-08-16

Initial standalone release. ReelSync is a fork of [MoneyPrinterTurbo](https://github.com/harry0703/MoneyPrinterTurbo) (MIT licensed) with the following additions and changes:

### Added

- **Intelligent multi-source media juggling** — concurrent search across Pexels, Pixabay, Coverr, and custom media APIs via a thread pool, with smart ranking by resolution, duration, and source priority.
- **Custom media API** — plug in any provider returning a list of video URLs (`standard`, `openai`, and `url_list` response formats), with hybrid fallback to Pexels.
- **Web scraping channel** — optional yt-dlp-based footage scraping with subprocess timeouts, process-group cleanup on Windows, and post-download container validation so corrupt downloads fail the material rather than the render.
- **"Mix" transition engine** — true cross-dissolve overlap blends between clips (plus an "Auto" smart transition selector).
- **Ken Burns motion** — pan-and-zoom on static images and AI-generated frames.
- **Agency typography** — bundled Montserrat / Roboto fonts, configurable subtitle colors, stroke outlines, rounded backgrounds, and 2–4 word phrase chunking to avoid single-word flashing.
- **Audio FX & Atmosphere Suite** — atmosphere beds and transition SFX from `resource/sfx/`, and Smart Audio Ducking that lowers background music during speech intervals.
- **Subtitle casing** — per-project text casing (Original / ALL CAPS / Title Case / lowercase) applied before frame rendering.
- **Dark redesigned WebUI** — four-panel workspace, rounded controls, hover motion, task manager, and multilingual UI.
- **CI workflow** — full pytest suite with coverage on every push/PR (Python 3.11 & 3.13 on Ubuntu, plus a Windows smoke job).

### Fixed

- Smart Audio Ducking previously created a second `SubtitlesClip` without `make_textclip`, leaving the ducking timeline empty; it now reuses the already-parsed subtitle timings.
- Scraped downloads are probed with `VideoFileClip` at download time so corrupt containers are rejected before the render stage.
- `uv.lock` version drift between `pyproject.toml` and the lockfile that broke `uv sync --frozen` in CI.

### Changed

- Rebranded from MoneyPrinterTurbo to ReelSync across code, docs, and WebUI.
- Primary README rewritten in English with honest upstream attribution; Chinese translation maintained alongside.

### License

MIT. Upstream: [harry0703/MoneyPrinterTurbo](https://github.com/harry0703/MoneyPrinterTurbo).
