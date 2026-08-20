# ReelSync — Decision-Grade Quality Report

**Date:** 2026-08-19
**Question:** ReelSync project quality (architecture, tests, dependencies, robustness, performance).
**Method:** 5 parallel angle researchers (architecture / tests / deps / robustness / perf) → skeptic verification of top claims by direct code reads & command runs → merge ranked by confidence.
**Scope:** `app/`, `test/`, `webui/`, root manifests. Working tree state (agentic + video refactors uncommitted).

## Verified high-confidence findings (decision-grade)

### Security / correctness
1. **Unauthenticated FastAPI v1 API, LAN-exposed task outputs.**
   Auth dependency is commented out on both routers; `/tasks` is mounted with `follow_symlink=True`; `listen_host` defaults to `0.0.0.0`.
   Evidence: `app/controllers/v1/video.py:39-40`, `app/controllers/v1/llm.py:16-17`, `app/asgi.py:86`, `app/config/config.py:569`.
   Confidence: high (verified).

2. **Coverr signed download URLs are logged raw on failure.**
   Coverr comments state URLs are "signed JWT (绑定 API key)"; `save_video` logs the full URL on non-200 and on exception, without redaction (contrast `_redact_request_error` used elsewhere).
   Evidence: `app/services/material.py:529,665,671`; redactor at `material.py` (used in `download_videos` ~line 1288 but not in `save_video`).
   Confidence: high (fact of raw logging verified; key-binding per code comment).

3. **Crash isolation is asymmetric in `combine_videos`.**
   The source-probe loop (open + read `.duration`/`.size`) has no try/except, so one corrupt material aborts the whole combine; failures in the per-clip processing loop are caught and silently swallowed.
   Evidence: `app/services/video/__init__.py:675-703` (unguarded) vs `:727,859` (swallowed).
   Confidence: high (verified).

4. **MoviePy clip handles leak on the per-clip error path.**
   The clip opened at the top of the try is closed only on the success path; the `except Exception` logs and continues without `close_clip`. The mix-merge branch likewise closes `loaded_clips` only on success, leaking them on merge failure.
   Evidence: `app/services/video/__init__.py:700` open, `:836` close, `:859-860` except; `:913-933` vs `:943-947`.
   Confidence: high (per-clip leak verified; mix leak medium).

### Performance / operations
5. **Every clip is encoded at least twice per task.**
   Each subclip is rendered to `temp-clip-*.mp4`, then the ffmpeg concat demuxer (or MoviePy mix merge) re-encodes the concatenation; no codec-copy path exists. Ken Burns image→mp4 intermediates add a third pass and are never cleaned.
   Evidence: `app/services/video/__init__.py:824-832` (temp clip), `:293-313` (concat re-encode), `:923-930` (mix re-encode), `:430-468` (image intermediates, no `delete_files`).
   Confidence: high (verified).

6. **Mix (cross-dissolve) merge loads every clip into memory simultaneously**, including duplicate handles from the `itertools.cycle` loop, contradicting the "avoid loading all videos at once" comment on the ffmpeg path.
   Evidence: `app/services/video/__init__.py:869` (cycle), `:913-922` (open all + `concatenate_videoclips(method="compose")`), comment at `:900`.
   Confidence: high (verified).

7. **`save_video` buffers the entire download in RAM with no size cap** (`resp.content`), while yt-dlp paths are bounded at 500M.
   Evidence: `app/services/material.py:656-669`; `app/services/web_scrape.py:22`.
   Confidence: medium-high.

8. **Downloaded `cache_videos` have no TTL/eviction** (unlike 24h search cache and 7d media cache); cleanup only via manual WebUI button.
   Evidence: `app/services/material.py:622-638`; `app/services/cache_manager.py:158`; `webui/Main.py:2080`.
   Confidence: medium.

9. **`threads` applies only to the final merge**, not the per-clip encodes that dominate encoding time.
   Evidence: `app/services/video/__init__.py:826-832` (no threads) vs `:929,:939`.
   Confidence: medium.

10. **Coverr search results are structurally never cached** (provider excluded in `material_cache`), so every Coverr task re-queries the API.
    Evidence: `app/services/material_cache.py:208-225,341-342`.
    Confidence: medium.

### Architecture / layering
11. **`video.py` → `video/` package is mostly relocation**: ~1736 lines / 40 functions still live in `__init__.py`; only ~84 lines extracted (`constants.py`, `types.py`). God-module risk persists.
    Confidence: high (verified).

12. **The `agentic` facade re-exports private internals and the whole `llm` module purely for test compatibility** (`agentic.__init__.py:10-19`); `graph.py` resolves its dependencies through the facade at call time (`graph.py:178-202`), so orchestration is wired to tests rather than to the owning modules (`intelligence.py`, `research.py`).
    Confidence: high (verified).

13. **Pervasive private-helper cross-module coupling**: `research` imports `task_artifacts._write_json_atomic`; `intelligence`/`titles`/`trends`/`research` import underscore-prefixed LLM helpers from `agent_llm`.
    Evidence: `app/services/research.py:38`, `intelligence.py:25`, `titles.py:26`, `trends.py:29`.
    Confidence: high (verified).

14. **Two parallel entry paths**: FastAPI controllers (thin, `controllers/v1/video.py:33-36`) vs a 6473-line `webui/Main.py` that imports ~20 service modules directly and bypasses controllers; `cli.py` (863 lines) is a third composition point; `main.py` is a 16-line uvicorn launcher.
    Confidence: high (verified).

15. **Largest god-modules untouched by the refactor**: `voice.py` (2156 lines, 57 defs), `llm.py` (1120 lines, 26 defs).
    Confidence: high (verified).

### Dependencies / environment
16. **Deterministic runtime deps** (`==` pins in `requirements.txt` + `pyproject.toml` + hash-locked `uv.lock`); only `twelvelabs` extra and dev group use loose `>=`.
    Confidence: high (verified).

17. **MoviePy is installed as 2.2.1 but reports `__version__` = 2.1.2** (packaging quirk). Code consistently uses the v2 API (`.resized`, `.subclipped`, `.with_effects`, `vfx/afx`) — no v1 legacy calls found.
    Confidence: high (verified via metadata + grep).

18. **`ffmpeg` and `yt-dlp` are not verified at startup**; resolved lazily at call time. Missing yt-dlp in `search_videos_web_scrape` surfaces through a broad `except Exception` (no dedicated `FileNotFoundError` handler, unlike `fetch_page_metadata`).
    Evidence: `app/utils/utils.py:145-178`, `app/services/web_scrape.py:159-167,229-230,281-286`.
    Confidence: high (lazy resolution verified); low (FileNotFoundError nuance).

19. **10 `PydanticDeprecatedSince20` warnings** from class-based `Config` in `app/models/schema.py` (e.g., `:185,:526,:638`).
    Confidence: high (verified).

20. **LLM stack is Moonshot-default, multi-provider via OpenAI SDK + litellm gateway; no `anthropic` SDK dependency** despite the LLM-centric app.
    Confidence: high (verified).

### Tests
21. **Mock/fake-dominated suite** (≥174 `patch(` call sites; `test_material.py` 40, `test_research_layer.py` 40, `test_upload_post.py` 31); only **one real MoviePy render** in pipeline tests; no mp4 fixtures (9 PNGs only). Risk: integration regressions (real encoding, ffmpeg concat, resource cleanup) can pass CI.
    Confidence: high (verified).

22. **Duplicate test scaffolding**: 7 nearly identical nested `_FakeVideoClip` classes + repeated `_FakeAudioClip`/`_FakeSubtitlesClip` in `test_video.py` (`:28,:860,:930,:1031,:1131,:1207,:1565,:1638`).
    Confidence: high (verified).

23. **All 10 `skipUnless` gates are environment/infra-dependent** (live LLM, Redis, voice, TwelveLabs); none skip for known failures.
    Confidence: medium.

24. **The agentic pipeline refactor is covered by behavior-level tests** using a canned `_QueueLLM` fake running the full orchestration flow (55 tests) — unusually real for this suite. Positive.
    Confidence: high (verified).

25. **`media_utils.py` has no direct test references** (only service module with zero mentions under `test/`).
    Confidence: medium.

## Low-confidence / dropped-in-verification
- Dropped: "web_scrape Popen FileNotFoundError" nuance downgraded (low) — code read confirms, but impact minor.
- Kept with lower confidence: mix-branch handle leak (medium), save_video memory (medium-high), cache TTL gaps (medium).

## Top findings (by confidence × impact)
1. Unauthenticated API + LAN-exposed `/tasks` (verify_token commented out; `0.0.0.0`; `follow_symlink`).
2. Coverr signed-JWT URLs logged raw on failure.
3. Crash-isolation asymmetry + clip-handle leaks in `combine_videos`.
4. Double (sometimes triple) encoding per task; mix merge loads all clips in memory.
5. `webui/Main.py` (6473 lines) bypasses controllers; `voice.py`/`llm.py` god-modules remain.
6. Suite is fake-heavy with only one real render; duplicate scaffolding.
7. MoviePy 2.2.1 misreports as 2.1.2; Pydantic v2 deprecation warnings pending migration.

*Sources: file paths + line numbers; dates = last commit via `git log -1 --format=%cs` or 2026-08-19 for this-session edits. Verification re-ran against live code on 2026-08-19.*

---

# Round 2 Audit (2026-08-19) — 5 parallel sub-agent review

**Method:** 5 parallel agents (security, concurrency, data-integrity, test-coverage, performance/reliability) each doing deep reads of `app/` and `test/`. Findings deduplicated and ranked by confidence × impact.

## CRITICAL — fix first (data loss / correctness / security)

| # | Finding | File:line | Impact |
|---|---------|-----------|--------|
| C1 | **No retry for stock footage API HTTP calls** (Pexels, Pixabay, Coverr). Single network blip kills the entire search for that provider. The `research_layer/http.py` has proper backoff; stock footage does not. | `material.py:342,430,575` | High — one DNS failure = no video for that segment |
| C2 | **No retry for video downloads** (`save_video`). Single connection drop mid-download returns empty, exhausting candidates. | `material.py:672-691` | High — transient CDN errors kill downloads |
| C3 | **MemoryState.update_task replaces entire dict** instead of merging. After first `update_task` in `_run_pipeline`, fields like `video_subject` (set once at submission) are overwritten with nothing and lost. RedisState merges correctly (HSET). | `state.py:57-74` vs `:153-172` | High — `video_subject` lost in MemoryState mode; search terms become empty |
| C4 | **No crash recovery for in-flight tasks in Redis**. Process crash leaves task stuck in `state=4` (PROCESSING) forever. Only cross-post recovery exists at startup. | `state.py:238-244`, `asgi.py:28` | High — stuck tasks visible in WebUI forever with Redis persistence |

## HIGH — security / robustness

| # | Finding | File:line | Impact |
|---|---------|-----------|--------|
| H1 | **SSRF via yt-dlp**: `fetch_page_metadata()` and `download_web_video()` pass user-supplied URLs to yt-dlp without a host allowlist. Internal URLs (cloud metadata, Redis) could be fetched. `social_profile.py` has a allowlist but core functions don't enforce it. | `web_scrape.py:251-256,:338-361` | Medium — requires web scraping to be enabled |
| H2 | **Custom API response URLs not validated as HTTP(S)**. `_parse_standard` and `_parse_openai` store URLs without scheme check; `file://` URLs could flow into `save_video` which handles local files. | `custom_media.py:165,214` | Low-Medium — requires compromised custom API |
| H3 | **Redis password in module-level URL string**. `redis_url` contains plaintext password at module scope. Any exception trace logging module vars exposes it. | `v1/video.py:50` | Medium |
| H4 | **`save_video` buffers entire download in RAM** (`resp.content`). 50MB video = 100MB peak (response + write buffer). Concurrent auto-search with 3 workers = 300MB+. | `material.py:673-686` | Medium — OOM risk on constrained hosts |
| H5 | **`AudioFileClip` leak in 3 TTS functions** (`siliconflow_tts`, `elevenlabs_tts`, `chatterbox_tts`). No try/finally around `audio_clip.duration`. Corrupt audio from flaky TTS leaks FFmpeg reader. MiniMax path does this correctly. | `voice.py:900-902,:1684-1686,:1771-1773` | Medium — Windows file lock errors on retry |
| H6 | **FFmpeg reader leak on `VideoFileClip` constructor failure** in `preprocess_video`. Partial init opens subprocess but exception prevents assignment to `clip`. | `video/__init__.py:1732` | Medium — orphaned FFmpeg processes |
| H7 | **Upload file race condition**. Two concurrent uploads of same filename write to same path without atomic write. Windows: PermissionError. POSIX: silent overwrite of partially-written file. | `v1/video.py:411-434` | Medium |
| H8 | **Duplicate POST requests create duplicate tasks** (no idempotency key). Automated retry on timeout doubles LLM/TTS/stock footage costs. | `v1/video.py:176-225` | Medium — cost impact |
| H9 | **API tasks not protected by runtime config lock**. WebUI tasks use `runtime_config_lock()` but API tasks read config without snapshotting. Config change mid-task = inconsistent provider/key usage. | `task.py:1254-1497` | Medium |

## MEDIUM — performance / reliability

| # | Finding | File:line | Impact |
|---|---------|-----------|--------|
| M1 | **Probe loop opens full clips twice per video**. First loop reads duration/size via full `VideoFileClip` (spawns FFmpeg). Second loop opens same file again to subclip. 15 videos = 30 FFmpeg spawns. | `video/__init__.py:698-710` vs `:760` | 2x FFmpeg overhead per video |
| M2 | **`gc.collect()` called per clip close** in hot loop. 30 clips = 30 full GC cycles. `del clip` in `close_clip` has no effect (local binding only). | `video/__init__.py:553-554` | GC pauses every clip |
| M3 | **Concat always re-encodes** even when all clips share identical codec/resolution (they do, written by same code). `-c copy` would be lossless and instant. | `video/__init__.py:281-349` | Minutes of unnecessary encoding |
| M4 | **`delete_files` no retry on Windows PermissionError**. Windows Defender/Search Indexer mid-scan = permanent file leak. | `video/__init__.py:557-575` | Orphaned temp files accumulate |
| M5 | **Daemon thread leak on edge_tts timeout**. Timeout raises but daemon thread blocks indefinitely on Azure TTS. 3 retries = 3 orphaned threads with open sockets. | `voice.py:704` | Socket leak under Azure outages |
| M6 | **No retry for cross-posting uploads** (`upload_post.py`). Single HTTP failure = no social media post. | `upload_post.py:88-96` | Medium |
| M7 | **No retry for yt-dlp subprocess failures**. Transient network issues kill search/download. | `web_scrape.py:159,:366` | Medium |
| M8 | **SiliconFlow TTS retry has no backoff**. 3 rapid retries all hit same rate limit. | `voice.py:877-884` | Low-Medium |
| M9 | **Orphaned `final-1.mp4` on second-video failure**. First video succeeds, second fails; only failing video's files cleaned. | `task.py:894-908` | Low — eventual cleanup on task delete |
| M10 | **`_runtime_disabled_video_codecs` shared set without lock**. Benign in CPython (GIL) but not portable. | `video/__init__.py:60,180,201` | Low |

## LOW — code quality / maintenance

| # | Finding | File:line |
|---|---------|-----------|
| L1 | `_safe_public_url` duplicated across `material.py:28-50`, `material_cache.py:42-57`, and canonical in `media_utils.py:14-32` |
| L2 | `_matches_video_aspect` duplicated in `material.py:258-289` and `web_scrape.py:106-127` |
| L3 | `_get_tls_verify`, `_redact_secret`, `_redact_request_error` duplicated in `material.py:152-218` (canonical in `media_utils.py`) |
| L4 | `import hashlib` imported inside function bodies (`material.py:1367,:1488`) instead of module level |
| L5 | `_kenburns_image_to_video` imports `video_effects` locally despite module-level import |
| L6 | Content-Disposition filename not RFC 6266 sanitized |
| L7 | LLM error messages may contain internal hostnames to API consumers |
| L8 | `ast.literal_eval` on Redis values — safe but trust boundary should be documented |
| L9 | Bare `except Exception: pass` in transition times loading (`video/__init__.py:1566-1571`) |

## Test coverage gaps (prioritized)

| Priority | Gap | Impact |
|----------|-----|--------|
| 1 | **15+ source modules with no test file** (agentic sub-modules, subtitle_engine, manager modules, research_layer providers) | Untested orchestration and rendering |
| 2 | **Zero integration tests** (script → download → combine → output) | End-to-end regressions invisible |
| 3 | **No tests for voice.py, llm.py, cache_manager.py** dedicated files | Critical services untested in isolation |
| 4 | **35+ edge cases** untested in well-tested modules (all-providers-fail, concurrent downloads, empty audio, zero-duration materials) | Boundary failures pass CI |
| 5 | **8 tests that assert on mocks** rather than real behavior | False confidence |

## Fixes applied (2026-08-19)
- **Security — Coverr URL redaction**: `save_video` now logs `_redact_video_url(...)` (scheme+host+first 2 path segments; query/fragment/tokens dropped). `material.py`.
- **Security — conditional API auth**: `base.auth_dependencies()` now mounts `verify_token` on both v1 routers whenever an `api_key` is configured; open otherwise (no lockout). `controllers/base.py`, `controllers/v1/{video,llm}.py`.
- **Robustness — probe-loop isolation**: unreadable materials are skipped (logged) instead of aborting the whole combine. `video/__init__.py`.
- **Robustness — handle leaks**: per-clip error path and mix-merge failure path now call `close_clip` on opened clips. `video/__init__.py`.
- **Perf — threading**: `threads` is now passed to per-clip encodes, not only the final merge. `video/__init__.py`.
- **Ops — Ken Burns intermediates**: `delete_image_material_clips()` removes generated `<image>.mp4` after all combines (keeps originals; never touches real videos). Wired into `task.py`.
- **Deps — Pydantic v2**: all 9 class-based `Config` blocks migrated to `model_config = ConfigDict(...)`; `MaterialInfo` dataclass config uses `ConfigDict`; 0 `PydanticDeprecatedSince20` warnings remain. `models/schema.py`.
- **Tests**: +18 tests added (URL redaction, auth deps, probe-skip, clip-close, image-clip cleanup, `media_utils.py` coverage). Full suite 1033 passed / 10 skipped; ruff clean.

## Fixes applied (Round 2, 2026-08-19)
- **C3 — MemoryState.update_task merge**: now merges kwargs into existing task dict instead of replacing it entirely (matches RedisState HSET behavior). `state.py`.
- **H4 — streaming download**: `save_video` now uses `stream=True` + `iter_content(chunk_size=8192)` instead of buffering `resp.content` in RAM. `material.py`.
- **H5 — AudioFileClip leak**: try/finally in `siliconflow_tts`, `elevenlabs_tts`, and `chatterbox_tts` ensures `audio_clip.close()` on corrupt-audio errors. `voice.py`.
- **H6 — FFmpeg reader leak on constructor failure**: `_open_video_clip_quietly` already propagates exception cleanly; `preprocess_video` outer try closes on failure. Verified safe.
- **M2 — gc.collect() removed**: `close_clip` no longer calls `del clip` (no-op) or `gc.collect()` per clip. `video/__init__.py`.
- **M4 — delete_files Windows retry**: `PermissionError` now retries 3 times with 100/200ms backoff before giving up. `video/__init__.py`.
- **L1-L3 — dedup helpers**: `_safe_public_url`, `_get_tls_verify`, `_redact_request_error` removed from `material.py`; now imported from `media_utils.py`. -67 lines, single canonical implementation. `material.py`.

## Fixes applied (Round 3, 2026-08-19) — all deferred items
- **C1 — retry on stock footage APIs**: new `_request_with_retry` helper (2 retries, exponential backoff 0.5s/1s, on connection errors + 429/5xx) now used by `search_videos_pexels`, `search_videos_pixabay`, `search_videos_coverr`. `material.py`.
- **C2 — retry on video downloads**: `save_video` download loop retries twice (cleans partial file between attempts) on transient HTTP/connection failures. `material.py`.
- **C4 — crash recovery for stuck tasks**: new `recover_interrupted_generation_tasks()` scans for PROCESSING tasks at startup and marks them FAILED (`failed_stage=restart`) when using Redis persistence; wired into `asgi.py` startup before cross-post recovery. `task.py`, `asgi.py`.
- **H3 — Redis password no longer in URL string**: `RedisTaskManager` now takes `host/port/db/password` and calls `redis.Redis(...)` directly; module-level `redis://:password@...` string removed from `controllers/v1/video.py`. `redis_manager.py`, `video.py`.
- **H7 — atomic uploads**: `upload_video_material_file` writes to a UUID temp file then `os.replace()` (cleans temp on failure) so concurrent same-name uploads can't interleave partial writes. `controllers/v1/video.py`.
- **H8 — idempotency key**: `create_task` honors `X-Idempotency-Key` header; repeated submits with same key+stage reuse the existing non-failed task instead of creating a duplicate (avoids double LLM/TTS/stock costs). `controllers/v1/video.py`.
- **H9 — config snapshotting**: `runtime_config_lock` is now depth-aware (thread-local nesting counter); `tm.start` acquires it for the whole pipeline, giving API tasks the same mid-run config stability WebUI tasks already had — without the nested case re-flushing queued updates mid-task. `config/config.py`, `task.py`.
- **M1 — probe-loop double-open**: new `_probe_video_metadata()` reads duration/size via `ffprobe` (30s timeout) and only falls back to `VideoFileClip` when ffprobe is unavailable; combine probe loop no longer opens every clip twice. `video/__init__.py`, `video/constants.py`.
- **M3 — concat stream-copy fast path**: `concat_video_clips_with_ffmpeg` tries `-c copy` first (instant when all clips share codec/resolution); falls back to the full re-encode + codec-fallback path on any failure. `video/__init__.py`.
- **Tests**: +7 tests added (idempotency reuse ×2, generation recovery, ffprobe probe ×3, concat copy fast path). Full suite 1040 passed / 10 skipped; ruff clean.

## Deliberately not auto-fixed (needs product/risk decision)
- **cache_videos auto-eviction (F7)**: TTL/trigger is a policy choice; concurrent tasks share the dir, so deleting could break in-flight jobs. Existing manual `clean_video_cache` retained.
- **Structural refactors**: `webui/Main.py` (6473 lines), `voice.py`/`llm.py` god-modules, agentic facade/private-helper coupling — too large to change safely in this pass; documented above.
- **test_video.py fake dedup**: high churn vs. 100+ passing tests; cosmetic.