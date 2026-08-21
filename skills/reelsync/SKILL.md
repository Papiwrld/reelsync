---
name: reelsync
description: Generate short videos with ReelSync. Create a video from a subject/topic, poll task status, list tasks, or draft a script. Use when the user wants to produce a short video, check on a generation, or write a video script.
---

# ReelSync — Short Video Generation

ReelSync generates AI short videos from a subject or keyword: it writes the
script, gathers materials (stock / AI images / web footage), adds TTS
voiceover, subtitles and BGM, and renders a 9:16 (or 16:9 / 1:1) video.

## Prerequisites

- The ReelSync API server must be running (`http://127.0.0.1:8080`).
- If the app is configured with an `api_key`, requests need
  `Authorization: Bearer <key>`; otherwise no auth is required (local default).

## MCP Tools

The pipeline is exposed over the Model Context Protocol at `/mcp`.

### create_video

Start a generation task. Required: `video_subject`. Optional: `video_script`,
`video_terms` (list), `video_source` (`auto`, `custom_api`, `gemini_image`,
`pexels`, `pixabay`, `coverr`, `pollinations`, `web_scrape`, `local`),
`video_aspect` (`9:16`, `16:9`, `1:1`), `paragraph_number`, `video_duration_seconds`,
`voice_name`, `subtitle_enabled`.

Returns a `task_id`.

### get_task_status

Poll a task. `task_id` required. Returns state, progress, videos, error.

States: `1` = completed, `0` = processing, `-1` = failed.

### list_tasks

List recent tasks. Optional `limit` (default 10).

### generate_script

Draft a script for a subject (no media generated). Required: `video_subject`.

## Workflow guidance

1. Call `create_video` with the subject.
2. Poll `get_task_status` every ~10s until state is `1` (done) or `-1` (failed).
3. On failure, read `failed_stage` + `error` and report to the user.
4. On success, report the video path(s) from `videos` / `output_copies`.

## Manual API examples (REST fallback)

Create a video:

```bash
curl -X POST http://127.0.0.1:8080/api/v1/videos \
  -H "Content-Type: application/json" \
  -d '{"video_subject": "Why cats purr", "video_source": "auto", "video_aspect": "9:16"}'
```

Poll status:

```bash
curl http://127.0.0.1:8080/api/v1/tasks/<task_id>
```

Docs: `http://127.0.0.1:8080/docs`
