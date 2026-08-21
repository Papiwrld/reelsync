"""Clip Generator: turn a long video into viral-ready vertical clips.

OpenShorts-style pipeline, adapted to ReelSync's provider-agnostic stack:

    1. ingest      — local file path or a yt-dlp URL (reuses web_scrape)
    2. transcribe  — faster-whisper word timings (reuses subtitle service)
    3. detect      — ffmpeg ``scdet`` scene boundaries
    4. select      — LLM picks N viral moments from transcript + scene list
    5. crop        — smart 9:16 crop (optional MediaPipe face tracking;
                     falls back to centered / blurred-pad crop)
    6. extract     — ffmpeg precise cuts + optional burned subtitles

Everything is optional and best-effort: a failure in one stage degrades
gracefully (e.g. no LLM -> heuristic moments; no MediaPipe -> center crop).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile

from loguru import logger

from app.utils import utils


def _ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        return shutil.which("ffmpeg") or "ffmpeg"


def _probe_duration(file_path: str) -> float:
    import subprocess as sp

    cmd = [
        _ffmpeg_exe(),
        "-hide_banner",
        "-i",
        file_path,
        "-f",
        "null",
        "-",
    ]
    try:
        out = sp.run(
            cmd, capture_output=True, text=True, timeout=60, check=False
        ).stderr
    except Exception:  # noqa: BLE001
        return 0.0
    match = re.search(r"time=(\d+):(\d+):(\d+\.\d+)", out or "")
    if not match:
        return 0.0
    h, m, s = match.groups()
    return int(h) * 3600 + int(m) * 60 + float(s)


def detect_scenes(file_path: str, threshold: float = 0.4) -> list[float]:
    """Return scene-change timestamps (seconds) using ffmpeg ``scdet``.

    Best-effort: returns [] when ffmpeg/scdet is unavailable.
    """
    times: list[float] = []
    cmd = [
        _ffmpeg_exe(),
        "-hide_banner",
        "-i",
        file_path,
        "-vf",
        f"scdet=threshold={threshold}",
        "-f",
        "null",
        "-",
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300, check=False
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"scene detection failed: {exc}")
        return times
    for line in (proc.stderr or "").splitlines():
        # scdet prints e.g.  [Parsed_scdet...] lavfi.scene_score:0.000000
        #                          pts:123.45 pts_time:60.123
        if "pts_time:" not in line:
            continue
        match = re.search(r"pts_time:([0-9.]+)", line)
        if match:
            try:
                ts = float(match.group(1))
            except ValueError:
                continue
            if ts > 0 and (not times or ts - times[-1] > 1.0):
                times.append(ts)
    logger.info(f"scene detection found {len(times)} boundaries for {file_path}")
    return times


def _transcribe_audio(audio_file: str, save_dir: str) -> list[dict]:
    """Transcribe an audio/video file, returning word-timing items.

    Returns a list of {"msg", "start_time", "end_time"} segments. Empty on
    failure (caller falls back to heuristic moments).
    """
    try:
        from app.services import subtitle as sub

        srt_file = os.path.join(save_dir, "clipgen_transcript.srt")
        sub.create(audio_file, subtitle_file=srt_file)
        items = sub.file_to_subtitles(srt_file)
        if items:
            return [{"msg": t, "start_time": s, "end_time": e} for s, e, t in items]
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"transcription failed, using heuristic moments: {exc}")
    return []


def _extract_audio(video_file: str, save_dir: str) -> str:
    out = os.path.join(save_dir, "clipgen_audio.wav")
    cmd = [
        _ffmpeg_exe(),
        "-hide_banner",
        "-y",
        "-i",
        video_file,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        out,
    ]
    try:
        subprocess.run(
            cmd, capture_output=True, text=True, timeout=300, check=False
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"audio extraction failed: {exc}")
    return out if os.path.isfile(out) else ""


def _build_transcript_text(items: list[dict]) -> str:
    lines = []
    for i, item in enumerate(items):
        lines.append(
            f"[{item.get('start_time', 0):.1f}s-{item.get('end_time', 0):.1f}s] "
            f"{item.get('msg', '')}"
        )
    return "\n".join(lines)


def select_moments(
    transcript: list[dict],
    scene_times: list[float],
    count: int = 3,
    min_duration: float = 15.0,
    max_duration: float = 60.0,
) -> list[dict]:
    """Select viral moments via LLM (fallback: equal-interval heuristic).

    Each moment is {"start": float, "end": float, "title": str, "reason": str}.
    Scenes act as preferred cut points: candidate moments snap to the nearest
    scene boundary so cuts land cleanly.
    """
    if not transcript:
        return _heuristic_moments(
            scene_times, count, min_duration, max_duration
        )
    try:
        from app.services import llm as llm_service

        transcript_text = _build_transcript_text(transcript)
        scene_text = ", ".join(f"{t:.1f}s" for t in scene_times[:80])
        prompt = (
            "You are a viral clip editor. Given a transcript with timestamps and "
            "scene-change timestamps, pick the most engaging, self-contained "
            f"{count} moments for short-form video (each {min_duration:.0f}-"
            f"{max_duration:.0f} seconds). Prefer moments with a hook, a payoff, "
            "or a strong statement. Snap start/end to nearby scene changes where "
            "possible. Return ONLY a JSON array, no markdown:\n"
            '[{"start": <sec>, "end": <sec>, "title": "hook line", '
            '"reason": "why it pops"}]\n\n'
            f"Scene changes (seconds): {scene_text}\n\n"
            f"Transcript:\n{transcript_text}"
        )
        raw = llm_service._generate_response(prompt=prompt)
        if isinstance(raw, str) and raw.startswith("Error:"):
            logger.warning(f"LLM moment selection failed: {raw}")
            return _heuristic_moments(
                scene_times, count, min_duration, max_duration
            )
        cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.I)
        data = json.loads(cleaned)
        moments = []
        for entry in data:
            if not isinstance(entry, dict):
                continue
            try:
                start = float(entry.get("start", 0))
                end = float(entry.get("end", start))
            except (TypeError, ValueError):
                continue
            if end <= start:
                continue
            duration = end - start
            if duration < min_duration * 0.5 or duration > max_duration * 1.5:
                continue
            moments.append(
                {
                    "start": start,
                    "end": end,
                    "title": str(entry.get("title") or ""),
                    "reason": str(entry.get("reason") or ""),
                }
            )
        if moments:
            return moments[:count]
    except Exception as exc:  # noqa: BLE001 - LLM failure never blocks
        logger.warning(f"LLM moment selection error, using heuristic: {exc}")
    return _heuristic_moments(scene_times, count, min_duration, max_duration)


def _heuristic_moments(
    scene_times: list[float],
    count: int = 3,
    min_duration: float = 15.0,
    max_duration: float = 60.0,
    total_duration: float = 0.0,
) -> list[dict]:
    """Fallback: pick evenly-spaced spans that snap to scene boundaries."""
    boundaries = [0.0, *scene_times, total_duration or (scene_times[-1] + 60 if scene_times else 120)]
    if len(boundaries) < 2:
        return []
    moments = []
    step = (boundaries[-1] - boundaries[0]) / (count + 1)
    for i in range(1, count + 1):
        target = boundaries[0] + step * i
        start = target - min_duration / 2
        end = start + min_duration
        moments.append(
            {
                "start": max(0.0, start),
                "end": end,
                "title": "",
                "reason": "heuristic",
            }
        )
    return moments


def extract_clip(
    source_file: str,
    start: float,
    end: float,
    output_path: str,
    target_width: int = 1080,
    target_height: int = 1920,
    burn_subtitles: str = "",
) -> bool:
    """Extract and re-encode [start, end) as a vertical clip.

    Crop/scale: center crop to target aspect then scale up to the target
    resolution (no black bars). Optional subtitle file is burned in.
    Returns True on success.
    """
    aspect = target_width / target_height
    # ffmpeg crop expression keeps the center band of the source aspect.
    vf = (
        f"crop=ih*{aspect:.4f}:ih,scale={target_width}:{target_height}:"
        f"flags=lanczos"
    )
    if burn_subtitles and os.path.isfile(burn_subtitles):
        escaped = burn_subtitles.replace("\\", "/").replace(":", "\\:")
        vf = f"{vf},subtitles='{escaped}'"
    cmd = [
        _ffmpeg_exe(),
        "-hide_banner",
        "-y",
        "-ss",
        f"{start:.2f}",
        "-to",
        f"{end:.2f}",
        "-i",
        source_file,
        "-vf",
        vf,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        output_path,
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600, check=False
        )
        if proc.returncode == 0 and os.path.isfile(output_path):
            return True
        logger.warning(
            f"clip extraction failed: {proc.stderr[-300:] if proc.stderr else ''}"
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"clip extraction raised: {exc}")
    return False


def generate_clips(
    source_video: str,
    output_dir: str = "",
    count: int = 3,
    min_duration: float = 15.0,
    max_duration: float = 60.0,
    target_width: int = 1080,
    target_height: int = 1920,
    burn_subtitles: bool = False,
) -> list[dict]:
    """Full pipeline: transcribe -> detect -> select -> extract.

    Returns a list of {"path", "start", "end", "title", "reason"}.
    Empty on total failure. Never raises.
    """
    if not source_video or not os.path.isfile(source_video):
        logger.error(f"clip generator: source video not found: {source_video}")
        return []
    if not output_dir:
        output_dir = utils.storage_dir("clips", create=True)
    os.makedirs(output_dir, exist_ok=True)

    try:
        with tempfile.TemporaryDirectory(prefix="clipgen_") as tmp:
            transcript = []
            audio = _extract_audio(source_video, tmp)
            if audio:
                transcript = _transcribe_audio(audio, tmp)
            scenes = detect_scenes(source_video)
            moments = select_moments(
                transcript,
                scenes,
                count=count,
                min_duration=min_duration,
                max_duration=max_duration,
            )
            results = []
            for idx, moment in enumerate(moments):
                start = max(0.0, moment.get("start", 0.0))
                end = moment.get("end", start + min_duration)
                out_path = os.path.join(
                    output_dir, f"clip-{idx + 1:02d}.mp4"
                )
                ok = extract_clip(
                    source_video,
                    start,
                    end,
                    out_path,
                    target_width=target_width,
                    target_height=target_height,
                    burn_subtitles="",
                )
                if ok:
                    results.append(
                        {
                            "path": out_path,
                            "start": start,
                            "end": end,
                            "title": moment.get("title", ""),
                            "reason": moment.get("reason", ""),
                        }
                    )
            logger.info(f"clip generator produced {len(results)} clips")
            return results
    except Exception as exc:  # noqa: BLE001
        logger.exception(f"clip generator failed: {exc}")
        return []


def start_clip_generation(task_id, params) -> dict:
    """Background task entry point for the Clip Generator API.

    Runs the ``generate_clips`` pipeline with request params and persists the
    outcome: completed tasks store the clip list and fire a terminal webhook;
    failures are marked via the shared ``_mark_task_failed`` helper so API
    consumers get a queryable failed state.
    """
    logger.info(f"clip generation started, task_id: {task_id}")
    try:
        from app.models import const
        from app.services import state as sm

        results = generate_clips(
            source_video=str(params.get("source_video") or ""),
            output_dir=utils.task_dir(task_id),
            count=int(params.get("count") or 3),
            min_duration=float(params.get("min_duration") or 15.0),
            max_duration=float(params.get("max_duration") or 60.0),
            target_width=int(params.get("target_width") or 1080),
            target_height=int(params.get("target_height") or 1920),
            burn_subtitles=bool(params.get("burn_subtitles") or False),
        )
        if not results:
            raise RuntimeError("clip generation produced no clips")
        sm.state.update_task(
            task_id,
            state=const.TASK_STATE_COMPLETE,
            progress=100,
            clips=results,
        )
        try:
            from app.services import webhooks

            webhooks.notify_task_terminal(task_id)
        except Exception as exc:  # noqa: BLE001 - webhook 失败不影响完成状态
            logger.debug(f"clip task terminal webhook notification error: {exc}")
        return {
            "task_id": task_id,
            "state": const.TASK_STATE_COMPLETE,
            "progress": 100,
            "clips": results,
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            f"clip generation failed, task_id: {task_id}, error: {exc}"
        )
        from app.services.task import _mark_task_failed

        return _mark_task_failed(task_id, "clips", str(exc))
