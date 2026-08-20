import glob
import json
import os
import re
import signal
import subprocess
import sys
from typing import Any, List

from loguru import logger

from app.models.schema import MaterialInfo, VideoAspect

# yt-dlp 搜索只做 JSON 元数据探测，超时窗口更短；下载需要拉取音视频流并
# 交给 ffmpeg 合并，允许更长的等待时间。
_SEARCH_TIMEOUT_SECONDS = 60
_DOWNLOAD_TIMEOUT_SECONDS = 300

def _get_max_height() -> int:
    """读取可配置的抓取分辨率上限，支持 4K（2160）。"""
    try:
        from app.config import config

        raw = int(config.app.get("web_scrape_max_height", _MAX_HEIGHT_DEFAULT) or _MAX_HEIGHT_DEFAULT)
        return max(720, min(4320, raw))
    except Exception:
        return _MAX_HEIGHT_DEFAULT


def _get_max_filesize() -> str:
    try:
        from app.config import config

        raw = str(config.app.get("web_scrape_max_filesize", _MAX_FILESIZE_DEFAULT) or _MAX_FILESIZE_DEFAULT).strip()
        return raw if raw else _MAX_FILESIZE_DEFAULT
    except Exception:
        return _MAX_FILESIZE_DEFAULT


_MAX_HEIGHT_DEFAULT = 1080
_MAX_FILESIZE_DEFAULT = "500M"
# 兼容旧常量：实际使用 _get_max_height() 动态读取，低端设备仍默认 1080p，
# 高配用户可在 config.toml 设置 web_scrape_max_height = 2160 以启用 4K。
_MAX_HEIGHT = 1080
_MAX_FILESIZE = "500M"

# 搜索阶段就过滤掉方向不符或过低清的素材，避免把横屏 YouTube 视频塞进
# 竖屏任务（成片出现黑边），也避免把 360p 内容放大到 1080p 变糊。
_MIN_MATERIAL_DIMENSION = 720

# yt-dlp 中断时会留下 <output>.part、<output>.ytdl，以及分片下载产生的
# <output>.f<id>.<ext>.part 等临时文件。统一按输出路径前缀清理。
_PARTIAL_FILE_SUFFIXES = (".part", ".ytdl")


def _is_windows() -> bool:
    return sys.platform == "win32"


def _popen_kwargs() -> dict:
    """
    让 yt-dlp 运行在独立的进程组/会话中。

    POSIX 下 ``start_new_session=True`` 使子进程成为新的会话和进程组组长，
    超时后才能用 ``killpg`` 一次终止整组（包含 yt-dlp 派生的 ffmpeg）；
    Windows 下使用独立进程组，配合 ``taskkill /T`` 按进程树终止。
    """
    if _is_windows():
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def _kill_process_tree(process: subprocess.Popen) -> None:
    """
    强制终止 yt-dlp 及其派生进程，并回收进程句柄防止僵尸进程。

    Windows 的 ``TerminateProcess`` 只终止直接子进程；yt-dlp 通过 ffmpeg
    合并音视频时，直接 kill 会留下孤儿 ffmpeg 继续写盘。这里在 Windows 上
    用 ``taskkill /PID <pid> /T /F`` 按进程树终止，POSIX 上使用 ``killpg``
    一次性终止整个进程组。最后通过 ``wait()`` 收割进程，避免超时之后留下
    僵尸进程。
    """
    if process.poll() is not None:
        return
    try:
        if _is_windows():
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=15,
            )
        else:
            os.killpg(process.pid, signal.SIGKILL)
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning(f"failed to kill yt-dlp process tree: {exc}")
    finally:
        try:
            process.wait()
        except OSError:
            pass


def _remove_partial_files(output_path: str) -> None:
    """
    清理 yt-dlp 为同一输出路径遗留的临时/分片文件。

    下载开始前清理历史残留，超时或失败后再次清理，避免中断的任务反复占用
    磁盘或干扰后续重试。yt-dlp 只有在完整下载并合并成功后才把临时文件原子
    改名为最终文件，因此成功路径不会误删任何内容。
    """
    candidates = [f"{output_path}{suffix}" for suffix in _PARTIAL_FILE_SUFFIXES]
    # 分片下载会生成 <output>.f<id>.<ext>.part；glob 按输出路径前缀匹配，
    # 避免误删其它任务的文件。
    candidates.extend(glob.glob(f"{glob.escape(output_path)}.*.part"))
    for candidate in candidates:
        try:
            os.remove(candidate)
            logger.debug(f"removed stale yt-dlp temp file: {candidate}")
        except FileNotFoundError:
            continue
        except OSError as exc:
            logger.warning(
                f"failed to remove stale yt-dlp temp file {candidate}: {exc}"
            )


def _matches_video_aspect(
    width: Any,
    height: Any,
    video_aspect: VideoAspect,
) -> bool:
    """判断 yt-dlp 返回的素材方向是否与目标一致（与 material.py 同口径）。

    无法确认方向的素材直接跳过，避免竖屏任务混入横屏素材并在成片中产生黑边。
    """
    aspect = VideoAspect(video_aspect)
    try:
        normalized_width = int(float(width))
        normalized_height = int(float(height))
    except (TypeError, ValueError):
        return False
    if normalized_width <= 0 or normalized_height <= 0:
        return False
    if aspect == VideoAspect.portrait:
        return normalized_height > normalized_width
    if aspect == VideoAspect.landscape:
        return normalized_width > normalized_height
    return normalized_width == normalized_height


def _rewrite_query_for_web_search(search_term: str) -> str:
    """Use LLM to rewrite a scene search term into a high-recall YouTube/TikTok query.

    Stock search terms are concrete phrases like 'growing bar chart' — good for
    Pexels but too narrow for YouTube. The rewrite expands to natural language
    that matches video titles, e.g. 'growing bar chart' → 'business chart growing animation'.
    Falls back to original term on any LLM failure (never blocks scraping).
    """
    try:
        from app.config import config
        from app.services import llm as _llm

        # Only rewrite when LLM is configured; otherwise keep original
        if not config.app.get("llm_provider") and not config.app.get("openai_api_key") and not config.app.get("gemini_api_key"):
            return search_term
        prompt = (
            "Rewrite this stock-footage search phrase into a short YouTube/TikTok search query "
            "(3-6 words) that will find real videos matching the scene. Keep it concrete and visual, "
            "no abstract concepts. Return ONLY the rewritten query, nothing else.\n"
            f"Phrase: \"{search_term}\""
        )
        rewritten = _llm._generate_response(prompt=prompt)
        if isinstance(rewritten, str) and rewritten.startswith("Error:"):
            return search_term
        cleaned = re.sub(r'^["\']|["\']$', "", (rewritten or "").strip().splitlines()[0].strip())
        if 3 <= len(cleaned) <= 80 and len(cleaned.split()) <= 8:
            logger.info(f"web search query rewritten: {search_term!r} -> {cleaned!r}")
            return cleaned
    except Exception as exc:
        logger.debug(f"web query rewrite fallback: {exc}")
    return search_term


def _rank_web_results_by_relevance(
    results: List[MaterialInfo], search_term: str, top_k: int = 5
) -> List[MaterialInfo]:
    """LLM re-rank web candidates by title/description relevance to the scene.

    Keeps top_k most relevant. Falls back to original order on LLM failure.
    """
    if len(results) <= top_k:
        return results
    try:
        from app.config import config
        from app.services import llm as _llm
        import json as _json

        if not config.app.get("llm_provider") and not config.app.get("openai_api_key") and not config.app.get("gemini_api_key"):
            return results[:top_k]

        # Build compact candidate list for LLM
        candidates = []
        for idx, r in enumerate(results[:10]):
            info = r.source_info or {}
            candidates.append(
                {"idx": idx, "title": info.get("title", "")[:80], "desc": info.get("description", "")[:120]}
            )
        prompt = (
            "Rank these YouTube candidates by how well they visually match the scene query. "
            "Return ONLY a JSON array of indices in ranked order, most relevant first.\n"
            f"Query: \"{search_term}\"\n"
            f"Candidates: {_json.dumps(candidates, ensure_ascii=False)}\n"
            "Example: [2,0,1]"
        )
        raw = _llm._generate_response(prompt=prompt)
        if isinstance(raw, str) and raw.startswith("Error:"):
            return results[:top_k]
        # Extract JSON array
        m = re.search(r"\[[\d,\s]+\]", raw or "")
        if not m:
            return results[:top_k]
        order = json.loads(m.group(0))
        ranked = []
        seen = set()
        for idx in order:
            if isinstance(idx, int) and 0 <= idx < len(results) and idx not in seen:
                ranked.append(results[idx])
                seen.add(idx)
        # Append any missing in original order
        for idx, r in enumerate(results):
            if idx not in seen:
                ranked.append(r)
        logger.info(f"web results re-ranked for {search_term!r}: {[r.source_info.get('title','')[:30] for r in ranked[:top_k]]}")
        return ranked[:top_k]
    except Exception as exc:
        logger.debug(f"web rank fallback: {exc}")
        return results[:top_k]


def search_videos_web_scrape(
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect,
) -> List[MaterialInfo]:
    """
    Search for web videos using yt-dlp. Returns the top 5 results matching the search term.

    Highly intelligent mode: rewrites the query via LLM for YouTube/TikTok recall and
    re-ranks candidates by title/description relevance. Falls back gracefully when
    LLM is not configured.
    """
    # Intelligent query rewrite for better YouTube/TikTok recall
    effective_term = _rewrite_query_for_web_search(search_term)
    logger.info(f"Searching web videos for {search_term!r} (effective: {effective_term!r}) via yt-dlp")
    results = []

    # ytsearch5: keyword limits search to 5 results
    command = [
        "yt-dlp",
        f"ytsearch5:{effective_term}",
        "--dump-json",
        "--default-search",
        "ytsearch",
        "--no-playlist",
        "--match-filter",
        f"duration >= {minimum_duration}",
        "--max-filesize",
        _get_max_filesize(),
        "--ignore-errors",
    ]

    try:
        # Run yt-dlp in its own process group so a timeout can kill the whole
        # tree (yt-dlp may spawn ffmpeg for probing).
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            **_popen_kwargs(),
        )
        try:
            stdout, stderr = process.communicate(timeout=_SEARCH_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            # 终止整个进程组并收割句柄，避免超时后留下僵尸 yt-dlp/ffmpeg。
            _kill_process_tree(process)
            process.communicate()
            logger.error(f"yt-dlp search timed out after {_SEARCH_TIMEOUT_SECONDS}s")
            return []

        for line in stdout.splitlines():
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                url = data.get("webpage_url") or data.get("url")
                duration = data.get("duration", minimum_duration)

                # yt-dlp resolution is sometimes widthxheight, or just width
                width = data.get("width")
                height = data.get("height")

                # 搜索阶段就过滤方向不符或低于 _MIN_MATERIAL_DIMENSION 的素材，
                # 否则横屏/低清内容会混入竖屏成片（黑边）或放大后变糊。
                if not _matches_video_aspect(width, height, video_aspect):
                    logger.debug(
                        f"skip web material with mismatched orientation: "
                        f"term={search_term!r}, size={width}x{height}"
                    )
                    continue
                try:
                    short_side = min(int(float(width)), int(float(height)))
                except (TypeError, ValueError):
                    continue
                if short_side < _MIN_MATERIAL_DIMENSION:
                    logger.debug(
                        f"skip web material below minimum resolution: "
                        f"term={search_term!r}, size={width}x{height}, "
                        f"minimum={_MIN_MATERIAL_DIMENSION}"
                    )
                    continue

                resolution = f"{width}x{height}" if width and height else ""

                if url:
                    results.append(
                        MaterialInfo(
                            provider="web_scrape",
                            url=url,
                            duration=duration,
                            resolution=resolution,
                            source_info={
                                "provider": "web_scrape",
                                "search_term": search_term,
                                "title": str(data.get("title") or ""),
                                "description": str(data.get("description") or ""),
                            },
                        )
                    )
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse yt-dlp json: {line}")

    except Exception as e:
        logger.error(f"Error executing yt-dlp: {e}")

    # Highly intelligent re-ranking: keep top-5 most scene-relevant
    if results:
        results = _rank_web_results_by_relevance(results, search_term, top_k=5)

    return results


# 主页/资料页元数据探测只需要 JSON 元数据，不拉取音视频流，超时窗口更短。
_METADATA_TIMEOUT_SECONDS = 20


def fetch_page_metadata(url: str, proxy: str = "") -> dict:
    """Fetch public page/profile metadata via yt-dlp (metadata only, no download).

    Runs ``yt-dlp --dump-json --skip-download`` on a user-supplied public URL
    (e.g. a YouTube channel or X profile) and returns a safe subset of the
    extracted metadata. Returns ``{}`` on any failure — this is a best-effort
    convenience path, never a blocker. ``proxy`` (optional) is forwarded to
    yt-dlp via ``--proxy`` so regions that block these sites can still fetch
    through the app's configured proxy. The caller is responsible for host
    allowlisting and for the user consent toggle (Web UI "Enable Web Video
    Scraping").
    """
    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        logger.debug(f"refusing metadata fetch for non-http(s) URL: {url!r}")
        return {}
    command = [
        "yt-dlp",
        url,
        "--dump-json",
        "--skip-download",
        "--no-playlist",
        "--ignore-errors",
        "--no-warnings",
        "--retries",
        "1",
        "--socket-timeout",
        "10",
    ]
    if proxy:
        command += ["--proxy", proxy]
    process = None
    try:
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                **_popen_kwargs(),
            )
        except FileNotFoundError:
            logger.warning(
                "yt-dlp executable not found; install yt-dlp or add it to PATH "
                "to enable public page metadata"
            )
            return {}
        try:
            stdout, _stderr = process.communicate(timeout=_METADATA_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            _kill_process_tree(process)
            process.communicate()
            logger.warning(f"yt-dlp metadata fetch timed out for: {url}")
            return {}
        if process.returncode != 0:
            logger.debug(f"yt-dlp metadata fetch failed for {url}")
            return {}
        first_line = next(
            (line for line in stdout.splitlines() if line.strip()), ""
        )
        if not first_line:
            return {}
        data = json.loads(first_line)
        if not isinstance(data, dict):
            return {}
        return {
            "title": str(data.get("title") or ""),
            # 频道/账号简介（bio/about）是推断内容赛道最重要的公开信号。
            "description": str(data.get("description") or ""),
            "channel": str(data.get("channel") or data.get("uploader") or ""),
            "channel_id": str(data.get("channel_id") or ""),
            # 公开的订阅/关注数，帮助判断账号规模与受众定位（不会用作用户画像）。
            "channel_follower_count": int(
                data.get("channel_follower_count") or 0
            ),
            "categories": [str(item) for item in (data.get("categories") or [])][:6],
            "tags": [str(tag) for tag in (data.get("tags") or [])][:10],
            "webpage_url": str(data.get("webpage_url") or ""),
        }
    except json.JSONDecodeError:
        logger.warning(f"yt-dlp metadata JSON parse failed for: {url}")
        return {}
    except Exception as exc:  # noqa: BLE001 - metadata fetch must never raise
        logger.warning(f"yt-dlp metadata fetch error for {url}: {exc}")
        return {}


def _maybe_remove_watermark(input_path: str) -> bool:
    """Optionally remove platform watermark via ffmpeg delogo (only when explicitly enabled).

    Default: disabled. When `app.enable_watermark_removal` is true, applies a
    generic delogo at bottom-right (common TikTok/YouTube Shorts watermark area).
    This is intended ONLY for user-owned content where you have the rights.
    Removing watermarks from copyrighted stock violates ToS — disabled by default.
    Returns True if removal was attempted.
    """
    try:
        from app.config import config

        if not config.app.get("enable_watermark_removal"):
            return False
    except Exception:
        return False

    # Generic bottom-right delogo (10-20% of frame) — not pixel-perfect but hides
    # most platform watermarks without cropping. Uses ffmpeg delogo filter.
    try:
        import shutil

        tmp = input_path + ".nowm.mp4"
        # delogo x/y/w/h as fractions via expressions (works for any resolution)
        # x=W-w*0.22:y=H-h*0.14:w=w*0.20:h=h*0.12:show=0
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-vf",
            "delogo=x=W-w*0.22:y=H-h*0.14:w=w*0.20:h=h*0.12:show=0",
            "-c:a",
            "copy",
            tmp,
        ]
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
        if proc.returncode == 0 and os.path.isfile(tmp) and os.path.getsize(tmp) > 0:
            shutil.move(tmp, input_path)
            logger.info(f"watermark delogo applied: {input_path}")
            return True
        try:
            os.remove(tmp)
        except Exception:
            pass
    except Exception as exc:
        logger.warning(f"watermark removal failed: {exc}")
    return False


def download_web_video(url: str, output_path: str) -> bool:
    """
    Download a single video using yt-dlp to the specified output path.

    Supports any yt-dlp extractor (YouTube, TikTok, Instagram, Twitter/X, etc. — 1800+ sites).
    Pass a direct video URL (e.g. https://www.tiktok.com/... or https://youtube.com/watch?v=...)
    to fetch that exact video; search mode uses YouTube via ytsearch.

    只有目标文件存在且非空时才返回 True。失败、超时或没有产出时清理残留的
    ``.part``/``.ytdl`` 临时文件，避免磁盘占用和后续重试被残留文件干扰。
    """
    logger.info(f"Downloading web video via yt-dlp: {url} to {output_path}")

    # 只接受 http/https 素材地址。file:// 等本地路径或选项形输入（以 - 开头）
    # 一律拒绝，防止把任意本地文件塞进任务或让 URL 被 yt-dlp 解析成命令行选项。
    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        logger.error(f"refusing to download non-http(s) web video URL: {url!r}")
        return False

    # 先清除同一输出路径的历史残留，防止上次中断的 .part 干扰本次下载。
    _remove_partial_files(output_path)

    command = [
        "yt-dlp",
        url,
        "-o",
        output_path,
        "-f",
        (
            f"bestvideo[ext=mp4][height<={_get_max_height()}]"
            f"+bestaudio[ext=m4a]/best[ext=mp4][height<={_get_max_height()}]"
            f"/best[height<={_get_max_height()}]/best"
        ),
        "--merge-output-format",
        "mp4",
        "--max-filesize",
        _get_max_filesize(),
        "--ignore-errors",
    ]

    success = False
    process = None
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            **_popen_kwargs(),
        )
        try:
            stdout, stderr = process.communicate(timeout=_DOWNLOAD_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            # 终止整个进程组并收割句柄，防止孤儿 ffmpeg 继续写盘。
            _kill_process_tree(process)
            process.communicate()
            logger.error(f"yt-dlp download timed out for: {url}")
            return False
        if process.returncode != 0:
            logger.error(f"yt-dlp download failed: {(stderr or '').strip()[:500]}")
            return False
        try:
            output_ok = os.path.isfile(output_path) and os.path.getsize(output_path) > 0
        except OSError:
            output_ok = False
        if output_ok:
            success = True
            # Optional watermark removal for user-owned content (disabled by default)
            _maybe_remove_watermark(output_path)
            return True
        logger.error(f"yt-dlp finished without producing output file: {output_path}")
        return False
    except (OSError, subprocess.SubprocessError) as exc:
        logger.error(f"failed to run yt-dlp download for {url}: {exc}")
        return False
    finally:
        if process is not None:
            try:
                process.wait()
            except OSError:
                pass
        if not success:
            # 失败或超时时清理分片残留；成功时 yt-dlp 已原子改名完成。
            _remove_partial_files(output_path)
