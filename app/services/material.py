import json
import os
import random
import re
import tempfile
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlsplit

import requests
from loguru import logger
from moviepy.video.io.VideoFileClip import VideoFileClip

from app.config import config
from app.models.schema import MaterialInfo, VideoAspect, VideoConcatMode
from app.services import material_cache, task_artifacts
from app.services.media_utils import (
    get_tls_verify as _get_tls_verify,
)
from app.services.media_utils import (
    redact_request_error as _redact_request_error,
)
from app.services.media_utils import (
    safe_public_url as _safe_public_url,
)
from app.utils import utils

# Thread-safe counter for API key rotation
_api_key_counter = 0
_api_key_lock = threading.Lock()

# auto 来源并发查询供应商的上限。5 个供应商同时查询在 8GB 内存的机器上
# 会与 ffmpeg/moviepy 渲染争抢资源；这里固定收敛到 3，任务时间线只取决于
# 最慢的供应商，多余并发只带来内存压力（G：低端设备自适应）。
_AUTO_SEARCH_MAX_WORKERS = 3

# 素材下载并发度。下载是网络 IO + 磁盘写，串行会让 8+ 关键词的片段逐个累加
# 10-60s 延迟；3 个 worker 能显著压缩等待，同时避免与 ffmpeg/moviepy 渲染
# 争抢资源（G：低端设备自适应，与 _AUTO_SEARCH_MAX_WORKERS 同量级）。
MAX_DOWNLOAD_WORKERS = 3
_MAX_DOWNLOAD_WORKERS_CAP = 5


def _effective_download_workers(required_clip_count: int | None = None) -> int:
    """根据成片所需片段数自适应并发，短视频保持 3，高时长自动扩到 5。"""
    if not required_clip_count or required_clip_count <= 6:
        return MAX_DOWNLOAD_WORKERS
    # 6 段以上每 6 段 +1 worker，上限 5
    workers = MAX_DOWNLOAD_WORKERS + (required_clip_count - 6 + 5) // 6
    return max(MAX_DOWNLOAD_WORKERS, min(_MAX_DOWNLOAD_WORKERS_CAP, workers))

# 素材搜索/下载是网络 IO，瞬时的 DNS、连接重置或供应商 5xx/限流不应让整个
# 片段静默缺素材。这里做有限次指数退避重试：失败成本低，重试能显著提高
# 成功率；但重试次数封顶，避免供应商持续故障时拖长任务时间。
_REQUEST_RETRY_ATTEMPTS = 2
_REQUEST_RETRY_BASE_DELAY_SECONDS = 0.5
_REQUEST_RETRY_STATUS_CODES = (429, 500, 502, 503, 504)


def _request_with_retry(
    url: str,
    *,
    headers: dict | None = None,
    params: dict | None = None,
    timeout: tuple,
    max_retries: int = _REQUEST_RETRY_ATTEMPTS,
    **request_kwargs,
):
    """发送 GET 请求，并对瞬时失败做指数退避重试后返回最后一次响应。

    连接类异常或 5xx/429 响应会重试；成功响应（含 4xx 等其它状态）直接返回，
    由调用方决定如何处理。所有尝试都用光后，连接类异常原样抛出，交给调用方
    统一记录脱敏日志。
    """
    delay = _REQUEST_RETRY_BASE_DELAY_SECONDS
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(
                url,
                headers=headers,
                params=params,
                proxies=config.proxy,
                verify=_get_tls_verify(),
                timeout=timeout,
                **request_kwargs,
            )
            if (
                getattr(resp, "status_code", 200) in _REQUEST_RETRY_STATUS_CODES
                and attempt < max_retries
            ):
                last_error = f"status={getattr(resp, 'status_code', 200)}"
                time.sleep(delay)
                delay *= 2
                continue
            return resp
        except requests.RequestException as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(delay)
                delay *= 2
    if isinstance(last_error, Exception):
        raise last_error
    raise RuntimeError(str(last_error))


def _creator_info(value: Any) -> dict[str, str] | None:
    """从不同供应商的作者结构中提取统一的公开字段。"""
    if isinstance(value, str) and value.strip():
        return {"name": value.strip()}
    if not isinstance(value, dict):
        return None

    creator: dict[str, str] = {}
    creator_id = value.get("id")
    creator_name = value.get("name") or value.get("username")
    creator_page = _safe_public_url(
        value.get("url") or value.get("profile_url") or value.get("profile_page")
    )
    if creator_id is not None:
        creator["id"] = str(creator_id)
    if creator_name:
        creator["name"] = str(creator_name)
    if creator_page:
        creator["profile_page"] = creator_page
    return creator or None


def _material_source_record(item: MaterialInfo, local_path: str) -> dict[str, Any]:
    """
    为成功下载的素材生成轻量来源记录。

    ``source_info`` 可能来自缓存，甚至来自外部构造的 ``MaterialInfo``，因此
    不能原样写入。这里按白名单重新构造，只保留公开页面、业务标识和尺寸，
    并只记录本地文件名，避免用户目录或 Docker 挂载路径进入任务文件。
    """
    source = item.source_info if isinstance(item.source_info, dict) else {}
    record: dict[str, Any] = {
        "provider": str(item.provider or source.get("provider") or ""),
        "local_file": Path(local_path).name,
        "duration": int(item.duration),
    }

    search_term = source.get("search_term")
    asset_id = source.get("asset_id")
    source_page = _safe_public_url(source.get("source_page"))
    if isinstance(search_term, str) and search_term.strip():
        record["search_term"] = search_term.strip()
    if asset_id not in (None, ""):
        record["asset_id"] = str(asset_id)
    if source_page:
        record["source_page"] = source_page

    # 文本元数据随任务记录保留，便于回溯 auto 来源的相关性排序依据。
    for field in ("title", "tags", "description"):
        value = source.get(field)
        if isinstance(value, str) and value.strip():
            record[field] = value.strip()

    creator = _creator_info(source.get("creator"))
    if creator:
        record["creator"] = creator

    raw_rendition = source.get("rendition")
    if isinstance(raw_rendition, dict):
        rendition = {}
        for field in ("id", "width", "height"):
            value = raw_rendition.get(field)
            if value not in (None, ""):
                rendition[field] = str(value) if field == "id" else value
        if rendition:
            record["rendition"] = rendition
    return record


def _persist_material_sources(
    task_id: str,
    material_sources: list[dict[str, Any]],
) -> None:
    """
    将当前实际下载成功的素材来源补充到任务清单。

    任务记录是辅助能力，不能改变视频下载函数的返回值，也不能因为写盘失败
    中断成片主流程。``patch_script_data`` 会负责原子替换和异常日志；这里仅在
    成功后记录数量，便于确认任务追溯信息是否已经落盘。
    """
    try:
        saved = task_artifacts.patch_script_data(
            task_id,
            material_sources=material_sources,
        )
        if saved:
            logger.info(
                f"saved material source records: "
                f"task_id={task_id}, count={len(material_sources)}"
            )
    except Exception as exc:
        # task_artifacts 自身已经按失败降级设计，这里仍保留最后一道隔离，
        # 防止未来实现调整或目录解析异常意外影响素材下载返回值。
        logger.warning(
            "failed to persist material source records: "
            f"task_id={task_id}, error={type(exc).__name__}, detail={exc}"
        )


def get_api_key(cfg_key: str):
    api_keys = config.app.get(cfg_key)
    if not api_keys:
        raise ValueError(
            f"\n\n##### {cfg_key} is not set #####\n\n"
            f"Please set it in the config.toml file: {config.config_file}\n"
        )

    # if only one key is provided, return it
    if isinstance(api_keys, str):
        return api_keys

    global _api_key_counter
    with _api_key_lock:
        _api_key_counter += 1
        return api_keys[_api_key_counter % len(api_keys)]


def _redact_video_url(video_url: str) -> str:
    """
    用于日志的下载地址脱敏：只保留 scheme + host + 前两段路径。

    库存供应商的直链可能内嵌签名令牌（如 Coverr 的 signed JWT 绑定 API key、
    Pixabay 的 key 查询参数）。查询串、fragment 和后续路径段全部丢弃，
    既避免密钥/令牌落盘，又保留可定位到供应商的有限上下文。
    """
    try:
        parts = urlsplit(video_url)
    except ValueError:
        return "<unparseable url>"
    segments = [segment for segment in parts.path.split("/") if segment]
    truncated_path = "/".join(segments[:2])
    return f"{parts.scheme}://{parts.netloc}/{truncated_path}"


def _is_cloudflare_challenge(response: requests.Response) -> bool:
    """
    识别 Cloudflare 返回的 HTML Challenge，而不是把它当成 Pixabay JSON。

    Cloudflare 通常会设置 `cf-mitigated: challenge`；部分部署只返回带有
    "Just a moment" 或 challenge-platform 的 HTML，因此保留内容特征兜底。
    响应正文仅在内存中判断，不写入日志，避免记录无价值的大段 HTML。
    """
    headers = getattr(response, "headers", {}) or {}
    if str(headers.get("cf-mitigated", "")).lower() == "challenge":
        return True

    content_type = str(headers.get("content-type", "")).lower()
    if "text/html" not in content_type:
        return False

    body = str(getattr(response, "text", "")).lower()
    return "just a moment" in body or "/cdn-cgi/challenge-platform/" in body


def _matches_video_aspect(
    width: Any,
    height: Any,
    video_aspect: VideoAspect,
    *,
    is_vertical: Any = None,
) -> bool:
    """
    判断远端素材是否与目标画面方向一致。

    Pexels、Pixabay 和 Coverr 的响应字段并不统一，因此先使用宽高做可靠判断；
    Coverr 部分历史响应缺少尺寸时，再使用明确的 ``is_vertical`` 布尔值兜底。
    无法确认方向的素材直接跳过，避免竖屏任务混入横屏素材并在成片中产生黑边。
    """
    aspect = VideoAspect(video_aspect)
    try:
        normalized_width = int(float(width))
        normalized_height = int(float(height))
    except (TypeError, ValueError):
        normalized_width = 0
        normalized_height = 0

    if normalized_width > 0 and normalized_height > 0:
        if aspect == VideoAspect.portrait:
            return normalized_height > normalized_width
        if aspect == VideoAspect.landscape:
            return normalized_width > normalized_height
        return normalized_width == normalized_height

    if isinstance(is_vertical, bool) and aspect != VideoAspect.square:
        return is_vertical == (aspect == VideoAspect.portrait)
    return False


def _filter_materials_by_aspect(
    items: list[MaterialInfo],
    video_aspect: VideoAspect,
) -> list[MaterialInfo]:
    """
    对缓存结果再次校验方向。

    素材搜索缓存最长保留 24 小时，升级前写入的缓存可能包含方向不匹配的素材。
    在统一缓存入口过滤可以让修复立即生效，也能防御第三方 Provider 或旧缓存
    遗漏远端筛选。无法读取 rendition 尺寸的旧条目按未验证处理并跳过。
    """
    aspect = VideoAspect(video_aspect)
    if aspect == VideoAspect.square:
        # Pixabay 和 Coverr 很少提供原生方形素材。方形输出沿用既有行为，
        # 接受可用候选并交给视频合成阶段裁剪，避免升级后 1:1 任务无素材。
        return list(items)

    filtered_items = []
    for item in items:
        source_info = item.source_info if isinstance(item.source_info, dict) else {}
        rendition = source_info.get("rendition")
        rendition = rendition if isinstance(rendition, dict) else {}
        if _matches_video_aspect(
            rendition.get("width"),
            rendition.get("height"),
            aspect,
        ):
            filtered_items.append(item)
    return filtered_items


def search_videos_pexels(
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect = VideoAspect.portrait,
) -> list[MaterialInfo]:
    aspect = VideoAspect(video_aspect)
    video_orientation = aspect.name
    video_width, video_height = aspect.to_resolution()
    api_key = get_api_key("pexels_api_keys")
    headers = {
        "Authorization": api_key,
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    }
    # Build URL
    params = {"query": search_term, "per_page": 20, "orientation": video_orientation}
    query_url = f"https://api.pexels.com/v1/videos/search?{urlencode(params)}"
    logger.info(f"searching videos on pexels: term={search_term!r}")

    try:
        r = _request_with_retry(
            query_url,
            headers=headers,
            timeout=(30, 60),
        )
        response = r.json()
        video_items = []
        if "videos" not in response:
            logger.error("pexels video search returned an unsupported response")
            return video_items
        videos = response["videos"]
        # loop through each video in the result
        for v in videos:
            duration = v["duration"]
            # check if video has desired minimum duration
            if duration < minimum_duration:
                continue
            video_files = v["video_files"]
            # 优先选取分辨率满足目标且面积最大的 rendition（支持 4K 向上兼容 1080p）
            best_video = None
            best_pixels = -1
            for video in video_files:
                try:
                    w = int(video["width"])
                    h = int(video["height"])
                except (KeyError, TypeError, ValueError):
                    continue
                if _matches_video_aspect(w, h, aspect) and w >= video_width and h >= video_height:
                    pixels = w * h
                    if pixels > best_pixels:
                        best_pixels = pixels
                        best_video = video
            if best_video is not None:
                video = best_video
                w = int(video["width"])
                h = int(video["height"])
                item = MaterialInfo()
                item.provider = "pexels"
                item.url = video["link"]
                item.duration = duration
                item.source_info = {
                    "provider": "pexels",
                    "search_term": search_term,
                    "title": str(v.get("title") or ""),
                    "asset_id": (
                        str(v.get("id")) if v.get("id") is not None else None
                    ),
                    "source_page": _safe_public_url(v.get("url")),
                    "creator": _creator_info(v.get("user")),
                    "rendition": {
                        "id": (
                            str(video.get("id"))
                            if video.get("id") is not None
                            else None
                        ),
                        "width": w,
                        "height": h,
                    },
                }
                video_items.append(item)
        return video_items
    except Exception as e:
        logger.error(
            "pexels video search failed: "
            f"error={type(e).__name__}, detail={_redact_request_error(e, api_key)}"
        )

    return []


def search_videos_pixabay(
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect = VideoAspect.portrait,
) -> list[MaterialInfo]:
    aspect = VideoAspect(video_aspect)

    video_width, video_height = aspect.to_resolution()

    api_key = get_api_key("pixabay_api_keys")
    # Build URL
    params = {
        "q": search_term,
        "video_type": "all",  # Accepted values: "all", "film", "animation"
        "per_page": 50,
        "key": api_key,
    }
    query_url = f"https://pixabay.com/api/videos/?{urlencode(params)}"
    logger.info(
        f"searching videos on pixabay: term={search_term!r}, "
        f"proxy_enabled={bool(config.proxy)}"
    )

    try:
        r = _request_with_retry(query_url, timeout=(30, 60))
        status_code = int(getattr(r, "status_code", 200))
        headers = getattr(r, "headers", {}) or {}
        content_type = str(headers.get("content-type", ""))
        retry_after = headers.get("retry-after")
        cf_ray = headers.get("cf-ray")

        if _is_cloudflare_challenge(r):
            logger.error(
                "pixabay search was blocked by a Cloudflare challenge: "
                f"status={status_code}, cf_ray={cf_ray or 'unknown'}. "
                "Check the server network or proxy, or use Pexels/Coverr instead."
            )
            return []

        if status_code == 429:
            logger.error(
                "pixabay API rate limit exceeded: "
                f"status=429, retry_after={retry_after or 'unknown'}"
            )
            return []

        if status_code >= 400:
            logger.error(
                "pixabay search request failed: "
                f"status={status_code}, content_type={content_type or 'unknown'}"
            )
            return []

        try:
            response = r.json()
        except ValueError:
            logger.error(
                "pixabay returned an unexpected non-JSON response: "
                f"status={status_code}, content_type={content_type or 'unknown'}"
            )
            return []

        video_items = []
        if "hits" not in response:
            logger.error("pixabay video search returned an unsupported response")
            return video_items
        videos = response["hits"]
        # loop through each video in the result
        for v in videos:
            duration = v["duration"]
            # check if video has desired minimum duration
            if duration < minimum_duration:
                continue
            video_files = v["videos"]
            # 选取满足分辨率且面积最大的 rendition，优先 4K
            best_video = None
            best_pixels = -1
            best_type = ""
            for video_type in video_files:
                video = video_files[video_type]
                try:
                    w = int(video["width"])
                    h = int(video["height"])
                except (KeyError, TypeError, ValueError):
                    continue
                orientation_matches = aspect == VideoAspect.square or (
                    _matches_video_aspect(w, h, aspect)
                )
                if orientation_matches and w >= video_width and h >= video_height:
                    pixels = w * h
                    if pixels > best_pixels:
                        best_pixels = pixels
                        best_video = video
                        best_type = video_type
            if best_video is not None:
                video = best_video
                w = int(best_video["width"])
                item = MaterialInfo()
                item.provider = "pixabay"
                item.url = video["url"]
                item.duration = duration
                item.source_info = {
                    "provider": "pixabay",
                    "search_term": search_term,
                    "tags": str(v.get("tags") or ""),
                    "asset_id": (
                        str(v.get("id")) if v.get("id") is not None else None
                    ),
                    "source_page": _safe_public_url(v.get("pageURL")),
                    "creator": _creator_info(
                        {
                            "id": v.get("user_id"),
                            "name": v.get("user"),
                        }
                    ),
                    "rendition": {
                        "id": best_type,
                        "width": w,
                        "height": video.get("height"),
                    },
                }
                video_items.append(item)
        return video_items
    except Exception as e:
        error_message = _redact_request_error(e, api_key)
        logger.error(
            "pixabay search request failed: "
            f"error={type(e).__name__}, detail={error_message}"
        )

    return []


def search_videos_coverr(
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect = VideoAspect.portrait,
) -> list[MaterialInfo]:
    """
    Coverr (https://coverr.co) - free HD/4K stock videos,
    subject to Coverr license terms (https://coverr.co/license).

    Coverr API notes (based on official docs at api.coverr.co/docs/):
      - 鉴权: Authorization: Bearer <api_key>
      - 搜索端点: GET /videos?query=...,响应结构 {"hits": [...], ...}
      - 加 ?urls=true 在搜索响应里直接返回 mp4 直链
      - URL 是 signed JWT(绑定 API key,无过期时间)
      - Coverr 支持通过 filter=is_vertical:true/false 筛选横竖屏素材；
        响应返回后仍根据 max_width/max_height 或 is_vertical 做本地校验
      - duration 字段同时存在 number 和 string 两种形态,本函数都接受

    本函数使用 urls.mp4_download 字段作为下载地址 —— 按 Coverr 官方文档
    (https://api.coverr.co/docs/videos/#download-a-video) 的说法,
    GET 这个 URL 本身就被 Coverr 当作一次合法的 download 事件计入统计,
    无需再调用 PATCH /videos/:id/stats/downloads。
    """
    aspect = VideoAspect(video_aspect)
    api_key = get_api_key("coverr_api_keys")
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {
        "query": search_term,
        "page_size": 20,
        "urls": "true",
        "sort": "popular",
    }
    # 服务端方向筛选可以直接从完整搜索结果中返回目标素材，避免先取热门结果再
    # 本地过滤导致竖屏候选为空。方形素材没有对应布尔条件，继续依赖本地宽高校验。
    if aspect == VideoAspect.portrait:
        params["filter"] = "is_vertical:true"
    elif aspect == VideoAspect.landscape:
        params["filter"] = "is_vertical:false"
    query_url = f"https://api.coverr.co/videos?{urlencode(params)}"
    logger.info(f"searching videos on coverr: term={search_term!r}")

    try:
        r = _request_with_retry(
            query_url,
            headers=headers,
            timeout=(30, 60),
        )
        response = r.json()
        video_items: list[MaterialInfo] = []

        if not isinstance(response, dict) or "hits" not in response:
            logger.error("coverr video search returned an unsupported response")
            return video_items

        for v in response["hits"]:
            # duration 在不同响应里可能是 number(11.625) 或 string("10.500000")
            try:
                duration = int(float(v.get("duration") or 0))
            except (TypeError, ValueError):
                continue
            if duration < minimum_duration:
                continue

            video_id = v.get("id")
            mp4_download_url = (v.get("urls") or {}).get("mp4_download")
            if not video_id or not mp4_download_url:
                continue
            if aspect != VideoAspect.square and not _matches_video_aspect(
                v.get("max_width"),
                v.get("max_height"),
                aspect,
                is_vertical=v.get("is_vertical"),
            ):
                continue

            item = MaterialInfo()
            item.provider = "coverr"
            item.url = mp4_download_url
            item.duration = duration
            item.source_info = {
                "provider": "coverr",
                "search_term": search_term,
                "title": str(v.get("title") or ""),
                "description": str(v.get("description") or ""),
                "asset_id": str(video_id),
                "source_page": _safe_public_url(v.get("canonical_url") or v.get("url")),
                "creator": _creator_info(v.get("creator") or v.get("author")),
                "rendition": {
                    "id": "mp4_download",
                    "width": v.get("max_width"),
                    "height": v.get("max_height"),
                },
            }
            video_items.append(item)
        return video_items
    except Exception as e:
        logger.error(
            "coverr video search failed: "
            f"error={type(e).__name__}, detail={_redact_request_error(e, api_key)}"
        )

    return []


def save_video(video_url: str, save_dir: str = "") -> str:
    if not save_dir:
        save_dir = utils.storage_dir("cache_videos")

    if not os.path.exists(save_dir):
        # 并发下载时多个线程可能同时通过存在性检查；exist_ok 让先建目录的
        # 线程赢，其它线程静默复用，避免 FileExistsError 误伤一次下载。
        os.makedirs(save_dir, exist_ok=True)

    url_without_query = video_url.split("?")[0]
    url_hash = utils.md5(url_without_query)
    video_id = f"vid-{url_hash}"

    # If a validated video or image already exists, return the path immediately
    for candidate_ext in ["mp4", "png", "jpg", "jpeg", "webp"]:
        candidate_path = f"{save_dir}/{video_id}.{candidate_ext}"
        if os.path.exists(candidate_path) and os.path.getsize(candidate_path) > 0:
            logger.info(f"material already exists: {candidate_path}")
            return candidate_path

    # AI-generated media (e.g. Gemini) is written to a local file during the
    # search phase; validate it and return it instead of trying to download.
    if os.path.isfile(video_url) and os.path.getsize(video_url) > 0:
        valid_path = _validate_saved_media(video_url)
        if valid_path:
            logger.info(f"material already saved locally: {valid_path}")
            return valid_path
        return ""

    video_path = f"{save_dir}/{video_id}.mp4"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    }

    # Download raw material. 连接中断、CDN 瞬时 5xx/限流都值得重试一次；
    # 失败成本只有几秒钟，重试能显著减少片段缺素材。所有尝试用光才放弃。
    last_error = None
    for attempt in range(_REQUEST_RETRY_ATTEMPTS + 1):
        try:
            resp = _request_with_retry(
                video_url,
                headers=headers,
                timeout=(60, 240),
                max_retries=0,
                stream=True,
            )
            if resp.status_code != 200:
                if (
                    resp.status_code in _REQUEST_RETRY_STATUS_CODES
                    and attempt < _REQUEST_RETRY_ATTEMPTS
                ):
                    time.sleep(_REQUEST_RETRY_BASE_DELAY_SECONDS * (2**attempt))
                    continue
                logger.warning(
                    f"failed to download material from {_redact_video_url(video_url)}: status={resp.status_code}"
                )
                return ""
            tmp_fd = None
            tmp_path = ""
            try:
                tmp_fd, tmp_path = tempfile.mkstemp(
                    dir=save_dir, prefix=f"{video_id}.", suffix=".tmp"
                )
                try:
                    with os.fdopen(tmp_fd, "wb") as f:
                        tmp_fd = None
                        for chunk in resp.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                except Exception:
                    if tmp_fd is not None:
                        try:
                            os.close(tmp_fd)
                        except OSError:
                            pass
                        tmp_fd = None
                    raise
                if os.path.getsize(tmp_path) == 0:
                    logger.warning(
                        f"downloaded empty file from {_redact_video_url(video_url)}"
                    )
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass
                    return ""
                try:
                    # Use hard link as atomic exclusive claim (no 0-byte placeholder window)
                    os.link(tmp_path, video_path)
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass
                except FileExistsError:
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass
                    if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
                        logger.info(f"material already exists (race): {video_path}")
                        return video_path
                    return ""
                except OSError:
                    # Fallback for filesystems where link not supported; avoid overwriting existing
                    if os.path.exists(video_path):
                        try:
                            os.remove(tmp_path)
                        except OSError:
                            pass
                        if os.path.getsize(video_path) > 0:
                            logger.info(f"material already exists (race): {video_path}")
                            return video_path
                        return ""
                    try:
                        os.replace(tmp_path, video_path)
                    except OSError:
                        try:
                            os.remove(tmp_path)
                        except OSError:
                            pass
                        raise
            finally:
                if tmp_fd is not None:
                    try:
                        os.close(tmp_fd)
                    except OSError:
                        pass
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        if os.path.abspath(tmp_path) != os.path.abspath(video_path):
                            os.remove(tmp_path)
                    except OSError:
                        pass
            break
        except Exception as exc:
            last_error = exc
            if attempt < _REQUEST_RETRY_ATTEMPTS:
                time.sleep(_REQUEST_RETRY_BASE_DELAY_SECONDS * (2**attempt))
                continue
            logger.warning(
                f"failed to fetch material from {_redact_video_url(video_url)}: {last_error}"
            )
            return ""

    if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
        # First attempt: validate as a video clip
        clip = None
        try:
            clip = VideoFileClip(video_path)
            duration = clip.duration
            fps = clip.fps
            if duration > 0 and fps > 0:
                _enforce_video_cache_limit_quietly()
                return video_path
        except Exception:
            pass
        finally:
            if clip is not None:
                try:
                    clip.close()
                except Exception:
                    pass

        # Second attempt: check if it is a valid image (e.g. AI-generated image)
        try:
            from PIL import Image

            with Image.open(video_path) as img:
                img_format = (img.format or "PNG").lower()
                img_ext = "jpg" if img_format in ["jpeg", "jpg"] else img_format
                img_path = f"{save_dir}/{video_id}.{img_ext}"
            os.replace(video_path, img_path)
            logger.info(f"material saved as image: {img_path}")
            _enforce_video_cache_limit_quietly()
            return img_path
        except Exception as img_err:
            logger.warning(
                f"invalid media file (not video or image): {video_path} => {img_err}"
            )
            try:
                os.remove(video_path)
            except Exception:
                pass

    return ""


def _validate_saved_media(file_path: str) -> str:
    """
    Return ``file_path`` if it is a playable video or readable image,
    otherwise return an empty string.
    """
    if not file_path or not os.path.isfile(file_path) or os.path.getsize(file_path) <= 0:
        return ""
    clip = None
    try:
        clip = VideoFileClip(file_path)
        if clip.duration > 0 and clip.fps > 0:
            return file_path
    except Exception:
        pass
    finally:
        if clip is not None:
            try:
                clip.close()
            except Exception:
                pass
    try:
        from PIL import Image

        with Image.open(file_path) as img:
            img.verify()
        return file_path
    except Exception:
        return ""


def _validate_scraped_video(video_path: str) -> str:
    """
    用 VideoFileClip 探测 yt-dlp 下载的文件，损坏容器返回空串。

    ``download_web_video`` 只检查文件存在且非空，yt-dlp 返回成功不代表
    容器完好（例如下载被中断、合并失败或扩展名与实际格式不符）。如果不
    在这里提前拦截，损坏文件会一直走到最终渲染阶段才因为无法解码而炸掉
    整条任务。探测失败时清理残留文件，避免脏文件污染后续重试。
    """
    if (
        not video_path
        or not os.path.isfile(video_path)
        or os.path.getsize(video_path) <= 0
    ):
        return ""

    clip = None
    try:
        clip = VideoFileClip(video_path)
        if clip.duration > 0 and clip.fps > 0:
            return video_path
        logger.warning(
            f"scraped video has no usable duration/fps, treating as invalid: {video_path}"
        )
    except Exception as exc:
        logger.warning(
            f"scraped video container is invalid, treating as download failure: "
            f"path={video_path}, error={type(exc).__name__}, detail={exc}"
        )
    finally:
        if clip is not None:
            try:
                clip.close()
            except Exception:
                pass

    try:
        os.remove(video_path)
    except Exception:
        pass
    return ""


def _get_synonym_terms(search_term: str) -> list[str]:
    """
    Ask LLM for 2 synonyms/alternative visual search terms. Bounded to max 1 call
    per original term to avoid cost explosion.
    """

    if not search_term or not search_term.strip():
        return []
    try:
        from app.services import llm as _llm_svc
    except Exception as exc:  # pragma: no cover - import failure
        logger.warning(f"synonym expansion unavailable: {exc}")
        return []
    prompt = (
        f'Provide 2 concise synonyms or alternative visual search terms for "{search_term.strip()}" '
        f"suitable for stock video search. "
        f'Return ONLY a JSON array of 2 strings, no explanation. '
        f'Example: ["synonym1", "synonym2"]'
    )
    try:
        response = _llm_svc._generate_response(prompt)
        if not isinstance(response, str) or not response.strip():
            return []
        if response.startswith("Error:"):
            logger.warning(
                f"synonym expansion LLM error for {search_term!r}: {response[:200]}"
            )
            return []
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z0-9]*\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
            cleaned = cleaned.strip()
        try:
            terms = json.loads(cleaned)
        except Exception:
            match = re.search(r"\[.*\]", cleaned, re.DOTALL)
            if not match:
                return []
            terms = json.loads(match.group())
        if not isinstance(terms, list):
            return []
        result: list[str] = []
        seen: set[str] = set()
        lower_original = search_term.strip().lower()
        for t in terms:
            if not isinstance(t, str):
                continue
            norm = t.strip()
            if not norm or norm.lower() == lower_original or norm.lower() in seen:
                continue
            seen.add(norm.lower())
            result.append(norm)
            if len(result) >= 2:
                break
        return result
    except Exception as exc:
        logger.warning(
            f"failed to get synonyms for {search_term!r}: {type(exc).__name__}: {exc}"
        )
        return []


def _enforce_video_cache_limit_quietly() -> None:
    """Best-effort LRU enforcement; never raises to caller."""

    try:
        from app.services import cache_manager as _cache_mgr

        _cache_mgr.enforce_cache_limit()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"cache LRU enforcement failed: {exc}")


def _search_videos_with_cache(
    provider: str,
    search_videos: Callable[..., list[MaterialInfo]],
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect,
) -> list[MaterialInfo]:
    """
    统一处理三个在线素材源的 24 小时搜索缓存。

    缓存只包裹搜索 API，不改变后续视频下载与去重逻辑。远端返回空列表时不写
    缓存，因为现有 provider 接口使用空列表同时表示“没有结果”和“请求失败”；
    在两者尚未拆分为明确结果类型前，宁可下次重试，也不能把临时故障缓存一天。
    """
    cache_args = {
        "provider": provider,
        "search_term": search_term,
        "minimum_duration": minimum_duration,
        "video_aspect": video_aspect,
    }

    def load_cache_safely() -> list[MaterialInfo] | None:
        try:
            return material_cache.load_material_search_cache(**cache_args)
        except Exception as exc:
            # 缓存是可选优化，任何缓存实现异常都必须按未命中处理，不能阻断
            # Pexels、Pixabay 或 Coverr 的正常远端搜索。
            logger.warning(
                "material search cache read failed, continue with remote search: "
                f"provider={provider}, error={type(exc).__name__}, detail={exc}"
            )
            return None

    def load_matching_cache() -> tuple[list[MaterialInfo] | None, int]:
        cached_items = load_cache_safely()
        if cached_items is None:
            return None, 0

        filtered_cached_items = _filter_materials_by_aspect(
            cached_items,
            video_aspect,
        )
        ignored_count = len(cached_items) - len(filtered_cached_items)
        if ignored_count:
            # 旧版本缓存可能混入其它方向的素材。即使仍有少量可用条目，也要刷新
            # 完整候选集，否则在缓存有效期内会反复使用同一批少量视频。
            return None, ignored_count
        return filtered_cached_items, 0

    cached_items, ignored_count = load_matching_cache()
    if cached_items is not None:
        return cached_items
    if ignored_count:
        logger.info(
            "material search cache contains mismatched orientations, "
            f"refresh from provider: provider={provider}, term={search_term!r}, "
            f"ignored={ignored_count}"
        )

    cache_lock = material_cache.get_material_search_cache_lock(**cache_args)
    with cache_lock:
        # 等待相同搜索条件的线程完成后再次读取，避免多个 API 任务在首次缓存
        # 未命中时同时请求远端，降低第三方接口限流和风控触发概率。
        cached_items, _ = load_matching_cache()
        if cached_items is not None:
            return cached_items

        items = search_videos(
            search_term=search_term,
            minimum_duration=minimum_duration,
            video_aspect=video_aspect,
        )
        # Provider 正常会写入当前关键词，但测试替身、第三方扩展或旧实现可能
        # 遗漏或携带错误值。缓存读取会根据缓存键恢复该字段，因此远端结果也在
        # 同一入口校正，保证首次搜索与缓存命中的任务来源记录保持一致。
        for item in items:
            if isinstance(item.source_info, dict):
                item.source_info = dict(item.source_info)
                item.source_info["search_term"] = search_term
        if items:
            try:
                material_cache.save_material_search_cache(
                    **cache_args,
                    items=items,
                )
            except Exception as exc:
                logger.warning(
                    "material search cache write failed, use remote results: "
                    f"provider={provider}, error={type(exc).__name__}, detail={exc}"
                )
        return items


def _search_videos_custom_api_with_fallback(
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect,
) -> list[MaterialInfo]:
    """
    Attempt the user-configured custom API first.

    If the custom API returns no results (or raises), and hybrid_video_mode is
    enabled in config, automatically sweep Pexels as a fallback.  This makes
    the custom source "additive": it wins when it has relevant content, but
    standard stock footage is always available as a safety net.
    """
    # Local import to avoid circular dependency at module load time.
    from app.services import custom_media as _custom_media_svc

    items = _search_videos_with_cache(
        provider="custom_api",
        search_videos=_custom_media_svc.search_media_custom_api,
        search_term=search_term,
        minimum_duration=minimum_duration,
        video_aspect=video_aspect,
    )

    if not items and config.app.get("hybrid_video_mode", True):
        logger.info(
            f"custom API returned no results for {search_term!r}; "
            "falling back to Pexels (hybrid_video_mode=true)"
        )
        # Only fall back to Pexels when keys are available; silently skip
        # if the user has not configured a Pexels key.
        if config.app.get("pexels_api_keys"):
            items = _search_videos_with_cache(
                provider="pexels",
                search_videos=search_videos_pexels,
                search_term=search_term,
                minimum_duration=minimum_duration,
                video_aspect=video_aspect,
            )
        else:
            logger.warning(
                "hybrid_video_mode is enabled but pexels_api_keys is not set; "
                "cannot fall back to Pexels"
            )

    return items


def _tokenize(text: str) -> set[str]:
    """将文本拆成小写字母数字词元集合，用于相关性比较。"""
    return set(re.findall(r"[a-z0-9]+", str(text or "").lower()))


def _relevance_text(item: MaterialInfo) -> str:
    """汇总素材的标题/标签/描述文本，用于与搜索词比较。"""
    source = item.source_info if isinstance(item.source_info, dict) else {}
    return " ".join(
        str(source.get(field) or "") for field in ("title", "tags", "description")
    )


def _relevance_penalty(item: MaterialInfo) -> float:
    """
    搜索词与素材文本的贴合度（0=完全贴合，1=毫无重合）。

    无文本元数据时返回中性值 0.5，避免某个来源缺字段导致整体被压到末尾，
    也避免只有部分来源有元数据时排序整体偏向它们。
    """
    source = item.source_info if isinstance(item.source_info, dict) else {}
    term_tokens = _tokenize(source.get("search_term") or "")
    clip_tokens = _tokenize(_relevance_text(item))
    if not term_tokens or not clip_tokens:
        return 0.5
    overlap = len(term_tokens & clip_tokens)
    jaccard = overlap / len(term_tokens | clip_tokens)
    return 1.0 - jaccard


# 素材方向与目标画幅不一致时的排名惩罚（tiebreak 级）。
#
# 该值刻意保持很小：它只在贴合度接近时生效，不会压过相关度主排序因子，
# 也小于 4K vs 720p 的分辨率差距（约 60 分），避免为了“方向匹配”而硬选
# 低清但同方向的素材。方向不匹配的素材仍可在成片中被中心裁切或铺模糊背景。
_ORIENTATION_MISMATCH_PENALTY = 20.0


def _material_dimensions(item: MaterialInfo) -> tuple[int, int] | None:
    """Best-effort width/height from search-time metadata, or None if unknown.

    Web 抓取素材在搜索阶段把尺寸写入 resolution 字段；库存供应商则放在
    source_info.rendition 里。统一回退读取，让所有来源都能参与方向判断。
    """
    resolution = getattr(item, "resolution", "") or ""
    if isinstance(resolution, str) and "x" in resolution:
        try:
            parts = resolution.split("x", 1)
            width = int(parts[0])
            height = int(parts[1])
            if width > 0 and height > 0:
                return (width, height)
        except (TypeError, ValueError):
            pass
    source = item.source_info if isinstance(item.source_info, dict) else {}
    rendition = source.get("rendition")
    if isinstance(rendition, dict):
        try:
            width = int(rendition.get("width") or 0)
            height = int(rendition.get("height") or 0)
            if width > 0 and height > 0:
                return (width, height)
        except (TypeError, ValueError):
            pass
    return None


def _material_orientation(item: MaterialInfo) -> str | None:
    """素材方向：portrait / landscape / square，未知时返回 None。"""
    dimensions = _material_dimensions(item)
    if not dimensions:
        return None
    width, height = dimensions
    if width == height:
        return "square"
    return "portrait" if height > width else "landscape"


def _aspect_orientation(video_aspect: VideoAspect) -> str | None:
    """目标画幅的方向，与 _matches_video_aspect（web_scrape）同口径。"""
    try:
        width, height = VideoAspect(video_aspect).to_resolution()
    except (TypeError, ValueError):
        return None
    if width == height:
        return "square"
    return "portrait" if height > width else "landscape"


def _rank_and_select_best_material(
    candidates: list[MaterialInfo], required_duration: int, video_aspect: VideoAspect
) -> list[MaterialInfo]:
    """
    相关性优先：先按素材与脚本搜索词的贴合度排序，再用来源优先级、
    时长接近度和分辨率作为同分时的次级依据。
    """
    if not candidates:
        return []

    # Priority mapping (lower number is higher priority)
    # web_scrape platform-aware search (YouTube vs TikTok) and embedding ranking are
    # handled inside web_scrape.search_videos_web_scrape; here we keep provider as tiebreak only.
    provider_priority = {
        "custom_api": 0,
        "pexels": 1,
        "pixabay": 2,
        "coverr": 3,
        "web_scrape": 4,
        "pollinations": 5,
    }

    def _resolution_pixels(item: MaterialInfo) -> float:
        dimensions = _material_dimensions(item)
        if not dimensions:
            return 0.0
        return float(dimensions[0] * dimensions[1])

    def score_material(item: MaterialInfo) -> float:
        # Score calculation. Lower score is better.
        # Relevance is the primary factor (0..1, scaled by 1000); a poor match
        # (penalty ~1.0) can be outranked by a strong match from any source.
        relevance_penalty = _relevance_penalty(item)

        # Source priority: small tiebreak among equally relevant candidates.
        priority_score = provider_priority.get(item.provider, 99)

        # 生成式 AI 图片（Gemini/Pollinations）在相同相关度下排在真实视频
        # 素材之后：视频素材不需要二次转换，成片质感也更稳定。加分很小，
        # 不会压过相关度这一主排序因子（F5 排名审核）。
        source_info = item.source_info if isinstance(item.source_info, dict) else {}
        image_penalty = 0.5 if source_info.get("media_type") == "image" else 0.0

        # Duration score: distance from required duration (in seconds)
        item_duration = float(item.duration) if item.duration is not None else 0.0
        duration_diff = abs(item_duration - float(required_duration))

        # Resolution score: We want HD if possible, but favor relevance mostly.
        # 继续放大分辨率权重，让同等相关度下高清/4K 素材明显胜过低清素材。
        # 相关度仍是主排序因子（相关度差 0.01 相当于约 10 分），分辨率只在
        # 贴合度接近时起决定性作用：4K vs 720p 的差距（约 60 分）仍小于
        # 0.06 的相关度差，不会让“高分但毫不相关”的素材压过真正匹配的素材。
        res_score = -_resolution_pixels(item) / 100000.0

        # Orientation: 素材方向与目标画幅一致时成片无需黑边或大幅裁切。方向
        # 不匹配只给 tiebreak 级小惩罚，不会压过相关度，也不会让低清同方向
        # 素材压过高清异方向素材（4K vs 720p 差距约 60 分 > 本惩罚）。
        target_orientation = _aspect_orientation(video_aspect)
        material_orientation = _material_orientation(item)
        orientation_penalty = 0.0
        if (
            target_orientation
            and material_orientation
            and material_orientation != target_orientation
        ):
            orientation_penalty = _ORIENTATION_MISMATCH_PENALTY

        return (
            relevance_penalty * 1000.0
            + duration_diff
            + priority_score
            + res_score
            + image_penalty
            + orientation_penalty
        )

    # Sort candidates by score
    ranked = sorted(candidates, key=score_material)
    return ranked


def _auto_provider_configs() -> list[tuple[str, Callable]]:
    """
    返回 auto 来源实际启用的素材供应商列表。

    只纳入用户已完整配置的供应商：缺 API Key 的来源（custom_api、coverr 等）
    会被跳过，而不是每次搜索都发出注定失败的请求。没有配置任何供应商时
    返回空列表，调用方据此直接失败并给出明确提示。

    若用户通过配置 ``auto_providers``（界面“Auto 素材来源”多选）指定了参与
    来源，则只返回用户选中的供应商；顺序仍按内置优先级
    custom_api → pexels → pixabay → coverr → web_scrape → pollinations，
    避免多选框的返回顺序给排序引入歧义。
    """
    from app.services import custom_media as _custom_media_svc

    providers: list[tuple[str, Callable]] = []
    if _custom_media_svc.is_custom_api_configured():
        providers.append(("custom_api", _custom_media_svc.search_media_custom_api))
    if _custom_media_svc.is_gemini_image_configured():
        # Nano Banana：高质量免费图片源（quota 耗尽自动回退 pollinations）。
        providers.append(
            ("gemini_image", _custom_media_svc.search_media_gemini_image)
        )
    if config.app.get("pexels_api_keys"):
        providers.append(("pexels", search_videos_pexels))
    if config.app.get("pixabay_api_keys"):
        providers.append(("pixabay", search_videos_pixabay))
    if config.app.get("coverr_api_keys"):
        providers.append(("coverr", search_videos_coverr))
    if config.app.get("enable_web_scraping", False):
        from app.services import web_scrape as _web_scrape_svc

        providers.append(("web_scrape", _web_scrape_svc.search_videos_web_scrape))
    if _custom_media_svc.is_pollinations_enabled():
        # 免费无密钥供应商放在最后：只在真实视频素材不足时兜底，不抢优先级。
        providers.append(
            ("pollinations", _custom_media_svc.search_media_pollinations)
        )

    user_selected = config.app.get("auto_providers")
    if isinstance(user_selected, list) and user_selected:
        selected = set(user_selected)
        providers = [
            (name, func) for name, func in providers if name in selected
        ]
    return providers


def _search_videos_auto_all_sources(
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect,
    failed_providers: set[str] | None = None,
) -> list[MaterialInfo]:
    """
    Concurrently query all configured providers and rank the results.

    ``failed_providers`` 是任务级的失败记录：某个供应商在本次任务里抛出
    异常后，后续搜索词不再请求它（F2：跳过失败供应商，不重复查询）。
    供应商“正常返回空结果”不视为失败，可能是该关键词确实没有素材。
    """
    import concurrent.futures

    providers = _auto_provider_configs()
    if failed_providers:
        providers = [
            (name, func) for name, func in providers if name not in failed_providers
        ]

    if not providers:
        logger.warning(
            "auto video source has no configured providers; "
            "set custom_api_url/custom_api_key or a stock footage API key"
        )
        return []

    all_candidates: list[MaterialInfo] = []

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(len(providers), _AUTO_SEARCH_MAX_WORKERS) or 1
    ) as executor:
        future_to_provider = {}
        for provider_name, search_func in providers:
            future = executor.submit(
                _search_videos_with_cache,
                provider=provider_name,
                search_videos=search_func,
                search_term=search_term,
                minimum_duration=minimum_duration,
                video_aspect=video_aspect,
            )
            future_to_provider[future] = provider_name

        for future in concurrent.futures.as_completed(future_to_provider):
            provider_name = future_to_provider[future]
            try:
                items = future.result()
                if items:
                    all_candidates.extend(items)
            except Exception as e:
                logger.warning(f"Error fetching from {provider_name}: {e}")
                if failed_providers is not None:
                    failed_providers.add(provider_name)

    # Rank and select the best material
    ranked_candidates = _rank_and_select_best_material(
        all_candidates, minimum_duration, video_aspect
    )

    # We return the ranked list so the caller (download_videos) can pick the top one as usual
    return ranked_candidates


def _is_image_material(item: MaterialInfo) -> bool:
    """True when the material is a still image (needs Ken Burns animation)."""
    source = item.source_info if isinstance(item.source_info, dict) else {}
    return source.get("media_type") == "image"


def _download_material_item(
    item: MaterialInfo,
    material_directory: str,
    *,
    search_term: str | None = None,
) -> str:
    """在工作线程内下载单个素材：只做网络 IO 与磁盘写入，返回本地路径，失败返回空串。

    ``search_term`` 仅用于保留脚本顺序路径的日志上下文；调用方统一处理两个
    下载循环的其它文案差异。
    """
    source_info = item.source_info if isinstance(item.source_info, dict) else {}
    if search_term is not None:
        logger.info(
            f"downloading ordered {item.provider} video for {search_term!r}: "
            f"asset_id={source_info.get('asset_id') or 'unknown'}"
        )
    else:
        logger.info(
            f"downloading {item.provider} video: "
            f"asset_id={source_info.get('asset_id') or 'unknown'}"
        )
    if item.provider == "web_scrape":
        import hashlib

        from app.services import web_scrape as _web_scrape_svc

        # Generate unique filename for the scraped video
        file_name = f"web_scrape_{hashlib.md5(item.url.encode()).hexdigest()}.mp4"
        saved_video_path = os.path.join(material_directory, file_name)
        success = _web_scrape_svc.download_web_video(item.url, saved_video_path)
        # yt-dlp 返回成功不代表容器完好，重新用 VideoFileClip 探测，损坏文件在
        # 下载阶段就按失败处理，避免最终渲染时整条任务炸掉。
        return _validate_scraped_video(saved_video_path) if success else ""
    return save_video(video_url=item.url, save_dir=material_directory)


def _concurrent_prefix_len(
    tasks, max_clip_duration: int, audio_duration: float
) -> int:
    """返回无论单个下载成败与否都必须先处理的前置片段数。

    串行循环在累计时长第一次超过 ``audio_duration`` 时才停止；此前所有片段都
    必然会被下载，可以放心并发。一旦超出该前缀，是否继续取决于前面结果的
    实际时长，必须回到顺序下载，否则会多下素材、改变返回集合。
    """
    total = 0.0
    for index, (item, _) in enumerate(tasks):
        total += min(max_clip_duration, item.duration)
        if total > audio_duration:
            return index
    return len(tasks)


def _download_with_concurrent_prefix(
    tasks,
    material_directory: str,
    max_clip_duration: int,
    audio_duration: float,
    video_paths: list[str],
    material_sources: list[dict[str, Any]],
) -> float:
    """有界并发下载 ``tasks``（``(item, search_term_or_None)``，顺序即串行处理顺序）。

    - 必下前缀：一次性提交给 ``MAX_DOWNLOAD_WORKERS`` 个 worker，结果按提交
      顺序处理，``video_paths`` 与 ``material_sources`` 顺序和串行实现完全一致。
    - 顺序尾部：逐个下载，保证"总时长超了就停"的判定与串行实现一致（前缀里
      若有片段下载失败，实际累计时长更低，尾部会继续补足，行为不变）。
    - 错误隔离：单个片段抛异常只记日志，不影响其它片段；返回集合不变。
    返回最终累计时长（调用方不再使用）。
    """
    total_duration = 0.0
    prefix_len = _concurrent_prefix_len(tasks, max_clip_duration, audio_duration)

    def record_success(item, saved_video_path, search_term) -> None:
        nonlocal total_duration
        if saved_video_path:
            logger.info(f"video saved: {saved_video_path}")
            video_paths.append(saved_video_path)
            try:
                material_sources.append(
                    _material_source_record(item, saved_video_path)
                )
            except Exception as source_error:
                # 来源记录异常不能把已经成功下载的素材视为下载失败，更不能
                # 阻断视频生成；保留供应商和异常类型用于后续定位。
                log_key = "ordered " if search_term is not None else ""
                logger.warning(
                    f"failed to prepare {log_key}material source record: "
                    f"provider={item.provider}, "
                    f"error={type(source_error).__name__}, detail={source_error}"
                )
            total_duration += min(max_clip_duration, item.duration)
            _enforce_video_cache_limit_quietly()

    def log_failure(item, search_term, error) -> None:
        log_key = "ordered " if search_term is not None else ""
        logger.error(
            f"failed to download {log_key}material video: "
            f"provider={item.provider}, error={type(error).__name__}, "
            f"detail={_redact_request_error(error, item.url)}"
        )

    if prefix_len:
        prefix_tasks = tasks[:prefix_len]
        effective_workers = _effective_download_workers(prefix_len)
        with ThreadPoolExecutor(max_workers=effective_workers) as pool:
            futures = {
                pool.submit(
                    _download_material_item,
                    item,
                    material_directory,
                    search_term=search_term,
                ): index
                for index, (item, search_term) in enumerate(prefix_tasks)
            }
            outcomes: dict[int, tuple[str | None, Exception | None]] = {}
            for future in as_completed(futures):
                index = futures[future]
                try:
                    outcomes[index] = (future.result(), None)
                except Exception as exc:  # noqa: BLE001 - 下载线程异常逐条隔离
                    outcomes[index] = (None, exc)
        for index, (item, search_term) in enumerate(prefix_tasks):
            saved_video_path, error = outcomes[index]
            if error is not None:
                log_failure(item, search_term, error)
                continue
            record_success(item, saved_video_path, search_term)
            if total_duration > audio_duration:
                logger.info(
                    f"total duration of downloaded videos: {total_duration} seconds, skip downloading more"
                )
                return total_duration

    # 前缀可能因部分失败而不足；剩余候选保持原有顺序逐个下载。
    for item, search_term in tasks[prefix_len:]:
        try:
            saved_video_path = _download_material_item(
                item, material_directory, search_term=search_term
            )
        except Exception as e:
            log_failure(item, search_term, e)
            continue
        record_success(item, saved_video_path, search_term)
        if total_duration > audio_duration:
            logger.info(
                f"total duration of downloaded videos: {total_duration} seconds, skip downloading more"
            )
            break
    return total_duration


def download_videos(
    task_id: str,
    search_terms: list[str],
    source: str = "pexels",
    video_aspect: VideoAspect = VideoAspect.portrait,
    video_concat_mode: VideoConcatMode = VideoConcatMode.random,
    audio_duration: float = 0.0,
    max_clip_duration: int = 5,
    match_script_order: bool = False,
    allow_images: bool = True,
    grouped_search_terms: list[list[str]] | None = None,
    scene_narrations: list[str] | None = None,
    scene_durations: list[float] | None = None,
) -> list[str]:
    provider = "pexels"
    remote_search_videos = search_videos_pexels
    if source == "pixabay":
        provider = "pixabay"
        remote_search_videos = search_videos_pixabay
    elif source == "coverr":
        provider = "coverr"
        remote_search_videos = search_videos_coverr
    elif source == "custom_api":
        # custom_api uses its own hybrid wrapper; bypass the generic cache path
        # so the fallback logic can itself call _search_videos_with_cache.
        provider = "custom_api"
        remote_search_videos = None
    elif source == "pollinations":
        provider = "pollinations"
        remote_search_videos = None
    elif source == "gemini_image":
        # Nano Banana 高质量免费图片源：quota 耗尽自动回退 pollinations。
        provider = "gemini_image"
        remote_search_videos = None
    elif source == "web_scrape":
        from app.services import web_scrape as _web_scrape_svc

        provider = "web_scrape"
        remote_search_videos = _web_scrape_svc.search_videos_web_scrape
    elif source == "auto":
        provider = "auto"
        remote_search_videos = None

    def search_videos(
        search_term: str,
        minimum_duration: int,
        video_aspect: VideoAspect,
        failed_providers: set[str] | None = None,
    ) -> list[MaterialInfo]:
        if provider == "custom_api":
            items = _search_videos_custom_api_with_fallback(
                search_term=search_term,
                minimum_duration=minimum_duration,
                video_aspect=video_aspect,
            )
        elif provider == "pollinations":
            from app.services import custom_media as _custom_media_svc

            items = _custom_media_svc.search_media_pollinations(
                search_term=search_term,
                minimum_duration=minimum_duration,
                video_aspect=video_aspect,
            )
        elif provider == "gemini_image":
            from app.services import custom_media as _custom_media_svc

            items = _custom_media_svc.search_media_gemini_image(
                search_term=search_term,
                minimum_duration=minimum_duration,
                video_aspect=video_aspect,
            )
        elif provider == "auto":
            items = _search_videos_auto_all_sources(
                search_term=search_term,
                minimum_duration=minimum_duration,
                video_aspect=video_aspect,
                failed_providers=failed_providers,
            )
        else:
            items = _search_videos_with_cache(
                provider=provider,
                search_videos=remote_search_videos,
                search_term=search_term,
                minimum_duration=minimum_duration,
                video_aspect=video_aspect,
            )
        if not allow_images:
            items = [
                item
                for item in items
                if not _is_image_material(item)
            ]
        # Second-pass synonym search: if no results, ask LLM for 2 alternatives and retry once
        if not items:
            synonyms = _get_synonym_terms(search_term)
            for syn in synonyms:
                logger.info(
                    f"synonym fallback: {search_term!r} -> {syn!r} (provider={provider})"
                )
                try:
                    if provider == "custom_api":
                        retry_items = _search_videos_custom_api_with_fallback(
                            search_term=syn,
                            minimum_duration=minimum_duration,
                            video_aspect=video_aspect,
                        )
                    elif provider == "pollinations":
                        from app.services import custom_media as _custom_media_svc2

                        retry_items = _custom_media_svc2.search_media_pollinations(
                            search_term=syn,
                            minimum_duration=minimum_duration,
                            video_aspect=video_aspect,
                        )
                    elif provider == "auto":
                        retry_items = _search_videos_auto_all_sources(
                            search_term=syn,
                            minimum_duration=minimum_duration,
                            video_aspect=video_aspect,
                            failed_providers=failed_providers,
                        )
                    else:
                        retry_items = _search_videos_with_cache(
                            provider=provider,
                            search_videos=remote_search_videos,
                            search_term=syn,
                            minimum_duration=minimum_duration,
                            video_aspect=video_aspect,
                        )
                except Exception as exc:
                    logger.warning(
                        f"synonym search failed for {syn!r}: {type(exc).__name__}: {exc}"
                    )
                    continue
                if not allow_images:
                    retry_items = [
                        item
                        for item in retry_items
                        if not _is_image_material(item)
                    ]
                if retry_items:
                    return retry_items
        return items

    material_directory = config.app.get("material_directory", "").strip()
    if material_directory == "task":
        material_directory = utils.task_dir(task_id)
    elif material_directory and not os.path.isdir(material_directory):
        material_directory = ""

    # 任务级失败记录：auto 来源中某个供应商抛异常后，后续关键词不再查询它。
    failed_providers: set[str] = set()

    # 场景分组精准匹配：当上游已按场景产出分组搜索词时，按场景顺序分配素材
    if grouped_search_terms:
        grouped_result = _download_videos_grouped(
            task_id=task_id,
            grouped_search_terms=grouped_search_terms,
            scene_narrations=scene_narrations,
            scene_durations=scene_durations,
            search_videos=search_videos,
            video_aspect=video_aspect,
            audio_duration=audio_duration,
            max_clip_duration=max_clip_duration,
            material_directory=material_directory,
            failed_providers=failed_providers,
        )
        if grouped_result:
            return grouped_result
        logger.warning("grouped scene download yielded no videos, falling back to flat search")

    if match_script_order:
        return _download_videos_by_script_order(
            task_id=task_id,
            search_terms=search_terms,
            search_videos=search_videos,
            video_aspect=video_aspect,
            audio_duration=audio_duration,
            max_clip_duration=max_clip_duration,
            material_directory=material_directory,
            failed_providers=failed_providers,
        )

    valid_video_items = []
    valid_video_urls = []
    found_duration = 0.0
    for search_term in search_terms:
        video_items = search_videos(
            search_term=search_term,
            minimum_duration=max_clip_duration,
            video_aspect=video_aspect,
            failed_providers=failed_providers,
        )
        logger.info(f"found {len(video_items)} videos for '{search_term}'")

        for item in video_items:
            if item.url not in valid_video_urls:
                valid_video_items.append(item)
                valid_video_urls.append(item.url)
                found_duration += item.duration

    logger.info(
        f"found total videos: {len(valid_video_items)}, required duration: {audio_duration} seconds, found duration: {found_duration} seconds"
    )
    video_paths = []
    material_sources: list[dict[str, Any]] = []

    concat_mode_value = getattr(video_concat_mode, "value", video_concat_mode)
    if concat_mode_value == VideoConcatMode.random.value:
        random.shuffle(valid_video_items)

    _download_with_concurrent_prefix(
        tasks=[(item, None) for item in valid_video_items],
        material_directory=material_directory,
        max_clip_duration=max_clip_duration,
        audio_duration=audio_duration,
        video_paths=video_paths,
        material_sources=material_sources,
    )
    logger.success(f"downloaded {len(video_paths)} videos")
    _persist_material_sources(task_id, material_sources)
    return video_paths


def _download_videos_by_script_order(
    task_id: str,
    search_terms: list[str],
    search_videos,
    video_aspect: VideoAspect,
    audio_duration: float,
    max_clip_duration: int,
    material_directory: str,
    failed_providers: set[str] | None = None,
) -> list[str]:
    """
    按脚本文案顺序下载素材。

    默认下载逻辑会把所有关键词的候选素材合并成一个大列表；如果第一个
    关键词返回很多结果，最终下载时可能一直消耗这个关键词的素材，后续
    脚本主题就排不上时间线。这里按关键词分组后轮询下载：
    第 1 轮取每个关键词的第 1 个候选，第 2 轮取每个关键词的第 2 个候选。
    这样在不重写视频合成引擎的前提下，尽量保证素材顺序贴近文案顺序。
    """
    logger.info("downloading videos with script-order material matching")
    candidate_groups = []
    valid_video_urls = set()
    found_duration = 0.0

    for search_term in search_terms:
        video_items = search_videos(
            search_term=search_term,
            minimum_duration=max_clip_duration,
            video_aspect=video_aspect,
            failed_providers=failed_providers,
        )
        logger.info(f"found {len(video_items)} videos for '{search_term}'")

        term_items = []
        for item in video_items:
            if item.url in valid_video_urls:
                continue
            term_items.append(item)
            valid_video_urls.add(item.url)
            found_duration += item.duration

        if term_items:
            candidate_groups.append((search_term, term_items))

    logger.info(
        f"found total ordered video candidates: {sum(len(items) for _, items in candidate_groups)}, "
        f"required duration: {audio_duration} seconds, found duration: {found_duration} seconds"
    )

    video_paths = []
    material_sources: list[dict[str, Any]] = []

    # 与串行循环完全相同的轮询顺序：第 1 轮取每个关键词的第 1 个候选，第 2 轮
    # 取第 2 个候选……展开成有序任务表后交给并发下载，返回顺序保持一致。
    ordered_tasks = []
    max_group_size = (
        max(len(items) for _, items in candidate_groups) if candidate_groups else 0
    )
    for candidate_index in range(max_group_size):
        for search_term, term_items in candidate_groups:
            if candidate_index < len(term_items):
                ordered_tasks.append((term_items[candidate_index], search_term))

    _download_with_concurrent_prefix(
        tasks=ordered_tasks,
        material_directory=material_directory,
        max_clip_duration=max_clip_duration,
        audio_duration=audio_duration,
        video_paths=video_paths,
        material_sources=material_sources,
    )
    logger.success(f"downloaded {len(video_paths)} ordered videos")
    _persist_material_sources(task_id, material_sources)
    return video_paths


def _download_videos_grouped(
    task_id: str,
    grouped_search_terms: list[list[str]],
    scene_narrations: list[str] | None,
    search_videos,
    video_aspect: VideoAspect,
    audio_duration: float,
    max_clip_duration: int,
    material_directory: str,
    failed_providers: set[str] | None = None,
    scene_durations: list[float] | None = None,
) -> list[str]:
    """
    按场景分组精准下载：每个场景的搜索词只为该场景服务，素材按场景顺序
    连续排列，确保最终成片的画面时序与文案段落一一对应。

    与轮询式顺序不同，这里每个场景的候选按场景块连续排列：
      scene0.term0 clips, scene0.term1 clips, scene1.term0 clips...
    最终视频按此顺序拼接时，scene0 的画面只出现在开头，sceneN 只在结尾。
    """
    logger.info(
        f"downloading videos grouped by scene: {len(grouped_search_terms)} scenes, "
        f"audio_duration={audio_duration:.1f}s"
    )
    # 按场景收集候选，避免跨场景 dedup 丢失场景归属
    scene_candidates: list[list[tuple[MaterialInfo, str]]] = []
    valid_urls: set[str] = set()
    total_candidates = 0

    for scene_idx, scene_terms in enumerate(grouped_search_terms):
        if not scene_terms:
            scene_candidates.append([])
            continue
        narration_preview = ""
        if scene_narrations and scene_idx < len(scene_narrations):
            narration_preview = scene_narrations[scene_idx][:80].replace("\n", " ")
        logger.info(f"scene {scene_idx} terms={scene_terms} | narration: {narration_preview}")

        scene_items: list[tuple[MaterialInfo, str]] = []
        for search_term in scene_terms:
            if not search_term or not search_term.strip():
                continue
            video_items = search_videos(
                search_term=search_term.strip(),
                minimum_duration=max_clip_duration,
                video_aspect=video_aspect,
                failed_providers=failed_providers,
            )
            logger.info(f"  scene {scene_idx} term '{search_term}' found {len(video_items)} videos")
            for item in video_items:
                if item.url in valid_urls:
                    continue
                scene_items.append((item, search_term))
                valid_urls.add(item.url)
                total_candidates += 1
        scene_candidates.append(scene_items)

    if not any(scene_candidates):
        logger.warning("no candidates found for grouped scene terms, falling back to flat")
        return []

    # 9.0+ 精准时长：优先使用基于 word timings 的 scene_durations，否则回退字符比例
    if (
        scene_durations
        and len(scene_durations) == len(grouped_search_terms)
        and audio_duration > 0
        and all(isinstance(d, (int, float)) and d > 0 for d in scene_durations)
    ):
        total = sum(scene_durations) or 1
        scale = audio_duration / total if total else 1
        scene_targets = [float(d) * scale for d in scene_durations]
        logger.info(f"using word-timing scene durations: {[round(t, 2) for t in scene_targets]}")
    elif scene_narrations and len(scene_narrations) == len(grouped_search_terms) and audio_duration > 0:
        total_chars = sum(len(n) for n in scene_narrations) or 1
        scene_targets = [
            audio_duration * len(n) / total_chars for n in scene_narrations
        ]
    else:
        per_scene = audio_duration / max(1, len(grouped_search_terms)) if audio_duration > 0 else 5.0
        scene_targets = [per_scene] * len(grouped_search_terms)

    # 可选的 TwelveLabs 视觉相关性重排：若配置了 twelvelabs_api_keys，
    # 按场景旁白对候选素材做语义重排，使画面更贴合文案。未启用或失败时保持原序。
    if scene_narrations and any(scene_narrations):
        try:
            from app.services import twelvelabs as _twelvelabs  # noqa: I001  # lazy to avoid circular import

            if _twelvelabs.is_enabled():
                import math as _math

                for _scene_idx, _scene_items in enumerate(scene_candidates):
                    if not _scene_items:
                        continue
                    _narration = ""
                    if scene_narrations and _scene_idx < len(scene_narrations):
                        _narration = str(scene_narrations[_scene_idx] or "").strip()
                    if not _narration:
                        continue
                    _narration_vec = _twelvelabs.embed_text(_narration)
                    if not _narration_vec:
                        continue
                    _scored: list[tuple] = []
                    for _item, _term in _scene_items:
                        _text = _relevance_text(_item) or str(_term or "")
                        _vec = _twelvelabs.embed_text(_text) if _text else None
                        if _vec is None:
                            _scored.append((_item, _term, -1.0))
                        else:
                            _dot = sum(x * y for x, y in zip(_narration_vec, _vec))
                            _na = _math.sqrt(sum(x * x for x in _narration_vec))
                            _nb = _math.sqrt(sum(y * y for y in _vec))
                            _cos = _dot / (_na * _nb) if _na and _nb else 0.0
                            _scored.append((_item, _term, _cos))
                    _scored.sort(key=lambda _x: _x[2], reverse=True)
                    scene_candidates[_scene_idx] = [(_it, _tm) for _it, _tm, _ in _scored]
                    logger.info(
                        f"twelvelabs reranked scene {_scene_idx} candidates: "
                        f"{len(_scored)} items by narration relevance"
                    )
        except Exception as _exc:  # noqa: BLE001 - optional feature never blocks pipeline
            logger.warning(
                f"twelvelabs scene rerank skipped: "
                f"error={type(_exc).__name__}, detail={_exc}"
            )

    # 每场景按目标时长截取候选，剩余候选作为溢出池按场景顺序追加
    ordered_tasks: list[tuple[MaterialInfo, str]] = []
    overflow: list[tuple[MaterialInfo, str]] = []
    for scene_idx, scene_items in enumerate(scene_candidates):
        target = scene_targets[scene_idx] if scene_idx < len(scene_targets) else 5.0
        acc = 0.0
        kept = 0
        for item, term in scene_items:
            # 估算该片段对目标时长的贡献
            contrib = min(float(max_clip_duration), float(getattr(item, "duration", max_clip_duration) or max_clip_duration))
            if acc >= target and kept >= 1:
                overflow.append((item, term))
            else:
                ordered_tasks.append((item, term))
                acc += contrib
                kept += 1
        logger.info(f"scene {scene_idx} target={target:.1f}s, selected {kept}/{len(scene_items)} clips, acc={acc:.1f}s")

    # 溢出池追加到末尾，保证长视频仍能填满总时长
    ordered_tasks.extend(overflow)

    logger.info(
        f"grouped candidates total={total_candidates}, grouped tasks={len(ordered_tasks)} (overflow {len(overflow)}), "
        f"required duration={audio_duration:.1f}s"
    )

    video_paths: list[str] = []
    material_sources: list[dict[str, Any]] = []
    _download_with_concurrent_prefix(
        tasks=ordered_tasks,
        material_directory=material_directory,
        max_clip_duration=max_clip_duration,
        audio_duration=audio_duration,
        video_paths=video_paths,
        material_sources=material_sources,
    )
    logger.success(f"downloaded {len(video_paths)} grouped-scene videos")
    _persist_material_sources(task_id, material_sources)
    return video_paths


if __name__ == "__main__":
    download_videos(
        "test123", ["Money Exchange Medium"], audio_duration=100, source="pixabay"
    )
