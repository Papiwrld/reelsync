"""
Generic custom media provider for ReelSync.

Supports any REST-based image or video generation API (e.g. Stability AI,
Runway ML, Google Veo, DALL-E, Kling AI, Midjourney, Leonardo, etc.) via a
configurable endpoint and a small adapter layer.

Configuration (config.toml [app] section):
    custom_api_url      Base URL of the provider endpoint. Used as the
                        video and image endpoint unless the dedicated keys
                        below are set.
    custom_api_video_url
                        Optional. Video endpoint, queried first. Overrides
                        custom_api_url for video requests.
    custom_api_image_url
                        Optional. Image endpoint, used automatically only
                        when the video endpoint returns no usable items.
                        Overrides custom_api_url for image requests.
    custom_api_key      Bearer token / API key for the provider.
    custom_api_method   HTTP method: "POST" (default) or "GET".
    custom_api_response_format
                        How to parse the response:
                        "standard"  (default) - expects:
                            {"videos": [{"url": "...", "duration": 5, "width": 1080, "height": 1920}]}
                        "openai"    - OpenAI Images API-style:
                            {"data": [{"url": "..."}]}
                        "url_list"  - plain list of URL strings:
                            ["https://...", "https://..."]
                        "gemini"    - Google Gemini Interactions API (Nano
                            Banana images / Veo or Omni videos). Responses
                            carry base64 image data or auth-gated video URIs,
                            so generated media is decoded and written to the
                            local storage/generated_media directory. Uses
                            custom_api_video_model / custom_api_image_model
                            (defaults: gemini-omni-flash-preview,
                            gemini-3.1-flash-image).
    custom_api_extra_headers
                        Optional JSON string of extra HTTP headers.
    custom_api_extra_body
                        Optional JSON string merged into the request body.
    custom_api_video_model
                        Model id used for the video endpoint with the
                        "gemini" format.
    custom_api_image_model
                        Model id used for the image endpoint with the
                        "gemini" format.
    hybrid_video_mode   Boolean. When true, fall back to Pexels automatically
                        if the custom API returns no results or fails.
"""

import base64
import hashlib
import json
import os
import tempfile
import threading
import time
import urllib.parse
from typing import Any, List

import requests
from loguru import logger

from app.config import config
from app.models.schema import MaterialInfo, VideoAspect, video_aspect_from_string
from app.services.media_utils import (
    get_tls_verify,
    redact_request_error,
    safe_public_url,
)
from app.utils import utils

_IMAGE_DEFAULT_DURATION = 5
_IMAGE_DEFAULT_WIDTH = 1080
_IMAGE_DEFAULT_HEIGHT = 1920


def _get_custom_api_cfg() -> dict:
    legacy_url = config.app.get("custom_api_url", "").strip()
    return {
        "url": legacy_url,
        "video_url": config.app.get("custom_api_video_url", "").strip()
        or legacy_url,
        "image_url": config.app.get("custom_api_image_url", "").strip()
        or legacy_url,
        "key": config.app.get("custom_api_key", "").strip(),
        "method": config.app.get("custom_api_method", "POST").strip().upper(),
        "response_format": config.app.get("custom_api_response_format", "standard")
        .strip()
        .lower(),
        "extra_headers": config.app.get("custom_api_extra_headers", ""),
        "extra_body": config.app.get("custom_api_extra_body", ""),
        "video_model": config.app.get(
            "custom_api_video_model", "gemini-omni-flash-preview"
        )
        .strip()
        or "gemini-omni-flash-preview",
        "image_model": config.app.get(
            "custom_api_image_model", "gemini-3.1-flash-image"
        )
        .strip()
        or "gemini-3.1-flash-image",
    }


def is_custom_api_configured() -> bool:
    cfg = _get_custom_api_cfg()
    return bool(cfg["key"] and (cfg["video_url"] or cfg["image_url"]))


def _build_headers(cfg: dict) -> dict:
    headers = {
        "Authorization": f"Bearer {cfg['key']}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }
    raw = cfg.get("extra_headers", "")
    if raw:
        try:
            extra = json.loads(raw)
            if isinstance(extra, dict):
                headers.update({str(k): str(v) for k, v in extra.items()})
        except (ValueError, TypeError):
            logger.warning(
                f"custom_api_extra_headers is not valid JSON, ignoring: {raw!r}"
            )
    return headers


def _build_body(search_term: str, video_aspect: VideoAspect, cfg: dict) -> dict:
    aspect = VideoAspect(video_aspect)
    width, height = aspect.to_resolution()
    body = {
        "prompt": search_term,
        "orientation": aspect.name,
        "width": width,
        "height": height,
    }
    raw = cfg.get("extra_body", "")
    if raw:
        try:
            extra = json.loads(raw)
            if isinstance(extra, dict):
                body.update(extra)
        except (ValueError, TypeError):
            logger.warning(
                f"custom_api_extra_body is not valid JSON, ignoring: {raw!r}"
            )
    return body


def _parse_standard(
    response: Any, search_term: str, video_aspect: VideoAspect
) -> List[MaterialInfo]:
    items = []
    if not isinstance(response, dict):
        return items
    entries = response.get("videos") or response.get("images") or []
    if not isinstance(entries, list):
        return items
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        url = str(entry.get("url", "")).strip()
        if not url:
            continue
        try:
            duration = int(float(entry.get("duration") or _IMAGE_DEFAULT_DURATION))
        except (TypeError, ValueError):
            duration = _IMAGE_DEFAULT_DURATION
        w = int(entry.get("width") or _IMAGE_DEFAULT_WIDTH)
        h = int(entry.get("height") or _IMAGE_DEFAULT_HEIGHT)
        raw_tags = entry.get("tags")
        tags = (
            ", ".join(str(tag) for tag in raw_tags)
            if isinstance(raw_tags, list)
            else str(raw_tags or "")
        )
        item = MaterialInfo()
        item.provider = "custom_api"
        item.url = url
        item.duration = max(duration, 1)
        item.source_info = {
            "provider": "custom_api",
            "search_term": search_term,
            "title": str(entry.get("title") or ""),
            "description": str(entry.get("description") or ""),
            "tags": tags,
            "asset_id": str(entry.get("id") or ""),
            "source_page": safe_public_url(entry.get("source_page") or url),
            "creator": entry.get("creator"),
            "rendition": {
                "id": str(entry.get("rendition_id") or "custom"),
                "width": w,
                "height": h,
            },
        }
        items.append(item)
    return items


def _parse_openai(
    response: Any, search_term: str, video_aspect: VideoAspect
) -> List[MaterialInfo]:
    items = []
    if not isinstance(response, dict):
        return items
    aspect = VideoAspect(video_aspect)
    width, height = aspect.to_resolution()
    for entry in response.get("data", []):
        if not isinstance(entry, dict):
            continue
        url = (entry.get("url") or "").strip()
        if not url:
            continue
        item = MaterialInfo()
        item.provider = "custom_api"
        item.url = url
        item.duration = _IMAGE_DEFAULT_DURATION
        item.source_info = {
            "provider": "custom_api",
            "search_term": search_term,
            "source_page": safe_public_url(url),
            "rendition": {"id": "openai", "width": width, "height": height},
        }
        items.append(item)
    return items


def _parse_url_list(
    response: Any, search_term: str, video_aspect: VideoAspect
) -> List[MaterialInfo]:
    items = []
    if not isinstance(response, list):
        return items
    aspect = VideoAspect(video_aspect)
    width, height = aspect.to_resolution()
    for entry in response:
        url = (str(entry) if not isinstance(entry, str) else entry).strip()
        if not url.startswith("http"):
            continue
        item = MaterialInfo()
        item.provider = "custom_api"
        item.url = url
        item.duration = _IMAGE_DEFAULT_DURATION
        item.source_info = {
            "provider": "custom_api",
            "search_term": search_term,
            "source_page": safe_public_url(url),
            "rendition": {"id": "url_list", "width": width, "height": height},
        }
        items.append(item)
    return items


_PARSERS = {
    "standard": _parse_standard,
    "openai": _parse_openai,
    "url_list": _parse_url_list,
}

_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"

# Gemini 高峰时段会返回 503（模型负载过高，官方建议稍后重试），429/500/502/504
# 也属于瞬时错误。做有限次数的退避重试，避免一次高峰抖动就让整轮素材搜索
# 从 Gemini 空手而归（其余供应商不受影响）。
_GEMINI_MAX_RETRIES = 3
_GEMINI_RETRY_BACKOFF_SECONDS = 3.0
_GEMINI_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


def _gemini_aspect_ratio(video_aspect: VideoAspect) -> str:
    return {
        "portrait": "9:16",
        "landscape": "16:9",
        "square": "1:1",
    }.get(VideoAspect(video_aspect).name, "9:16")


def _build_gemini_body(
    search_term: str,
    video_aspect: VideoAspect,
    cfg: dict,
    media_type: str,
) -> dict:
    if media_type == "image":
        response_format = {
            "type": "image",
            "mime_type": "image/jpeg",
            "aspect_ratio": _gemini_aspect_ratio(video_aspect),
        }
    else:
        response_format = {
            "type": "video",
            "aspect_ratio": _gemini_aspect_ratio(video_aspect),
            "delivery": "uri",
        }
    body = {
        "model": cfg[f"{media_type}_model"],
        "input": search_term,
        "response_format": response_format,
    }
    raw = cfg.get("extra_body", "")
    if raw:
        try:
            extra = json.loads(raw)
            if isinstance(extra, dict):
                body.update(extra)
        except (ValueError, TypeError):
            logger.warning(
                f"custom_api_extra_body is not valid JSON, ignoring: {raw!r}"
            )
    return body


def _download_gemini_uri(uri: str, api_key: str) -> bytes:
    """
    Download an auth-gated Gemini file URI (delivery=uri), polling until the
    file is ACTIVE. The URI is only guaranteed in the creation response.
    """
    headers = {
        "x-goog-api-key": api_key,
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }
    deadline = time.time() + 180
    while time.time() < deadline:
        try:
            resp = requests.get(
                uri,
                headers=headers,
                proxies=config.proxy,
                verify=get_tls_verify(),
                timeout=(30, 60),
            )
            if resp.status_code == 200 and resp.content:
                return resp.content
            if resp.status_code in _GEMINI_RETRYABLE_STATUS_CODES:
                logger.warning(
                    f"gemini file download hit transient status={resp.status_code}, "
                    f"polling again"
                )
            elif resp.status_code not in (404, 202):
                logger.warning(
                    f"gemini file download failed: status={resp.status_code}, "
                    f"response={resp.text[:300]!r}"
                )
                return b""
        except Exception as exc:
            logger.warning(
                f"gemini file download raised an exception: "
                f"error={type(exc).__name__}, detail={redact_request_error(exc, api_key)}"
            )
        time.sleep(5)
    logger.warning(f"gemini file download timed out: uri={uri}")
    return b""


def _save_gemini_media(
    media_type: str,
    raw: bytes,
    identifier: str,
    save_dir: str,
) -> str:
    if not raw:
        return ""
    if media_type == "video":
        ext = "mp4"
    else:
        ext = "jpg"
    name = f"gemini-{media_type}-{hashlib.md5(identifier.encode()).hexdigest()}.{ext}"
    path = os.path.join(save_dir, name)
    try:
        with open(path, "wb") as f:
            f.write(raw)
        return path
    except Exception as exc:
        logger.error(
            f"failed to save gemini {media_type} media: error={type(exc).__name__}, detail={exc}"
        )
        return ""


def _search_gemini(
    search_term: str,
    video_aspect: VideoAspect,
    cfg: dict,
    api_url: str,
    media_type: str,
    minimum_duration: int,
    save_dir: str,
) -> List[MaterialInfo]:
    """
    Query the Google Gemini Interactions API (REST, non-streaming) and decode
    the generated media into local files.

    Images arrive as base64 blocks in ``steps[].content[]``; videos use
    ``delivery: uri`` and are downloaded with the ``x-goog-api-key`` header.
    Returns an empty list on any error.
    """
    api_key = cfg["key"]
    if not api_url or not api_key:
        return []

    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }
    raw_extra = cfg.get("extra_headers", "")
    if raw_extra:
        try:
            extra = json.loads(raw_extra)
            if isinstance(extra, dict):
                headers.update({str(k): str(v) for k, v in extra.items()})
        except (ValueError, TypeError):
            logger.warning(
                f"custom_api_extra_headers is not valid JSON, ignoring: {raw_extra!r}"
            )

    body = _build_gemini_body(search_term, video_aspect, cfg, media_type)
    logger.info(
        f"searching Gemini {media_type} media: term={search_term!r}, "
        f"model={body.get('model')!r}, url={api_url!r}"
    )

    try:
        resp = requests.post(
            api_url,
            headers=headers,
            json=body,
            proxies=config.proxy,
            verify=get_tls_verify(),
            timeout=(30, 300),
        )
    except Exception as exc:
        logger.error(
            f"Gemini {media_type} request raised an exception: "
            f"error={type(exc).__name__}, detail={redact_request_error(exc, api_key)}"
        )
        return []

    for attempt in range(_GEMINI_MAX_RETRIES):
        if resp.status_code not in _GEMINI_RETRYABLE_STATUS_CODES:
            break
        logger.warning(
            f"Gemini {media_type} request hit transient status={resp.status_code} "
            f"(attempt {attempt + 1}/{_GEMINI_MAX_RETRIES}), retrying in "
            f"{_GEMINI_RETRY_BACKOFF_SECONDS * (attempt + 1)}s"
        )
        time.sleep(_GEMINI_RETRY_BACKOFF_SECONDS * (attempt + 1))
        try:
            resp = requests.post(
                api_url,
                headers=headers,
                json=body,
                proxies=config.proxy,
                verify=get_tls_verify(),
                timeout=(30, 300),
            )
        except Exception as exc:
            logger.error(
                f"Gemini {media_type} retry raised an exception: "
                f"error={type(exc).__name__}, detail={redact_request_error(exc, api_key)}"
            )
            return []

    if resp.status_code >= 400:
        logger.error(
            f"Gemini {media_type} request failed: status={resp.status_code}, "
            f"response={resp.text[:500]!r}"
        )
        return []

    try:
        data = resp.json()
    except ValueError:
        logger.error(
            f"Gemini {media_type} returned non-JSON: status={resp.status_code}, "
            f"body={resp.text[:500]!r}"
        )
        return []

    if not isinstance(data, dict):
        return []

    items: List[MaterialInfo] = []
    seen = set()
    for step in data.get("steps", []):
        if not isinstance(step, dict):
            continue
        content = step.get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != media_type:
                continue
            identifier = block.get("uri") or block.get("data") or ""
            if not identifier or identifier in seen:
                continue
            seen.add(identifier)
            raw = b""
            if media_type == "video":
                uri = block.get("uri")
                if uri:
                    raw = _download_gemini_uri(uri, api_key)
                else:
                    raw = base64.b64decode(block.get("data") or "")
            else:
                raw = base64.b64decode(block.get("data") or "")
            if not raw:
                logger.warning(
                    f"Gemini {media_type} block could not be decoded for term={search_term!r}"
                )
                continue
            path = _save_gemini_media(media_type, raw, identifier, save_dir)
            if not path:
                continue
            item = MaterialInfo()
            item.provider = "custom_api"
            item.url = path
            item.duration = max(minimum_duration, 5)
            item.source_info = {
                "provider": "custom_api",
                "search_term": search_term,
                "media_type": media_type,
                "model": str(body.get("model") or ""),
                "asset_id": str(data.get("id") or ""),
                "source_page": safe_public_url(uri if media_type == "video" else ""),
                "rendition": {"id": "gemini", "width": 0, "height": 0},
            }
            items.append(item)

    logger.info(
        f"Gemini {media_type} returned {len(items)} usable items for term={search_term!r}"
    )
    return items


def _search_media_at_url(
    search_term: str,
    video_aspect: VideoAspect,
    cfg: dict,
    api_url: str,
    minimum_duration: int,
) -> List[MaterialInfo]:
    """
    Query one endpoint and parse the response.

    Returns an empty list on any error so callers can apply fallbacks.
    """
    api_key = cfg["key"]
    if not api_url or not api_key:
        return []

    headers = _build_headers(cfg)
    method = cfg["method"]
    logger.info(
        f"searching media via custom API: term={search_term!r}, method={method}, url={api_url!r}"
    )

    try:
        if method == "GET":
            resp = requests.get(
                api_url,
                headers=headers,
                params={"prompt": search_term},
                proxies=config.proxy,
                verify=get_tls_verify(),
                timeout=(30, 120),
            )
        else:
            body = _build_body(search_term, video_aspect, cfg)
            resp = requests.post(
                api_url,
                headers=headers,
                json=body,
                proxies=config.proxy,
                verify=get_tls_verify(),
                timeout=(30, 120),
            )

        if resp.status_code >= 400:
            logger.error(
                f"custom API request failed: status={resp.status_code}, response={resp.text[:500]!r}"
            )
            return []

        try:
            data = resp.json()
        except ValueError:
            logger.error(
                f"custom API returned non-JSON: status={resp.status_code}, body={resp.text[:500]!r}"
            )
            return []

        fmt = cfg["response_format"]
        parser = _PARSERS.get(fmt)
        if parser is None:
            logger.warning(
                f"unknown custom_api_response_format={fmt!r}, using 'standard' parser"
            )
            parser = _parse_standard

        items = parser(data, search_term, video_aspect)
        for item in items:
            if item.duration < minimum_duration:
                item.duration = max(item.duration, minimum_duration)
        logger.info(
            f"custom API returned {len(items)} usable items for term={search_term!r}"
        )
        return items

    except Exception as exc:
        logger.error(
            f"custom API request raised an exception: error={type(exc).__name__}, "
            f"detail={redact_request_error(exc, api_key)}"
        )
        return []


def _mark_media_type(item: MaterialInfo, media_type: str) -> None:
    source = item.source_info if isinstance(item.source_info, dict) else {}
    if not isinstance(source, dict):
        return
    source["media_type"] = media_type
    item.source_info = source


def search_media_custom_api(
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect = VideoAspect.portrait,
    save_dir: str = "",
) -> List[MaterialInfo]:
    """
    Fetch media from the configured custom API endpoint(s).

    Provider-agnostic: any image or video generation service can be wired up
    by setting custom_api_url (or the dedicated custom_api_video_url /
    custom_api_image_url keys), custom_api_key, and optionally
    custom_api_response_format and custom_api_extra_body in config.toml.

    Video is preferred: the video endpoint is queried first; if it returns
    no usable items (or fails), the image endpoint is queried automatically
    and the resulting stills are returned as image materials.

    With the "gemini" response format, both endpoints hit the Google Gemini
    Interactions API (different models); generated media is decoded to local
    files under save_dir (defaults to storage/generated_media).

    Returns an empty list on any error so the caller can apply the Pexels
    fallback without additional handling.
    """
    cfg = _get_custom_api_cfg()
    video_url = cfg["video_url"]
    image_url = cfg["image_url"]
    fmt = cfg["response_format"]

    if not cfg["key"] or not (video_url or image_url):
        logger.warning(
            "custom_api_url/key is not configured; skipping custom API search"
        )
        return []

    if not save_dir:
        save_dir = utils.storage_dir("generated_media", create=True)

    if fmt == "gemini":
        video_items = _search_gemini(
            search_term,
            video_aspect,
            cfg,
            video_url,
            "video",
            minimum_duration,
            save_dir,
        )
        if video_items:
            return video_items
        if image_url:
            logger.info(
                f"Gemini video endpoint returned no usable items for term={search_term!r}; "
                "falling back to the image endpoint"
            )
            image_items = _search_gemini(
                search_term,
                video_aspect,
                cfg,
                image_url,
                "image",
                minimum_duration,
                save_dir,
            )
            for item in image_items:
                _mark_media_type(item, "image")
            return image_items
        return []

    video_items = _search_media_at_url(
        search_term, video_aspect, cfg, video_url, minimum_duration
    )
    if video_items:
        for item in video_items:
            _mark_media_type(item, "video")
        return video_items

    if image_url and image_url != video_url:
        logger.info(
            f"custom API video endpoint returned no usable items for term={search_term!r}; "
            "falling back to the image endpoint"
        )
        image_items = _search_media_at_url(
            search_term, video_aspect, cfg, image_url, minimum_duration
        )
        for item in image_items:
            _mark_media_type(item, "image")
        return image_items

    return []


# ---------------------------------------------------------------------------
# Pollinations: 免费无密钥图片生成供应商（Feature C + H 缓存）
#
# 使用仍保持免鉴权的旧版端点 `image.pollinations.ai/prompt/{prompt}`：
# 无需任何 API Key，模型默认 flux，支持 width/height/seed/nologo 参数。
# 生成结果按 (provider, prompt, width, height, model) 缓存到磁盘，避免
# 相同关键词反复消耗上游额度；缓存损坏（PIL 校验失败）时自动删除重建。
# ---------------------------------------------------------------------------

_POLLINATIONS_BASE_URL = "https://image.pollinations.ai/prompt/"
_POLLINATIONS_DEFAULT_MODEL = "flux"
_POLLINATIONS_TIMEOUT = (10, 180)
_POLLINATIONS_MAX_RETRIES = 2
_POLLINATIONS_RETRY_BACKOFF_SECONDS = 2.0
_POLLINATIONS_IMAGE_COUNT = 1
_POLLINATIONS_CHUNK_SIZE = 64 * 1024

# 生成图片缓存：7 天 TTL，目录位于 storage/cache_generated_media。
GENERATED_IMAGE_CACHE_TTL_SECONDS = 7 * 24 * 60 * 60
_GENERATED_IMAGE_CACHE_CLEANUP_INTERVAL_SECONDS = 60 * 60
_generated_cache_cleanup_lock = threading.Lock()
_last_generated_cache_cleanup_monotonic: float | None = None


def is_pollinations_enabled() -> bool:
    """
    Pollinations 是否参与素材搜索。

    默认策略（未显式配置 enable_pollinations 时）：没有任何其他可用供应商
    时自动启用——这就是“免费模式”：零 API Key 也能出片。已有 Pexels /
    Pixabay / Coverr / custom_api / web_scrape 任一供应商时默认关闭，
    避免每次都生成无谓的占位图片消耗上游免费额度；用户可在 config.toml
    显式设置 ``enable_pollinations = true`` 强制开启。
    """
    configured = config.app.get("enable_pollinations", None)
    if configured is not None:
        return bool(configured)
    if (
        config.app.get("pexels_api_keys")
        or config.app.get("pixabay_api_keys")
        or config.app.get("coverr_api_keys")
        or is_custom_api_configured()
        or config.app.get("enable_web_scraping", False)
    ):
        return False
    return True


def _pollinations_image_url(
    search_term: str, width: int, height: int, model: str, seed: int
) -> str:
    encoded_prompt = urllib.parse.quote(search_term, safe="")
    encoded_model = urllib.parse.quote(model, safe="")
    return (
        f"{_POLLINATIONS_BASE_URL}{encoded_prompt}"
        f"?width={int(width)}&height={int(height)}"
        f"&model={encoded_model}&nologo=true&seed={int(seed)}"
    )


def _generated_media_cache_dir() -> str:
    return utils.storage_dir("cache_generated_media", create=True)


def _generated_media_cache_key(
    provider: str, prompt: str, width: int, height: int, model: str
) -> str:
    payload = json.dumps(
        {
            "provider": str(provider).strip().lower(),
            "prompt": str(prompt).strip(),
            "width": int(width),
            "height": int(height),
            "model": str(model).strip().lower(),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _generated_media_cache_path(
    provider: str, prompt: str, width: int, height: int, model: str
) -> str:
    digest = _generated_media_cache_key(
        provider, prompt, width, height, model
    )
    return os.path.join(_generated_media_cache_dir(), f"{digest}.jpg")


def _is_valid_image_file(path: str) -> bool:
    """用 PIL 校验文件确实是完整可读的图片，损坏缓存直接判失效。"""
    if not path or not os.path.isfile(path) or os.path.getsize(path) == 0:
        return False
    try:
        from PIL import Image

        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def _load_generated_media_cache(
    provider: str, prompt: str, width: int, height: int, model: str
) -> str:
    """
    返回仍然新鲜且校验通过的缓存图片路径；过期或损坏时删除并返回空串。
    """
    cache_path = _generated_media_cache_path(provider, prompt, width, height, model)
    try:
        if not os.path.isfile(cache_path):
            return ""
        if time.time() - os.path.getmtime(cache_path) >= GENERATED_IMAGE_CACHE_TTL_SECONDS:
            os.remove(cache_path)
            return ""
        if not _is_valid_image_file(cache_path):
            logger.warning(
                f"removing corrupt generated media cache: {os.path.basename(cache_path)}"
            )
            os.remove(cache_path)
            return ""
        return cache_path
    except OSError as exc:
        logger.warning(
            f"failed to read generated media cache: file={cache_path}, error={exc}"
        )
        return ""


def _store_generated_media_cache(
    provider: str, prompt: str, width: int, height: int, model: str, raw: bytes
) -> str:
    """原子写入生成图片缓存，返回缓存路径；失败返回空串。"""
    temp_path = None
    try:
        cache_path = _generated_media_cache_path(provider, prompt, width, height, model)
        cache_dir = os.path.dirname(cache_path)
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=cache_dir,
            prefix=".gen-",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = temp_file.name
            temp_file.write(raw)
            temp_file.flush()
            os.fsync(temp_file.fileno())
        if not _is_valid_image_file(temp_path):
            raise ValueError("generated media failed PIL validation")
        os.replace(temp_path, cache_path)
        temp_path = None
        return cache_path
    except Exception as exc:
        logger.warning(
            "failed to store generated media cache: "
            f"error={type(exc).__name__}, detail={exc}"
        )
        return ""
    finally:
        if temp_path is not None:
            try:
                os.remove(temp_path)
            except OSError:
                pass


def _cleanup_expired_generated_media_cache(
    *, now: float | None = None, force: bool = False
) -> int:
    """
    低频清理过期生成图片缓存（默认每小时最多扫一次目录）。

    只删除本项目生成的 SHA-256 命名文件，不触碰用户放入目录的其它文件。
    """
    global _last_generated_cache_cleanup_monotonic

    monotonic_now = time.monotonic()
    with _generated_cache_cleanup_lock:
        if (
            not force
            and _last_generated_cache_cleanup_monotonic is not None
            and monotonic_now - _last_generated_cache_cleanup_monotonic
            < _GENERATED_IMAGE_CACHE_CLEANUP_INTERVAL_SECONDS
        ):
            return 0
        _last_generated_cache_cleanup_monotonic = monotonic_now

    cache_dir = _generated_media_cache_dir()
    current_time = time.time() if now is None else now
    deleted_count = 0
    try:
        for entry in os.scandir(cache_dir):
            if not entry.is_file(follow_symlinks=False):
                continue
            name = entry.name
            if not (len(name) == 68 and name.endswith(".jpg") and name[:64].isalnum()):
                continue
            try:
                if 0 <= current_time - entry.stat(follow_symlinks=False).st_mtime < GENERATED_IMAGE_CACHE_TTL_SECONDS:
                    continue
                try:
                    os.unlink(entry.path)
                except PermissionError:
                    # Windows 下杀毒/索引服务可能短暂持有刚写入文件的句柄，
                    # 高负载（如完整测试套件）下偶发超过单次重试窗口；
                    # 按短退避重试数次，避免清理计数偶发偏差。
                    for delay in (0.05, 0.1, 0.2):
                        time.sleep(delay)
                        try:
                            os.unlink(entry.path)
                            break
                        except PermissionError:
                            continue
                    else:
                        raise
                deleted_count += 1
            except OSError as exc:
                logger.warning(
                    f"failed to delete generated media cache file: "
                    f"file={name}, error={exc}"
                )
    except OSError as exc:
        logger.warning(f"failed to scan generated media cache: error={exc}")
    return deleted_count


def _download_pollinations_image(
    url: str, cache_path: str, model: str
) -> str:
    """
    流式下载 Pollinations 生成图片到缓存路径，PIL 校验通过后原子发布。

    网络错误和 5xx 会重试 ``_POLLINATIONS_MAX_RETRIES`` 次（指数退避）；
    最终仍然失败时返回空串并清理残留临时文件。
    """
    if not cache_path:
        logger.warning(
            "Pollinations download aborted: cache_path is empty, "
            "cannot determine target directory"
        )
        return ""
    temp_path = None
    for attempt in range(_POLLINATIONS_MAX_RETRIES + 1):
        try:
            with requests.get(
                url,
                stream=True,
                proxies=config.proxy,
                verify=get_tls_verify(),
                timeout=_POLLINATIONS_TIMEOUT,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36"
                    )
                },
            ) as resp:
                if resp.status_code >= 400:
                    logger.warning(
                        f"Pollinations image request failed: status={resp.status_code} "
                        f"(attempt {attempt + 1}/{_POLLINATIONS_MAX_RETRIES + 1})"
                    )
                    if attempt < _POLLINATIONS_MAX_RETRIES:
                        time.sleep(
                            _POLLINATIONS_RETRY_BACKOFF_SECONDS * (attempt + 1)
                        )
                    continue
                cache_dir = os.path.dirname(cache_path)
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    dir=cache_dir,
                    prefix=".pollinations-",
                    suffix=".tmp",
                    delete=False,
                ) as temp_file:
                    temp_path = temp_file.name
                    for chunk in resp.iter_content(
                        chunk_size=_POLLINATIONS_CHUNK_SIZE
                    ):
                        if chunk:
                            temp_file.write(chunk)
                    temp_file.flush()
                    os.fsync(temp_file.fileno())
                break
        except Exception as exc:
            logger.warning(
                f"Pollinations image download raised an exception: "
                f"error={type(exc).__name__}, detail={redact_request_error(exc, '')} "
                f"(attempt {attempt + 1}/{_POLLINATIONS_MAX_RETRIES + 1})"
            )
            if temp_path is not None:
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
                temp_path = None
            if attempt < _POLLINATIONS_MAX_RETRIES:
                time.sleep(_POLLINATIONS_RETRY_BACKOFF_SECONDS * (attempt + 1))

    if not temp_path:
        return ""
    try:
        if not _is_valid_image_file(temp_path):
            raise ValueError("downloaded image failed PIL validation")
        os.replace(temp_path, cache_path)
        return cache_path
    except Exception as exc:
        logger.warning(
            f"Pollinations image validation failed: error={type(exc).__name__}, "
            f"detail={exc}"
        )
        return ""
    finally:
        if temp_path is not None and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def search_media_pollinations(
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect = VideoAspect.portrait,
    save_dir: str = "",
) -> List[MaterialInfo]:
    """
    用 Pollinations（免费无密钥）根据关键词生成一张图片并返回为图片素材。

    图片分辨率按画幅取中心化映射（9:16 -> 1080x1920 等）。结果落盘在
    生成的图片缓存目录，缓存键含 provider/prompt/width/height/model；
    命中且校验通过时直接复用，不重复请求上游。图片素材后续由合成管线
    转成 Ken Burns 推镜片段，因此这里不做任何视频相关的处理。
    """
    if not is_pollinations_enabled():
        return []

    aspect = video_aspect_from_string(video_aspect)
    width, height = aspect.to_resolution()
    model = str(
        config.app.get("pollinations_image_model", _POLLINATIONS_DEFAULT_MODEL)
        or _POLLINATIONS_DEFAULT_MODEL
    ).strip()

    if save_dir:
        _generated_media_cache_dir()

    seed = int(
        hashlib.sha256(search_term.strip().encode("utf-8")).hexdigest()[:8], 16
    )
    generated_key = ("pollinations", search_term, width, height, model)
    cache_path = _load_generated_media_cache(*generated_key)
    if cache_path:
        logger.info(
            f"Pollinations image cache hit: term={search_term!r}, "
            f"model={model!r}, {width}x{height}"
        )
    else:
        url = _pollinations_image_url(search_term, width, height, model, seed)
        logger.info(
            f"generating Pollinations image: term={search_term!r}, "
            f"model={model!r}, {width}x{height}"
        )
        cache_path = _download_pollinations_image(
            url, _generated_media_cache_path(*generated_key), model
        )
        if cache_path:
            _cleanup_expired_generated_media_cache()

    if not cache_path:
        logger.warning(
            f"Pollinations could not generate an image for term={search_term!r}"
        )
        return []

    items: List[MaterialInfo] = []
    for _ in range(_POLLINATIONS_IMAGE_COUNT):
        item = MaterialInfo()
        item.provider = "pollinations"
        item.url = cache_path
        item.duration = max(minimum_duration, _IMAGE_DEFAULT_DURATION)
        # 不写入 title/tags，避免生成图片因自带关键词而相关性恒为满分，
        # 反而压过真实视频素材（F5 排名审核：同相关度下视频优先）。
        item.source_info = {
            "provider": "pollinations",
            "search_term": search_term,
            "media_type": "image",
            "model": model,
            "width": width,
            "height": height,
        }
        items.append(item)
    return items
