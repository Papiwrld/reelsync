"""Persistent response cache for the research layer.

Storage lives in ``storage/research_cache/`` next to the agentic research
cache, using the ``zk-`` prefix so the two never collide. Entries carry a
per-source TTL; expired or corrupt entries are lazily deleted. Concurrent
identical requests are coalesced via 256 sharded locks with a double-check
read, mirroring the project's material-search cache pattern.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from app.utils import utils

_CACHE_PREFIX = "zk-"
_CACHE_FORMAT_VERSION = 1
_CLOCK_SKEW_TOLERANCE_SECONDS = 30.0
_MIN_CACHE_TTL_SECONDS = 60
_EMPTY_RESULT_TTL_SECONDS = 3600

_CACHE_LOCKS = tuple(threading.Lock() for _ in range(256))


def _cache_dir() -> str:
    return utils.storage_dir("research_cache", create=True)


def _canonical_params(params: Dict[str, Any]) -> str:
    return json.dumps(params, sort_keys=True, ensure_ascii=False, default=str)


def cache_key(provider: str, operation: str, params: Dict[str, Any]) -> str:
    payload = f"{provider}|{operation}|{_canonical_params(params)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_path(key: str) -> str:
    return os.path.join(_cache_dir(), f"{_CACHE_PREFIX}{key}.json")


def get_cache_lock(provider: str, operation: str, params: Dict[str, Any]) -> threading.Lock:
    digest = cache_key(provider, operation, params)
    return _CACHE_LOCKS[int(digest[:2], 16)]


def _read_cache_entry(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            entry = json.load(handle)
    except (OSError, ValueError, UnicodeError):
        return None
    if not isinstance(entry, dict):
        return None
    if entry.get("version") != _CACHE_FORMAT_VERSION:
        return None
    if not isinstance(entry.get("results"), list):
        return None
    return entry


def _entry_expired(entry: Dict[str, Any], now: float) -> bool:
    expires_at = entry.get("expires_at")
    return not isinstance(expires_at, (int, float)) or expires_at <= now


def _remove_path(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def load_cached(
    provider: str,
    operation: str,
    params: Dict[str, Any],
    now: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Return a valid, non-expired cache entry or None (removing stale files)."""
    if now is None:
        now = time.time()
    path = _cache_path(cache_key(provider, operation, params))
    entry = _read_cache_entry(path)
    if entry is None or _entry_expired(entry, now):
        _remove_path(path)
        return None
    return entry


def save_cached(
    provider: str,
    operation: str,
    params: Dict[str, Any],
    results: List[Dict[str, Any]],
    metadata: Dict[str, Any],
    ttl_seconds: int,
    now: Optional[float] = None,
) -> None:
    """Atomically persist a cache entry (temp file + os.replace)."""
    if now is None:
        now = time.time()
    if not isinstance(ttl_seconds, (int, float)) or ttl_seconds <= 0:
        ttl_seconds = _MIN_CACHE_TTL_SECONDS
    if not results:
        ttl_seconds = min(ttl_seconds, _EMPTY_RESULT_TTL_SECONDS)
    entry = {
        "version": _CACHE_FORMAT_VERSION,
        "provider": provider,
        "operation": operation,
        "params": _canonical_params(params),
        "created_at": now,
        "expires_at": now + ttl_seconds,
        "results": results,
        "metadata": metadata or {},
    }
    path = _cache_path(cache_key(provider, operation, params))
    directory = os.path.dirname(path)
    try:
        os.makedirs(directory, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(prefix=".zk-", suffix=".tmp", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(entry, handle, ensure_ascii=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
        finally:
            if os.path.exists(temp_path):
                _remove_path(temp_path)
    except OSError as exc:
        logger.warning(f"failed to persist research cache: {exc}")


def cleanup_expired(force: bool = False) -> int:
    """Delete expired entries; throttled to once per hour unless forced."""
    now = time.time()
    directory = _cache_dir()
    removed = 0
    try:
        for filename in os.listdir(directory):
            if not filename.startswith(_CACHE_PREFIX) or not filename.endswith(".json"):
                continue
            path = os.path.join(directory, filename)
            entry = _read_cache_entry(path)
            if entry is None or _entry_expired(entry, now):
                if os.path.getmtime(path) + _CLOCK_SKEW_TOLERANCE_SECONDS <= now:
                    _remove_path(path)
                    removed += 1
    except OSError as exc:
        logger.warning(f"research cache cleanup failed: {exc}")
    return removed


class CachedSearch:
    """Coalescing wrapper: lockless peek, then locked double-check, then load.

    ``loader`` runs only when the entry is genuinely missing; concurrent
    callers for the same key share the single external request.
    """

    def __init__(
        self,
        provider: str,
        operation: str,
        params: Dict[str, Any],
        ttl_seconds: int,
        metrics,
        now: Optional[float] = None,
    ) -> None:
        self.provider = provider
        self.operation = operation
        self.params = params
        self.ttl_seconds = ttl_seconds
        self.metrics = metrics
        self._now = now

    def get(self, loader) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        now = self._now if self._now is not None else time.time()
        cached = load_cached(self.provider, self.operation, self.params, now=now)
        if cached is not None:
            self.metrics.increment("cache_hits")
            return cached["results"], cached.get("metadata", {})
        with get_cache_lock(self.provider, self.operation, self.params):
            cached = load_cached(self.provider, self.operation, self.params, now=now)
            if cached is not None:
                self.metrics.increment("cache_hits")
                self.metrics.increment("coalesced_requests")
                return cached["results"], cached.get("metadata", {})
            self.metrics.increment("cache_misses")
            results, metadata = loader()
            if results is not None:
                save_cached(
                    self.provider,
                    self.operation,
                    self.params,
                    results,
                    metadata,
                    self.ttl_seconds,
                    now=now,
                )
            return results or [], metadata or {}