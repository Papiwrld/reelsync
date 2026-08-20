"""HTTP client for the research layer.

Implements the reliability contract shared by every provider: meaningful
User-Agent, configurable timeouts, retries with exponential backoff and
jitter, HTTP 429 handling (honoring Retry-After and reporting the event to
the request manager), 5xx retry, and graceful failure (returns None, never
raises). Network access is confined to this module.
"""

from __future__ import annotations

import random
import threading
import time
from typing import Any, Dict, Optional

import requests
from loguru import logger

from app import __version__

DEFAULT_USER_AGENT = (
    f"ReelSync/{__version__} (https://github.com/Papiwrld/reelsync; research layer)"
)
DEFAULT_TIMEOUT = (10, 30)
DEFAULT_MAX_RETRIES = 3
BACKOFF_BASE_SECONDS = 1.0
BACKOFF_MAX_SECONDS = 16.0
RETRY_AFTER_CAP_SECONDS = 30.0

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class ResearchHttpClient:
    """Synchronous, retrying GET client with centralized failure handling."""

    def __init__(
        self,
        user_agent: str = DEFAULT_USER_AGENT,
        contact_email: str = "",
        proxy: Optional[Dict[str, str]] = None,
        tls_verify: bool = True,
        timeout: tuple = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_base: float = BACKOFF_BASE_SECONDS,
        request_manager=None,
        metrics=None,
        sleep=time.sleep,
    ) -> None:
        self.user_agent = user_agent
        self.contact_email = contact_email or ""
        self.proxy = proxy or {}
        self.tls_verify = bool(tls_verify)
        self.timeout = timeout
        self.max_retries = max(0, int(max_retries))
        self.backoff_base = backoff_base
        self.request_manager = request_manager
        self.metrics = metrics
        self._sleep = sleep
        self._lock = threading.Lock()

    def build_user_agent(self) -> str:
        if not self.contact_email:
            return self.user_agent
        return f"{self.user_agent} ({self.contact_email})"

    def _backoff_delay(self, attempt: int, retry_after: float = 0.0) -> float:
        if retry_after > 0:
            return min(retry_after, RETRY_AFTER_CAP_SECONDS)
        exponential = min(
            BACKOFF_MAX_SECONDS, self.backoff_base * (2 ** max(0, attempt - 1))
        )
        return exponential + random.uniform(0, 0.25 * exponential)

    def get(
        self,
        provider: str,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        retries: Optional[int] = None,
    ) -> Optional[requests.Response]:
        """Perform a GET with retries; returns None on final failure."""
        retries = self.max_retries if retries is None else max(0, int(retries))
        request_headers = {"User-Agent": self.build_user_agent()}
        if headers:
            request_headers.update(headers)
        started = time.time()
        for attempt in range(1, retries + 2):
            try:
                response = requests.get(
                    url,
                    params=params,
                    headers=request_headers,
                    proxies=self.proxy or None,
                    verify=self.tls_verify,
                    timeout=self.timeout,
                )
                if self.metrics is not None:
                    self.metrics.record_response_time(time.time() - started)
                if response.status_code == 429:
                    retry_after = _parse_retry_after(response.headers)
                    if self.metrics is not None:
                        self.metrics.record_rate_limit(retry_after)
                    if self.request_manager is not None:
                        self.request_manager.mark_rate_limited(provider, retry_after)
                    if attempt <= retries:
                        logger.warning(
                            f"research provider rate limited: provider={provider}, "
                            f"retry_after={retry_after}, attempt={attempt}"
                        )
                        self._sleep(self._backoff_delay(attempt, retry_after))
                        continue
                    logger.warning(
                        f"research provider gave up after rate limiting: provider={provider}"
                    )
                    return None
                if response.status_code in _RETRYABLE_STATUS and attempt <= retries:
                    logger.warning(
                        f"research provider transient failure: provider={provider}, "
                        f"status={response.status_code}, attempt={attempt}"
                    )
                    self._sleep(self._backoff_delay(attempt))
                    continue
                if response.status_code >= 400:
                    if self.metrics is not None:
                        self.metrics.increment("failed_requests")
                    logger.warning(
                        f"research provider request failed: provider={provider}, "
                        f"status={response.status_code}"
                    )
                    return None
                return response
            except requests.RequestException as exc:
                if self.metrics is not None:
                    self.metrics.increment("failed_requests")
                if attempt <= retries:
                    logger.warning(
                        f"research provider request error: provider={provider}, "
                        f"error={type(exc).__name__}, attempt={attempt}"
                    )
                    self._sleep(self._backoff_delay(attempt))
                    continue
                logger.warning(
                    f"research provider gave up after request error: "
                    f"provider={provider}, error={type(exc).__name__}"
                )
                return None
        return None


def _parse_retry_after(headers) -> float:
    raw = headers.get("Retry-After")
    if not raw:
        return 0.0
    try:
        return max(0.0, min(float(raw), RETRY_AFTER_CAP_SECONDS))
    except (TypeError, ValueError):
        return 0.0