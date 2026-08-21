"""Task completion webhooks for ReelSync.

When a generation task reaches a terminal state (completed / failed), a JSON
payload is POSTed to the configured ``webhook_url`` (optionally signed with
HMAC-SHA256 when ``webhook_secret`` is set). This lets external automation
(n8n, Zapier, cron, MCP agents) react to results without polling the API.

Configuration (config.toml [app] section):
    webhook_url      URL to receive task-completion notifications.
    webhook_secret   Optional shared secret; when set, the request carries a
                     ``X-ReelSync-Signature`` header with the HMAC-SHA256 of
                     the raw body. Empty secret -> no signature header.

Delivery is best-effort and never blocks the pipeline: sends happen on a
background thread with bounded retries. The final outcome (``delivered`` /
``failed`` / ``skipped``) is recorded in the task's ``webhook_state`` field and
appended to its ``warnings`` list so callers can see what happened without
polling logs. Failures are logged, never raised.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import threading
import time
import urllib.error
import urllib.request

from loguru import logger

_WEBHOOK_TIMEOUT_SECONDS = 10
_WEBHOOK_MAX_RETRIES = 3
_WEBHOOK_RETRY_BACKOFF_SECONDS = (1.0, 2.0, 4.0)
_WEBHOOK_SIGNATURE_HEADER = "X-ReelSync-Signature"


def _webhook_config(app_config) -> tuple[str, str]:
    """Return (webhook_url, webhook_secret) from the runtime app config."""
    try:
        url = str((app_config or {}).get("webhook_url", "") or "").strip()
        secret = str((app_config or {}).get("webhook_secret", "") or "").strip()
    except Exception:  # noqa: BLE001 - config read must never raise
        return "", ""
    return url, secret


def is_webhook_configured(app_config=None) -> bool:
    from app.config import config

    url, _ = _webhook_config(app_config if app_config is not None else config.app)
    return bool(url)


def _build_payload(task: dict) -> dict:
    """A compact, stable subset of a task for webhook consumers."""
    return {
        "event": "task.completed"
        if str((task or {}).get("state")) == "1"
        else "task.failed",
        "task_id": (task or {}).get("task_id", ""),
        "state": (task or {}).get("state"),
        "progress": (task or {}).get("progress", 0),
        "video_subject": (task or {}).get("video_subject", ""),
        "failed_stage": (task or {}).get("failed_stage", ""),
        "error": (task or {}).get("error", ""),
        "videos": (task or {}).get("videos", []) or [],
        "output_copies": (task or {}).get("output_copies", []) or [],
        "cross_post_state": (task or {}).get("cross_post_state"),
        "warnings": (task or {}).get("warnings") or [],
        "timestamp": time.time(),
    }


def _sign_body(body: bytes, secret: str) -> str:
    return hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()


def _deliver_once(url: str, secret: str, payload: dict) -> bool:
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "ReelSync-webhook/1.3",
    }
    if secret:
        headers[_WEBHOOK_SIGNATURE_HEADER] = _sign_body(body, secret)
    request = urllib.request.Request(
        url, data=body, headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(  # noqa: S310 - user-configured endpoint
            request, timeout=_WEBHOOK_TIMEOUT_SECONDS
        ) as response:
            if 200 <= response.status < 300:
                return True
            logger.warning(
                f"webhook {url!r} returned status={response.status}"
            )
            return False
    except urllib.error.HTTPError as exc:
        logger.warning(f"webhook {url!r} returned HTTP {exc.code}")
        return False
    except (urllib.error.URLError, OSError, ValueError) as exc:
        logger.debug(f"webhook {url!r} transport error: {exc}")
        return False


def _record_webhook_outcome(task_id: str, webhook_state: str, message: str) -> None:
    """Persist the delivery outcome onto the task (best-effort, never raises).

    Records ``webhook_state`` ("delivered" / "failed" / "skipped") while
    preserving the task's terminal state/progress, and appends a human-readable
    summary to the task's existing ``warnings`` list.
    """
    from app.services import state as sm

    try:
        task = sm.state.get_task(task_id)
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"webhook: could not load task {task_id} for outcome: {exc}")
        return
    if not task:
        logger.debug(f"webhook: task {task_id} gone before outcome recorded")
        return

    warnings = list(task.get("warnings") or [])
    warnings.append(message)
    try:
        sm.state.update_task(
            task_id,
            state=task.get("state", 0),
            progress=task.get("progress", 0),
            webhook_state=webhook_state,
            warnings=warnings,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"webhook: could not record outcome for task {task_id}: {exc}")


def _send_with_retries(task_id: str, url: str, secret: str, payload: dict) -> bool:
    for attempt, delay in enumerate(_WEBHOOK_RETRY_BACKOFF_SECONDS):
        if _deliver_once(url, secret, payload):
            _record_webhook_outcome(task_id, "delivered", "webhook delivered")
            return True
        if attempt + 1 < _WEBHOOK_MAX_RETRIES:
            time.sleep(delay)
    logger.warning(f"webhook delivery failed after retries: {url!r}")
    _record_webhook_outcome(
        task_id, "failed", f"webhook delivery failed after {_WEBHOOK_MAX_RETRIES} attempts"
    )
    return False


def notify_task_terminal(task_id: str, app_config=None) -> None:
    """Fire a completion webhook for a terminal task (best-effort, async).

    Runs on a daemon thread so generation is never blocked by slow or
    unreachable webhook endpoints. The delivery outcome is recorded back on the
    task (``webhook_state`` + ``warnings``); any failure is logged and swallowed.
    """
    from app.services import state as sm

    try:
        task = sm.state.get_task(task_id)
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"webhook: could not load task {task_id}: {exc}")
        return
    if not task:
        logger.debug(f"webhook: task {task_id} not found, skipping")
        return

    from app.config import config

    url, secret = _webhook_config(
        app_config if app_config is not None else config.app
    )
    if not url:
        # 没有配置 webhook_url 时什么都不写：记录 "skipped" 会污染每个任务的
        # 状态（webhook_state + warnings），且对用户没有价值。
        logger.debug("webhook skipped: no webhook_url configured")
        return

    payload = _build_payload(task)
    threading.Thread(
        target=_send_with_retries,
        args=(task_id, url, secret, payload),
        daemon=True,
        name=f"webhook-{task_id}",
    ).start()
