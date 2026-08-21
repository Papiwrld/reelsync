"""ReelSync MCP server (Model Context Protocol) using FastMCP.

Exposes the ReelSync generation pipeline as MCP tools so AI agents
(Claude, Cursor, OpenCode, …) can create and monitor videos directly:

    claude mcp add reelsync http://127.0.0.1:8080/mcp

Tools:
    create_video       start a generation task
    get_task_status    poll a task by id
    list_tasks         list recent tasks
    generate_script    draft a script for a subject (no media)
"""

from __future__ import annotations

import json

from loguru import logger

from mcp.server.fastmcp import FastMCP

from app.config import config
from mcp.server.transport_security import TransportSecuritySettings

# streamable_http_path="/" so the app serves at its root; asgi.py mounts it
# under /mcp, so external requests to /mcp reach the internal "/" route.
_MCP = FastMCP(
    "ReelSync",
    log_level="WARNING",
    streamable_http_path="/",
    # 嵌入在本地 FastAPI 中时，DNS 反绑定保护（默认 localhost 自动启用）
    # 会拒绝来自其他 host 头的请求（如 testserver）。此处显式关闭，因为
    # FastAPI 层已经通过 listen_host 和 CORS 提供了访问控制。
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=False,
    ),
)


def _authorized() -> bool:
    expected = config.app.get("api_key", "")
    return not expected  # local mode, no key required


@_MCP.tool()
def create_video(video_subject: str, video_script: str = "", video_terms: list[str] | None = None,
                  video_source: str = "auto", video_aspect: str = "9:16",
                  paragraph_number: int = 1, video_duration_seconds: float = 0.0,
                  voice_name: str = "", subtitle_enabled: bool = True) -> str:
    """Create a short video generation task. Returns a task_id to poll with get_task_status."""
    if not video_subject or not video_subject.strip():
        return "Error: video_subject is required"
    from app.models.schema import TaskVideoRequest
    from app.controllers.v1.video import create_task

    class _FakeRequest:
        headers = {}

    body = TaskVideoRequest(
        video_subject=video_subject.strip(),
        video_script=video_script.strip(),
        video_terms=video_terms or [],
        video_source=video_source,
        video_aspect=video_aspect,
        paragraph_number=paragraph_number,
        video_duration_seconds=video_duration_seconds,
        voice_name=voice_name.strip(),
        subtitle_enabled=subtitle_enabled,
    )
    try:
        response = create_task(_FakeRequest(), body, stop_at="video")
        data = response.get("data") or {}
        task_id = data.get("task_id")
        if task_id:
            return f"Task created: task_id={task_id}"
        return f"Error: create_task returned no task_id: {response}"
    except Exception as exc:
        logger.exception("mcp create_video failed")
        return f"Error: create_video failed: {exc}"


@_MCP.tool()
def get_task_status(task_id: str) -> str:
    """Get the current state of a generation task. Returns a JSON object with state, progress, videos, error."""
    from app.services import state as sm

    try:
        task = sm.state.get_task(task_id)
    except Exception as exc:
        return f"Error: failed to read task: {exc}"
    if not task:
        return f"Error: task not found: {task_id}"
    return json.dumps(task, ensure_ascii=False, indent=2)


@_MCP.tool()
def list_tasks(limit: int = 10) -> str:
    """List recent generation tasks (newest first)."""
    from app.services import state as sm

    limit = min(limit, 50)
    try:
        tasks, _ = sm.state.get_all_tasks(1, limit)
    except Exception as exc:
        return f"Error: failed to list tasks: {exc}"
    summary = []
    for task in tasks:
        summary.append({
            "task_id": task.get("task_id"),
            "state": task.get("state"),
            "progress": task.get("progress"),
            "video_subject": task.get("video_subject"),
            "failed_stage": task.get("failed_stage"),
            "error": task.get("error"),
        })
    return json.dumps(summary, ensure_ascii=False, indent=2)


@_MCP.tool()
def generate_script(video_subject: str, language: str = "", paragraph_number: int = 1) -> str:
    """Draft a video script for a subject using the configured LLM (no media generated)."""
    if not video_subject or not video_subject.strip():
        return "Error: video_subject is required"
    from app.services import llm as llm_service

    try:
        script = llm_service.generate_script(
            video_subject=video_subject.strip(),
            language=language.strip() or None,
            paragraph_number=paragraph_number,
        )
    except Exception as exc:
        return f"Error: generate_script failed: {exc}"
    if script and script.startswith("Error:"):
        return script
    return str(script or "")


def _strip_prefix_middleware(inner_app, prefix: str):
    """ASGI wrapper rewriting the path seen by the mounted MCP app.

    Starlette/FastAPI mounts set ``scope['root_path']`` to the mount prefix
    but leave ``scope['path']`` with the full URL path (e.g. ``/mcp/``),
    while the MCP Streamable HTTP transport serves at ``/``. We strip the
    mount prefix so the transport's own routing matches.
    """

    async def wrapped(scope, receive, send):
        if scope.get("type") == "http":
            path = scope.get("path", "")
            if path == prefix or path == prefix + "/":
                scope = dict(scope)
                scope["path"] = "/"
                scope["raw_path"] = b"/"
            elif path.startswith(prefix + "/"):
                scope = dict(scope)
                scope["path"] = path[len(prefix):]
                scope["raw_path"] = scope["path"].encode("utf-8")
        await inner_app(scope, receive, send)

    return wrapped


def get_mcp_app():
    """Return the FastMCP Streamable HTTP Starlette app to mount at /mcp."""
    return _strip_prefix_middleware(_MCP.streamable_http_app(), "/mcp")


def get_mcp_server() -> FastMCP:
    return _MCP