"""Entry point adapter for the task pipeline."""

from __future__ import annotations

from typing import Any, Callable, Optional

from app.services.intelligence import ContentRequest
from app.services.agentic.graph import plan_video_content


def _content_request_from_params(params: Any) -> Optional[ContentRequest]:
    """Build the content configuration from VideoParams (or None when the
    user left every intelligence input empty)."""
    request = ContentRequest(
        niche=getattr(params, "niche", "") or "",
        sub_niche=getattr(params, "sub_niche", "") or "",
        audience=getattr(params, "audience", "") or "",
        platform=getattr(params, "platform", "") or "",
        format=getattr(params, "content_format", "") or "",
        content_goal=getattr(params, "content_goal", "") or "",
        automation_level=getattr(params, "automation_level", "") or "",
        trend_preference=getattr(params, "trend_preference", "") or "",
        sources=[str(item) for item in (getattr(params, "sources", None) or []) if str(item).strip()],
        fact_check_override=getattr(params, "fact_check_level", "") or "",
        research_depth_override=getattr(params, "research_depth", "") or "",
    )
    return request if request.has_context else None


def plan_video_content_from_params(
    params: Any,
    app_config: Any = None,
    task_id: str = "",
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Any:
    """Entry point used by the task pipeline (video generation)."""
    return plan_video_content(
        subject=params.video_subject,
        language=params.video_language or "",
        profile_name=params.content_profile or "",
        paragraph_number=getattr(params, "paragraph_number", 1),
        target_duration_seconds=getattr(params, "video_duration_seconds", 0) or 0,
        app_config=app_config,
        user_context=_content_request_from_params(params),
        task_id=task_id,
        script_style=getattr(params, "script_style", "") or "",
        progress_cb=progress_cb,
    )