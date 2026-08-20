"""Deterministic, safe-zone-aware subtitle positioning.

``SubtitlePositionEngine`` computes the top-left ``(x, y)`` of a subtitle box
for a given anchor. It never lets subtitles leave the video frame and never
produces random movement:

- BOTTOM / CENTER / TOP use fixed anchors plus an explicit vertical offset
  (-200 up ... +200 down, 0 = historical position);
- DYNAMIC raises the baseline deterministically as a function of line count
  and box height so multi-line subtitles keep clear of bottom UI overlays;
- CUSTOM preserves the historical percentage-based position.

Safe-zone margins scale with video size so portrait and landscape videos
behave consistently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.models.schema import SubtitlePosition

# 通用边距：字幕与画面边缘的最小距离（相对高度的比例）。
_SAFE_MARGIN_RATIO = 0.03
# 底部安全区：为底部 UI 覆盖层（进度条、按钮、手势区域）保留的区间，
# 多行字幕或 DYNAMIC 模式下按行数与字幕高度把文字抬出该区间。
_BOTTOM_SAFE_ZONE_RATIO = 0.12
# DYNAMIC 模式的最大抬升幅度（相对高度）。
_MAX_DYNAMIC_RAISE_RATIO = 0.30


@dataclass(frozen=True)
class SubtitlePositionInput:
    video_width: int
    video_height: int
    box_width: int
    box_height: int
    line_count: int = 1
    anchor: SubtitlePosition = SubtitlePosition.BOTTOM
    vertical_offset: int = 0
    custom_position: float = 70.0
    safe_margin: Optional[int] = None
    dynamic_auto_avoidance: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "video_width", max(1, int(self.video_width))
        )
        object.__setattr__(
            self, "video_height", max(1, int(self.video_height))
        )


@dataclass(frozen=True)
class SubtitlePositionResult:
    x: int
    y: int
    bounds: tuple  # (left, top, right, bottom)

    @property
    def width(self) -> int:
        return self.bounds[2] - self.bounds[0]

    @property
    def height(self) -> int:
        return self.bounds[3] - self.bounds[1]


class SubtitlePositionEngine:
    """根据锚点与安全区计算字幕的确定位置。"""

    def compute(self, input_: SubtitlePositionInput) -> SubtitlePositionResult:
        width = input_.video_width
        height = input_.video_height
        box_width = max(1, int(input_.box_width))
        box_height = max(1, int(input_.box_height))
        margin = (
            input_.safe_margin
            if input_.safe_margin is not None
            else max(4, int(height * _SAFE_MARGIN_RATIO))
        )
        offset = int(input_.vertical_offset or 0)

        anchor = input_.anchor
        if anchor == SubtitlePosition.BOTTOM:
            base_y = height * 0.95 - box_height
        elif anchor == SubtitlePosition.TOP:
            base_y = height * 0.05
        elif anchor == SubtitlePosition.CENTER:
            base_y = (height - box_height) / 2.0
        elif anchor == SubtitlePosition.DYNAMIC:
            base_y = self._dynamic_baseline(input_)
        else:  # CUSTOM：历史百分比定位
            base_y = (height - box_height) * (float(input_.custom_position) / 100.0)

        y = int(round(base_y)) + offset
        y = _clamp(y, margin, height - box_height - margin)

        x = int(round((width - box_width) / 2.0))
        x = max(margin, min(x, width - box_width - margin))

        return SubtitlePositionResult(
            x=x,
            y=y,
            bounds=(x, y, x + box_width, y + box_height),
        )

    def _dynamic_baseline(self, input_: SubtitlePositionInput) -> float:
        """DYNAMIC 锚点的确定性基线。

        基础位置是底部。根据行数与字幕高度把文字抬升，使多行字幕保持在
        底部安全区之上；抬升量是行数与字幕高度的确定性函数，绝不随机。
        """
        height = input_.video_height
        box_height = max(1, int(input_.box_height))
        bottom_y = height * 0.95 - box_height
        margin = max(4, int(height * _SAFE_MARGIN_RATIO))

        if not input_.dynamic_auto_avoidance:
            return bottom_y

        safe_zone_px = height * _BOTTOM_SAFE_ZONE_RATIO
        line_raise = min(1.0, max(0.0, (input_.line_count - 1) / 2.0))
        # 字幕本身过高（长句多行）时按“底部留白不够”的程度继续抬升。
        height_raise = min(
            1.0,
            max(0.0, (box_height - safe_zone_px) / max(safe_zone_px, 1.0)),
        )
        raise_ratio = min(1.0, max(line_raise, height_raise))
        return max(margin, bottom_y - height * _MAX_DYNAMIC_RAISE_RATIO * raise_ratio)


def _clamp(value: float, low: float, high: float) -> float:
    if high < low:
        high = low
    return max(low, min(value, high))
