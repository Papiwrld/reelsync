"""Deterministic subtitle animations.

All animation curves are pure functions of ``(time, duration, ...)`` so the
same input always produces the same output (reproducible renders). Animations
are opt-in: legacy behavior renders every subtitle statically unless one of
the animation toggles is explicitly enabled.
"""

from __future__ import annotations

import math
from typing import Optional


# 弹出弹跳：仅在前 0.15 秒内从 0 快速放大到 1.1，再回落稳定到 1.0。
# 曲线与历史实现逐点一致（分段线性：0 -> peak -> 1.0），保证旧配置
# 渲染结果不变。字幕时长小于该窗口时整个曲线压缩到字幕时长内。
POP_IN_WINDOW_SECONDS = 0.15
POP_IN_OVERSHOOT = 0.1
# 漂浮运动：在字幕持续时间内做一次平滑正弦漂移，幅度为视频高度的 2.5%
# （最小 4px），不会漂出上下安全区；水平漂移幅度为其一半，更加克制。
KINETIC_FLOAT_AMPLITUDE_RATIO = 0.025
KINETIC_FLOAT_MIN_AMPLITUDE = 4
KINETIC_HORIZONTAL_RATIO = 0.5
# 动态缩放：短文本最大放大 1.5 倍，长文本最多缩小到 0.6 倍。
DYNAMIC_SCALE_MAX_RATIO = 1.5
DYNAMIC_SCALE_MIN_RATIO = 0.6
DYNAMIC_UPSCALE_TRIGGER_RATIO = 0.6
DYNAMIC_UPSCALE_FILL_RATIO = 0.55
DYNAMIC_MAX_LINES = 2


def pop_in_scale(t: float, duration: float) -> float:
    """返回时刻 ``t`` 的弹出弹跳缩放系数（0.0 -> 1.1 -> 1.0）。

    曲线是确定性的分段线性函数：前半程从 0 放大到 1+overshoot，
    后半程回落并稳定到 1.0。字幕时长短于动画窗口时，把整个曲线压缩到
    字幕时长内，避免短字幕来不及完成动画；动画窗口之后的时刻恒为 1.0。
    """
    if duration <= 0 or t < 0 or t >= duration:
        return 1.0
    window = min(POP_IN_WINDOW_SECONDS, duration)
    if t >= window:
        return 1.0
    progress = t / window
    peak = 1.0 + POP_IN_OVERSHOOT
    if progress <= 0.5:
        return peak * (progress / 0.5)
    return peak - POP_IN_OVERSHOOT * ((progress - 0.5) / 0.5)


def kinetic_float_offset(
    t: float,
    duration: float,
    video_height: int,
    seed: Optional[int] = None,
) -> tuple[float, float]:
    """返回时刻 ``t`` 的 (垂直, 水平) 漂移偏移（像素，正值向右/下）。

    使用正弦曲线在 +amp / -amp 之间平滑往返一次，起点和终点都在 0；
    幅度限制在视频高度的一小部分，保证任何锚点都不会漂出安全区。
    ``seed`` 只改变水平漂移的相位，保证每个字幕的漂移方向有细微差异，
    且相同 seed 永远得到相同结果（确定性）。
    """
    if duration <= 0:
        return 0.0, 0.0
    amplitude = max(
        KINETIC_FLOAT_MIN_AMPLITUDE,
        int(video_height * KINETIC_FLOAT_AMPLITUDE_RATIO),
    )
    progress = min(max(t / duration, 0.0), 1.0)
    phase = ((seed or 0) % 10) * 0.31
    vertical = amplitude * math.sin(2 * math.pi * progress)
    horizontal = (
        amplitude * KINETIC_HORIZONTAL_RATIO * math.sin(2 * math.pi * progress + phase)
    )
    return vertical, horizontal


def dynamic_font_size(
    phrase: str,
    base_size: int,
    max_width: int,
    font_path: str,
) -> int:
    """按实际文本宽度计算动态字号（测量与换行基准和渲染层一致）。"""
    from PIL import ImageFont

    max_width = max(1, int(max_width))
    base_size = max(1, int(base_size))
    try:
        font = ImageFont.truetype(font_path, base_size)
        width = font.getbbox(phrase)[2] - font.getbbox(phrase)[0]
    except Exception:
        return base_size

    if width <= max_width * DYNAMIC_UPSCALE_TRIGGER_RATIO:
        ratio = min(
            DYNAMIC_SCALE_MAX_RATIO,
            max(1.0, (max_width * DYNAMIC_UPSCALE_FILL_RATIO) / max(width, 1)),
        )
        return max(1, int(round(base_size * ratio)))

    ratio = 1.0
    while ratio > DYNAMIC_SCALE_MIN_RATIO:
        wrapped = measure_wrapped_lines(
            phrase,
            max_width=max_width,
            font_path=font_path,
            font_size=int(round(base_size * ratio)),
        )
        if len(wrapped) <= DYNAMIC_MAX_LINES:
            break
        ratio = round(ratio - 0.1, 2)
    return max(1, int(round(base_size * max(ratio, DYNAMIC_SCALE_MIN_RATIO))))


def measure_wrapped_lines(
    text: str, max_width: int, font_path: str, font_size: int
) -> list[str]:
    """按目标宽度换行并返回行列表（供动态字号/预览使用）。"""
    from app.services.subtitle_engine.text import wrap_text

    wrapped, _height = wrap_text(
        text,
        max_width=int(max_width),
        font=font_path,
        fontsize=int(font_size),
    )
    return wrapped.split("\n") if wrapped else []