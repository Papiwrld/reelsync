"""Subtitle renderer: from subtitle item to a positioned, animated MoviePy clip.

The renderer owns text measurement/wrapping, background compositing, word
highlight variants, multilingual font runs, positioning and animation. It has
two internal paths:

- **legacy path** (default): pixel-identical to the historical
  ``create_text_clip`` implementation (TextClip + supersampling + rounded
  background), so old configurations render exactly as before;
- **enhanced path**: PIL-drawn frames used when active-word highlighting or
  multilingual run splitting is needed. It shares the same layout pipeline,
  paddings and colors so switching between presets is visually consistent.

Missing fonts never crash generation: font resolution falls back through the
registry, and per-subtitle failures degrade to a warning instead of aborting
the whole video.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

from loguru import logger
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.services.subtitle_engine.animation import (
    POP_IN_WINDOW_SECONDS,
    dynamic_font_size,
    kinetic_float_offset,
    pop_in_scale,
)
from app.services.subtitle_engine.fonts import (
    DEFAULT_FALLBACK_FONT,
    FontRegistry,
    get_font_registry,
)
from app.services.subtitle_engine.position import (
    SubtitlePositionEngine,
    SubtitlePositionInput,
)
from app.services.subtitle_engine.styles import (
    SubtitleStyleConfig,
    apply_subtitle_casing,
)
from app.services.subtitle_engine.text import wrap_text
from app.services.subtitle_engine.timing import (
    PhraseWord,
    WordTiming,
    match_words_to_phrase,
)

_SUBTITLE_SUPERSAMPLE_SCALE = 2

# MoviePy 2.1.x TextClip 的 margin 参数类型别名，仅用于类型标注。
TextClipFactory = Callable[..., object]


@dataclass
class TokenLayout:
    """一行内的一个词 token（用于逐词高亮与多语言 run 绘制）。"""

    text: str
    font_path: str
    line: int
    x: int  # 相对整块画布的 x（超采样坐标系）
    width: int


@dataclass
class SubtitleLayout:
    """一次字幕布局的完整测量结果（超采样坐标系）。"""

    box_width: int
    box_height: int
    font_size: int
    stroke_width: int
    interline: int
    margin_y: int
    ascent: int
    line_height: int
    lines: List[str] = field(default_factory=list)
    tokens: List[TokenLayout] = field(default_factory=list)
    background: Optional[Tuple[int, int, int, int]] = None  # (r, g, b, a)
    background_radius: int = 0
    line_offsets: List[int] = field(default_factory=list)  # 每行基线 y

    @property
    def line_count(self) -> int:
        return len(self.lines)


class SubtitleRenderer:
    """把一个字幕帧渲染为带定位/动画的 MoviePy clip。"""

    def __init__(
        self,
        style: SubtitleStyleConfig,
        video_width: int,
        video_height: int,
        font_registry: Optional[FontRegistry] = None,
        text_clip_factory=None,
        composite_clip_factory=None,
        image_clip_factory=None,
        position_engine: Optional[SubtitlePositionEngine] = None,
    ):
        self.style = style
        self.video_width = max(1, int(video_width))
        self.video_height = max(1, int(video_height))
        self.font_registry = font_registry or get_font_registry()
        self.text_clip_factory = text_clip_factory
        self.composite_clip_factory = composite_clip_factory
        self.image_clip_factory = image_clip_factory
        self.position_engine = position_engine or SubtitlePositionEngine()
        # 延迟导入，避免渲染器在 MoviePy 不可用时阻断静态工具函数。
        if self.text_clip_factory is None or self.image_clip_factory is None:
            from moviepy import CompositeVideoClip, ImageClip, TextClip

            self.text_clip_factory = self.text_clip_factory or TextClip
            self.image_clip_factory = self.image_clip_factory or ImageClip
            self.composite_clip_factory = (
                self.composite_clip_factory or CompositeVideoClip
            )

    # ------------------------------------------------------------------
    # 入口
    # ------------------------------------------------------------------
    def render(
        self,
        subtitle_item: Tuple[Tuple[float, float], str],
        word_timings: Optional[List[WordTiming]] = None,
    ):
        """渲染一个字幕帧，返回已定时（start/end/duration）的 clip。

        ``subtitle_item`` 形如 ``((start, end), text)``（MoviePy SubtitlesClip
        的条目格式）。返回 None 表示该帧被安全跳过（不应发生）。
        """
        (start, end), text = subtitle_item
        duration = float(end) - float(start)
        if duration <= 0 or not text:
            return None

        phrase = apply_subtitle_casing(text, self.style.casing)
        if not phrase:
            return None

        # 字体解析与多语言 run 分段；失败时回退默认字体。
        font_path = self._safe_font_path(self.style.font_path)
        runs = self.font_registry.split_runs(phrase, font_path)

        # 词时间轴对齐（可选）：仅在开启高亮且能匹配到时启用增强路径。
        words: List[PhraseWord] = []
        if self.style.active_word_highlight and word_timings:
            words = match_words_to_phrase(
                word_timings, phrase, float(start), float(end)
            )
        use_enhanced = bool(words) or len(runs) > 1

        layout = self._build_layout(phrase, runs, font_path)
        if layout is None:
            return None

        if use_enhanced:
            variants = self._render_enhanced_frames(
                start, end, phrase, layout, words, runs
            )
            clips = []
            for variant_start, variant_end, image in variants:
                if variant_end <= variant_start:
                    continue
                clip = self.image_clip_factory(image, transparent=True)
                clip = clip.with_start(variant_start)
                clip = clip.with_end(min(variant_end, float(end)))
                clip = clip.with_duration(
                    max(0.0, min(variant_end, float(end)) - variant_start)
                )
                clip = self._finalize_visuals(clip, layout)
                clips.append(clip)
            return clips if clips else None

        clip = self._render_legacy(start, end, phrase, layout, font_path)
        if clip is None:
            return None
        clip = clip.with_start(float(start))
        clip = clip.with_end(float(end))
        clip = clip.with_duration(duration)
        clip = self._finalize_visuals(clip, layout)
        return clip

    def _finalize_visuals(self, clip, layout: SubtitleLayout):
        """把布局/动画/定位应用到单个 clip（不重复设置时间轴）。"""
        if _SUBTITLE_SUPERSAMPLE_SCALE > 1:
            from moviepy import vfx

            clip = clip.with_effects(
                [vfx.Resize(1 / _SUBTITLE_SUPERSAMPLE_SCALE)]
            )
        clip = self._apply_animations(clip, getattr(clip, "duration", 0.0) or 0.0)
        clip = self._apply_position(clip, layout)
        return clip

    # ------------------------------------------------------------------
    # 布局
    # ------------------------------------------------------------------
    def _build_layout(
        self,
        phrase: str,
        runs: Sequence,
        font_path: str,
    ) -> Optional[SubtitleLayout]:
        style = self.style
        ss = _SUBTITLE_SUPERSAMPLE_SCALE
        max_width = self.video_width * 0.9 * ss
        bg_color = style.background
        rounded_bg = bool(style.rounded_background and bg_color)
        padding_ratio = 0.4 if rounded_bg else 0.6
        pad_x = (
            int(style.font_size * padding_ratio) * ss if bg_color else 0
        )
        text_max_width = max(1, int(max_width) - 2 * pad_x)

        font_size = style.font_size
        if style.dynamic_scaling:
            font_size = dynamic_font_size(
                phrase, font_size, int(text_max_width / ss), font_path
            )
        font_size = font_size * ss
        stroke_width = int(round(style.outline_width * ss))

        if len(runs) > 1:
            wrapped_lines = wrap_runs_text(
                runs, max_width=text_max_width, font_size=font_size
            )
        else:
            wrapped_text, _ = wrap_text(
                phrase,
                max_width=text_max_width,
                font=font_path,
                fontsize=font_size,
            )
            wrapped_lines = wrapped_text.split("\n") if wrapped_text else [""]
        wrapped_lines = [line for line in wrapped_lines if line is not None]
        if not wrapped_lines:
            return None

        line_count = len(wrapped_lines)
        interline = int(font_size * 0.25)
        vertical_padding = int(font_size * 0.35)
        margin_y = max(int(font_size * 0.3), int(stroke_width * 2))

        try:
            probe = ImageFont.truetype(font_path, font_size)
            ascent, descent = probe.getmetrics()
        except Exception:
            ascent, descent = int(font_size * 0.8), int(font_size * 0.2)
        line_height = ascent + descent
        txt_height = line_height * line_count
        clip_h = int(txt_height + vertical_padding + (interline * line_count))

        # 行级 token 布局（用于高亮与 run 绘制）
        tokens: List[TokenLayout] = []
        line_offsets: List[int] = []
        for line_index, line in enumerate(wrapped_lines):
            line_offsets.append(
                margin_y + ascent + line_index * (line_height + interline)
            )
            tokens.extend(
                self._layout_line_tokens(
                    line, line_index, runs, font_size, line_offsets[-1]
                )
            )

        box_width = int(max_width)
        background = None
        background_radius = 0
        if bg_color:
            alpha = int(round(255 * style.background_opacity))
            if rounded_bg:
                line_widths = [
                    self._measure_run_line(line, runs, font_size)[0]
                    for line in wrapped_lines
                ]
                text_w = max(line_widths) if line_widths else int(max_width)
                box_width = max(1, min(int(max_width), int(text_w) + 2 * pad_x))
                background_radius = max(8, int(font_size * 0.4))
            background = (*_hex_to_rgb(bg_color), alpha)

        return SubtitleLayout(
            box_width=box_width,
            box_height=clip_h,
            font_size=font_size,
            stroke_width=stroke_width,
            interline=interline,
            margin_y=margin_y,
            ascent=ascent,
            line_height=line_height,
            lines=wrapped_lines,
            tokens=tokens,
            background=background,
            background_radius=background_radius,
            line_offsets=line_offsets,
        )

    def _layout_line_tokens(
        self,
        line: str,
        line_index: int,
        runs: Sequence,
        font_size: int,
        baseline_y: int,
    ) -> List[TokenLayout]:
        """把一行拆成 token，并测量每个 token 的 x 偏移（超采样坐标系）。"""
        from app.services.subtitle_engine.timing import tokenize

        tokens: List[TokenLayout] = []
        cursor = 0
        x = 0
        for token in tokenize(line):
            start = line.find(token, cursor)
            if start < 0:
                continue
            # token 之前的空白/标点也要计入 x 前进量。
            prefix = line[cursor:start]
            prefix_w = self._measure_text_segment(prefix, runs, font_size)
            token_w = self._measure_text_segment(token, runs, font_size)
            x += prefix_w
            tokens.append(
                TokenLayout(
                    text=token,
                    font_path=self._font_for_text(token, runs),
                    line=line_index,
                    x=x,
                    width=token_w,
                )
            )
            x += token_w
            cursor = start + len(token)
        return tokens

    def _measure_text_segment(
        self, text: str, runs: Sequence, font_size: int
    ) -> int:
        if not text:
            return 0
        total = 0
        for char in text:
            font_path = self._font_for_text(char, runs)
            try:
                font = ImageFont.truetype(font_path, font_size)
                total += font.getbbox(char)[2] - font.getbbox(char)[0]
            except Exception:
                total += font_size // 2
        return total

    def _font_for_text(self, text: str, runs: Sequence) -> str:
        for run in runs:
            if text in run.text:
                return run.font_path
        # 跨 run 边界的 token：按首个字符所在 run 取字体。
        for run in runs:
            if run.text.startswith(text[0]) if text else False:
                return run.font_path
        return getattr(runs[0], "font_path", "") if runs else ""

    def _measure_run_line(self, line: str, runs: Sequence, font_size: int) -> Tuple[int, int]:
        total = 0
        for char in line:
            total += self._measure_text_segment(char, runs, font_size)
        return total, 0

    # ------------------------------------------------------------------
    # legacy 渲染路径（与历史行为逐像素一致）
    # ------------------------------------------------------------------
    def _render_legacy(
        self,
        start: float,
        end: float,
        phrase: str,
        layout: SubtitleLayout,
        font_path: str,
    ):
        style = self.style
        ss = _SUBTITLE_SUPERSAMPLE_SCALE
        max_width = self.video_width * 0.9 * ss
        bg_color = style.background
        rounded_bg = bool(style.rounded_background and bg_color)
        padding_ratio = 0.4 if rounded_bg else 0.6
        pad_x = int(style.font_size * padding_ratio) * ss if bg_color else 0

        wrapped_txt = "\n".join(layout.lines)
        font_size = layout.font_size
        interline = layout.interline
        clip_h = layout.box_height
        text_clip_margin_y = layout.margin_y
        stroke_width = layout.stroke_width
        TextClip = self.text_clip_factory
        CompositeVideoClip = self.composite_clip_factory

        if rounded_bg:
            box_w = layout.box_width
            radius = layout.background_radius
            try:
                font = ImageFont.truetype(font_path, font_size)
                text_w = max(
                    int(font.getbbox(line)[2] - font.getbbox(line)[0])
                    for line in wrapped_txt.split("\n")
                )
            except Exception:
                text_w = int(max_width)
            box_w = max(1, min(int(max_width), text_w + 2 * pad_x))
            text_clip = TextClip(
                text=wrapped_txt,
                font=font_path,
                font_size=font_size,
                color=style.color,
                bg_color=None,
                stroke_color=style.outline_color,
                stroke_width=stroke_width,
                interline=interline,
                size=(box_w, None),
                text_align="center",
                margin=(0, text_clip_margin_y),
            )
            clip_h = max(clip_h, text_clip.h)
            bg_clip = _rounded_subtitle_background_clip(
                width=box_w,
                height=clip_h,
                color=style.background or "#000000",
                alpha=int(round(255 * style.background_opacity)),
                radius=radius,
            )
            text_position = _get_visible_center_position(text_clip, box_w, clip_h)
            _clip = CompositeVideoClip(
                [bg_clip, text_clip.with_position(text_position)],
                size=(box_w, clip_h),
            )
        elif bg_color:
            size = (int(max_width), clip_h)
            text_clip = TextClip(
                text=wrapped_txt,
                font=font_path,
                font_size=font_size,
                color=style.color,
                bg_color=None,
                stroke_color=style.outline_color,
                stroke_width=stroke_width,
                interline=interline,
                size=(int(max_width), None),
                text_align="center",
                margin=(0, text_clip_margin_y),
            )
            size = (size[0], max(size[1], text_clip.h))
            bg_clip = _rounded_subtitle_background_clip(
                width=size[0],
                height=size[1],
                color=bg_color,
                alpha=255,
                radius=0,
            )
            text_position = _get_visible_center_position(text_clip, size[0], size[1])
            _clip = CompositeVideoClip(
                [bg_clip, text_clip.with_position(text_position)],
                size=size,
            )
        else:
            size = (int(max_width), clip_h)
            _clip = TextClip(
                text=wrapped_txt,
                font=font_path,
                font_size=font_size,
                color=style.color,
                bg_color=None,
                stroke_color=style.outline_color,
                stroke_width=stroke_width,
                interline=interline,
                size=size,
                text_align="center",
            )
        return _clip

    # ------------------------------------------------------------------
    # 增强渲染路径（逐词高亮 / 多语言 run）
    # ------------------------------------------------------------------
    def _render_enhanced_frames(
        self,
        start: float,
        end: float,
        phrase: str,
        layout: SubtitleLayout,
        words: List[PhraseWord],
        runs: Sequence,
    ):
        """生成基础帧 + 每个词的高亮帧，返回 ``(窗口开始, 窗口结束, RGBA 帧)``。

        所有帧共享同一布局（token x 偏移固定），因此高亮词切换时文字
        不会抖动。词窗口由调用方转成 ``with_start/with_end`` 的 clip 叠加
        在同一时间轴上。
        """
        variants: List[Tuple[float, float, np.ndarray]] = []

        if words:
            cursor = start
            for index, word in enumerate(words):
                variants.append((cursor, word.start, None))
                variants.append((word.start, word.end, index))
                cursor = word.end
            if cursor < end:
                variants.append((cursor, end, None))
        else:
            variants = [(start, end, None)]

        frames = []
        for variant_start, variant_end, active_index in variants:
            if variant_end <= variant_start:
                continue
            frames.append(
                (
                    variant_start,
                    variant_end,
                    self._draw_frame(phrase, layout, runs, active_index),
                )
            )
        return frames

    def _draw_frame(
        self,
        phrase: str,
        layout: SubtitleLayout,
        runs: Sequence,
        active_token_index: Optional[int],
    ) -> np.ndarray:
        """用 PIL 绘制一个字幕帧（RGBA，超采样坐标系）。"""
        style = self.style
        image = Image.new("RGBA", (layout.box_width, layout.box_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)

        if layout.background is not None:
            if layout.background_radius > 0:
                draw.rounded_rectangle(
                    [0, 0, max(0, layout.box_width - 1), max(0, layout.box_height - 1)],
                    radius=layout.background_radius,
                    fill=layout.background,
                )
            else:
                draw.rectangle(
                    [0, 0, max(0, layout.box_width - 1), max(0, layout.box_height - 1)],
                    fill=layout.background,
                )

        # 逐 token 绘制：token 的 x 来自共享布局，保证各帧像素对齐。
        for index, token in enumerate(layout.tokens):
            is_active = active_token_index == index
            fill = (
                style.highlight_color if is_active else style.color
            )
            stroke = (
                style.highlight_color if is_active else style.outline_color
            )
            baseline_y = layout.line_offsets[token.line]
            line_width = self._line_pixel_width(layout, token.line)
            centered_x = (
                layout.box_width - line_width
            ) // 2
            x = centered_x + token.x
            try:
                font = ImageFont.truetype(token.font_path, layout.font_size)
                # 与 TextClip 一致使用 baseline 锚点；描边向两侧展开。
                draw.text(
                    (x, baseline_y),
                    token.text,
                    font=font,
                    fill=fill,
                    stroke_width=layout.stroke_width,
                    stroke_fill=stroke,
                    anchor="ls",
                )
            except Exception as exc:
                logger.warning(
                    f"failed to draw subtitle token, skipping: {exc}"
                )
        return np.array(image)

    def _line_pixel_width(self, layout: SubtitleLayout, line_index: int) -> int:
        width = 0
        for token in layout.tokens:
            if token.line == line_index:
                width = max(width, token.x + token.width)
        return width

    # ------------------------------------------------------------------
    # 定位与动画（所有路径共享）
    # ------------------------------------------------------------------
    def _apply_animations(self, clip, duration: float):
        style = self.style
        from moviepy import vfx

        if style.pop_bounce:
            if duration >= POP_IN_WINDOW_SECONDS:
                clip_w, clip_h = clip.w, clip.h
                clip = clip.with_effects(
                    [
                        vfx.Resize(
                            new_size=lambda t: (
                                max(
                                    1,
                                    int(round(clip_w * pop_in_scale(t, duration))),
                                ),
                                max(
                                    1,
                                    int(round(clip_h * pop_in_scale(t, duration))),
                                ),
                            )
                        )
                    ]
                )
        return clip

    def _apply_position(self, clip, layout: SubtitleLayout):
        style = self.style
        result = self.position_engine.compute(
            SubtitlePositionInput(
                video_width=self.video_width,
                video_height=self.video_height,
                box_width=layout.box_width // _SUBTITLE_SUPERSAMPLE_SCALE,
                box_height=layout.box_height // _SUBTITLE_SUPERSAMPLE_SCALE,
                line_count=layout.line_count,
                anchor=style.position,
                vertical_offset=style.vertical_offset,
                custom_position=style.custom_position,
                dynamic_auto_avoidance=style.dynamic_auto_avoidance,
            )
        )

        if style.kinetic_float:
            clip = clip.with_position(
                lambda t: (
                    result.x
                    + kinetic_float_offset(t, clip.duration or 0.0, self.video_height, seed=result.y)[1],
                    result.y
                    + kinetic_float_offset(t, clip.duration or 0.0, self.video_height, seed=result.y)[0],
                )
            )
        else:
            clip = clip.with_position((result.x, result.y))
        return clip

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------
    def _safe_font_path(self, font_path: str) -> str:
        import os

        if font_path and os.path.isfile(font_path):
            return font_path
        resolved = self.font_registry.resolve(font_path)
        if resolved and os.path.isfile(resolved):
            return resolved
        fallback = self.font_registry.resolve(DEFAULT_FALLBACK_FONT)
        return fallback if os.path.isfile(fallback) else (resolved or "")


# ---------------------------------------------------------------------------
# 多语言 run 的换行（按每个字符所属字体的真实宽度测量）
# ---------------------------------------------------------------------------


def wrap_runs_text(runs: Sequence, max_width: int, font_size: int) -> List[str]:
    """按 run 的真实字形宽度换行（用于混合字体文本）。

    与 ``wrap_text`` 相同的策略（空格优先、超长 token 字符级拆分），
    但每个字符用其所属 run 的字体测量，中文/日文等无空格脚本也能正确
    换行且不会因为首选字体缺少字形而测出错误宽度。
    """
    max_width = int(max_width)
    combined = "".join(run.text for run in runs)
    lines: List[str] = []
    current = ""
    for word in combined.split(" "):
        candidate = f"{current} {word}".strip() if current else word
        if _measure_mixed(candidate, runs, font_size) <= max_width:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = word
        if _measure_mixed(word, runs, font_size) > max_width:
            for piece in _split_long_mixed(word, runs, max_width, font_size):
                lines.append(piece)
            current = ""
    if current:
        lines.append(current)
    return [line.strip() for line in lines if line.strip()]


def _char_font(char: str, runs: Sequence) -> str:
    for run in runs:
        if char in run.text:
            return run.font_path
    return getattr(runs[0], "font_path", "") if runs else ""


def _measure_mixed(text: str, runs: Sequence, font_size: int) -> int:
    total = 0
    for char in text:
        try:
            font = ImageFont.truetype(_char_font(char, runs), font_size)
            total += font.getbbox(char)[2] - font.getbbox(char)[0]
        except Exception:
            total += font_size // 2
    return total


def _split_long_mixed(token: str, runs: Sequence, max_width: int, font_size: int) -> List[str]:
    pieces = []
    current = ""
    for char in token:
        candidate = f"{current}{char}"
        if _measure_mixed(candidate, runs, font_size) <= max_width or not current:
            current = candidate
            continue
        pieces.append(current)
        current = char
    if current:
        pieces.append(current)
    return pieces


# ---------------------------------------------------------------------------
# 与历史实现保持一致的辅助函数
# ---------------------------------------------------------------------------


def _hex_to_rgb(color: str) -> Tuple[int, int, int]:
    if isinstance(color, str) and color.startswith("#") and len(color) == 7:
        try:
            return (int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16))
        except ValueError:
            pass
    return (0, 0, 0)


def _rounded_subtitle_background_clip(
    width: int,
    height: int,
    color: str,
    alpha: int = 140,
    radius: int = 16,
):
    from moviepy import ImageClip

    rgb = _hex_to_rgb(color)
    safe_alpha = max(0, min(255, int(alpha)))
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    if radius > 0:
        draw.rounded_rectangle(
            [0, 0, max(0, width - 1), max(0, height - 1)],
            radius=max(0, int(radius)),
            fill=(rgb[0], rgb[1], rgb[2], safe_alpha),
        )
    else:
        draw.rectangle(
            [0, 0, max(0, width - 1), max(0, height - 1)],
            fill=(rgb[0], rgb[1], rgb[2], safe_alpha),
        )
    return ImageClip(np.array(img), transparent=True)


def _get_visible_center_position(text_clip, container_width: int, container_height: int) -> Tuple[int, int]:
    """按文字真实可见像素把 TextClip 放到背景容器中心（历史行为）。"""
    x = int(round((container_width - text_clip.w) / 2))
    y = int(round((container_height - text_clip.h) / 2))

    try:
        if text_clip.mask is None:
            return x, y

        mask_frame = text_clip.mask.get_frame(0)
        ys, _ = np.where(mask_frame > 0.01)
        if len(ys) == 0:
            return x, y

        visible_top = int(ys.min())
        visible_bottom = int(ys.max())
        visible_height = visible_bottom - visible_top + 1
        y = int(round((container_height - visible_height) / 2 - visible_top))
    except Exception as exc:
        logger.debug(f"failed to center subtitle text by visible mask: {str(exc)}")

    return x, y


# ---------------------------------------------------------------------------
# ASS (libass) 矢量字幕生成：作为 PIL 位图路径的矢量替代（Gap 2）
# ---------------------------------------------------------------------------


def _ass_escape_text(text: str) -> str:
    """Escape ASS special characters but preserve intentional overrides."""
    return (
        str(text or "")
        .replace("\\", r"\\")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def _format_ass_time(seconds: float) -> str:
    """Format seconds as ASS timestamp H:MM:SS.CS (centiseconds)."""
    total_cs = max(0, int(round(float(seconds) * 100)))
    cs = total_cs % 100
    total_s = total_cs // 100
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _hex_to_ass_color(hex_color: str, opacity: float = 0.0) -> str:
    """Convert #RRGGBB + opacity (0 opaque …1 transparent) to ASS &HAABBGGRR."""
    rgb = _hex_to_rgb(hex_color)
    try:
        alpha = int(round(max(0.0, min(1.0, float(opacity))) * 255))
    except Exception:
        alpha = 0
    # ASS uses BGR order and inverted alpha (00 opaque)
    return f"&H{alpha:02X}{rgb[2]:02X}{rgb[1]:02X}{rgb[0]:02X}"


def _ass_font_name(style) -> str:
    font_name = getattr(style, "font_name", "") or "Montserrat-Bold.ttf"
    base = str(font_name).strip()
    # strip path and extension, turn hyphen/underscore into space
    import os as _os

    base = _os.path.basename(base)
    if "." in base:
        base = base.rsplit(".", 1)[0]
    base = base.replace("-", " ").replace("_", " ").strip()
    return base or "Arial"


def _ass_alignment_from_style(style) -> int:
    """Map subtitle position to ASS Alignment (1..9)."""
    try:
        pos = str(getattr(style, "position", "") or "").lower()
        # also handle enum values like SubtitlePosition.BOTTOM
        pos = pos.split(".")[-1]
    except Exception:
        pos = "bottom"
    if pos in ("top",):
        return 8  # top center
    if pos in ("center", "middle"):
        return 5  # middle center
    # bottom, dynamic, custom all default to bottom center
    return 2


def _build_ass_dialogue_text(
    phrase: str,
    phrase_start: float,
    phrase_end: float,
    style,
    word_timings: Optional[List],
) -> str:
    """Build ASS dialogue text with optional {\\k} karaoke from word timings."""
    if not phrase:
        return ""
    base = _ass_escape_text(phrase)
    # no highlight or no timings -> plain
    if not getattr(style, "active_word_highlight", False) or not word_timings:
        return base
    try:
        from app.services.subtitle_engine.timing import match_words_to_phrase

        words = match_words_to_phrase(word_timings, phrase, float(phrase_start), float(phrase_end))
    except Exception:
        return base
    if not words:
        return base
    # Build karaoke: each word becomes {\k<dur>}word
    # \k is centiseconds (10ms). Use word duration.
    parts: List[str] = []
    for word in words:
        dur_cs = max(1, int(round((float(word.end) - float(word.start)) * 100)))
        # escape per-word text separately
        word_text = _ass_escape_text(word.text)
        parts.append(f"{{\\k{dur_cs}}}{word_text}")
        # preserve a space between words (karaoke expects spaces outside tags)
        parts.append(" ")
    karaoke = "".join(parts).strip()
    # If karaoke word count mismatched due to tokenization differences, fallback
    # to plain to avoid losing characters.
    if not karaoke:
        return base
    # Heuristic: if karaoke covers drastically fewer chars, keep plain
    plain_chars = len("".join(ch for ch in phrase if ch.strip()))
    karaoke_chars = len("".join(w.text for w in words))
    if karaoke_chars < max(1, plain_chars * 0.5):
        return base
    return karaoke


def build_ass_content(
    subtitle_items,
    video_width: int,
    video_height: int,
    style,
    word_timings: Optional[List] = None,
) -> str:
    """Build ASS file content from subtitle items and style.

    subtitle_items: iterable of ((start, end), text) or (start, end, text)
    """
    vw = max(1, int(video_width))
    vh = max(1, int(video_height))
    font_name = _ass_font_name(style)
    font_size = int(getattr(style, "font_size", 60) or 60)
    # clamp font size to reasonable range for ASS (libass scales with PlayRes)
    font_size = max(12, min(160, font_size))
    primary = _hex_to_ass_color(getattr(style, "color", "#FFFFFF") or "#FFFFFF", 0.0)
    secondary = _hex_to_ass_color(
        getattr(style, "highlight_color", "#FFD60A") or "#FFD60A", 0.0
    )
    outline = _hex_to_ass_color(
        getattr(style, "outline_color", "#000000") or "#000000", 0.0
    )
    # background: opacity from style, invert for ASS alpha
    bg_raw = getattr(style, "background", None)
    # style.background may be None or "#RRGGBB"
    if bg_raw:
        bg_opacity = 1.0 - float(getattr(style, "background_opacity", 0.55) or 0.55)
        # clamp 0 transparent …1 opaque for ASS alpha? Actually we computed alpha as transparency
        # So bg alpha = 1 - opacity -> transparency
        back = _hex_to_ass_color(str(bg_raw), max(0.0, min(1.0, bg_opacity)))
        border_style = 3  # opaque box
    else:
        back = _hex_to_ass_color("#000000", 1.0)  # fully transparent
        border_style = 1  # outline only
    outline_width = max(0, float(getattr(style, "outline_width", 1.5) or 1.5))
    alignment = _ass_alignment_from_style(style)
    # Margins: use vertical_offset to shift MarginV
    try:
        v_offset = int(getattr(style, "vertical_offset", 0) or 0)
    except Exception:
        v_offset = 0
    # For bottom alignment, MarginV is distance from bottom; for top, from top.
    # Keep base 24px plus offset (negative moves up for bottom)
    base_margin_v = 24
    if alignment == 2:  # bottom
        margin_v = max(0, base_margin_v - v_offset)
    elif alignment == 8:  # top
        margin_v = max(0, base_margin_v + v_offset)
    else:
        margin_v = base_margin_v
    # Bold detection: font name contains Bold
    bold = -1 if "bold" in font_name.lower() else 0
    # Header
    lines: List[str] = []
    lines.append("[Script Info]")
    lines.append("Title: ReelSync Subtitles")
    lines.append("ScriptType: v4.00+")
    lines.append(f"PlayResX: {vw}")
    lines.append(f"PlayResY: {vh}")
    lines.append("WrapStyle: 0")
    lines.append("ScaledBorderAndShadow: yes")
    lines.append("YCbCr Matrix: TV.709")
    lines.append("")
    lines.append("[V4+ Styles]")
    lines.append(
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
        "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding"
    )
    style_line = (
        f"Style: Default,{font_name},{font_size},{primary},{secondary},"
        f"{outline},{back},{bold},0,0,0,100,100,0,0,{border_style},"
        f"{outline_width:.1f},0,{alignment},20,20,{margin_v},1"
    )
    lines.append(style_line)
    lines.append("")
    lines.append("[Events]")
    lines.append(
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"
    )
    for item in subtitle_items or []:
        try:
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], (list, tuple)):
                (start, end), text = item
            elif isinstance(item, tuple) and len(item) == 3:
                start, end, text = item
            else:
                continue
            start_f = float(start)
            end_f = float(end)
            if end_f <= start_f or not text:
                continue
            # apply casing already done by caller? Ensure we use text as-is
            dialogue_text = _build_ass_dialogue_text(
                str(text), start_f, end_f, style, word_timings
            )
            # escape newlines -> \N
            dialogue_text = dialogue_text.replace("\n", r"\N")
            lines.append(
                f"Dialogue: 0,{_format_ass_time(start_f)},{_format_ass_time(end_f)},"
                f"Default,,0,0,0,,{dialogue_text}"
            )
        except Exception as exc:
            logger.debug(f"skip ASS dialogue item due to error: {exc}")
            continue
    return "\n".join(lines) + "\n"


def generate_ass_file(
    subtitle_path: str,
    output_ass_path: str,
    video_width: int,
    video_height: int,
    style,
    word_timings: Optional[List] = None,
) -> bool:
    """Generate ASS sidecar from SRT + word timings. Returns True on success.

    Tries pysubs2 if available for higher fidelity, otherwise falls back to
    manual ASS generation. Never raises — failures are logged and return False.
    """
    try:
        # Prefer pysubs2 when installed (optional dependency)
        try:
            import pysubs2  # type: ignore

            # pysubs2 path: load SRT and convert, then inject karaoke if needed
            subs = pysubs2.load(subtitle_path, encoding="utf-8")
            # ensure PlayRes
            subs.info["PlayResX"] = str(int(video_width))
            subs.info["PlayResY"] = str(int(video_height))
            # Convert styles if needed — keep default style from style object
            # Fallback to manual generation if karaoke requested but complex
            if getattr(style, "active_word_highlight", False) and word_timings:
                # pysubs2 karaoke injection is non-trivial; use manual builder for karaoke
                raise ImportError("use manual for karaoke")
            # otherwise use pysubs2 dump with custom style
            # Minimal: just dump and return
            # Build a simple style override via manual header injection is easier -> manual
            raise ImportError("prefer manual for style fidelity")
        except Exception:
            pass

        # Manual path: parse SRT via subtitle file_to_subtitles helper
        try:
            from app.services.subtitle import file_to_subtitles as _file_to_subtitles

            srt_items = _file_to_subtitles(subtitle_path)
            # convert to ((start,end), text) with numeric times

            def _srt_to_sec(value: str) -> float:
                value = str(value or "").strip()
                try:
                    h, m, s = value.split(":")
                    s, _, ms = s.partition(",")
                    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms or 0) / 1000.0
                except Exception:
                    return 0.0

            numeric_items = []
            for _, times, text in srt_items:
                if "-->" not in times:
                    continue
                a, _, b = times.partition("-->")
                numeric_items.append((( _srt_to_sec(a), _srt_to_sec(b)), text))
        except Exception:
            # fallback: read via moviepy SubtitlesClip if parser failed
            numeric_items = []

        if not numeric_items:
            logger.warning(f"no subtitle items to generate ASS: {subtitle_path}")
            return False

        content = build_ass_content(
            numeric_items, int(video_width), int(video_height), style, word_timings
        )
        # write atomically
        import os as _os

        tmp = output_ass_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fp:
            fp.write(content)
        _os.replace(tmp, output_ass_path)
        logger.info(f"ASS subtitle sidecar generated: {output_ass_path}")
        return True
    except Exception as exc:
        logger.warning(
            f"failed to generate ASS sidecar: {output_ass_path}, "
            f"error={type(exc).__name__}: {exc}"
        )
        return False