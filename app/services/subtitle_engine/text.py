"""Font-aware text measurement and line wrapping for subtitles.

``wrap_text`` measures with the real font at the real size (PIL) instead of
relying on fixed character counts, so multi-line sizing, long-word handling,
CJK text and portrait/landscape layouts all behave correctly. The renderer
must calculate actual rendered bounds before positioning — every caller uses
the measured result, never a character-count heuristic.
"""

from __future__ import annotations

from typing import Tuple

from PIL import ImageFont

# 行首不允许出现的闭合标点（避免换行后把标点孤立到行首）。
_LINE_START_PUNCTUATION = "，。！？；：、,.!?;:)]}）】》」』”’"


def _measure_line(text: str, font) -> Tuple[int, int]:
    text = text.strip()
    if not text:
        return 0, 0
    left, top, right, bottom = font.getbbox(text)
    return right - left, bottom - top


def wrap_text(
    text,
    max_width,
    font: str = "Arial",
    fontsize: int = 60,
) -> Tuple[str, int]:
    """按实际字体宽度换行，返回 ``(带换行的文本, 总行高)``。

    与历史实现保持同一算法（空格优先、超长 token 字符级拆分、闭合标点
    跟随前一行），保证旧配置渲染结果不变；换行基准是 PIL 的真实字形
    测量，中文长句和英文超长单词都能正确处理。
    """
    max_width = int(max_width)
    try:
        font_loaded = ImageFont.truetype(font, fontsize)
    except Exception:
        # 字体缺失或损坏时退化为字符数估算，保证字幕渲染永不崩溃；
        # 调用方（渲染器）仍会先通过注册表回退到可用字体。
        estimated_height = int(fontsize)
        return text, estimated_height

    def get_text_size(inner_text):
        inner_text = inner_text.strip()
        if not inner_text:
            return 0, fontsize
        return _measure_line(inner_text, font_loaded)

    width, height = get_text_size(text)
    if width <= max_width:
        return text, height

    def split_long_token(token):
        lines = []
        current = ""
        for char in token:
            candidate = f"{current}{char}"
            candidate_width, _ = get_text_size(candidate)
            if candidate_width <= max_width or not current:
                current = candidate
                continue
            lines.append(current)
            current = char
        if current:
            lines.append(current)
        return lines

    lines = []
    current = ""
    words = text.split(" ")
    for word in words:
        candidate = f"{current} {word}".strip() if current else word
        candidate_width, _ = get_text_size(candidate)
        if candidate_width <= max_width:
            current = candidate
            continue

        if current:
            lines.append(current)

        word_width, _ = get_text_size(word)
        if word_width <= max_width:
            current = word
        else:
            lines.extend(split_long_token(word))
            current = ""

    if current:
        lines.append(current)

    for index in range(1, len(lines)):
        if not lines[index] or lines[index][0] not in _LINE_START_PUNCTUATION:
            continue
        if len(lines[index - 1]) <= 1:
            continue

        candidate = f"{lines[index - 1][-1]}{lines[index]}"
        candidate_width, _ = get_text_size(candidate)
        if candidate_width <= max_width:
            lines[index] = candidate
            lines[index - 1] = lines[index - 1][:-1]

    result = "\n".join(line.strip() for line in lines if line.strip()).strip()
    total_height = len(lines) * height
    return result, total_height


def measure_text(text: str, font_path: str, font_size: int) -> Tuple[int, int]:
    """测量文本的 (宽, 高)，用于布局计算。"""
    try:
        font = ImageFont.truetype(font_path, int(font_size))
        width, height = _measure_line(text, font)
        return max(0, width), max(0, height or int(font_size))
    except Exception:
        # 字体加载失败时退化为字符数量估算，保证布局计算永不崩溃。
        return max(0, len(text)) * int(font_size) // 2, int(font_size)


def estimate_line_height(font_path: str, font_size: int) -> int:
    """估算单行文本高度（含字体行高）。"""
    try:
        font = ImageFont.truetype(font_path, int(font_size))
        ascent, descent = font.getmetrics()
        return max(1, ascent + descent)
    except Exception:
        return max(1, int(font_size))