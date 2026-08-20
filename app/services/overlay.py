"""图文叠加层规划器（Overlay plan builder）。

从文案、字幕分句和成片时长里推导一组叠加层卡片：片头标题卡、数据/事实卡
与关键句 callout。输出是纯数据（OverlayItem 列表），不依赖 MoviePy，方便
单元测试；合成逻辑放在 video.py 的 generate_video 里按位置与时间轴渲染。
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional

# 事实/数据卡识别规则：包含百分比、货币、大数字、年份、引用等信号。
FACT_PATTERNS = [
    re.compile(r"\d+(?:[.,]\d+)?\s*%"),
    re.compile(r"\d+(?:[.,]\d+)?\s*percent", re.IGNORECASE),
    re.compile(r"[$€£¥]\s?\d[\d,.]*"),
    re.compile(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b"),
    re.compile(r"\b\d+(?:\.\d+)?\s*(?:million|billion|trillion|thousand|k)\b", re.IGNORECASE),
    re.compile(r"\b\d+(?:\.\d+)?\s*(?:万|亿|百万|千万|万亿)"),
    re.compile(r"\b(?:19|20)\d{2}\b"),
    re.compile(r"\b\d+(?:\.\d+)?\s*(?:x|×|times)\b", re.IGNORECASE),
    re.compile(r"\b\d+\s*倍"),
    re.compile(r"according to", re.IGNORECASE),
    re.compile(r"\b(?:data|statistics|studies?|reports?)\b", re.IGNORECASE),
]

# 关键句 callout 识别：短句 + 转折/强调/结论性词。
CALLOUT_HOOKS = re.compile(
    r"\b(?:but|however|yet|key|crucial|important|the truth|in fact|remember|"
    r"imagine|here.s|look at|that.s why|what if|the point is|in short)\b",
    re.IGNORECASE,
)

# 事实卡出现在左下角，callout 出现在顶部，标题卡始终在顶部居中。
FACT_POSITION = "lower_third"
CALLOUT_POSITION = "top"
TITLE_POSITION = "top_center"

# 标题卡最长展示时间（秒）。
TITLE_MAX_DURATION = 3.5
# 事实/数据卡最短展示时长（秒），避免一闪而过。
CARD_MIN_DURATION = 1.2
# 从一句话到字幕时间窗的对齐失败时，使用音频时长的 2% 兜底。
_PHRASE_FALLBACK_FRACTION = 0.02


@dataclass
class OverlayItem:
    kind: str  # "title" | "fact" | "callout"
    text: str
    start: float
    end: float
    position: str = FACT_POSITION
    style: str = "default"
    source_phrase_index: Optional[int] = None
    extra: dict = field(default_factory=dict)


def detect_fact(text: str) -> bool:
    """判断一句话是否包含数据/事实信号（数字、百分比、年份、引用等）。"""
    if not text:
        return False
    return any(pattern.search(text) for pattern in FACT_PATTERNS)


def detect_callout(text: str) -> bool:
    """判断一句话是否适合做关键句 callout：短小且有转折/强调语气。"""
    if not text:
        return False
    words = [w for w in re.split(r"\s+", text.strip()) if w]
    if not 2 <= len(words) <= 12:
        return False
    if detect_fact(text):
        return False
    return bool(CALLOUT_HOOKS.search(text))


def _parse_srt_timestamp(ts: str) -> float:
    """把 SRT 时间戳 "00:00:01,735" 解析成秒。"""
    body = ts.strip()
    if "," in body:
        body = body.replace(",", ".")
    parts = body.split(":")
    if len(parts) == 3:
        hours, minutes, seconds = (float(p) for p in parts)
        return hours * 3600 + minutes * 60 + seconds
    if len(parts) == 2:
        minutes, seconds = (float(p) for p in parts)
        return minutes * 60 + seconds
    return float(body)


def parse_subtitle_phrases(subtitle_lines: list) -> list:
    """
    把 subtitle.file_to_subtitles 的输出转成 (start, end, text) 三元组。

    subtitle_lines 形如 [[1, "00:00:00,100 --> 00:00:00,917", "It is hard to"], ...]
    无法解析的项会被跳过，保证空/损坏字幕不中断规划。
    """
    phrases = []
    for item in subtitle_lines:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        ts = item[1]
        text = str(item[2]).strip()
        if not text or not ts or "-->" not in ts:
            continue
        try:
            start_str, end_str = ts.split("-->")
            start = _parse_srt_timestamp(start_str)
            end = _parse_srt_timestamp(end_str)
        except (ValueError, TypeError):
            continue
        if end <= start:
            end = start + CARD_MIN_DURATION
        phrases.append((start, end, text))
    return phrases


def build_overlay_plan(
    params,
    subject: str,
    script: str,
    subtitle_phrases: Optional[list] = None,
    video_duration: float = 0.0,
) -> List[OverlayItem]:
    """
    生成叠加层卡片列表，按字幕时间轴对齐。

    - 标题卡：subject，成片开头显示，最多 TITLE_MAX_DURATION 秒。
    - 事实卡：包含数据信号的短语，在对应时间窗左下角显示。
    - callout：短小有强调语气且不含数据信号的短语，顶部显示。

    overlay_enabled=False 时返回空列表。样式由 overlay_style 控制：
      title_fact（默认）| title_only | facts_only | callouts_only | full。
    """
    if not getattr(params, "overlay_enabled", False):
        return []

    style = getattr(params, "overlay_style", "title_fact") or "title_fact"
    show_title = getattr(params, "overlay_title_card", True)
    show_facts = getattr(params, "overlay_fact_cards", True)
    show_callouts = getattr(params, "overlay_callouts", False)

    if style == "title_only":
        show_facts = False
        show_callouts = False
    elif style == "facts_only":
        show_title = False
        show_callouts = False
    elif style == "callouts_only":
        show_title = False
        show_facts = False
    elif style == "full":
        show_title = True
        show_facts = True
        show_callouts = True

    items: List[OverlayItem] = []

    phrases = subtitle_phrases or parse_subtitle_phrases([])
    first_start = phrases[0][0] if phrases else 0.0
    first_end = phrases[0][1] if phrases else video_duration

    if show_title and subject and subject.strip():
        items.append(
            OverlayItem(
                kind="title",
                text=subject.strip(),
                start=0.0,
                end=min(TITLE_MAX_DURATION, max(first_start + 0.2, first_end)),
                position=TITLE_POSITION,
            )
        )

    for index, (start, end, text) in enumerate(phrases):
        if show_facts and detect_fact(text):
            items.append(
                OverlayItem(
                    kind="fact",
                    text=text,
                    start=start,
                    end=max(end, start + CARD_MIN_DURATION),
                    position=FACT_POSITION,
                    source_phrase_index=index,
                )
            )
        elif show_callouts and detect_callout(text):
            items.append(
                OverlayItem(
                    kind="callout",
                    text=text,
                    start=start,
                    end=max(end, start + CARD_MIN_DURATION),
                    position=CALLOUT_POSITION,
                    source_phrase_index=index,
                )
            )

    return items