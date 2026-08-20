"""Provider-neutral word timing.

``WordTiming`` is the single representation of word-level timestamps used by
the subtitle renderer. TTS providers (edge-tts WordBoundary cues, Azure word
boundary events, Whisper word timestamps) are *producers* of this
representation; the renderer only ever consumes ``WordTiming`` objects, so new
providers can be added without touching rendering code.

Word timings are persisted next to the SRT as a small JSON sidecar
(``<subtitle>.words.json``) so the renderer can be fed them without keeping
the live TTS objects alive across pipeline stages.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional

from loguru import logger

# 中文没有空格分词，按“单词/连续字母数字/单字符”拆 token，保证高亮粒度合理。
_TOKEN_RE = re.compile(
    r"[A-Za-z0-9]+(?:['\u2019-][A-Za-z0-9]+)*"
    r"|[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]"
)
_STRIP_RE = re.compile(r"[^A-Za-z0-9\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")


@dataclass(frozen=True)
class WordTiming:
    """一个词的开始/结束时间（秒）与文本。"""

    text: str
    start: float
    end: float


def word_timings_to_json(
    timings: Iterable[WordTiming], output_file: str
) -> bool:
    """把词时间轴写入 JSON sidecar；失败返回 False 且不抛异常。"""
    try:
        payload = {
            "words": [
                {"text": timing.text, "start": timing.start, "end": timing.end}
                for timing in timings
            ]
        }
        with open(output_file, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False)
        return True
    except Exception as exc:
        logger.warning(
            f"failed to write word timings: {output_file}, "
            f"error={type(exc).__name__}: {exc}"
        )
        return False


def load_word_timings_from_json(json_file: str) -> List[WordTiming]:
    """从 JSON sidecar 加载词时间轴；缺失/损坏时返回空列表。"""
    if not json_file or not os.path.isfile(json_file):
        return []
    try:
        with open(json_file, "r", encoding="utf-8") as fp:
            payload = json.load(fp)
        timings = []
        for item in payload.get("words", []):
            text = str(item.get("text", "") or "")
            start = _safe_float(item.get("start"))
            end = _safe_float(item.get("end"))
            if not text or start is None or end is None:
                continue
            if end < start:
                end = start
            timings.append(WordTiming(text=text, start=start, end=end))
        timings.sort(key=lambda timing: timing.start)
        return timings
    except Exception as exc:
        logger.warning(
            f"failed to load word timings: {json_file}, "
            f"error={type(exc).__name__}: {exc}"
        )
        return []


def extract_word_timings_from_submaker(sub_maker: Any) -> List[WordTiming]:
    """从 TTS 返回的 SubMaker 提取词时间轴（秒）。

    兼容两种结构：
    - edge_tts 7.x：``cues``（WordBoundary cue，``content``/``start``/``end``）；
    - 旧版/其它 provider：``subs`` + ``offset``（100ns 单位）。
    """
    if sub_maker is None:
        return []
    timings: List[WordTiming] = []

    cues = getattr(sub_maker, "cues", None)
    if cues:
        for cue in cues:
            content = getattr(cue, "content", "")
            if not content:
                continue
            start = _cue_seconds(getattr(cue, "start", None))
            end = _cue_seconds(getattr(cue, "end", None))
            if start is None:
                continue
            if end is None or end < start:
                end = start
            timings.append(WordTiming(text=content, start=start, end=end))

    if not timings:
        offsets = getattr(sub_maker, "offset", None) or []
        subs = getattr(sub_maker, "subs", None) or []
        for offset, sub in zip(offsets, subs):
            text = str(sub or "")
            if not text.strip():
                continue
            try:
                start_100ns, end_100ns = offset
                start = float(start_100ns) / 10_000_000.0
                end = float(end_100ns) / 10_000_000.0
            except (TypeError, ValueError):
                continue
            if end < start:
                end = start
            timings.append(WordTiming(text=text, start=start, end=end))

    timings.sort(key=lambda timing: timing.start)
    return timings


def _cue_seconds(value: Any) -> Optional[float]:
    try:
        return float(value.total_seconds())
    except AttributeError:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# 词时间轴与字幕帧文本的匹配 / 活动词选择
# ---------------------------------------------------------------------------


@dataclass
class PhraseWord:
    """字幕帧内一个词及其在绝对时间轴上的窗口。"""

    text: str
    start: float
    end: float


def match_words_to_phrase(
    timings: List[WordTiming],
    phrase_text: str,
    phrase_start: float,
    phrase_end: float,
) -> List[PhraseWord]:
    """把词时间轴按顺序对齐到单个字幕帧的文本上。

    匹配策略（确定性、容错）：
    1. 只考虑与帧窗口有交集的词；
    2. 去掉标点/空白后按顺序与帧文本的 token 序列匹配（大小写不敏感）；
    3. 时间窗口回退到帧窗口边界，保证连续。

    无法对齐（时间轴缺失、文本不一致）时返回空列表，调用方回退到
    整帧渲染——绝不因为匹配失败而中断生成。
    """
    if not timings or not phrase_text:
        return []

    tokens = tokenize(phrase_text)
    if not tokens:
        return []

    relevant = [
        timing
        for timing in timings
        if timing.end >= phrase_start and timing.start <= phrase_end
    ]
    if not relevant:
        return []

    matched: List[WordTiming] = []
    token_index = 0
    for timing in relevant:
        if token_index >= len(tokens):
            break
        if _normalize(timing.text) == _normalize(tokens[token_index]):
            matched.append(timing)
            token_index += 1
        else:
            # 宽容匹配：时间轴文本可能是标点粘连（"word,"），剥离后再比。
            stripped = _normalize(_STRIP_RE.sub("", timing.text))
            if stripped and stripped == _normalize(tokens[token_index]):
                matched.append(timing)
                token_index += 1

    if not matched:
        return []

    words: List[PhraseWord] = []
    for index, timing in enumerate(matched):
        start = max(phrase_start, timing.start)
        end = (
            matched[index + 1].start
            if index + 1 < len(matched)
            else min(phrase_end, timing.end)
        )
        end = max(end, start)
        words.append(PhraseWord(text=tokens[index], start=start, end=end))
    return words


def active_word_at(
    words: List[PhraseWord], t: float
) -> Optional[PhraseWord]:
    """返回时刻 ``t`` 正在朗读的词；没有词处于该时刻时返回 None。"""
    if not words:
        return None
    for word in words:
        if word.start <= t < word.end:
            return word
    return None


def tokenize(text: str) -> List[str]:
    """把文本拆成词 token（保留顺序与大小写，用于对齐与高亮）。"""
    return _TOKEN_RE.findall(str(text or ""))


def _normalize(text: str) -> str:
    return _STRIP_RE.sub("", str(text or "")).lower()