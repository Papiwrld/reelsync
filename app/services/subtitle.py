import json
import os.path
import re
import threading
from timeit import default_timer as timer
from typing import List, Tuple

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None
from loguru import logger

from app.config import config
from app.utils import utils

model_size = config.whisper.get("model_size", "medium")
device = config.whisper.get("device", "cpu")
compute_type = config.whisper.get("compute_type", "int8")
# 允许通过 video_language 提升识别准确率（尤其短句）；默认自动检测
whisper_language = str(config.whisper.get("language", "") or "").strip().lower() or None
initial_prompt = config.whisper.get("initial_prompt", "") or None
model = None
_model_lock = threading.Lock()


def create(audio_file, subtitle_file: str = ""):
    global model
    if WhisperModel is None:
        logger.warning(
            "faster_whisper not available, skipping whisper subtitle generation"
        )
        return ""
    with _model_lock:
        if not model:
            model_path = f"{utils.root_dir()}/models/whisper-{model_size}"
            model_bin_file = f"{model_path}/model.bin"
            if not os.path.isdir(model_path) or not os.path.isfile(model_bin_file):
                model_path = model_size

            logger.info(
                f"loading model: {model_path}, device: {device}, compute_type: {compute_type}"
            )
            try:
                model = WhisperModel(
                    model_size_or_path=model_path, device=device, compute_type=compute_type
                )
            except Exception as e:
                logger.error(
                    f"failed to load model: {e} \n\n"
                    f"********************************************\n"
                    f"this may be caused by network issue. \n"
                    f"please download the model manually and put it in the 'models' folder. \n"
                    f"see [README.md FAQ](https://github.com/Papiwrld/reelsync) for more details.\n"
                    f"********************************************\n\n"
                )
                return None

    logger.info(f"start, output file: {subtitle_file}")
    if not subtitle_file:
        subtitle_file = f"{audio_file}.srt"

    transcribe_kwargs: dict = dict(
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )
    if initial_prompt:
        transcribe_kwargs["initial_prompt"] = initial_prompt
    if whisper_language and whisper_language not in ("auto", ""):
        transcribe_kwargs["language"] = whisper_language
    segments, info = model.transcribe(audio_file, **transcribe_kwargs)

    logger.info(
        f"detected language: '{info.language}', probability: {info.language_probability:.2f}"
    )

    start = timer()
    subtitles = []
    word_timings = []

    def recognized(seg_text, seg_start, seg_end):
        seg_text = seg_text.strip()
        if not seg_text:
            return

        msg = "[%.2fs -> %.2fs] %s" % (seg_start, seg_end, seg_text)
        logger.debug(msg)

        subtitles.append(
            {"msg": seg_text, "start_time": seg_start, "end_time": seg_end}
        )

    for segment in segments:
        words_idx = 0
        words_len = len(segment.words)

        seg_start = 0
        seg_end = 0
        seg_text = ""

        if segment.words:
            is_segmented = False
            for word in segment.words:
                if not is_segmented:
                    seg_start = word.start
                    is_segmented = True

                seg_end = word.end
                # If it contains punctuation, then break the sentence.
                seg_text += word.word

                # 逐词时间戳（供渲染层逐词高亮）：Whisper 的词通常带前导
                # 空格，去掉后保留 文本+起止时间；单个词的异常时间被丢弃。
                word_text = (word.word or "").strip()
                if word_text:
                    word_timings.append(
                        {
                            "text": word_text,
                            "start": float(word.start),
                            "end": float(word.end),
                        }
                    )

                if utils.str_contains_punctuation(word.word):
                    # remove last char
                    seg_text = seg_text[:-1]
                    if not seg_text:
                        continue

                    recognized(seg_text, seg_start, seg_end)

                    is_segmented = False
                    seg_text = ""

                if words_idx == 0 and segment.start < word.start:
                    seg_start = word.start
                if words_idx == (words_len - 1) and segment.end > word.end:
                    seg_end = word.end
                words_idx += 1

        if not seg_text:
            continue

        recognized(seg_text, seg_start, seg_end)

    end = timer()

    diff = end - start
    logger.info(f"complete, elapsed: {diff:.2f} s")

    idx = 1
    lines = []
    for subtitle in subtitles:
        text = subtitle.get("msg")
        if text:
            lines.append(
                utils.text_to_srt(
                    idx, text, subtitle.get("start_time"), subtitle.get("end_time")
                )
            )
            idx += 1

    sub = "\n".join(lines) + "\n"
    with open(subtitle_file, "w", encoding="utf-8") as f:
        f.write(sub)
    logger.info(f"subtitle file created: {subtitle_file}")

    # 词时间轴 sidecar：逐词高亮依赖它。写失败只影响高亮，不影响字幕。
    if word_timings:
        try:
            from app.services.subtitle_engine.timing import (
                WordTiming,
                word_timings_to_json,
            )

            word_timings_to_json(
                [
                    WordTiming(
                        text=timing["text"],
                        start=float(timing["start"]),
                        end=float(timing["end"]),
                    )
                    for timing in word_timings
                ],
                subtitle_file + ".words.json",
            )
        except Exception as exc:
            logger.warning(
                f"failed to write whisper word timings sidecar: "
                f"error={type(exc).__name__}, detail={exc}"
            )


# 短语分句的“可计数单元”：拉丁语系按空格分词（保留撇号/连字符连词），
# 中日韩文本没有空格，按单个字符计数。
_PHRASE_UNIT_RE = re.compile(
    r"[A-Za-z0-9]+(?:['’\x27-][A-Za-z0-9]+)*|"
    r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]"
)
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")
# CJK 没有空格分词，2~4 个词折算为 4~8 个字符。
_CJK_WORD_SCALE = 2


def _srt_time_to_seconds(value: str) -> float:
    """把 SRT 时间 ``HH:MM:SS,mmm`` 解析为秒；非法输入返回 0.0。"""
    value = str(value or "").strip()
    try:
        hours, minutes, seconds = value.split(":")
        seconds, _, millis = seconds.partition(",")
        return (
            int(hours) * 3600
            + int(minutes) * 60
            + int(seconds)
            + int(millis or 0) / 1000.0
        )
    except (TypeError, ValueError):
        return 0.0


def _count_phrase_units(text: str) -> Tuple[int, bool]:
    """
    统计文本的“短语单元”数量，并判断是否包含 CJK 字符。

    返回 ``(单元数, 是否含 CJK)``。调用方据此把 2~4 词的英文目标折算成
    4~8 个字符的中文目标，保证两种语言的字幕帧密度一致。
    """
    spans = list(_PHRASE_UNIT_RE.finditer(text))
    return len(spans), bool(_CJK_RE.search(text))


def _slice_text_by_units(text: str, max_units: int) -> List[str]:
    """
    把文本按单元切成每段不超过 ``max_units`` 个单元的片段。

    切分边界取下一段首个单元的开始位置，因此逗号、句号等标点始终跟随
    前一个词而不是被孤立到下一行；原始字符（包括空格和标点）全部保留。
    """
    spans = list(_PHRASE_UNIT_RE.finditer(text))
    if not spans:
        return [text.strip()] if text.strip() else []
    if len(spans) <= max_units:
        return [text.strip()] if text.strip() else []

    # 把单元尽量均匀分到各组，避免最后只留下一个单词的碎帧。
    # 例如 5 个单元、上限 4：切成 3+2 而不是 4+1。
    total_units = len(spans)
    part_count = (total_units + max_units - 1) // max_units
    part_size = (total_units + part_count - 1) // part_count

    boundaries = [0]
    for index in range(part_size, total_units, part_size):
        boundaries.append(spans[index].start())
    boundaries.append(len(text))

    slices = []
    for index in range(len(boundaries) - 1):
        piece = text[boundaries[index] : boundaries[index + 1]].strip()
        if piece:
            slices.append(piece)
    return slices


def chunk_subtitle_items(
    items,
    min_words: int = 2,
    max_words: int = 4,
    max_gap_seconds: float = 0.35,
):
    """
    把过短的字幕帧合并、过长的帧拆分，使每个字幕帧包含 2~4 个词。

    输入 ``items`` 为 ``(start_seconds, end_seconds, text)`` 列表，返回同样
    结构的分句结果。设计目标：

    - **避免单字闪烁**：Whisper/Edge 按标点切出的单字帧会与相邻帧合并，
      只有被明显停顿（超过 ``max_gap_seconds``）隔开的单字才会保留。
    - **保留自然语感**：合并尊重静音间隙，悬挂的单个词并入下一短语而不
      是单独闪一帧；CJK 文本按 4~8 字符折算，避免把中文切成碎字。
    - **上限约束**：单个条目超过上限时按单元数比例分配时间并拆分，保证
      每帧不超过 ``max_words``（CJK 为两倍字符）。

    超出上限的长帧拆分的切分点位于词边界，标点跟随前一个词，因此即使
    硬性截断也保持可读。
    """
    if min_words < 1 or max_words < 1:
        raise ValueError("min_words and max_words must be >= 1")
    if max_words < min_words:
        raise ValueError("max_words must be >= min_words")

    normalized = []
    for start, end, text in items:
        text = str(text or "").strip()
        if not text:
            continue
        try:
            start = max(0.0, float(start))
            end = max(float(end), start + 0.001)
        except (TypeError, ValueError):
            continue
        normalized.append((start, end, text))

    if not normalized:
        return []

    # 拆分阶段：把超过上限的单条字幕按单元数比例切分。
    pieces = []  # (start, end, text, is_cjk, units)
    for start, end, text in normalized:
        units, is_cjk = _count_phrase_units(text)
        max_units = max_words * (_CJK_WORD_SCALE if is_cjk else 1)
        if units <= max_units:
            pieces.append((start, end, text, is_cjk, units))
            continue
        slices = _slice_text_by_units(text, max_units)
        part_count = len(slices)
        for index, slice_text in enumerate(slices):
            part_start = start + (end - start) * index / part_count
            part_end = start + (end - start) * (index + 1) / part_count
            part_units, part_is_cjk = _count_phrase_units(slice_text)
            pieces.append((part_start, part_end, slice_text, part_is_cjk, part_units))

    # 合并阶段：贪心合并相邻条目为 2~max 单元的短语。
    chunks = []
    current = None  # [start, end, [texts], units, is_cjk]

    def close_current():
        nonlocal current
        if current is None:
            return
        start, end, texts, units, is_cjk = current
        separator = "" if is_cjk else " "
        merged_text = separator.join(part.strip() for part in texts).strip()
        chunks.append((start, end, merged_text))
        current = None

    for start, end, text, is_cjk, units in pieces:
        if current is None:
            current = [start, end, [text], units, is_cjk]
            continue

        _, _, _, current_units, current_is_cjk = current
        scale = _CJK_WORD_SCALE if current_is_cjk else 1
        max_units = max_words * scale
        min_units = min_words * scale
        gap = start - current[1]

        if start < current[0]:
            # 时间倒退的条目（例如 correct() 补出的零时长占位）不能参与
            # 合并，否则会合并出结束时间早于开始时间的坏帧。单独保留。
            close_current()
            current = [start, end, [text], units, is_cjk]
        elif gap > max_gap_seconds:
            # 明显停顿是天然断句点，即使当前只有单个词也保留独立帧。
            close_current()
            current = [start, end, [text], units, is_cjk]
        elif current_units + units > max_units:
            if current_units < min_units and gap <= max_gap_seconds:
                # 悬挂的单个词并入下一短语：宁稍微超过上限，也不产生
                # 一闪而过的单字帧。
                current[1] = end
                current[2].append(text)
                current[3] += units
                current[4] = current[4] or is_cjk
            else:
                close_current()
                current = [start, end, [text], units, is_cjk]
        else:
            current[1] = end
            current[2].append(text)
            current[3] += units
            current[4] = current[4] or is_cjk

    close_current()

    # 时间轴去重叠：Whisper/Edge 的片段可能轻微交叠，correct() 合并时也可能
    # 让相邻帧时间交叉。若直接写回 SRT，同一时刻会有两条字幕同时显示。
    # 这里把每个 chunk 的起点钳制到前一 chunk 的终点之后；被完全吞没的帧
    # 并入前一条文本，避免丢字也避免零时长坏帧。
    merged = []
    for start, end, text in chunks:
        if not merged:
            merged.append((start, end, text))
            continue
        prev_start, prev_end, prev_text = merged[-1]
        if start >= prev_end:
            merged.append((start, end, text))
            continue
        if start < prev_start:
            # 时间回退的帧（零时长占位等）已在合并循环中隔离，不再重叠处理。
            merged.append((start, end, text))
            continue
        clamped_start = prev_end
        if clamped_start >= end:
            _, prev_is_cjk = _count_phrase_units(prev_text)
            separator = "" if prev_is_cjk else " "
            merged[-1] = (
                prev_start,
                max(prev_end, end),
                (prev_text + separator + text).strip(),
            )
        else:
            merged[-1] = (prev_start, prev_end, prev_text)
            merged.append((clamped_start, end, text))
    return merged


def chunk_subtitle_file(subtitle_file: str) -> bool:
    """
    读取 SRT 文件，应用 2~4 词短语分句并写回。

    文件缺失、为空或没有可解析条目时返回 False 且不修改文件。
    """
    items = file_to_subtitles(subtitle_file)
    if not items:
        return False

    numeric = []
    for _, times, text in items:
        # file_to_subtitles 保留的是 "HH:MM:SS,mmm --> HH:MM:SS,mmm"。
        start_str, _, end_str = times.partition("-->")
        numeric.append(
            (
                _srt_time_to_seconds(start_str),
                _srt_time_to_seconds(end_str),
                text,
            )
        )

    chunked = chunk_subtitle_items(numeric)
    if not chunked:
        return False

    lines = []
    for index, (start, end, text) in enumerate(chunked, start=1):
        # 兜底保证每帧时长为正，避免极端输入写出结束早于开始的非法 SRT。
        end = max(float(end), float(start) + 0.001)
        lines.append(utils.text_to_srt(index, text, start, end))
    with open(subtitle_file, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines) + "\n")
    logger.info(f"chunked subtitle into {len(chunked)} phrase frames: {subtitle_file}")
    return True


def file_to_subtitles(filename):
    if not filename or not os.path.isfile(filename):
        return []

    times_texts = []
    current_times = None
    current_text = ""
    index = 0
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            times = re.findall("([0-9]*:[0-9]*:[0-9]*,[0-9]*)", line)
            if times:
                current_times = line
            elif line.strip() == "" and current_times:
                index += 1
                times_texts.append((index, current_times.strip(), current_text.strip()))
                current_times, current_text = None, ""
            elif current_times:
                current_text += line

    # Flush the final block. SRT files whose last subtitle is not followed by a
    # trailing blank line never hit the blank-line branch above, so without this
    # the last subtitle would be silently dropped.
    if current_times:
        index += 1
        times_texts.append((index, current_times.strip(), current_text.strip()))
    return times_texts


def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def similarity(a, b):
    distance = levenshtein_distance(a.lower(), b.lower())
    max_length = max(len(a), len(b))
    return 1 - (distance / max_length)


def correct(subtitle_file, video_script):
    subtitle_items = file_to_subtitles(subtitle_file)
    normalized_script = utils.normalize_script_for_subtitle_matching(video_script)
    script_lines = utils.split_string_by_punctuations(normalized_script)

    corrected = False
    new_subtitle_items = []
    script_index = 0
    subtitle_index = 0

    while script_index < len(script_lines) and subtitle_index < len(subtitle_items):
        script_line = script_lines[script_index].strip()
        subtitle_line = subtitle_items[subtitle_index][2].strip()

        if script_line == subtitle_line:
            new_subtitle_items.append(subtitle_items[subtitle_index])
            script_index += 1
            subtitle_index += 1
        else:
            combined_subtitle = subtitle_line
            start_time = subtitle_items[subtitle_index][1].split(" --> ")[0]
            end_time = subtitle_items[subtitle_index][1].split(" --> ")[1]
            next_subtitle_index = subtitle_index + 1

            while next_subtitle_index < len(subtitle_items):
                next_subtitle = subtitle_items[next_subtitle_index][2].strip()
                if similarity(
                    script_line, combined_subtitle + " " + next_subtitle
                ) > similarity(script_line, combined_subtitle):
                    combined_subtitle += " " + next_subtitle
                    end_time = subtitle_items[next_subtitle_index][1].split(" --> ")[1]
                    next_subtitle_index += 1
                else:
                    break

            if similarity(script_line, combined_subtitle) > 0.8:
                logger.warning(
                    f"Merged/Corrected - Script: {script_line}, Subtitle: {combined_subtitle}"
                )
                new_subtitle_items.append(
                    (
                        len(new_subtitle_items) + 1,
                        f"{start_time} --> {end_time}",
                        script_line,
                    )
                )
                corrected = True
            else:
                logger.warning(
                    f"Mismatch - Script: {script_line}, Subtitle: {combined_subtitle}"
                )
                new_subtitle_items.append(
                    (
                        len(new_subtitle_items) + 1,
                        f"{start_time} --> {end_time}",
                        script_line,
                    )
                )
                corrected = True

            script_index += 1
            subtitle_index = next_subtitle_index

    # Process the remaining lines of the script.
    while script_index < len(script_lines):
        logger.warning(f"Extra script line: {script_lines[script_index]}")
        if subtitle_index < len(subtitle_items):
            new_subtitle_items.append(
                (
                    len(new_subtitle_items) + 1,
                    subtitle_items[subtitle_index][1],
                    script_lines[script_index],
                )
            )
            subtitle_index += 1
        else:
            new_subtitle_items.append(
                (
                    len(new_subtitle_items) + 1,
                    "00:00:00,000 --> 00:00:00,000",
                    script_lines[script_index],
                )
            )
        script_index += 1
        corrected = True

    if corrected:
        with open(subtitle_file, "w", encoding="utf-8") as fd:
            for i, item in enumerate(new_subtitle_items):
                fd.write(f"{i + 1}\n{item[1]}\n{item[2]}\n\n")
        logger.info("Subtitle corrected")
    else:
        logger.success("Subtitle is correct")


if __name__ == "__main__":
    task_id = "c12fd1e6-4b0a-4d65-a075-c87abe35a072"
    task_dir = utils.task_dir(task_id)
    subtitle_file = f"{task_dir}/subtitle.srt"
    audio_file = f"{task_dir}/audio.mp3"

    subtitles = file_to_subtitles(subtitle_file)
    print(subtitles)

    script_file = f"{task_dir}/script.json"
    with open(script_file, "r") as f:
        script_content = f.read()
    s = json.loads(script_content)
    script = s.get("script")

    correct(subtitle_file, script)

    subtitle_file = f"{task_dir}/subtitle-test.srt"
    create(audio_file, subtitle_file)
