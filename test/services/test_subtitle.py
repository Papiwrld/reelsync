import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

# 测试文件直接运行时，也能从仓库根目录导入 app 包。
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services import subtitle


class TestSubtitleService(unittest.TestCase):
    def test_file_to_subtitles_returns_empty_for_missing_input(self):
        """空路径和不存在的文件都应安全返回空列表。"""
        self.assertEqual(subtitle.file_to_subtitles(""), [])
        with tempfile.TemporaryDirectory() as tmp_dir:
            missing_file = Path(tmp_dir) / "missing.srt"
            self.assertEqual(subtitle.file_to_subtitles(str(missing_file)), [])

    def test_levenshtein_distance_and_similarity_cover_common_boundaries(self):
        """
        字幕校正依赖编辑距离选择是否继续合并相邻字幕，因此覆盖空字符串、
        参数交换、大小写忽略和明显不相似四种边界，防止算法调整后误合并。
        """
        self.assertEqual(subtitle.levenshtein_distance("kitten", "sitting"), 3)
        self.assertEqual(subtitle.levenshtein_distance("a", "longer"), 6)
        self.assertEqual(subtitle.levenshtein_distance("hello", ""), 5)
        self.assertEqual(subtitle.similarity("Hello", "hello"), 1.0)
        self.assertLess(subtitle.similarity("hello", "world"), 0.5)

    def test_create_returns_empty_when_whisper_is_unavailable(self):
        """可选 Whisper 依赖未安装时应跳过，而不是在任务线程中抛异常。"""
        with patch.object(subtitle, "WhisperModel", None):
            self.assertEqual(subtitle.create("audio.mp3"), "")

    def test_create_returns_none_when_whisper_model_cannot_load(self):
        """模型下载或初始化失败时必须返回失败结果，并允许任务层更新状态。"""
        with (
            patch.object(subtitle, "model", None),
            patch.object(
                subtitle,
                "WhisperModel",
                side_effect=RuntimeError("model unavailable"),
            ),
        ):
            self.assertIsNone(subtitle.create("audio.mp3"))

    def test_create_writes_punctuated_and_trailing_segments(self):
        """
        使用假的 Whisper 模型覆盖逐词时间戳处理，不访问网络也不加载真实模型。
        一个 segment 同时包含标点断句和末尾无标点文本，可验证两条关键写入路径。
        """

        class _FakeWhisperModel:
            def __init__(self, **kwargs):
                self.init_kwargs = kwargs

            def transcribe(self, audio_file, **kwargs):
                words = [
                    SimpleNamespace(start=0.0, end=0.4, word="Hello"),
                    SimpleNamespace(start=0.4, end=0.9, word=" world."),
                    SimpleNamespace(start=1.0, end=1.5, word="Again"),
                ]
                segment = SimpleNamespace(
                    start=0.0,
                    end=1.8,
                    words=words,
                )
                info = SimpleNamespace(language="en", language_probability=0.99)
                return [segment], info

        with tempfile.TemporaryDirectory() as tmp_dir:
            subtitle_file = Path(tmp_dir) / "generated.srt"
            with (
                patch.object(subtitle, "model", None),
                patch.object(
                    subtitle,
                    "WhisperModel",
                    _FakeWhisperModel,
                ),
            ):
                subtitle.create("audio.mp3", str(subtitle_file))

            items = subtitle.file_to_subtitles(str(subtitle_file))

        self.assertEqual([item[2] for item in items], ["Hello world", "Again"])

    def test_correct_ignores_markdown_separator_lines(self):
        """
        Whisper fallback 校正阶段也必须忽略 `---` 这类不可发声脚本行。

        如果这里继续保留 Markdown 分隔符，`correct()` 会认为脚本行数多于
        字幕行数，并补出 `00:00:00,000 --> 00:00:00,000`，剪辑软件会把
        生成的 SRT 判定为不可导入。
        """
        original_srt = (
            "1\n"
            "00:00:00,100 --> 00:00:01,000\n"
            "第一段\n\n"
            "2\n"
            "00:00:01,100 --> 00:00:02,000\n"
            "第二段\n\n"
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            subtitle_file = Path(tmp_dir) / "subtitle.srt"
            subtitle_file.write_text(original_srt, encoding="utf-8")

            subtitle.correct(
                subtitle_file=str(subtitle_file),
                video_script="第一段\n---\n第二段",
            )

            corrected_srt = subtitle_file.read_text(encoding="utf-8")

        self.assertIn("第一段", corrected_srt)
        self.assertIn("第二段", corrected_srt)
        self.assertNotIn("---", corrected_srt)
        self.assertNotIn("00:00:00,000 --> 00:00:00,000", corrected_srt)

    def test_correct_merges_adjacent_subtitles_for_one_script_sentence(self):
        """
        Whisper 可能把一句文案拆成多个时间块。校正逻辑应合并时间范围并恢复
        原始脚本文本，避免最终字幕出现不必要的碎片。
        """
        original_srt = (
            "1\n00:00:00,100 --> 00:00:01,000\nHello\n\n"
            "2\n00:00:01,000 --> 00:00:02,000\nworld\n\n"
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            subtitle_file = Path(tmp_dir) / "subtitle.srt"
            subtitle_file.write_text(original_srt, encoding="utf-8")

            subtitle.correct(str(subtitle_file), "Hello world")
            items = subtitle.file_to_subtitles(str(subtitle_file))

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0][1], "00:00:00,100 --> 00:00:02,000")
        self.assertEqual(items[0][2], "Hello world")

    def test_correct_replaces_mismatch_and_appends_missing_script_line(self):
        """
        转写结果与脚本完全不一致时仍应以脚本为准；脚本多出的句子没有可复用
        时间轴时使用明确的零时间占位，避免丢失文本且保持现有兼容行为。
        """
        original_srt = "1\n00:00:00,100 --> 00:00:01,000\nWrong text\n\n"

        with tempfile.TemporaryDirectory() as tmp_dir:
            subtitle_file = Path(tmp_dir) / "subtitle.srt"
            subtitle_file.write_text(original_srt, encoding="utf-8")

            subtitle.correct(str(subtitle_file), "Expected sentence. Extra sentence.")
            items = subtitle.file_to_subtitles(str(subtitle_file))

        self.assertEqual(
            [item[2] for item in items],
            ["Expected sentence", "Extra sentence"],
        )
        self.assertEqual(items[1][1], "00:00:00,000 --> 00:00:00,000")

    def test_file_to_subtitles_keeps_last_block_without_trailing_newline(self):
        """
        The final subtitle must be parsed even when the SRT file does not end
        with a trailing blank line. Many tools omit it, and previously the last
        block was silently dropped because only a blank line flushed a block.
        """
        srt_without_trailing_blank = (
            "1\n"
            "00:00:00,000 --> 00:00:01,000\n"
            "Hello\n\n"
            "2\n"
            "00:00:01,000 --> 00:00:02,000\n"
            "World"
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            subtitle_file = Path(tmp_dir) / "subtitle.srt"
            subtitle_file.write_text(srt_without_trailing_blank, encoding="utf-8")

            items = subtitle.file_to_subtitles(str(subtitle_file))

        self.assertEqual(len(items), 2)
        self.assertEqual(items[0][2], "Hello")
        self.assertEqual(items[1][2], "World")

    def test_file_to_subtitles_parses_blocks_with_trailing_newline(self):
        """A normal SRT ending in a blank line still parses all blocks."""
        srt_with_trailing_blank = (
            "1\n"
            "00:00:00,000 --> 00:00:01,000\n"
            "Hello\n\n"
            "2\n"
            "00:00:01,000 --> 00:00:02,000\n"
            "World\n\n"
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            subtitle_file = Path(tmp_dir) / "subtitle.srt"
            subtitle_file.write_text(srt_with_trailing_blank, encoding="utf-8")

            items = subtitle.file_to_subtitles(str(subtitle_file))

        self.assertEqual([item[2] for item in items], ["Hello", "World"])


class TestSubtitlePhraseChunking(unittest.TestCase):
    """2~4 词短语分句：合并单字帧、拆分超长帧、保留自然语感。"""

    def test_merges_single_word_frames_into_one_phrase(self):
        items = [
            (0.0, 0.4, "I"),
            (0.4, 0.9, "am"),
            (0.9, 1.5, "going"),
            (1.5, 2.0, "home"),
        ]
        chunked = subtitle.chunk_subtitle_items(items)
        self.assertEqual(chunked, [(0.0, 2.0, "I am going home")])

    def test_pause_gap_is_a_natural_phrase_boundary(self):
        items = [(0.0, 0.5, "Hello"), (2.0, 2.6, "world")]
        chunked = subtitle.chunk_subtitle_items(items)
        self.assertEqual(
            chunked,
            [(0.0, 0.5, "Hello"), (2.0, 2.6, "world")],
        )

    def test_splits_oversized_entry_proportionally_and_balanced(self):
        # 8 个单元、上限 4：切成 4+4 两段，时间按片段数量均分。
        items = [(0.0, 4.0, "One two three four five six seven eight")]
        chunked = subtitle.chunk_subtitle_items(items)
        self.assertEqual(
            chunked,
            [
                (0.0, 2.0, "One two three four"),
                (2.0, 4.0, "five six seven eight"),
            ],
        )

    def test_split_avoids_single_word_remainder(self):
        # 5 个单元应按 3+2 切分，而不是 4+1 留下一个单词的碎帧；
        # 时间按两个片段均分。
        items = [(0.0, 2.0, "Hello, brave new world today")]
        chunked = subtitle.chunk_subtitle_items(items)
        self.assertEqual(
            chunked,
            [
                (0.0, 1.0, "Hello, brave new"),
                (1.0, 2.0, "world today"),
            ],
        )

    def test_punctuation_stays_attached_to_preceding_word(self):
        # 切分边界取下一段首词的位置，逗号应保留在 "Hello," 上。
        items = [(0.0, 2.0, "Hello, brave new world today")]
        chunked = subtitle.chunk_subtitle_items(items)
        self.assertTrue(chunked[0][2].startswith("Hello,"))
        self.assertNotIn(",", chunked[1][2])

    def test_dangling_single_word_merges_into_next_phrase(self):
        # "Wow." 只有 1 个词，紧随其后的 5 词句被拆成 3+2 后，
        # 单字帧并入第一段，不产生一闪而过的单字。
        items = [
            (0.0, 0.5, "Wow."),
            (0.5, 3.0, "incredible and truly amazing stuff"),
        ]
        chunked = subtitle.chunk_subtitle_items(items)
        self.assertEqual(
            chunked,
            [
                (0.0, 1.75, "Wow. incredible and truly"),
                (1.75, 3.0, "amazing stuff"),
            ],
        )

    def test_punctuated_sentence_keeps_natural_phrasing_when_merged(self):
        items = [(0.0, 0.8, "Hello."), (0.8, 2.0, "How are you?")]
        chunked = subtitle.chunk_subtitle_items(items)
        self.assertEqual(chunked, [(0.0, 2.0, "Hello. How are you?")])

    def test_cjk_frames_use_character_scaled_limits(self):
        # CJK 按 4~8 字符折算：2+6=8 个字符仍在同一帧内，且不加空格。
        items = [(0.0, 0.6, "人工"), (0.6, 1.5, "智能正在改变")]
        chunked = subtitle.chunk_subtitle_items(items)
        self.assertEqual(chunked, [(0.0, 1.5, "人工智能正在改变")])

    def test_empty_and_invalid_items_return_empty(self):
        self.assertEqual(subtitle.chunk_subtitle_items([]), [])
        self.assertEqual(subtitle.chunk_subtitle_items([(0.0, 1.0, "  ")]), [])
        self.assertEqual(subtitle.chunk_subtitle_items([("bad", 1.0, "x")]), [])

    def test_invalid_limits_raise(self):
        with self.assertRaises(ValueError):
            subtitle.chunk_subtitle_items([(0.0, 1.0, "x")], max_words=0)
        with self.assertRaises(ValueError):
            subtitle.chunk_subtitle_items([(0.0, 1.0, "x")], min_words=3, max_words=2)

    def test_chunk_subtitle_file_rewrites_srt_into_phrase_frames(self):
        srt = (
            "1\n00:00:00,000 --> 00:00:00,400\nI\n\n"
            "2\n00:00:00,400 --> 00:00:00,900\nam\n\n"
            "3\n00:00:00,900 --> 00:00:01,500\ngoing\n\n"
            "4\n00:00:01,500 --> 00:00:02,000\nhome\n\n"
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            subtitle_file = Path(tmp_dir) / "subtitle.srt"
            subtitle_file.write_text(srt, encoding="utf-8")

            self.assertTrue(subtitle.chunk_subtitle_file(str(subtitle_file)))
            items = subtitle.file_to_subtitles(str(subtitle_file))

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0][1], "00:00:00,000 --> 00:00:02,000")
        self.assertEqual(items[0][2], "I am going home")

    def test_chunk_subtitle_file_missing_returns_false(self):
        self.assertFalse(subtitle.chunk_subtitle_file(""))
        with tempfile.TemporaryDirectory() as tmp_dir:
            missing = Path(tmp_dir) / "missing.srt"
            self.assertFalse(subtitle.chunk_subtitle_file(str(missing)))

    def test_backwards_zero_duration_placeholder_stays_isolated(self):
        # correct() 在脚本行多于字幕行时可能补出 00:00:00,000 --> 00:00:00,000
        # 的零时长占位。它不能与前面的帧合并成结束早于开始的坏帧。
        items = [
            (0.1, 1.0, "Expected sentence"),
            (0.0, 0.0, "Extra sentence"),
        ]
        chunked = subtitle.chunk_subtitle_items(items)
        self.assertEqual(
            chunked,
            [
                (0.1, 1.0, "Expected sentence"),
                (0.0, 0.001, "Extra sentence"),
            ],
        )
        for start, end, _ in chunked:
            self.assertGreaterEqual(end, start)

    def test_overlapping_input_frames_are_clamped_not_duplicated(self):
        # Whisper/Edge 片段轻微交叠时（例如 correct() 合并后相邻帧时间交叉），
        # 写回 SRT 会造成同一时刻两条字幕同时显示。合并后的 chunk 必须把起点
        # 钳制到前一 chunk 的终点之后，且不产生零时长坏帧。
        items = [
            (0.0, 3.0, "one two three four five six"),
            (2.8, 5.5, "seven eight"),
        ]
        chunked = subtitle.chunk_subtitle_items(items)
        for i in range(1, len(chunked)):
            self.assertGreaterEqual(chunked[i][0], chunked[i - 1][1])
            self.assertGreater(chunked[i][1], chunked[i][0])
        texts = [c[2] for c in chunked]
        self.assertEqual(" ".join(texts), "one two three four five six seven eight")

    def test_fully_swallowed_overlap_merges_into_previous_frame(self):
        # 后一帧完全被前一帧覆盖时不能丢字：文本并入前一条，时间取两者并集。
        items = [
            (0.0, 4.0, "one two three four"),
            (1.0, 2.5, "five six"),
            (2.5, 6.0, "seven eight"),
        ]
        chunked = subtitle.chunk_subtitle_items(items)
        for i in range(1, len(chunked)):
            self.assertGreaterEqual(chunked[i][0], chunked[i - 1][1])
        joined = " ".join(c[2] for c in chunked)
        for word in ("one", "two", "three", "four", "five", "six", "seven", "eight"):
            self.assertIn(word, joined)

    def test_srt_time_to_seconds_parses_standard_format(self):
        self.assertAlmostEqual(subtitle._srt_time_to_seconds("00:01:02,500"), 62.5)
        self.assertEqual(subtitle._srt_time_to_seconds("not a time"), 0.0)


if __name__ == "__main__":
    unittest.main()
