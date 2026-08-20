"""质量类视频功能单测：可配置 FPS、loudnorm 集成、ASS 字幕引擎与 ffmpeg 烧录。

覆盖 Gap 1 / Gap 2 新增功能：
  1. ``app.services.video.constants`` 的 FPS 校验与 loudnorm 配置读取；
  2. ``app.services.video`` 的 loudnorm 分支（AFX 优先、ffmpeg 兜底）与
     ``_get_effective_video_fps`` 优先级解析；
  3. ``app.services.subtitle_engine.renderer`` 的 ASS 纯函数与侧车文件生成；
  4. ``_burn_ass_subtitles_via_ffmpeg`` 的 ffmpeg 调用与失败降级。

``generate_video`` 采用与 ``test_video.py`` 相同的整函数 mock 方式直接测试：
现有测试已经证明该 mock 面（_open_video_clip_quietly / AudioFileClip /
_write_videofile_with_codec_fallback 等）足够稳定，无需降级为只测小助手函数。
"""

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# add project root to python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.config import config
from app.models.schema import VideoParams
from app.services import video as vd
from app.services.subtitle_engine.renderer import (
    _ass_escape_text,
    _hex_to_ass_color,
    build_ass_content,
    generate_ass_file,
)
from app.services.subtitle_engine.styles import SubtitleStyleConfig
from app.services.subtitle_engine.timing import WordTiming
from app.services.video import constants as vc


class _FakeMoviePyClip:
    """为 generate_video 单测提供最小 MoviePy 接口，并记录施加的特效列表。"""

    def __init__(self, *, duration=5, fps=44100):
        self.duration = duration
        self.fps = fps
        self.close_calls = 0
        self.effects_applied = []
        self.with_audio_result = self

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

    def close(self):
        self.close_calls += 1

    def with_effects(self, effects):
        self.effects_applied.append(effects)
        return self

    def with_audio(self, _audio):
        return self.with_audio_result


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (24, 24),
        (30, 30),
        (60, 60),
        ("30", 30),
        (23, 30),
        (61, 30),
        ("abc", 30),
        (None, 30),
    ],
)
def test_validate_video_fps(value, expected):
    """24-60 接受，越界/非法/缺失回退默认 30。"""
    assert vc._validate_video_fps(value) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("{hello}", r"\{hello\}"),
        ("a{b}c", r"a\{b\}c"),
        ("a\\b", r"a\\b"),
        # 换行由 _ass_escape_text 原样保留，后续 build_ass_content 转成 \N。
        ("line1\nline2", "line1\nline2"),
    ],
)
def test_ass_escape_text(text, expected):
    assert _ass_escape_text(text) == expected


@pytest.mark.parametrize(
    ("hex_color", "opacity", "expected"),
    [
        ("#FFFFFF", 0.0, "&H00FFFFFF"),
        # ASS 使用 BGR 顺序：R=FF, G=D6, B=0A -> 0A D6 FF。
        ("#FFD60A", 0.0, "&H000AD6FF"),
        # opacity 1.0（全透明）-> alpha FF。
        ("#000000", 1.0, "&HFF000000"),
        ("#FFFFFF", 0.5, "&H80FFFFFF"),
    ],
)
def test_hex_to_ass_color(hex_color, opacity, expected):
    assert _hex_to_ass_color(hex_color, opacity) == expected


class TestVideoConstants(unittest.TestCase):
    """app.services.video.constants：FPS 配置读取与 loudnorm 开关。"""

    def setUp(self):
        self.original_app_config = dict(config.app)
        config.app.pop("video_fps", None)
        config.app.pop("audio_loudnorm", None)

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)

    def test_get_configured_video_fps_returns_default_30(self):
        """配置缺失时返回稳定默认值 30。"""
        self.assertEqual(vc.get_configured_video_fps(), 30)

    def test_get_configured_video_fps_reads_valid_config_value(self):
        """合法配置值（24-60）应原样返回。"""
        config.app["video_fps"] = 48
        self.assertEqual(vc.get_configured_video_fps(), 48)

    def test_get_configured_video_fps_falls_back_on_bad_value(self):
        """越界配置必须回退 30，不能把 23/61 直接交给编码器。"""
        config.app["video_fps"] = 23
        self.assertEqual(vc.get_configured_video_fps(), 30)
        config.app["video_fps"] = 61
        self.assertEqual(vc.get_configured_video_fps(), 30)

    def test_is_audio_loudnorm_enabled_defaults_true(self):
        """未配置 loudnorm 时默认开启（-14 LUFS），保证混音不削波。"""
        self.assertTrue(vc.is_audio_loudnorm_enabled())

    def test_is_audio_loudnorm_enabled_reads_config(self):
        config.app["audio_loudnorm"] = True
        self.assertTrue(vc.is_audio_loudnorm_enabled())

    def test_is_audio_loudnorm_enabled_accepts_string_flags(self):
        config.app["audio_loudnorm"] = "true"
        self.assertTrue(vc.is_audio_loudnorm_enabled())
        config.app["audio_loudnorm"] = "1"
        self.assertTrue(vc.is_audio_loudnorm_enabled())
        config.app["audio_loudnorm"] = "off"
        self.assertFalse(vc.is_audio_loudnorm_enabled())

    def test_get_audio_loudnorm_ffmpeg_params_shape(self):
        """ffmpeg loudnorm 参数必须是 -filter:a + 完整滤镜串的二元列表。"""
        self.assertEqual(
            vc.get_audio_loudnorm_ffmpeg_params(),
            ["-filter:a", "loudnorm=I=-14:TP=-1:LRA=11"],
        )


class TestVideoQualityIntegration(unittest.TestCase):
    """generate_video 的 loudnorm 分支与有效 FPS 优先级解析。"""

    def setUp(self):
        self.original_app_config = dict(config.app)
        config.app.pop("video_fps", None)
        config.app.pop("audio_loudnorm", None)
        vd._runtime_disabled_video_codecs.clear()

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)
        vd._runtime_disabled_video_codecs.clear()

    # ---------------- _is_loudnorm_enabled ----------------

    def test_is_loudnorm_enabled_uses_params_when_set(self):
        """params 显式设置 audio_loudnorm=True 时优先于配置。"""
        params = VideoParams(video_subject="test", audio_loudnorm=True)
        self.assertTrue(vd._is_loudnorm_enabled(params))

    def test_is_loudnorm_enabled_falls_back_to_config(self):
        """params 未显式设置时回退配置；配置也没有则默认开启。"""
        params = VideoParams(video_subject="test")
        self.assertTrue(vd._is_loudnorm_enabled(params))
        config.app["audio_loudnorm"] = False
        self.assertFalse(vd._is_loudnorm_enabled(params))

    def test_is_loudnorm_enabled_params_false_overrides_config(self):
        """显式 False 必须压过配置里的 True，不能因为配置开启而削波。"""
        config.app["audio_loudnorm"] = True
        params = VideoParams(video_subject="test", audio_loudnorm=False)
        self.assertFalse(vd._is_loudnorm_enabled(params))

    # ---------------- _get_effective_video_fps ----------------

    def test_get_effective_video_fps_explicit_wins(self):
        """显式参数优先级最高，覆盖 params 与配置。"""
        params = VideoParams(video_subject="test", video_fps=48)
        self.assertEqual(vd._get_effective_video_fps(params=params, explicit=24), 24)

    def test_get_effective_video_fps_explicit_invalid_falls_back(self):
        """显式值越界时回退默认 30。"""
        self.assertEqual(vd._get_effective_video_fps(explicit=23), 30)

    def test_get_effective_video_fps_uses_params_when_set(self):
        """params 显式设置 video_fps 时使用该值。"""
        params = VideoParams(video_subject="test", video_fps=48)
        self.assertEqual(vd._get_effective_video_fps(params=params), 48)

    def test_get_effective_video_fps_config_fallback(self):
        """params 未设置时回退配置，配置缺失时用默认 30。"""
        config.app["video_fps"] = 48
        self.assertEqual(vd._get_effective_video_fps(), 48)

        config.app.pop("video_fps", None)
        self.assertEqual(vd._get_effective_video_fps(), 30)

        # params 默认值 30 不应掩盖配置覆盖。
        config.app["video_fps"] = 55
        params = VideoParams(video_subject="test")
        self.assertEqual(vd._get_effective_video_fps(params=params), 55)

    # ---------------- generate_video loudnorm 分支 ----------------

    def _run_generate_video_with_loudnorm(self, params, temp_dir):
        """按 test_video.py 的 mock 约定跑 generate_video 最小路径。"""

        source_video = _FakeMoviePyClip()
        voice_source = _FakeMoviePyClip()
        final_video = _FakeMoviePyClip()
        source_video.with_audio_result = final_video
        output_file = os.path.join(temp_dir, "final.mp4")

        with (
            patch.object(vd, "_open_video_clip_quietly", return_value=source_video),
            patch.object(vd, "AudioFileClip", return_value=voice_source),
            patch.object(
                vd,
                "_write_videofile_with_codec_fallback",
                side_effect=lambda clip, output_file, **kwargs: Path(
                    output_file
                ).write_bytes(b"video-data"),
            ) as writer,
            patch.object(vd, "_get_configured_video_codec", return_value="libx264"),
        ):
            result = vd.generate_video(
                video_path="combined.mp4",
                audio_path="voice.mp3",
                subtitle_path="",
                output_file=output_file,
                params=params,
            )
        return result, writer, voice_source

    def test_generate_video_uses_afx_when_audio_normalize_available(self):
        """
        loudnorm 开启且 moviepy.afx.AudioNormalize 可用时，应走 AFX 特效路径，
        不再附加 ffmpeg loudnorm filter。
        """
        params = VideoParams(
            video_subject="test",
            subtitle_enabled=False,
            bgm_type="random",
            bgm_volume=0.0,
            audio_loudnorm=True,
        )
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch.object(vd.afx, "AudioNormalize", create=True) as audio_normalize,
        ):
            result, writer, voice_source = (
                self._run_generate_video_with_loudnorm(params, temp_dir)
            )

        self.assertTrue(result)
        audio_normalize.assert_called_once()
        # 旁白 MultiplyVolume 之后紧跟 AudioNormalize 特效。
        self.assertEqual(len(voice_source.effects_applied), 2)
        self.assertIs(voice_source.effects_applied[1][0], audio_normalize.return_value)
        self.assertNotIn("ffmpeg_params", writer.call_args.kwargs)

    def test_generate_video_falls_back_to_ffmpeg_loudnorm_when_afx_missing(self):
        """
        AudioNormalize 缺失（如 moviepy 版本不支持）时，ffmpeg_params
        必须附加 loudnorm 滤镜参数，保证响度归一化仍生效。
        """
        params = VideoParams(
            video_subject="test",
            subtitle_enabled=False,
            bgm_type="random",
            bgm_volume=0.0,
            audio_loudnorm=True,
        )
        fake_afx = types.SimpleNamespace(
            MultiplyVolume=MagicMock(),
            AudioFadeOut=MagicMock(),
            AudioLoop=MagicMock(),
        )
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch.object(vd, "afx", fake_afx),
        ):
            result, writer, _voice_source = (
                self._run_generate_video_with_loudnorm(params, temp_dir)
            )

        self.assertTrue(result)
        self.assertEqual(
            writer.call_args.kwargs["ffmpeg_params"],
            ["-filter:a", "loudnorm=I=-14:TP=-1:LRA=11"],
        )

    def test_generate_video_skips_loudnorm_when_disabled(self):
        """loudnorm 显式关闭时既不调用 AudioNormalize，也不附加滤镜。"""
        params = VideoParams(
            video_subject="test",
            subtitle_enabled=False,
            bgm_type="random",
            bgm_volume=0.0,
            audio_loudnorm=False,
        )
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch.object(vd.afx, "AudioNormalize", create=True) as audio_normalize,
        ):
            result, writer, voice_source = (
                self._run_generate_video_with_loudnorm(params, temp_dir)
            )

        self.assertTrue(result)
        audio_normalize.assert_not_called()
        self.assertEqual(len(voice_source.effects_applied), 1)
        self.assertNotIn("ffmpeg_params", writer.call_args.kwargs)


class TestAssSubtitleEngine(unittest.TestCase):
    """app.services.subtitle_engine.renderer 的 ASS 内容生成纯函数。"""

    def test_build_ass_content_contains_header_styles_and_dialogue(self):
        """ASS 内容必须包含画幅、样式段与 Dialogue 行。"""
        content = build_ass_content(
            [((1.0, 2.5), "hello world")], 1080, 1920, SubtitleStyleConfig()
        )
        self.assertIn("PlayResX: 1080", content)
        self.assertIn("PlayResY: 1920", content)
        self.assertIn("[V4+ Styles]", content)
        self.assertIn("[Events]", content)
        self.assertIn("Dialogue: 0,0:00:01.00,0:00:02.50,Default,", content)
        self.assertIn("hello world", content)

    def test_build_ass_content_builds_karaoke_from_word_timings(self):
        """开启逐词高亮且有词时间轴时，Dialogue 文本应生成 {\\k} 卡拉 OK。"""
        style = SubtitleStyleConfig(active_word_highlight=True)
        word_timings = [
            WordTiming(text="hello", start=1.0, end=1.4),
            WordTiming(text="world", start=1.4, end=2.0),
        ]
        content = build_ass_content(
            [((1.0, 2.5), "hello world")], 1080, 1920, style, word_timings
        )
        self.assertIn(r"{\k40}hello {\k60}world", content)

    def test_build_ass_content_escapes_braces_in_dialogue(self):
        """对话文本里的花括号必须转义，避免被 libass 当成 override tag。"""
        content = build_ass_content(
            [((1.0, 2.5), "use {keep}")], 1080, 1920, SubtitleStyleConfig()
        )
        self.assertIn(r"use \{keep\}", content)

    def test_build_ass_content_converts_newlines_to_ass_break(self):
        """多行文本应转成 ASS 的 \\N 换行标记。"""
        content = build_ass_content(
            [((1.0, 2.5), "first\nsecond")], 1080, 1920, SubtitleStyleConfig()
        )
        self.assertIn("first\\Nsecond", content)

    def test_build_ass_content_skips_invalid_items(self):
        """结束时间不晚于开始时间或空文本的字幕条目必须跳过。"""
        content = build_ass_content(
            [
                ((2.0, 1.0), "backwards"),
                ((3.0, 4.0), ""),
            ],
            1080,
            1920,
            SubtitleStyleConfig(),
        )
        self.assertNotIn("backwards", content)


def test_generate_ass_file_writes_sidecar(tmp_path):
    """SRT 应转换为同目录 .ass 侧车文件并返回 True。"""
    srt_path = tmp_path / "subtitle.srt"
    srt_path.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\nfirst phrase\n\n"
        "2\n00:00:03,000 --> 00:00:04,000\nsecond phrase\n\n",
        encoding="utf-8",
    )
    ass_path = tmp_path / "subtitle.ass"

    ok = generate_ass_file(
        str(srt_path), str(ass_path), 1080, 1920, SubtitleStyleConfig()
    )

    assert ok is True
    assert ass_path.is_file()
    content = ass_path.read_text(encoding="utf-8")
    assert "PlayResX: 1080" in content
    assert "Dialogue: 0,0:00:01.00,0:00:02.00,Default," in content
    assert "first phrase" in content


def _prepare_burn_files(tmp_path):
    input_video = tmp_path / "input.mp4"
    ass_path = tmp_path / "subtitle.ass"
    output_video = tmp_path / "burned.mp4"
    input_video.write_bytes(b"video")
    ass_path.write_text("[Script Info]\n", encoding="utf-8")
    return input_video, ass_path, output_video


def test_burn_ass_subtitles_via_ffmpeg_calls_with_ass_filter(tmp_path):
    """ffmpeg 必须收到 -vf ass=<path> 滤镜并成功返回 True。"""
    input_video, ass_path, output_video = _prepare_burn_files(tmp_path)
    fake_result = types.SimpleNamespace(returncode=0, stdout="", stderr="")

    with (
        patch.object(vd.utils, "get_ffmpeg_binary", return_value="/tmp/ffmpeg"),
        patch.object(vd.subprocess, "run", return_value=fake_result) as run,
    ):
        ok = vd._burn_ass_subtitles_via_ffmpeg(
            str(input_video), str(ass_path), str(output_video), threads=2
        )

    assert ok is True
    run.assert_called_once()
    command = run.call_args.args[0]
    vf_index = command.index("-vf")
    assert command[vf_index + 1].startswith("ass=")
    assert "subtitle.ass" in command[vf_index + 1]
    assert str(output_video) in command


def test_burn_ass_subtitles_via_ffmpeg_returns_false_on_failure(tmp_path):
    """ffmpeg 非零退出码必须返回 False 而不是抛异常。"""
    input_video, ass_path, output_video = _prepare_burn_files(tmp_path)
    fake_result = types.SimpleNamespace(
        returncode=1, stdout="", stderr="filter not found"
    )

    with (
        patch.object(vd.utils, "get_ffmpeg_binary", return_value="/tmp/ffmpeg"),
        patch.object(vd.subprocess, "run", return_value=fake_result),
    ):
        ok = vd._burn_ass_subtitles_via_ffmpeg(
            str(input_video), str(ass_path), str(output_video), threads=2
        )

    assert ok is False


def test_burn_ass_subtitles_via_ffmpeg_returns_false_on_exception(tmp_path):
    """subprocess 抛异常（如 ffmpeg 缺失）时优雅降级为 False。"""
    input_video, ass_path, output_video = _prepare_burn_files(tmp_path)

    with (
        patch.object(vd.utils, "get_ffmpeg_binary", return_value="/tmp/ffmpeg"),
        patch.object(vd.subprocess, "run", side_effect=RuntimeError("no ffmpeg")),
    ):
        ok = vd._burn_ass_subtitles_via_ffmpeg(
            str(input_video), str(ass_path), str(output_video), threads=2
        )

    assert ok is False


def test_burn_ass_subtitles_via_ffmpeg_returns_false_when_files_missing(tmp_path):
    """输入文件或 ASS 文件缺失时直接返回 False，不调用 ffmpeg。"""
    with patch.object(vd.subprocess, "run") as run:
        ok = vd._burn_ass_subtitles_via_ffmpeg(
            str(tmp_path / "missing.mp4"),
            str(tmp_path / "missing.ass"),
            str(tmp_path / "out.mp4"),
        )

    assert ok is False
    run.assert_not_called()