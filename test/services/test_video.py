import json
import os
import shutil
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from moviepy import (
    ImageClip,
    VideoFileClip,
)

# add project root to python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.config import config
from app.models.schema import MaterialInfo
from app.services import video as vd
from app.utils import utils

resources_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources")


class _FakeMoviePyClip:
    """为最终混音单测提供最小 MoviePy 接口，避免 CI 真实编码大型视频。"""

    def __init__(self, *, duration=5, fps=44100, w=100, h=50):
        self.duration = duration
        self.fps = fps
        self.w = w
        self.h = h
        self.close_calls = 0
        self.with_audio_result = self

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

    def close(self):
        self.close_calls += 1

    def with_effects(self, _effects):
        return self

    def with_audio(self, _audio):
        return self.with_audio_result

    def with_start(self, _start):
        return self

    def with_end(self, _end):
        return self

    def with_duration(self, _duration):
        return self

    def with_position(self, _position):
        return self


class TestVideoService(unittest.TestCase):
    def setUp(self):
        self.original_app_config = dict(config.app)
        self.test_img_path = os.path.join(resources_dir, "1.png")
        vd._runtime_disabled_video_codecs.clear()
        vd._ffmpeg_encoder_exists.cache_clear()

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)
        vd._runtime_disabled_video_codecs.clear()
        vd._ffmpeg_encoder_exists.cache_clear()

    def test_delete_files_deduplicates_paths_and_ignores_missing_files(self):
        """
        循环片段会让同一路径在拼接列表中重复出现，清理时每个路径只能删除一次。

        已不存在的文件属于幂等清理的正常状态，不应再产生误导用户的失败日志。
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            existing_file = os.path.join(temp_dir, "temp-clip-1.mp4")
            missing_file = os.path.join(temp_dir, "already-removed.mp4")
            Path(existing_file).write_bytes(b"temporary clip")

            original_remove = os.remove
            with (
                patch.object(vd.os, "remove", wraps=original_remove) as remove,
                patch.object(vd.logger, "warning") as warning,
            ):
                vd.delete_files(
                    [
                        existing_file,
                        existing_file,
                        missing_file,
                        missing_file,
                    ]
                )

        self.assertEqual(
            [item.args[0] for item in remove.call_args_list],
            [existing_file, missing_file],
        )
        warning.assert_not_called()

    def test_delete_files_logs_actionable_os_errors(self):
        """权限等真实清理失败必须保留路径和系统错误，方便定位残留文件。"""
        with (
            patch.object(
                vd.os,
                "remove",
                side_effect=PermissionError("permission denied"),
            ),
            patch.object(vd.logger, "warning") as warning,
        ):
            vd.delete_files(["protected-temp-clip.mp4"])

        warning.assert_called_once()
        message = warning.call_args.args[0]
        self.assertIn("protected-temp-clip.mp4", message)
        self.assertIn("permission denied", message)

    def test_generate_video_reports_successful_bgm_mix_and_closes_sources(self):
        """BGM 混合成功后应返回 True，并释放所有原始文件 reader。"""
        params = vd.VideoParams(
            video_subject="test",
            subtitle_enabled=False,
            bgm_type="sonilo",
        )
        source_video = _FakeMoviePyClip()
        voice_source = _FakeMoviePyClip()
        bgm_source = _FakeMoviePyClip()
        mixed_audio = _FakeMoviePyClip(fps=48000)
        final_video = _FakeMoviePyClip()
        source_video.with_audio_result = final_video

        def fake_writer(clip, output_file, **kwargs):
            # 模拟编码成功：写出临时文件，供原子改名流程消费。
            Path(output_file).write_bytes(b"video-data")

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = os.path.join(temp_dir, "final.mp4")
            with (
                patch.object(vd, "_open_video_clip_quietly", return_value=source_video),
                patch.object(vd, "AudioFileClip", side_effect=[voice_source, bgm_source]),
                patch.object(vd, "CompositeAudioClip", return_value=mixed_audio),
                patch.object(
                    vd, "_write_videofile_with_codec_fallback", side_effect=fake_writer
                ) as writer,
                patch.object(vd, "_get_configured_video_codec", return_value="libx264"),
            ):
                result = vd.generate_video(
                    video_path="combined.mp4",
                    audio_path="voice.mp3",
                    subtitle_path="",
                    output_file=output_file,
                    params=params,
                    bgm_file_override="sonilo.m4a",
                )

        self.assertTrue(result)
        writer.assert_called_once()
        self.assertEqual(writer.call_args.kwargs["audio_fps"], 48000)
        self.assertEqual(writer.call_args.kwargs["output_file"], output_file + ".tmp.mp4")
        self.assertEqual(source_video.close_calls, 1)
        self.assertEqual(voice_source.close_calls, 1)
        self.assertEqual(bgm_source.close_calls, 1)
        self.assertEqual(final_video.close_calls, 1)

    def test_generate_video_keeps_output_and_reports_failed_bgm_mix(self):
        """BGM 打开失败时仍应只写一次无 BGM 视频，并返回 False。"""
        params = vd.VideoParams(
            video_subject="test",
            subtitle_enabled=False,
            bgm_type="sonilo",
        )
        source_video = _FakeMoviePyClip()
        voice_source = _FakeMoviePyClip()
        final_video = _FakeMoviePyClip()
        source_video.with_audio_result = final_video

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = os.path.join(temp_dir, "final.mp4")
            with (
                patch.object(vd, "_open_video_clip_quietly", return_value=source_video),
                patch.object(
                    vd,
                    "AudioFileClip",
                    side_effect=[voice_source, RuntimeError("invalid BGM")],
                ),
                patch.object(vd, "CompositeAudioClip") as composite_audio,
                patch.object(
                    vd,
                    "_write_videofile_with_codec_fallback",
                    side_effect=lambda clip, output_file, **kwargs: Path(
                        output_file
                    ).write_bytes(b"video-data"),
                ) as writer,
                patch.object(vd, "_get_configured_video_codec", return_value="libx264"),
                patch.object(vd.logger, "exception") as log_exception,
            ):
                result = vd.generate_video(
                    video_path="combined.mp4",
                    audio_path="voice.mp3",
                    subtitle_path="",
                    output_file=output_file,
                    params=params,
                    bgm_file_override="broken.m4a",
                )

        self.assertFalse(result)
        writer.assert_called_once()
        composite_audio.assert_not_called()
        log_exception.assert_called_once()
        self.assertEqual(source_video.close_calls, 1)
        self.assertEqual(voice_source.close_calls, 1)
        self.assertEqual(final_video.close_calls, 1)

    def test_generate_video_skips_every_bgm_source_when_volume_is_zero(self):
        """0 音量必须在解析文件前统一短路当前来源和未来提供商。"""
        test_cases = [
            ("random", None),
            ("custom", None),
            ("sonilo", "sonilo.m4a"),
            ("future_provider", "future-provider.wav"),
        ]
        for bgm_type, bgm_override in test_cases:
            with self.subTest(bgm_type=bgm_type):
                params = vd.VideoParams(
                    video_subject="test",
                    subtitle_enabled=False,
                    bgm_type=bgm_type,
                    bgm_file="missing-background.mp3",
                    bgm_volume=0.0,
                )
                source_video = _FakeMoviePyClip()
                voice_source = _FakeMoviePyClip()
                final_video = _FakeMoviePyClip()
                source_video.with_audio_result = final_video

                with (
                    patch.object(
                        vd,
                        "_open_video_clip_quietly",
                        return_value=source_video,
                    ),
                    patch.object(
                        vd, "AudioFileClip", return_value=voice_source
                    ) as audio_file_clip,
                    patch.object(vd, "get_bgm_file") as get_bgm_file,
                    patch.object(vd, "CompositeAudioClip") as composite_audio,
                    patch.object(
                    vd,
                    "_write_videofile_with_codec_fallback",
                    side_effect=lambda clip, output_file, **kwargs: Path(
                        output_file
                    ).write_bytes(b"video-data"),
                ) as writer,
                    patch.object(
                        vd, "_get_configured_video_codec", return_value="libx264"
                    ),
                ):
                    result = vd.generate_video(
                        video_path="combined.mp4",
                        audio_path="voice.mp3",
                        subtitle_path="",
                        output_file=os.path.join(tempfile.gettempdir(), "final.mp4"),
                        params=params,
                        bgm_file_override=bgm_override,
                    )

                self.assertTrue(result)
                audio_file_clip.assert_called_once_with("voice.mp3")
                get_bgm_file.assert_not_called()
                composite_audio.assert_not_called()
                writer.assert_called_once()
                self.assertEqual(source_video.close_calls, 1)
                self.assertEqual(voice_source.close_calls, 1)
                self.assertEqual(final_video.close_calls, 1)

    def test_generate_video_chooses_looping_by_bgm_file_source(self):
        """默认曲库需要循环，任务层提供的时长适配文件不应依赖提供商名称。"""
        test_cases = [
            ("random", None, True),
            ("custom", None, True),
            ("sonilo", "sonilo.m4a", False),
            ("future_provider", "future-provider.wav", False),
        ]
        for bgm_type, bgm_override, should_loop in test_cases:
            with self.subTest(bgm_type=bgm_type, bgm_override=bgm_override):
                params = vd.VideoParams(
                    video_subject="test",
                    subtitle_enabled=False,
                    bgm_type=bgm_type,
                    bgm_file="library.mp3",
                    bgm_volume=0.2,
                )
                source_video = _FakeMoviePyClip()
                voice_source = _FakeMoviePyClip()
                bgm_source = _FakeMoviePyClip()
                mixed_audio = _FakeMoviePyClip()
                final_video = _FakeMoviePyClip()
                source_video.with_audio_result = final_video

                with (
                    patch.object(
                        vd,
                        "_open_video_clip_quietly",
                        return_value=source_video,
                    ),
                    patch.object(
                        vd,
                        "AudioFileClip",
                        side_effect=[voice_source, bgm_source],
                    ),
                    patch.object(vd, "get_bgm_file", return_value="library.mp3"),
                    patch.object(vd, "CompositeAudioClip", return_value=mixed_audio),
                    patch.object(vd.afx, "AudioLoop") as audio_loop,
                    patch.object(
                    vd,
                    "_write_videofile_with_codec_fallback",
                    side_effect=lambda clip, output_file, **kwargs: Path(
                        output_file
                    ).write_bytes(b"video-data"),
                ),
                    patch.object(
                        vd, "_get_configured_video_codec", return_value="libx264"
                    ),
                ):
                    result = vd.generate_video(
                        video_path="combined.mp4",
                        audio_path="voice.mp3",
                        subtitle_path="",
                        output_file=os.path.join(tempfile.gettempdir(), "final.mp4"),
                        params=params,
                        bgm_file_override=bgm_override,
                    )

                self.assertTrue(result)
                if should_loop:
                    audio_loop.assert_called_once_with(duration=source_video.duration)
                else:
                    audio_loop.assert_not_called()

    def test_generate_video_applies_ducking_from_parsed_subtitles(self):
        """
        Ducking 时间轴必须来自带 make_textclip 的 SubtitlesClip 实例。

        旧实现为读取时间轴新建了一个不带 make_textclip 的 SubtitlesClip，
        moviepy 会以 “Argument font is required” 拒绝解析，导致 ducking
        时间轴恒为空、MultiplyVolume 永远不会按字幕区间压低 BGM。
        """

        class _FakeSubtitlesClip:
            def __init__(self, _subtitles=None, **_kwargs):
                self.subtitles = [
                    ((1.0, 2.0), "first phrase"),
                    ((3.0, 4.0), "second phrase"),
                ]
                self.close_calls = 0

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                self.close()

            def close(self):
                self.close_calls += 1

        params = vd.VideoParams(
            video_subject="test",
            subtitle_enabled=True,
            audio_ducking_enabled=True,
            audio_ducking_intensity=0.5,
            bgm_type="random",
            bgm_volume=0.2,
            subtitle_position="bottom",
            font_name="Montserrat-Bold.ttf",
        )
        source_video = _FakeMoviePyClip()
        voice_source = _FakeMoviePyClip()
        bgm_source = _FakeMoviePyClip()
        mixed_audio = _FakeMoviePyClip()
        final_video = _FakeMoviePyClip()
        source_video.with_audio_result = final_video
        fake_subtitles = _FakeSubtitlesClip()
        fake_text_clip = _FakeMoviePyClip()

        with tempfile.TemporaryDirectory() as temp_dir:
            subtitle_path = os.path.join(temp_dir, "subtitle.srt")
            Path(subtitle_path).write_text(
                "1\n00:00:01,000 --> 00:00:02,000\nfirst phrase\n\n"
                "2\n00:00:03,000 --> 00:00:04,000\nsecond phrase\n\n",
                encoding="utf-8",
            )
            with (
                patch.object(vd, "_open_video_clip_quietly", return_value=source_video),
                patch.object(
                    vd, "AudioFileClip", side_effect=[voice_source, bgm_source]
                ),
                patch.object(
                    vd, "SubtitlesClip", return_value=fake_subtitles
                ) as subtitles_clip,
                patch.object(vd, "TextClip", return_value=fake_text_clip),
                patch.object(vd, "CompositeVideoClip", return_value=source_video),
                patch.object(vd, "CompositeAudioClip", return_value=mixed_audio),
                patch.object(vd, "get_bgm_file", return_value="library.mp3"),
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
                    subtitle_path=subtitle_path,
                    output_file=os.path.join(tempfile.gettempdir(), "final.mp4"),
                    params=params,
                )

        self.assertTrue(result)
        subtitles_clip.assert_called_once()
        # 只有带 make_textclip 的那次构造；不再为 ducking 新建第二个实例。
        self.assertIn("make_textclip", subtitles_clip.call_args.kwargs)
        writer.assert_called_once()

    def test_preprocess_video(self):
        if not os.path.exists(self.test_img_path):
            self.fail(f"test image not found: {self.test_img_path}")

        local_videos_dir = utils.storage_dir("local_videos", create=True)
        safe_img_path = os.path.join(local_videos_dir, "test-preprocess-1.png")
        shutil.copy2(self.test_img_path, safe_img_path)

        # test preprocess_video function
        m = MaterialInfo()
        m.url = os.path.basename(safe_img_path)
        m.provider = "local"
        print(m)

        try:
            materials = vd.preprocess_video([m], clip_duration=4)
            print(materials)

            # verify result
            self.assertIsNotNone(materials)
            self.assertEqual(len(materials), 1)
            self.assertTrue(materials[0].url.endswith(".mp4"))

            # moviepy get video info
            clip = VideoFileClip(materials[0].url)
            try:
                print(clip)
            finally:
                clip.close()

            # clean generated test video file
            if os.path.exists(materials[0].url):
                os.remove(materials[0].url)
        finally:
            if os.path.exists(safe_img_path):
                os.remove(safe_img_path)

    def test_preprocess_video_rejects_material_outside_local_videos(self):
        """
        local 素材路径来自 API 参数，不能允许任意绝对路径进入 MoviePy。
        这里验证非 local_videos 白名单目录内的路径会被跳过，避免任意文件读取。
        """
        m = MaterialInfo(provider="local", url=self.test_img_path)

        materials = vd.preprocess_video([m], clip_duration=4)

        self.assertEqual(materials, [])

    def test_get_bgm_file_accepts_song_directory_filename(self):
        """
        BGM 列表接口现在只暴露文件名；生成视频时应能把文件名安全解析回
        resource/songs 白名单目录，保持正常使用路径可用。
        """
        song_dir = utils.song_dir()
        bgm_path = os.path.join(song_dir, "test-safe-bgm.mp3")
        Path(bgm_path).write_bytes(b"fake-mp3")

        try:
            self.assertEqual(vd.get_bgm_file(bgm_file="test-safe-bgm.mp3"), bgm_path)
        finally:
            if os.path.exists(bgm_path):
                os.remove(bgm_path)

    def test_get_bgm_file_accepts_project_relative_song_path(self):
        """
        用户在 WebUI 中可能直接填写 ./resource/songs/xxx.mp3。该路径虽然是
        项目根目录相对路径，但实际文件仍在 resource/songs 白名单目录内，
        应该被接受，避免自定义背景音乐被误判为不存在。
        """
        song_dir = utils.song_dir()
        bgm_path = os.path.join(song_dir, "test-relative-bgm.mp3")
        Path(bgm_path).write_bytes(b"fake-mp3")

        try:
            self.assertEqual(
                vd.get_bgm_file(bgm_file="./resource/songs/test-relative-bgm.mp3"),
                bgm_path,
            )
        finally:
            if os.path.exists(bgm_path):
                os.remove(bgm_path)

    def test_get_bgm_file_rejects_path_outside_song_directory(self):
        """
        用户传入的 bgm_file 不能直接作为本地路径打开，否则可能读取系统文件。
        即使外部文件存在，也必须因为不在 songs 目录内被拒绝。
        """
        with tempfile.NamedTemporaryFile(suffix=".mp3") as temp_bgm:
            self.assertEqual(vd.get_bgm_file(bgm_file=temp_bgm.name), "")

    def test_get_ffmpeg_binary_uses_configured_env_path(self):
        """配置中显式指定 ffmpeg 时，应优先使用该路径。"""
        with patch.dict(
            os.environ, {"IMAGEIO_FFMPEG_EXE": "/tmp/custom-ffmpeg"}, clear=True
        ):
            self.assertEqual(utils.get_ffmpeg_binary(), "/tmp/custom-ffmpeg")

    def test_get_ffmpeg_binary_falls_back_to_imageio_ffmpeg(self):
        """
        Windows 便携包里系统 PATH 可能没有 ffmpeg，但 moviepy 依赖的
        imageio-ffmpeg 通常会提供可执行文件。这里验证该兜底路径可用。
        """
        fake_imageio_ffmpeg = types.SimpleNamespace(
            get_ffmpeg_exe=lambda: "/tmp/bundled-ffmpeg"
        )

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(utils.shutil, "which", return_value=None),
            patch.dict(sys.modules, {"imageio_ffmpeg": fake_imageio_ffmpeg}),
        ):
            self.assertEqual(utils.get_ffmpeg_binary(), "/tmp/bundled-ffmpeg")

    def test_get_effective_video_codec_falls_back_when_encoder_missing(self):
        """
        用户选择的硬件编码器必须先经过 FFmpeg encoder 列表检测。检测不到
        时直接回退 libx264，避免生成任务在写文件阶段才失败。
        """
        config.app["video_codec"] = "h264_nvenc"

        with patch.object(vd, "_ffmpeg_encoder_exists", return_value=False):
            self.assertEqual(vd._get_effective_video_codec(), "libx264")

    def test_get_configured_video_codec_uses_stable_default_when_unset(self):
        """
        WebUI 的“默认”模式不会持久化 video_codec。后端必须在配置缺失时继续
        明确返回 libx264，不能把空值直接交给 MoviePy 或 FFmpeg 自行决定。
        """
        config.app.pop("video_codec", None)

        self.assertEqual(vd._get_configured_video_codec(), "libx264")

    def test_get_configured_video_codec_preserves_explicit_libx264(self):
        """
        用户明确选择 libx264 时需要保持固定选择。它与“跟随项目默认策略”当前
        结果相同，但配置语义不同，未来调整默认值时不能影响显式选择。
        """
        config.app["video_codec"] = "libx264"

        self.assertEqual(vd._get_configured_video_codec(), "libx264")

    def test_ffmpeg_encoder_exists_falls_back_when_probe_fails(self):
        """
        Windows 上用户配置的 ffmpeg 可能因为路径损坏、权限或杀软拦截而无法
        正常执行。encoder 探测失败时必须返回 False，让上层稳定回退 libx264。
        """
        with patch.object(
            vd.subprocess,
            "run",
            side_effect=OSError("permission denied"),
        ):
            self.assertFalse(
                vd._ffmpeg_encoder_exists("C:/ffmpeg/bin/ffmpeg.exe", "h264_nvenc")
            )

    def test_write_videofile_falls_back_after_runtime_encoder_failure(self):
        """
        FFmpeg 声明支持某个硬件编码器，不代表当前显卡或驱动一定可用。
        首次实际编码失败后，应立即用 libx264 重试，并在本进程禁用该编码器。
        """

        class _FakeClip:
            def __init__(self):
                self.codecs = []

            def write_videofile(self, output_file, codec, **kwargs):
                self.codecs.append(codec)
                if codec == "h264_nvenc":
                    raise RuntimeError("nvenc device not available")

        fake_clip = _FakeClip()

        with patch.object(vd, "_ffmpeg_encoder_exists", return_value=True):
            used_codec = vd._write_videofile_with_codec_fallback(
                fake_clip,
                "/tmp/fake.mp4",
                codec="h264_nvenc",
                logger=None,
                fps=30,
            )

        self.assertEqual(used_codec, "libx264")
        self.assertEqual(fake_clip.codecs, ["h264_nvenc", "libx264"])
        self.assertIn("h264_nvenc", vd._runtime_disabled_video_codecs)

    def test_write_videofile_does_not_disable_codec_when_fallback_also_fails(self):
        """
        如果 libx264 兜底也失败，失败原因更可能是输出路径、权限、文件占用等
        通用问题，不能误判为硬件编码器不可用。
        """

        class _FakeClip:
            def write_videofile(self, output_file, codec, **kwargs):
                raise RuntimeError(f"{codec} cannot write output")

        with patch.object(vd, "_ffmpeg_encoder_exists", return_value=True):
            with self.assertRaises(RuntimeError):
                vd._write_videofile_with_codec_fallback(
                    _FakeClip(),
                    "/tmp/fake.mp4",
                    codec="h264_nvenc",
                    logger=None,
                    fps=30,
                )

        self.assertNotIn("h264_nvenc", vd._runtime_disabled_video_codecs)

    def test_format_ffmpeg_concat_path_normalizes_windows_path(self):
        """
        concat demuxer 的文件列表对 Windows 反斜杠较敏感，写入 list 前统一
        转成正斜杠，并继续保留单引号转义。
        """
        with patch.object(
            vd.os.path,
            "abspath",
            return_value=r"C:\Users\Test User's Videos\clip.mp4",
        ):
            self.assertEqual(
                vd._format_ffmpeg_concat_path(r"C:\Users\Test User's Videos\clip.mp4"),
                "C:/Users/Test User'\\''s Videos/clip.mp4",
            )

    def test_concat_video_clips_falls_back_after_runtime_encoder_failure(self):
        """
        最终 ffmpeg concat 阶段也要具备同样的回退能力。这里用 mock 模拟
        流复制失败且 h264_nvenc 编码失败，确认会自动再用 libx264 执行一次。
        """
        config.app["video_codec"] = "h264_nvenc"

        def fake_run(command, capture_output, text, check, timeout=None):
            codec_index = command.index("-c:v") + 1
            codec = command[codec_index]
            if codec == "copy":
                return types.SimpleNamespace(
                    returncode=1,
                    stdout="",
                    stderr="concat inputs have differing resolution",
                )
            if codec == "h264_nvenc":
                return types.SimpleNamespace(
                    returncode=1,
                    stdout="",
                    stderr="nvenc device not available",
                )
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as temp_dir:
            clip_file = os.path.join(temp_dir, "clip.mp4")
            output_file = os.path.join(temp_dir, "combined.mp4")
            Path(clip_file).write_bytes(b"fake")

            with patch.object(vd, "_ffmpeg_encoder_exists", return_value=True):
                with patch.object(vd.subprocess, "run", side_effect=fake_run) as run:
                    vd.concat_video_clips_with_ffmpeg(
                        clip_files=[clip_file],
                        output_file=output_file,
                        threads=1,
                        output_dir=temp_dir,
                    )

        used_codecs = [
            call.args[0][call.args[0].index("-c:v") + 1] for call in run.call_args_list
        ]
        self.assertEqual(used_codecs, ["copy", "h264_nvenc", "libx264"])
        self.assertIn("h264_nvenc", vd._runtime_disabled_video_codecs)

    def test_concat_video_clips_does_not_disable_codec_when_fallback_also_fails(self):
        """
        concat 阶段如果 libx264 也失败，说明可能是输入 list、路径或输出权限
        问题，不能把硬件编码器加入运行时禁用列表。
        """
        config.app["video_codec"] = "h264_nvenc"

        def fake_run(command, capture_output, text, check, timeout=None):
            codec_index = command.index("-c:v") + 1
            codec = command[codec_index]
            return types.SimpleNamespace(
                returncode=1,
                stdout="",
                stderr=f"{codec} cannot write output",
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            clip_file = os.path.join(temp_dir, "clip.mp4")
            output_file = os.path.join(temp_dir, "combined.mp4")
            Path(clip_file).write_bytes(b"fake")

            with patch.object(vd, "_ffmpeg_encoder_exists", return_value=True):
                with patch.object(vd.subprocess, "run", side_effect=fake_run):
                    with self.assertRaises(RuntimeError):
                        vd.concat_video_clips_with_ffmpeg(
                            clip_files=[clip_file],
                            output_file=output_file,
                            threads=1,
                            output_dir=temp_dir,
                        )

        self.assertNotIn("h264_nvenc", vd._runtime_disabled_video_codecs)

    def test_open_video_clip_quietly_suppresses_moviepy_stdout(self):
        """
        MoviePy 2.1.x 的 FFMPEG_VideoReader 会直接向 stdout 打印 metadata
        和 ffmpeg 命令。项目服务层应屏蔽这类依赖库噪声，避免用户把
        `audio_found: False` 误判为最终视频没有音频。
        """
        # 测试只关心服务层是否屏蔽 MoviePy 的读取噪声，不应长期保存一份由 PNG
        # 编码而来的二进制 MP4 fixture。运行时生成短视频既能保持测试独立，也能
        # 避免 fixture 因不同编码参数产生帧间闪烁后被误用于视觉效果验证。
        image_path = os.path.join(resources_dir, "1.png")
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = os.path.join(temp_dir, "image-fixture.mp4")
            source_clip = ImageClip(image_path).with_duration(0.2)
            try:
                source_clip.write_videofile(
                    video_path,
                    codec="libx264",
                    fps=5,
                    audio=False,
                    logger=None,
                )
            finally:
                source_clip.close()

            stdout = StringIO()
            with redirect_stdout(stdout):
                clip = vd._open_video_clip_quietly(video_path)

            try:
                self.assertEqual(stdout.getvalue(), "")
                self.assertIsNone(clip.audio)
                self.assertGreater(clip.duration, 0)
            finally:
                vd.close_clip(clip)

    def test_combine_videos_closes_audio_clip_when_duration_read_fails(self):
        """
        `combine_videos()` 只需要读取旁白音频时长。即使读取 duration
        时发生异常，也必须关闭 AudioFileClip，避免文件句柄泄漏。
        """

        class _FakeAudioReader:
            def __init__(self):
                self.closed = False

            def close(self):
                self.closed = True

        class _BrokenAudioClip:
            def __init__(self):
                self.reader = _FakeAudioReader()

            @property
            def duration(self):
                raise RuntimeError("failed to read duration")

        fake_audio_clip = _BrokenAudioClip()

        with patch.object(vd, "AudioFileClip", return_value=fake_audio_clip):
            with self.assertRaises(RuntimeError):
                vd.combine_videos(
                    combined_video_path="/tmp/unused-combined.mp4",
                    video_paths=[],
                    audio_file="/tmp/unused-audio.mp3",
                )

        self.assertTrue(fake_audio_clip.reader.closed)

    def test_combine_videos_handles_none_transition_mode(self):
        """
        Ensure `combine_videos` safely handles
        `video_transition_mode=None`.
        """

        class _FakeAudioClip:
            @property
            def duration(self):
                return 10.0

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as temp_dir:
            combined_video_path = os.path.join(temp_dir, "combined.mp4")
            audio_file = os.path.join(temp_dir, "audio.mp3")

            with patch.object(vd, "AudioFileClip", return_value=_FakeAudioClip()):
                # 空素材列表时没有任何片段可合并，按失败处理抛出明确错误，
                # 而不是返回一个并不存在的“合成产物”路径。
                with self.assertRaisesRegex(
                    RuntimeError, "no clips available for merging"
                ):
                    vd.combine_videos(
                        combined_video_path=combined_video_path,
                        video_paths=[],
                        audio_file=audio_file,
                        video_transition_mode=None,
                    )

    def test_concat_timeout_raises_clear_error(self):
        """ffmpeg concat 卡死时必须超时并抛出明确错误，而不是无限阻塞。"""

        def fake_run(command, capture_output, text, check, timeout=None):
            raise vd.subprocess.TimeoutExpired(cmd=command, timeout=timeout)

        with tempfile.TemporaryDirectory() as temp_dir:
            clip_file = os.path.join(temp_dir, "clip.mp4")
            output_file = os.path.join(temp_dir, "combined.mp4")
            Path(clip_file).write_bytes(b"fake")

            with patch.object(vd.subprocess, "run", side_effect=fake_run):
                with self.assertRaisesRegex(
                    RuntimeError, "ffmpeg concat timed out"
                ):
                    vd.concat_video_clips_with_ffmpeg(
                        clip_files=[clip_file],
                        output_file=output_file,
                        threads=1,
                        output_dir=temp_dir,
                    )

    def test_combine_videos_cleans_temp_clips_on_merge_failure(self):
        """合并阶段抛异常时必须清理已写出的 temp-clip 文件并继续上抛。"""

        class _FakeAudioClip:
            duration = 10.0

            def close(self):
                pass

        class _FakeVideoClip:
            duration = 5.0
            size = (1080, 1920)
            w = 1080
            h = 1920

            def subclipped(self, start_time, end_time):
                return _FakeVideoClip()

            def close(self):
                pass

        deleted = []

        with tempfile.TemporaryDirectory() as temp_dir:
            combined_video_path = os.path.join(temp_dir, "combined.mp4")
            with (
                patch.object(vd, "AudioFileClip", return_value=_FakeAudioClip()),
                patch.object(
                    vd,
                    "_open_video_clip_quietly",
                    return_value=_FakeVideoClip(),
                ),
                patch.object(
                    vd,
                    "_write_videofile_with_codec_fallback",
                    side_effect=lambda clip, output_file, **kwargs: Path(
                        output_file
                    ).write_bytes(b"video-data"),
                ),
                patch.object(
                    vd,
                    "concat_video_clips_with_ffmpeg",
                    side_effect=RuntimeError("merge exploded"),
                ),
                patch.object(vd, "delete_files", side_effect=lambda files: deleted.extend(files)),
            ):
                with self.assertRaisesRegex(RuntimeError, "merge exploded"):
                    vd.combine_videos(
                        combined_video_path=combined_video_path,
                        video_paths=["clip.mp4"],
                        audio_file="audio.mp3",
                        video_transition_mode=vd.VideoTransitionMode.none,
                    )

        self.assertTrue(deleted)  # 失败路径也清理了临时片段
        self.assertTrue(
            all("temp-clip" in name for name in deleted),
            f"unexpected cleanup targets: {deleted}",
        )

    def _capture_source_ranges_for_clip_speed(
        self,
        *,
        source_duration,
        audio_duration,
        clip_speed,
        max_clip_duration=3,
    ):
        """使用轻量假视频记录 combine_videos 实际读取的源时间范围。"""

        source_ranges = []
        written_durations = []

        class _FakeAudioClip:
            duration = audio_duration

            def close(self):
                pass

        class _FakeVideoClip:
            def __init__(self, duration, records_source_range=False):
                self.duration = duration
                self.size = (1080, 1920)
                self.w = 1080
                self.h = 1920
                self.records_source_range = records_source_range

            def subclipped(self, start_time, end_time):
                # 只记录直接从源文件读取的范围。变速后的安全裁剪也会调用
                # subclipped，但它不代表新的源时间段，不能混入断层判断。
                if self.records_source_range:
                    source_ranges.append((start_time, end_time))
                return _FakeVideoClip(end_time - start_time)

            def with_speed_scaled(self, factor):
                return _FakeVideoClip(self.duration / factor)

            def close(self):
                pass

        def _open_fake_video_clip(_video_path):
            return _FakeVideoClip(source_duration, records_source_range=True)

        def _capture_written_clip(clip, *_args, **_kwargs):
            written_durations.append(clip.duration)

        with tempfile.TemporaryDirectory() as temp_dir:
            combined_video_path = os.path.join(temp_dir, "combined.mp4")
            with (
                patch.object(vd, "AudioFileClip", return_value=_FakeAudioClip()),
                patch.object(
                    vd,
                    "_open_video_clip_quietly",
                    side_effect=_open_fake_video_clip,
                ),
                patch.object(
                    vd,
                    "_write_videofile_with_codec_fallback",
                    side_effect=_capture_written_clip,
                ),
                # random 模式默认会打乱同一源视频的切片。这里保持生成顺序，
                # 才能精确验证相邻源时间段是否连续。
                patch.object(
                    vd,
                    "_prioritize_unique_source_clips",
                    side_effect=lambda subclipped_items, concat_mode: subclipped_items,
                ),
                patch.object(vd, "concat_video_clips_with_ffmpeg"),
                patch.object(vd, "delete_files"),
            ):
                vd.combine_videos(
                    combined_video_path=combined_video_path,
                    video_paths=["clip.mp4"],
                    audio_file="audio.mp3",
                    video_concat_mode=vd.VideoConcatMode.random,
                    max_clip_duration=max_clip_duration,
                    clip_speed=clip_speed,
                )

        return source_ranges, written_durations

    def test_combine_videos_slow_speed_keeps_source_timeline_continuous(self):
        """0.5 倍慢放应连续读取 1.5 秒源片段，不能跳过中间画面。"""

        source_ranges, written_durations = self._capture_source_ranges_for_clip_speed(
            source_duration=4.0,
            audio_duration=5.9,
            clip_speed=0.5,
        )

        self.assertEqual(source_ranges, [(0, 1.5), (1.5, 3.0)])
        self.assertEqual(written_durations, [3.0, 3.0])

    def test_combine_videos_fast_speed_reads_enough_source_content(self):
        """2 倍快放应读取 6 秒源画面，使最终片段仍保持 3 秒。"""

        source_ranges, written_durations = self._capture_source_ranges_for_clip_speed(
            source_duration=8.0,
            audio_duration=2.9,
            clip_speed=2.0,
        )

        self.assertEqual(source_ranges, [(0, 6.0)])
        self.assertEqual(written_durations, [3.0])

    def test_combine_videos_keeps_small_duration_safety_margin(self):
        """
        音频和素材累计时长刚好相等时，仍应继续追加一个短片段作为安全余量。

        FFmpeg 按帧率拼接后可能让最终视频比理论时长短几十毫秒。如果这里
        在 10.0s == 10.0s 时立即停止，成片末尾就可能出现音频还在播放但
        视频素材已经结束的边界问题。
        """

        class _FakeAudioClip:
            duration = 10.0

            def close(self):
                pass

        class _FakeVideoClip:
            def __init__(self, duration):
                self.duration = duration
                self.size = (1080, 1920)
                self.w = 1080
                self.h = 1920

            def subclipped(self, start_time, end_time):
                return _FakeVideoClip(end_time - start_time)

        video_durations = {
            "clip-1.mp4": 3.0,
            "clip-2.mp4": 4.0,
            "clip-3.mp4": 3.0,
            "clip-4.mp4": 2.0,
        }

        def _open_fake_video_clip(video_path):
            return _FakeVideoClip(video_durations[video_path])

        with tempfile.TemporaryDirectory() as temp_dir:
            combined_video_path = os.path.join(temp_dir, "combined.mp4")

            with patch.object(vd, "AudioFileClip", return_value=_FakeAudioClip()):
                with patch.object(
                    vd, "_open_video_clip_quietly", side_effect=_open_fake_video_clip
                ):
                    with patch.object(
                        vd, "_write_videofile_with_codec_fallback"
                    ) as write_mock:
                        with patch.object(
                            vd, "concat_video_clips_with_ffmpeg"
                        ) as concat_mock:
                            with patch.object(vd, "delete_files"):
                                result = vd.combine_videos(
                                    combined_video_path=combined_video_path,
                                    video_paths=list(video_durations.keys()),
                                    audio_file=os.path.join(temp_dir, "audio.mp3"),
                                    video_aspect=vd.VideoAspect.portrait,
                                    video_concat_mode=vd.VideoConcatMode.sequential,
                                    video_transition_mode=None,
                                    max_clip_duration=10,
                                )

        self.assertEqual(result, combined_video_path)
        self.assertEqual(write_mock.call_count, 4)
        self.assertEqual(concat_mock.call_args.kwargs["max_duration"], 10.0)

    def test_cover_size_always_fully_covers_target(self):
        """scale-to-cover 尺寸必须让两条边都盖住目标画幅。"""
        self.assertEqual(vd._cover_size(1920, 1080, 1080, 1920), (3413, 1920))
        self.assertEqual(vd._cover_size(1080, 1920, 1920, 1080), (1920, 3413))
        self.assertEqual(vd._cover_size(1080, 1920, 1080, 1920), (1080, 1920))
        self.assertEqual(vd._cover_size(640, 360, 1920, 1080), (1920, 1080))

    def test_build_blurred_background_covers_darkens_and_centers(self):
        """模糊背景应覆盖目标画幅、压暗并居中，缩小时底图不放大回原尺寸。"""

        resized_sizes = []

        class _ResizableClip:
            def __init__(self, w, h):
                self.w = w
                self.h = h
                self.position = None
                self.effects = None

            def resized(self, new_size):
                resized_sizes.append(tuple(new_size))
                return _ResizableClip(*new_size)

            def with_effects(self, effects):
                self.effects = effects
                return self

            def with_position(self, position):
                self.position = position
                return self

        clip = _ResizableClip(1920, 1080)
        bg = vd._build_blurred_background(clip, 1080, 1920)

        self.assertEqual((bg.w, bg.h), (3413, 1920))
        self.assertEqual(
            resized_sizes,
            [(3413, 1920), (426, 240), (3413, 1920)],
        )
        self.assertEqual(bg.position, "center")
        self.assertIsNotNone(bg.effects)
        self.assertAlmostEqual(bg.effects[0].factor, 0.5)

    def test_combine_videos_falls_back_to_black_background_on_blur_failure(self):
        """模糊背景构建失败时必须回退到黑底，而不是中断合成。"""

        class _FakeAudioClip:
            duration = 5.0

            def close(self):
                pass

        class _FakeVideoClip:
            def __init__(self, duration=5.0):
                self.duration = duration
                self.size = (1920, 1080)
                self.w = 1920
                self.h = 1080

            def subclipped(self, start_time, end_time):
                return _FakeVideoClip(end_time - start_time)

            def resized(self, **_kwargs):
                return self

            def with_position(self, _position):
                return self

            def close(self):
                pass

        composed_composites = []

        class _FakeComposite:
            def __init__(self, clips, size=None):
                composed_composites.append(self)
                self.layers = list(clips)
                self.size = size
                self.duration = max(c.duration for c in clips)

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as temp_dir:
            combined_video_path = os.path.join(temp_dir, "combined.mp4")
            with (
                patch.object(vd, "AudioFileClip", return_value=_FakeAudioClip()),
                patch.object(
                    vd,
                    "_open_video_clip_quietly",
                    side_effect=lambda _p: _FakeVideoClip(),
                ),
                patch.object(
                    vd,
                    "_build_blurred_background",
                    side_effect=RuntimeError("blur failed"),
                ),
                patch.object(vd, "CompositeVideoClip", _FakeComposite),
                patch.object(vd, "_write_videofile_with_codec_fallback"),
                patch.object(vd, "concat_video_clips_with_ffmpeg"),
                patch.object(vd, "delete_files"),
            ):
                result = vd.combine_videos(
                    combined_video_path=combined_video_path,
                    video_paths=["clip.mp4"],
                    audio_file=os.path.join(temp_dir, "audio.mp3"),
                    video_aspect=vd.VideoAspect.portrait,
                    video_concat_mode=vd.VideoConcatMode.sequential,
                    video_transition_mode=None,
                    max_clip_duration=5,
                )

        self.assertEqual(result, combined_video_path)
        self.assertEqual(len(composed_composites), 1)
        # 回退背景必须是纯黑 ColorClip，而不是模糊层。
        self.assertIsInstance(composed_composites[0].layers[0], vd.ColorClip)
        # 合成帧必须显式钉在目标画幅，不能随 cover 背景膨胀。
        self.assertEqual(composed_composites[0].size, (1080, 1920))

    def test_combine_videos_uses_blurred_background_on_ratio_mismatch(self):
        """画幅不匹配时合成层应使用模糊背景，且帧尺寸钉在目标画幅。"""

        class _FakeAudioClip:
            duration = 5.0

            def close(self):
                pass

        class _FakeVideoClip:
            def __init__(self, duration=5.0):
                self.duration = duration
                self.size = (1920, 1080)
                self.w = 1920
                self.h = 1080

            def subclipped(self, start_time, end_time):
                return _FakeVideoClip(end_time - start_time)

            def resized(self, **_kwargs):
                return self

            def with_position(self, _position):
                return self

            def close(self):
                pass

        composed_composites = []

        class _FakeComposite:
            def __init__(self, clips, size=None):
                composed_composites.append(self)
                self.layers = list(clips)
                self.size = size
                self.duration = max(c.duration for c in clips)

            def close(self):
                pass

        class _FakeBlurredBackground:
            def __init__(self):
                self.duration = 5.0

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as temp_dir:
            combined_video_path = os.path.join(temp_dir, "combined.mp4")
            with (
                patch.object(vd, "AudioFileClip", return_value=_FakeAudioClip()),
                patch.object(
                    vd,
                    "_open_video_clip_quietly",
                    side_effect=lambda _p: _FakeVideoClip(),
                ),
                patch.object(
                    vd,
                    "_build_blurred_background",
                    return_value=_FakeBlurredBackground(),
                ),
                patch.object(vd, "CompositeVideoClip", _FakeComposite),
                patch.object(vd, "_write_videofile_with_codec_fallback"),
                patch.object(vd, "concat_video_clips_with_ffmpeg"),
                patch.object(vd, "delete_files"),
            ):
                result = vd.combine_videos(
                    combined_video_path=combined_video_path,
                    video_paths=["clip.mp4"],
                    audio_file=os.path.join(temp_dir, "audio.mp3"),
                    video_aspect=vd.VideoAspect.portrait,
                    video_concat_mode=vd.VideoConcatMode.sequential,
                    video_transition_mode=None,
                    max_clip_duration=5,
                )

        self.assertEqual(result, combined_video_path)
        self.assertEqual(len(composed_composites), 1)
        self.assertIsInstance(composed_composites[0].layers[0], _FakeBlurredBackground)
        self.assertEqual(composed_composites[0].size, (1080, 1920))

    def test_combine_videos_skips_unreadable_material_in_probe(self):
        """探测阶段遇到损坏素材应跳过该文件，而不是中断整个合成。"""

        class _FakeAudioClip:
            duration = 5.0

            def close(self):
                pass

        class _FakeVideoClip:
            def __init__(self, duration=3.0):
                self.duration = duration
                self.size = (1080, 1920)
                self.w = 1080
                self.h = 1920

            def subclipped(self, start_time, end_time):
                return _FakeVideoClip(end_time - start_time)

            def close(self):
                pass

        def _open_fake_video_clip(video_path):
            if video_path == "broken.mp4":
                raise RuntimeError("corrupt container")
            return _FakeVideoClip(3.0)

        with tempfile.TemporaryDirectory() as temp_dir:
            combined_video_path = os.path.join(temp_dir, "combined.mp4")
            with (
                patch.object(vd, "AudioFileClip", return_value=_FakeAudioClip()),
                patch.object(
                    vd,
                    "_open_video_clip_quietly",
                    side_effect=_open_fake_video_clip,
                ),
                patch.object(vd, "_write_videofile_with_codec_fallback") as write_mock,
                patch.object(vd, "concat_video_clips_with_ffmpeg"),
                patch.object(vd, "delete_files"),
            ):
                result = vd.combine_videos(
                    combined_video_path=combined_video_path,
                    video_paths=["broken.mp4", "good.mp4"],
                    audio_file=os.path.join(temp_dir, "audio.mp3"),
                    video_concat_mode=vd.VideoConcatMode.sequential,
                    video_transition_mode=None,
                    max_clip_duration=5,
                )

        self.assertEqual(result, combined_video_path)
        self.assertEqual(write_mock.call_count, 1)

    def test_combine_videos_closes_clip_on_processing_failure(self):
        """片段处理失败时必须关闭本次打开的 MoviePy clip，避免句柄泄漏。"""

        class _FakeAudioClip:
            duration = 5.0

            def close(self):
                pass

        created = []

        class _FakeVideoClip:
            def __init__(self):
                created.append(self)
                self.duration = 5.0
                self.size = (1080, 1920)
                self.w = 1080
                self.h = 1920

            def subclipped(self, start_time, end_time):
                return _FakeVideoClip()

            def close(self):
                pass

        closed = []

        def _recording_close(clip):
            closed.append(clip)

        with tempfile.TemporaryDirectory() as temp_dir:
            combined_video_path = os.path.join(temp_dir, "combined.mp4")
            with (
                patch.object(vd, "AudioFileClip", return_value=_FakeAudioClip()),
                patch.object(
                    vd,
                    "_open_video_clip_quietly",
                    return_value=_FakeVideoClip(),
                ),
                patch.object(vd, "close_clip", side_effect=_recording_close),
                patch.object(
                    vd,
                    "_write_videofile_with_codec_fallback",
                    side_effect=RuntimeError("encode failed"),
                ),
                patch.object(vd, "concat_video_clips_with_ffmpeg"),
                patch.object(vd, "delete_files"),
            ):
                with self.assertRaises(RuntimeError):
                    vd.combine_videos(
                        combined_video_path=combined_video_path,
                        video_paths=["clip.mp4"],
                        audio_file=os.path.join(temp_dir, "audio.mp3"),
                        video_concat_mode=vd.VideoConcatMode.sequential,
                        video_transition_mode=None,
                        max_clip_duration=5,
                    )

        # 探测阶段和逐片段处理阶段打开的 clip 都应被关闭。
        self.assertIn(created[-1], closed)

    def test_delete_image_material_clips_removes_only_generated_intermediates(self):
        """只删除由图片生成的 Ken Burns 中间 .mp4，保留原始图片和真实视频。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            image = os.path.join(temp_dir, "ai.png")
            generated = f"{image}.mp4"
            real_video = os.path.join(temp_dir, "clip.mp4")
            Path(image).write_bytes(b"img")
            Path(generated).write_bytes(b"generated")
            Path(real_video).write_bytes(b"real")

            vd.delete_image_material_clips([generated, real_video, "", None])

            self.assertFalse(os.path.exists(generated))
            self.assertTrue(os.path.exists(image))
            self.assertTrue(os.path.exists(real_video))

    def test_concat_video_clips_limits_output_to_audio_duration(self):
        """最终拼接时应裁到音频时长，避免安全余量带来明显静音尾巴。"""

        def fake_run(command, capture_output, text, check, timeout=None):
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as temp_dir:
            clip_file = os.path.join(temp_dir, "clip.mp4")
            output_file = os.path.join(temp_dir, "combined.mp4")
            Path(clip_file).write_bytes(b"fake")

            with patch.object(vd.subprocess, "run", side_effect=fake_run) as run:
                vd.concat_video_clips_with_ffmpeg(
                    clip_files=[clip_file],
                    output_file=output_file,
                    threads=1,
                    output_dir=temp_dir,
                    max_duration=10.0,
                )

        command = run.call_args.args[0]
        self.assertEqual(command[command.index("-t") + 1], "10.000")
        self.assertLess(command.index("-t"), command.index(output_file))

    def test_concat_video_clips_uses_stream_copy_when_possible(self):
        """所有片段参数一致时，concat 应走 -c copy 秒级路径，避免重复编码。"""

        def fake_run(command, capture_output, text, check, timeout=None):
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as temp_dir:
            clip_file = os.path.join(temp_dir, "clip.mp4")
            output_file = os.path.join(temp_dir, "combined.mp4")
            Path(clip_file).write_bytes(b"fake")

            with patch.object(vd.subprocess, "run", side_effect=fake_run) as run:
                vd.concat_video_clips_with_ffmpeg(
                    clip_files=[clip_file],
                    output_file=output_file,
                    threads=1,
                    output_dir=temp_dir,
                )

        self.assertEqual(len(run.call_args_list), 1)
        used_codecs = [
            call.args[0][call.args[0].index("-c:v") + 1]
            for call in run.call_args_list
        ]
        self.assertEqual(used_codecs, ["copy"])

    def test_probe_video_metadata_uses_ffprobe_when_available(self):
        """探测优先走 ffprobe，不拉起 MoviePy 帧读取管线。"""
        fake_meta = {
            "streams": [{"width": 1080, "height": 1920, "duration": "5.5"}],
            "format": {"duration": "5.5"},
        }

        with (
            patch.object(vd.shutil, "which", return_value="C:/ffprobe.exe"),
            patch.object(
                vd.subprocess,
                "run",
                return_value=types.SimpleNamespace(
                    returncode=0, stdout=json.dumps(fake_meta), stderr=""
                ),
            ) as run,
            patch.object(vd, "_open_video_clip_quietly") as open_clip,
        ):
            metadata = vd._probe_video_metadata("clip.mp4")

        self.assertEqual(metadata, (5.5, 1080, 1920))
        open_clip.assert_not_called()
        self.assertIn("-show_entries", run.call_args.args[0])

    def test_probe_video_metadata_falls_back_to_moviepy_without_ffprobe(self):
        """没有 ffprobe 时必须回退到 MoviePy 打开路径，保证正确性。"""

        class _FakeClip:
            duration = 3.0
            size = (720, 1280)

        with (
            patch.object(vd.shutil, "which", return_value=None),
            patch.object(vd, "_open_video_clip_quietly", return_value=_FakeClip()),
            patch.object(vd, "close_clip") as close_clip,
        ):
            metadata = vd._probe_video_metadata("clip.mp4")

        self.assertEqual(metadata, (3.0, 720, 1280))
        close_clip.assert_called_once()

    def test_probe_video_metadata_returns_none_for_unreadable_file(self):
        """无法探测的素材返回 None，由 combine_videos 跳过该文件。"""

        with (
            patch.object(vd.shutil, "which", return_value=None),
            patch.object(
                vd,
                "_open_video_clip_quietly",
                side_effect=RuntimeError("corrupt container"),
            ),
        ):
            self.assertIsNone(vd._probe_video_metadata("broken.mp4"))

    def test_prioritize_unique_source_clips_uses_each_source_before_reuse(self):
        """
        随机模式下，一个长素材会被拆成多个片段。调度层应先让每个源素材
        至少出现一次，再使用同一源素材的其他切片，降低用户感知到的重复。
        """
        clips = [
            vd.SubClippedVideoClip("a.mp4", 0, 4, source_file_path="a.mp4"),
            vd.SubClippedVideoClip("a.mp4", 4, 8, source_file_path="a.mp4"),
            vd.SubClippedVideoClip("b.mp4", 0, 4, source_file_path="b.mp4"),
            vd.SubClippedVideoClip("b.mp4", 4, 8, source_file_path="b.mp4"),
            vd.SubClippedVideoClip("c.mp4", 0, 4, source_file_path="c.mp4"),
        ]

        ordered_clips = vd._prioritize_unique_source_clips(
            subclipped_items=clips,
            concat_mode=vd.VideoConcatMode.random,
        )

        self.assertCountEqual(ordered_clips, clips)
        first_round_sources = [clip.source_file_path for clip in ordered_clips[:3]]
        self.assertCountEqual(first_round_sources, ["a.mp4", "b.mp4", "c.mp4"])

    def test_prioritize_unique_source_clips_keeps_sequential_order(self):
        """
        顺序模式本身只取每个素材的首段，不应被随机调度逻辑改变顺序。
        """
        clips = [
            vd.SubClippedVideoClip("a.mp4", 0, 4, source_file_path="a.mp4"),
            vd.SubClippedVideoClip("b.mp4", 0, 4, source_file_path="b.mp4"),
            vd.SubClippedVideoClip("c.mp4", 0, 4, source_file_path="c.mp4"),
        ]

        ordered_clips = vd._prioritize_unique_source_clips(
            subclipped_items=clips,
            concat_mode=vd.VideoConcatMode.sequential,
        )

        self.assertEqual(ordered_clips, clips)

    def test_prioritize_unique_source_clips_prefers_long_primary_clip(self):
        """
        同一个源素材的最后一个切片可能短于目标片段时长。首轮去重时应优先
        选择较长片段，否则会因为累计时长不足而提前复用素材。
        """
        short_tail = vd.SubClippedVideoClip("a.mp4", 6, 6.5, source_file_path="a.mp4")
        full_clip = vd.SubClippedVideoClip("a.mp4", 0, 3, source_file_path="a.mp4")
        other_source = vd.SubClippedVideoClip("b.mp4", 0, 3, source_file_path="b.mp4")

        ordered_clips = vd._prioritize_unique_source_clips(
            subclipped_items=[short_tail, full_clip, other_source],
            concat_mode=vd.VideoConcatMode.random,
        )

        first_a_clip = next(
            clip for clip in ordered_clips if clip.source_file_path == "a.mp4"
        )
        self.assertEqual(first_a_clip, full_clip)

    def test_wrap_text(self):
        """test text wrapping function"""
        try:
            font_path = os.path.join(utils.font_dir(), "MicrosoftYaHeiNormal.ttc")
            if not os.path.exists(font_path):
                self.fail(f"font file not found: {font_path}")

            # test english text wrapping
            test_text_en = (
                "This is a test text for wrapping long sentences in english language"
            )

            wrapped_text_en, text_height_en = vd.wrap_text(
                text=test_text_en, max_width=300, font=font_path, fontsize=30
            )
            print(wrapped_text_en, text_height_en)
            # verify text is wrapped
            self.assertIn("\n", wrapped_text_en)

            # test chinese text wrapping
            test_text_zh = (
                "这是一段用来测试中文长句换行的文本内容，应该会根据宽度限制进行换行处理"
            )
            wrapped_text_zh, text_height_zh = vd.wrap_text(
                text=test_text_zh, max_width=300, font=font_path, fontsize=30
            )
            print(wrapped_text_zh, text_height_zh)
            # verify chinese text is wrapped
            self.assertIn("\n", wrapped_text_zh)
        except Exception as e:
            self.fail(f"test wrap_text failed: {str(e)}")

    def test_subtitle_pop_in_scale_stays_static_outside_animation_window(self):
        """时长无效、时间越界时缩放恒为 1.0，不产生任何动画。"""
        self.assertEqual(vd.subtitle_pop_in_scale(-1, 1.0), 1.0)
        self.assertEqual(vd.subtitle_pop_in_scale(0, 0), 1.0)
        self.assertEqual(vd.subtitle_pop_in_scale(2.0, 1.0), 1.0)

    def test_subtitle_pop_in_scale_overshoots_then_settles(self):
        """0.15 秒窗口内：起点 0 -> 峰值 1.1 -> 终点回落到 1.0。"""
        window = vd._SUBTITLE_POP_IN_WINDOW_SECONDS
        self.assertAlmostEqual(vd.subtitle_pop_in_scale(0, 1.0), 0.0)
        peak = vd.subtitle_pop_in_scale(window / 2, 1.0)
        self.assertAlmostEqual(peak, 1.1, places=6)
        self.assertAlmostEqual(vd.subtitle_pop_in_scale(window, 1.0), 1.0)
        self.assertAlmostEqual(vd.subtitle_pop_in_scale(window + 0.001, 1.0), 1.0)

    def test_subtitle_pop_in_scale_compresses_window_for_short_clips(self):
        """字幕时长短于动画窗口时，曲线压缩到字幕时长内完成。"""
        duration = 0.1
        self.assertAlmostEqual(vd.subtitle_pop_in_scale(0, duration), 0.0)
        self.assertAlmostEqual(
            vd.subtitle_pop_in_scale(duration / 2, duration), 1.1, places=6
        )
        self.assertAlmostEqual(vd.subtitle_pop_in_scale(duration, duration), 1.0)

    def test_subtitle_pop_in_scale_is_monotonic_up_then_down(self):
        """前半程单调放大、后半程单调回落，避免抖动。"""
        window = vd._SUBTITLE_POP_IN_WINDOW_SECONDS
        samples = [vd.subtitle_pop_in_scale(t / 20 * window, 1.0) for t in range(21)]
        rising = samples[:11]
        falling = samples[10:]
        self.assertEqual(rising, sorted(rising))
        self.assertEqual(falling, sorted(falling, reverse=True))

    def test_subtitle_float_offset_is_zero_for_invalid_duration(self):
        self.assertEqual(vd.subtitle_float_offset(1.0, 0, 1080), 0.0)
        self.assertEqual(vd.subtitle_float_offset(1.0, -1, 1080), 0.0)

    def test_subtitle_float_offset_returns_to_zero_at_start_mid_end(self):
        """正弦漂移在起点、中点和终点都回到 0，视觉上是来回浮动。"""
        self.assertAlmostEqual(vd.subtitle_float_offset(0.0, 2.0, 1080), 0.0)
        self.assertAlmostEqual(vd.subtitle_float_offset(1.0, 2.0, 1080), 0.0)
        self.assertAlmostEqual(vd.subtitle_float_offset(2.0, 2.0, 1080), 0.0)

    def test_subtitle_float_offset_peaks_at_quarter_points(self):
        """1/4 处达到 +amp，3/4 处达到 -amp，且幅度随视频高度缩放。"""
        video_height = 1920
        amplitude = max(
            vd._SUBTITLE_FLOAT_MIN_AMPLITUDE,
            int(video_height * vd._SUBTITLE_FLOAT_AMPLITUDE_RATIO),
        )
        self.assertAlmostEqual(
            vd.subtitle_float_offset(0.5, 2.0, video_height), amplitude, places=6
        )
        self.assertAlmostEqual(
            vd.subtitle_float_offset(1.5, 2.0, video_height), -amplitude, places=6
        )

    def test_subtitle_float_offset_stays_within_safe_amplitude(self):
        """任意时刻的偏移量都被限制在安全幅度内，不会漂出安全区。"""
        video_height = 720
        amplitude = max(
            vd._SUBTITLE_FLOAT_MIN_AMPLITUDE,
            int(video_height * vd._SUBTITLE_FLOAT_AMPLITUDE_RATIO),
        )
        for t in (0.1, 0.7, 1.3, 1.9):
            offset = vd.subtitle_float_offset(t, 2.0, video_height)
            self.assertLessEqual(abs(offset), amplitude)

    def test_dynamic_subtitle_font_size_bounds_short_text_at_max_ratio(self):
        """短文本（如 STOP!）放大但不超过 1.5 倍上限。"""
        font_path = os.path.join(utils.font_dir(), "MicrosoftYaHeiNormal.ttc")
        if not os.path.exists(font_path):
            self.skipTest(f"font file not found: {font_path}")
        size = vd.dynamic_subtitle_font_size(
            "STOP!", 60, 1200, font_path
        )
        self.assertLessEqual(size, 60 * vd._SUBTITLE_DYNAMIC_MAX_RATIO)
        self.assertGreater(size, 60)

    def test_dynamic_subtitle_font_size_reduces_long_text_to_fit_two_lines(self):
        """长文本缩小字号，且按计算结果换行不超过两行。"""
        font_path = os.path.join(utils.font_dir(), "MicrosoftYaHeiNormal.ttc")
        if not os.path.exists(font_path):
            self.skipTest(f"font file not found: {font_path}")
        long_text = (
            "This is a fairly long subtitle sentence that wraps into multiple "
            "lines when rendered at the base font size"
        )
        base_size = 60
        max_width = 900
        size = vd.dynamic_subtitle_font_size(long_text, base_size, max_width, font_path)
        self.assertGreaterEqual(size, base_size * vd._SUBTITLE_DYNAMIC_MIN_RATIO)
        self.assertLessEqual(size, base_size)
        wrapped, _ = vd.wrap_text(
            long_text, max_width=max_width, font=font_path, fontsize=size
        )
        self.assertLessEqual(wrapped.count("\n") + 1, vd._SUBTITLE_DYNAMIC_MAX_LINES)

    def test_dynamic_subtitle_font_size_never_drops_below_min_ratio(self):
        """极端长文本触底后停在最小比例，而不是无限缩小。"""
        font_path = os.path.join(utils.font_dir(), "MicrosoftYaHeiNormal.ttc")
        if not os.path.exists(font_path):
            self.skipTest(f"font file not found: {font_path}")
        very_long_text = (
            "This is an extremely long subtitle sentence that keeps going and "
            "going far beyond what any single subtitle should ever contain in "
            "a typical short video production pipeline"
        )
        size = vd.dynamic_subtitle_font_size(
            very_long_text, 60, 900, font_path
        )
        self.assertEqual(size, int(round(60 * vd._SUBTITLE_DYNAMIC_MIN_RATIO)))

    def test_dynamic_subtitle_font_size_keeps_normal_text_at_base(self):
        """长度适中的文本不放大也不缩小，保持基准字号。"""
        font_path = os.path.join(utils.font_dir(), "MicrosoftYaHeiNormal.ttc")
        if not os.path.exists(font_path):
            self.skipTest(f"font file not found: {font_path}")
        size = vd.dynamic_subtitle_font_size("Normal subtitle", 60, 900, font_path)
        self.assertGreaterEqual(size, 60 * vd._SUBTITLE_DYNAMIC_MIN_RATIO)
        self.assertLessEqual(size, 60 * vd._SUBTITLE_DYNAMIC_MAX_RATIO)

    def test_dynamic_subtitle_font_size_falls_back_on_bad_font(self):
        """字体加载失败时返回基准字号，不中断渲染。"""
        self.assertEqual(
            vd.dynamic_subtitle_font_size("STOP!", 60, 900, "missing-font.ttf"), 60
        )

    def test_rounded_subtitle_background_clip_has_transparent_corners(self):
        """
        圆角字幕背景只在用户显式开启时使用。这里直接验证生成的 RGBA
        背景具备透明圆角和半透明中心，避免后续改动把圆角效果退化成实心矩形。
        """
        clip = vd._rounded_subtitle_background_clip(
            width=120,
            height=48,
            color="#123456",
            alpha=140,
            radius=16,
        )
        try:
            frame = clip.get_frame(0)
            mask = clip.mask.get_frame(0)

            self.assertEqual(frame.shape[0:2], (48, 120))
            self.assertEqual(tuple(frame[24, 60]), (18, 52, 86))
            self.assertEqual(mask[0, 0], 0)
            self.assertGreater(mask[24, 60], 0.5)
            self.assertLess(mask[24, 60], 0.6)
        finally:
            clip.close()

    def test_get_temp_audio_dir_returns_system_temp_on_windows(self):
        with patch("sys.platform", "win32"):
            result = vd._get_temp_audio_dir("/some/output/dir")
            self.assertEqual(result, tempfile.gettempdir())

    def test_get_temp_audio_dir_returns_output_dir_on_non_windows(self):
        for platform in ("linux", "darwin"):
            with self.subTest(platform=platform):
                with patch("sys.platform", platform):
                    result = vd._get_temp_audio_dir("/some/output/dir")
                    self.assertEqual(result, "/some/output/dir")

    def test_combine_videos_mix_transition_uses_cross_dissolve_merge(self):
        """
        Mix 转场必须能走完整个拼接流程：重叠时长参与进度累计，最终通过
        MoviePy 交叉溶解合并（而不是 ffmpeg concat）。回归保护：此前
        combine_videos 引用了不存在的 params 和未导入的
        concatenate_videoclips，选择 Mix 模式必然崩溃。
        """

        class _FakeVideoClip:
            def __init__(self, duration):
                self.duration = duration
                self.size = (1080, 1920)
                self.w = 1080
                self.h = 1920

            def subclipped(self, start_time, end_time):
                return _FakeVideoClip(end_time - start_time)

        class _FakeAudioClip:
            duration = 6.0

            def close(self):
                pass

        def _open_fake_video_clip(video_path):
            name = os.path.basename(video_path)
            if name.startswith("temp-clip-"):
                return _FakeVideoClip(5.0)
            return _FakeVideoClip(8.0)

        with tempfile.TemporaryDirectory() as temp_dir:
            combined_video_path = os.path.join(temp_dir, "combined.mp4")

            with (
                patch.object(vd, "AudioFileClip", return_value=_FakeAudioClip()),
                patch.object(
                    vd, "_open_video_clip_quietly", side_effect=_open_fake_video_clip
                ),
                patch.object(
                    vd,
                    "_write_videofile_with_codec_fallback",
                    side_effect=lambda clip, output_file, **kwargs: Path(
                        output_file
                    ).write_bytes(b"video-data"),
                ),
                patch.object(
                    vd, "concat_video_clips_with_ffmpeg"
                ) as concat_ffmpeg_mock,
                patch.object(
                    vd.video_effects, "crossfade_transition"
                ) as crossfade_mock,
                patch.object(vd, "concatenate_videoclips") as concat_video_mock,
                patch.object(vd, "delete_files"),
            ):
                result = vd.combine_videos(
                    combined_video_path=combined_video_path,
                    video_paths=["clip-1.mp4", "clip-2.mp4"],
                    audio_file=os.path.join(temp_dir, "audio.mp3"),
                    video_aspect=vd.VideoAspect.portrait,
                    video_concat_mode=vd.VideoConcatMode.sequential,
                    video_transition_mode=vd.VideoTransitionMode.mix,
                    max_clip_duration=5,
                    mix_overlap_duration=1.0,
                )

        self.assertEqual(result, combined_video_path)
        # Mix 模式走 MoviePy 交叉溶解，不应调用 ffmpeg concat。
        concat_ffmpeg_mock.assert_not_called()
        concat_video_mock.assert_called_once()
        self.assertEqual(concat_video_mock.call_args.kwargs["padding"], -1.0)
        self.assertEqual(concat_video_mock.call_args.kwargs["method"], "compose")
        # 第二个片段开始应用一次交叉溶解转场。
        self.assertEqual(crossfade_mock.call_count, 1)
        self.assertEqual(crossfade_mock.call_args.args[1], 1.0)

    def test_combine_videos_mix_overlap_is_clamped_below_clip_duration(self):
        """
        重叠时长大于等于单片段时长时，累计进度会停滞并可能死循环。
        combine_videos 必须把重叠收敛到严格小于片段时长。
        """

        class _FakeVideoClip:
            def __init__(self, duration):
                self.duration = duration
                self.size = (1080, 1920)
                self.w = 1080
                self.h = 1920

            def subclipped(self, start_time, end_time):
                return _FakeVideoClip(end_time - start_time)

        class _FakeAudioClip:
            duration = 12.0

            def close(self):
                pass

        def _open_fake_video_clip(video_path):
            name = os.path.basename(video_path)
            if name.startswith("temp-clip-"):
                return _FakeVideoClip(2.0)
            return _FakeVideoClip(6.0)

        with tempfile.TemporaryDirectory() as temp_dir:
            combined_video_path = os.path.join(temp_dir, "combined.mp4")

            with (
                patch.object(vd, "AudioFileClip", return_value=_FakeAudioClip()),
                patch.object(
                    vd, "_open_video_clip_quietly", side_effect=_open_fake_video_clip
                ),
                patch.object(
                    vd,
                    "_write_videofile_with_codec_fallback",
                    side_effect=lambda clip, output_file, **kwargs: Path(
                        output_file
                    ).write_bytes(b"video-data"),
                ),
                patch.object(vd, "concat_video_clips_with_ffmpeg"),
                patch.object(vd.video_effects, "crossfade_transition"),
                patch.object(vd, "concatenate_videoclips") as concat_video_mock,
                patch.object(vd, "delete_files"),
            ):
                vd.combine_videos(
                    combined_video_path=combined_video_path,
                    video_paths=["clip-1.mp4", "clip-2.mp4"],
                    audio_file=os.path.join(temp_dir, "audio.mp3"),
                    video_aspect=vd.VideoAspect.portrait,
                    video_concat_mode=vd.VideoConcatMode.random,
                    video_transition_mode=vd.VideoTransitionMode.mix,
                    max_clip_duration=2,
                    mix_overlap_duration=2.0,
                )

        # 2 秒片段 + 2.0 秒重叠会被收敛到 1.95 秒，进度保持为正。
        self.assertEqual(concat_video_mock.call_args.kwargs["padding"], -1.95)

    def test_combine_videos_mix_concat_is_memory_bounded(self):
        """
        Mix 拼接必须分块处理：任意时刻同时打开的 clip 不超过 chunk 大小，
        而不是像旧实现那样把全部片段一次性加载进内存。这里通过统计
        _open_video_clip_quietly 的打开与 close_clip 的关闭，跟踪峰值并发。
        """

        class _FakeVideoClip:
            def __init__(self, duration):
                self.duration = duration
                self.size = (1080, 1920)
                self.w = 1080
                self.h = 1920

            def subclipped(self, start_time, end_time):
                return _FakeVideoClip(end_time - start_time)

        class _FakeAudioClip:
            duration = 40.0

            def close(self):
                pass

        def _open_fake_video_clip(video_path):
            name = os.path.basename(video_path)
            if name.startswith("temp-clip-"):
                return _FakeVideoClip(5.0)
            return _FakeVideoClip(8.0)

        open_count = 0
        close_count = 0
        max_open = 0

        def _recording_open(video_path):
            nonlocal open_count, max_open
            open_count += 1
            max_open = max(max_open, open_count - close_count)
            return _open_fake_video_clip(video_path)

        def _recording_close(_clip):
            nonlocal close_count
            close_count += 1

        with tempfile.TemporaryDirectory() as temp_dir:
            combined_video_path = os.path.join(temp_dir, "combined.mp4")

            with (
                patch.object(vd, "AudioFileClip", return_value=_FakeAudioClip()),
                patch.object(
                    vd, "_open_video_clip_quietly", side_effect=_recording_open
                ),
                patch.object(vd, "close_clip", side_effect=_recording_close),
                patch.object(
                    vd,
                    "_write_videofile_with_codec_fallback",
                    side_effect=lambda clip, output_file, **kwargs: Path(
                        output_file
                    ).write_bytes(b"video-data"),
                ),
                patch.object(vd, "concat_video_clips_with_ffmpeg"),
                patch.object(vd.video_effects, "crossfade_transition"),
                patch.object(vd, "concatenate_videoclips"),
                patch.object(vd, "delete_files"),
            ):
                result = vd.combine_videos(
                    combined_video_path=combined_video_path,
                    video_paths=[f"clip-{i}.mp4" for i in range(7)],
                    audio_file=os.path.join(temp_dir, "audio.mp3"),
                    video_aspect=vd.VideoAspect.portrait,
                    video_concat_mode=vd.VideoConcatMode.sequential,
                    video_transition_mode=vd.VideoTransitionMode.mix,
                    max_clip_duration=5,
                    mix_overlap_duration=1.0,
                )

        self.assertEqual(result, combined_video_path)
        # 40s 音频会触发素材循环，processed_clips 远多于 chunk 大小；
        # 分块拼接下同时打开的 clip 峰值必须被限制在 chunk 大小内。
        self.assertGreaterEqual(open_count, vd._MIX_CONCAT_CHUNK_SIZE * 3)
        self.assertLessEqual(max_open, vd._MIX_CONCAT_CHUNK_SIZE)


class TestMaterialResolutionTolerance(unittest.TestCase):
    def test_accepts_material_at_the_nominal_minimum(self):
        self.assertTrue(vd.is_material_resolution_acceptable(480, 480))

    def test_accepts_whatsapp_recompressed_portrait_clip(self):
        # WhatsApp delivers 9:16 clips as 478x850, two pixels under the
        # nominal 480 minimum. Rejecting them fails the whole task.
        self.assertTrue(vd.is_material_resolution_acceptable(478, 850))

    def test_accepts_material_exactly_at_the_tolerance_bound(self):
        bound = vd._MIN_MATERIAL_DIMENSION - vd._MIN_DIMENSION_TOLERANCE
        self.assertTrue(vd.is_material_resolution_acceptable(bound, bound))

    def test_rejects_material_just_below_the_tolerance_bound(self):
        bound = vd._MIN_MATERIAL_DIMENSION - vd._MIN_DIMENSION_TOLERANCE
        self.assertFalse(vd.is_material_resolution_acceptable(bound - 1, 850))
        self.assertFalse(vd.is_material_resolution_acceptable(850, bound - 1))

    def test_rejects_genuinely_low_resolution_material(self):
        self.assertFalse(vd.is_material_resolution_acceptable(320, 240))


class TestSubtitleCasing(unittest.TestCase):
    def test_original_leaves_text_unchanged(self):
        text = "This is an end to end verification"
        self.assertEqual(
            vd.apply_subtitle_casing(text, vd.SUBTITLE_CASING_ORIGINAL), text
        )
        self.assertEqual(vd.apply_subtitle_casing(text, None), text)
        self.assertEqual(vd.apply_subtitle_casing(text, ""), text)

    def test_upper_applies_full_uppercase(self):
        self.assertEqual(
            vd.apply_subtitle_casing(
                "This is an end to end verification", vd.SUBTITLE_CASING_UPPER
            ),
            "THIS IS AN END TO END VERIFICATION",
        )

    def test_title_case_capitalizes_each_word(self):
        self.assertEqual(
            vd.apply_subtitle_casing(
                "the mix transition should cycle smoothly", vd.SUBTITLE_CASING_TITLE
            ),
            "The Mix Transition Should Cycle Smoothly",
        )

    def test_lower_applies_full_lowercase(self):
        self.assertEqual(
            vd.apply_subtitle_casing(
                "THIS IS AN END TO END VERIFICATION", vd.SUBTITLE_CASING_LOWER
            ),
            "this is an end to end verification",
        )

    def test_mode_is_case_insensitive_and_trimmed(self):
        self.assertEqual(
            vd.apply_subtitle_casing("hello world", " UPPER "),
            "HELLO WORLD",
        )

    def test_unknown_mode_falls_back_to_original(self):
        text = "Mixed Case Text"
        self.assertEqual(vd.apply_subtitle_casing(text, "sticky_note"), text)

    def test_empty_text_is_safe(self):
        self.assertEqual(vd.apply_subtitle_casing("", vd.SUBTITLE_CASING_UPPER), "")

    def test_phrase_chunks_render_cleanly_across_modes(self):
        # 2~4 词短语在所有模式下都不应产生异常或破坏换行符号。
        phrases = ["This is an end", "to end verification", "The Mix transition should"]
        for mode in (
            vd.SUBTITLE_CASING_ORIGINAL,
            vd.SUBTITLE_CASING_UPPER,
            vd.SUBTITLE_CASING_TITLE,
            vd.SUBTITLE_CASING_LOWER,
        ):
            for phrase in phrases:
                transformed = vd.apply_subtitle_casing(phrase, mode)
                self.assertTrue(transformed)
                self.assertEqual(len(transformed.split()), len(phrase.split()))


class TestOverlayCompositing(unittest.TestCase):
    def test_build_overlay_clips_returns_empty_when_disabled(self):
        """叠加层总开关关闭时，即使有规划项也不合成任何片段。"""
        params = types.SimpleNamespace(
            overlay_enabled=False,
            overlay_text_color="#FFFFFF",
            overlay_bg_color="#000000",
            overlay_image_opacity=0.85,
        )
        item = types.SimpleNamespace(
            kind="title",
            text="都市通勤",
            start=0.0,
            end=2.0,
            position="top_center",
        )
        self.assertEqual(vd.build_overlay_clips([item], "", 720, 1280, params), [])

    def test_build_overlay_clips_returns_empty_when_no_items(self):
        params = types.SimpleNamespace(
            overlay_enabled=True,
            overlay_text_color="#FFFFFF",
            overlay_bg_color="#000000",
            overlay_image_opacity=0.85,
        )
        self.assertEqual(vd.build_overlay_clips([], "", 720, 1280, params), [])

    def test_build_overlay_image_clip_returns_none_for_missing_file(self):
        params = types.SimpleNamespace(overlay_image_opacity=0.85)
        self.assertIsNone(
            vd.build_overlay_image_clip(
                "C:/definitely/does/not/exist.png", 720, 1280, params, 10.0
            )
        )

    def test_build_overlay_image_clip_returns_none_for_empty_path(self):
        params = types.SimpleNamespace(overlay_image_opacity=0.85)
        self.assertIsNone(vd.build_overlay_image_clip("", 720, 1280, params, 10.0))


if __name__ == "__main__":
    unittest.main()
