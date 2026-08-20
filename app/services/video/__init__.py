import itertools
import io
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from contextlib import ExitStack, redirect_stdout
from functools import lru_cache
from typing import List
from loguru import logger
from moviepy import (
    AudioFileClip,
    ColorClip,
    CompositeAudioClip,
    CompositeVideoClip,
    ImageClip,
    TextClip,
    VideoFileClip,
    afx,
    vfx,
    concatenate_videoclips,
)
from moviepy.video.tools.subtitles import SubtitlesClip
from PIL import Image, ImageFont

from app.config import config
from app.models import const
from app.models.schema import (
    MaterialInfo,
    VideoAspect,
    VideoConcatMode,
    VideoParams,
    VideoTransitionMode,
)
from app.services import bgm as bgm_service
from app.services.utils import video_effects
from app.utils import file_security, utils

from .constants import (
    audio_codec,
    audio_bitrate,
    fps,
    _VIDEO_DURATION_SAFETY_MARGIN,
    _MIN_MATERIAL_DIMENSION,
    _MIN_DIMENSION_TOLERANCE,
    _DEFAULT_VIDEO_CODEC,
    _DEFAULT_VIDEO_PRESET,
    _DEFAULT_VIDEO_CRF,
    _SUPPORTED_VIDEO_CODECS,
    _FFMPEG_CONCAT_TIMEOUT_SECONDS,
    _FFPROBE_TIMEOUT_SECONDS,
    _get_required_video_duration,
    is_material_resolution_acceptable,
)
from .types import SubClippedVideoClip

# Tests patch this set directly (vd._runtime_disabled_video_codecs.clear()).
# Keep it here so the exact same set object is shared with all functions.
_runtime_disabled_video_codecs = set()


def _prioritize_unique_source_clips(
    subclipped_items: List[SubClippedVideoClip],
    concat_mode: VideoConcatMode,
) -> List[SubClippedVideoClip]:
    """
    优先让每个源素材只出现一次，降低成片里同一素材反复出现的概率。

    线上素材经常会遇到“一个长视频被切成多个短片段”的情况。旧逻辑在
    random 模式下直接打乱所有短片段，导致同一个源视频的多个切片可能
    分布在开头和中间，用户会感知为素材重复。本函数只调整片段顺序：
    先放每个源文件里最长的一个片段，剩余片段作为兜底；当素材总时长不足时，
    仍然允许后续片段补齐音频长度，避免破坏视频生成成功率。优先选择最长
    片段是为了避免随机选中视频尾部的零碎短片段，导致明明有足够素材却过早复用。
    """
    if not subclipped_items:
        return []

    concat_mode_value = getattr(concat_mode, "value", concat_mode)
    if concat_mode_value != VideoConcatMode.random.value:
        return subclipped_items

    grouped_items: dict[str, list[SubClippedVideoClip]] = {}
    for item in subclipped_items:
        grouped_items.setdefault(item.source_file_path, []).append(item)

    primary_items = []
    overflow_items = []
    for items in grouped_items.values():
        primary_item = max(items, key=lambda item: item.duration)
        primary_items.append(primary_item)
        overflow_items.extend(item for item in items if item is not primary_item)

    random.shuffle(primary_items)
    random.shuffle(overflow_items)
    logger.info(
        "prioritized unique video materials, "
        f"sources: {len(grouped_items)}, "
        f"primary clips: {len(primary_items)}, "
        f"fallback clips: {len(overflow_items)}"
    )
    return primary_items + overflow_items


def get_ffmpeg_binary():
    """
    兼容历史上直接从 video 服务读取 FFmpeg 路径的调用方。

    真正的解析逻辑已经抽到 `app.utils.utils.get_ffmpeg_binary()`，视频、语音
    和后续新增链路都应复用同一套优先级；这里保留薄包装，避免外部脚本或
    旧测试直接导入 `app.services.video.get_ffmpeg_binary` 时出现 AttributeError。
    """
    return utils.get_ffmpeg_binary()


def _get_configured_video_codec() -> str:
    """
    读取用户配置的视频编码器。

    该配置面向高级用户，用于尝试启用 NVENC/AMF/QSV/VideoToolbox 等硬件
    编码。这里刻意只允许固定白名单，避免开放任意 FFmpeg 参数后，用户填错
    参数导致输出格式不可控，甚至让生成任务在后续阶段才失败。
    """
    configured_codec = str(
        config.app.get("video_codec", _DEFAULT_VIDEO_CODEC) or _DEFAULT_VIDEO_CODEC
    ).strip()
    if configured_codec not in _SUPPORTED_VIDEO_CODECS:
        logger.warning(
            f"unsupported video codec configured: {configured_codec}, "
            f"fallback to {_DEFAULT_VIDEO_CODEC}"
        )
        return _DEFAULT_VIDEO_CODEC
    return configured_codec


def _get_video_encode_args() -> dict:
    """Return libx264 encode kwargs (preset/crf/bitrate) resolved from config.

    Read at call time through ``config.app.get`` so the per-task config snapshot
    overlay (thread-local) is honored. Unset values fall back to conservative
    defaults, keeping output size predictable while making quality tunable.
    """
    preset = str(
        config.app.get("video_preset", _DEFAULT_VIDEO_PRESET) or _DEFAULT_VIDEO_PRESET
    ).strip()
    try:
        crf = int(config.app.get("video_crf", _DEFAULT_VIDEO_CRF) or _DEFAULT_VIDEO_CRF)
    except (TypeError, ValueError):
        crf = _DEFAULT_VIDEO_CRF
    args: dict = {"preset": preset, "crf": crf}
    bitrate = str(config.app.get("video_bitrate", "") or "").strip()
    if bitrate:
        args["bitrate"] = bitrate
    return args


@lru_cache(maxsize=16)
def _ffmpeg_encoder_exists(ffmpeg_binary: str, codec: str) -> bool:
    """
    检查当前 FFmpeg 是否声明支持指定编码器。

    这只能证明 FFmpeg 编译时包含该 encoder，不能证明当前机器硬件和驱动
    一定可用。因此实际编码失败时仍会再回退到 libx264。
    """
    try:
        result = subprocess.run(
            [ffmpeg_binary, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning(
            "failed to inspect ffmpeg encoders, "
            f"fallback to {_DEFAULT_VIDEO_CODEC}: {str(exc)}"
        )
        return False

    if result.returncode != 0:
        logger.warning(
            "failed to inspect ffmpeg encoders, "
            f"fallback to {_DEFAULT_VIDEO_CODEC}: {(result.stderr or result.stdout or '').strip()}"
        )
        return False
    return codec in result.stdout


def _get_effective_video_codec(preferred_codec: str | None = None) -> str:
    """
    返回本次实际使用的视频编码器。

    用户选择硬件编码器时，先做 FFmpeg encoder 列表检测；如果本进程里已经
    实际编码失败过，也直接回退，避免一个任务里每个片段都重复失败。
    """
    selected_codec = preferred_codec or _get_configured_video_codec()
    if selected_codec == _DEFAULT_VIDEO_CODEC:
        return _DEFAULT_VIDEO_CODEC

    if selected_codec in _runtime_disabled_video_codecs:
        logger.warning(
            f"video codec {selected_codec} was disabled after a runtime failure, "
            f"fallback to {_DEFAULT_VIDEO_CODEC}"
        )
        return _DEFAULT_VIDEO_CODEC

    ffmpeg_binary = utils.get_ffmpeg_binary()
    if not _ffmpeg_encoder_exists(ffmpeg_binary, selected_codec):
        logger.warning(
            f"ffmpeg encoder {selected_codec} is not available, "
            f"fallback to {_DEFAULT_VIDEO_CODEC}"
        )
        return _DEFAULT_VIDEO_CODEC

    return selected_codec


def _disable_runtime_video_codec(codec: str, reason: str):
    if codec == _DEFAULT_VIDEO_CODEC:
        return
    _runtime_disabled_video_codecs.add(codec)
    logger.warning(
        f"video codec {codec} failed, fallback to {_DEFAULT_VIDEO_CODEC}. "
        f"reason: {reason}"
    )


def _get_temp_audio_dir(output_dir: str) -> str:
    """
    Return the directory to use for MoviePy's temporary audio file.

    On Windows, Windows Defender can lock files written to the task output
    directory while scanning them, causing MoviePy to fail with a
    PermissionError (WinError 32) on the TEMP_MPY_wvf_snd temp file and
    leaving the final MP4 at 0 bytes.  Using the system temp directory
    sidesteps the scan without changing behaviour on other platforms.

    On Linux/macOS/Docker the output directory is returned unchanged so
    existing behaviour is preserved.
    """
    if sys.platform == "win32":
        return tempfile.gettempdir()
    return output_dir


def _translate_encode_args(kwargs: dict, codec: str) -> dict:
    """Map quality knobs onto ``write_videofile`` kwargs for a specific codec.

    MoviePy's ``write_videofile`` accepts ``preset``/``bitrate`` but has no
    ``crf`` kwarg; CRF must travel via ``ffmpeg_params``. Hardware encoders
    (e.g. h264_nvenc) do not support ``-crf``, so it is dropped there and only
    applied when encoding with libx264.
    """
    translated = dict(kwargs)
    crf = translated.pop("crf", None)
    if crf is not None and codec == _DEFAULT_VIDEO_CODEC:
        params = list(translated.get("ffmpeg_params") or [])
        params.extend(["-crf", str(crf)])
        translated["ffmpeg_params"] = params
    return translated


def _fallback_write_videofile(
    clip, output_file: str, failed_codec: str, reason: str, **kwargs
):
    """
    硬件编码失败后用 libx264 重试，只有重试成功才禁用该硬件编码器。

    Windows 上 FFmpeg 失败原因比较复杂：可能是显卡/驱动不支持，也可能是输出
    文件被占用、目录权限、杀软拦截等通用 IO 问题。只有 libx264 能成功写出时，
    才能判断原始失败大概率来自硬件编码器本身，避免误伤后续任务。
    """
    clip.write_videofile(
        output_file,
        codec=_DEFAULT_VIDEO_CODEC,
        **_translate_encode_args(kwargs, _DEFAULT_VIDEO_CODEC),
    )
    _disable_runtime_video_codec(failed_codec, reason)
    return _DEFAULT_VIDEO_CODEC


def _write_videofile_with_codec_fallback(clip, output_file: str, codec: str, **kwargs):
    """
    使用指定编码器写出视频，失败时自动用 libx264 重试一次。

    硬件编码器是否可用不仅取决于 FFmpeg，还取决于显卡、驱动和当前运行环境。
    生成任务不能因为高级编码器不可用而整体失败，所以这里把回退集中处理。
    """
    effective_codec = _get_effective_video_codec(codec)
    try:
        clip.write_videofile(
            output_file,
            codec=effective_codec,
            **_translate_encode_args(kwargs, effective_codec),
        )
        return effective_codec
    except Exception as exc:
        if effective_codec == _DEFAULT_VIDEO_CODEC:
            raise
        return _fallback_write_videofile(
            clip,
            output_file,
            failed_codec=effective_codec,
            reason=str(exc),
            **kwargs,
        )


def _escape_ffmpeg_concat_path(file_path: str) -> str:
    # concat demuxer 使用单引号包裹路径，路径中的单引号需要先转义。
    return file_path.replace("'", "'\\''")


def _format_ffmpeg_concat_path(file_path: str) -> str:
    """
    生成 concat demuxer 文件列表中的路径。

    FFmpeg 官方文档要求 concat list 中的特殊字符和空格需要转义；Windows
    绝对路径里的反斜杠也容易被解析成转义字符。这里统一转成正斜杠形式，
    让 `C:\\Users\\...` 变成 `C:/Users/...`，再处理单引号，兼容 macOS/Linux。
    """
    absolute_path = os.path.abspath(file_path)
    return _escape_ffmpeg_concat_path(absolute_path.replace("\\", "/"))


def concat_video_clips_with_ffmpeg(
    clip_files: List[str],
    output_file: str,
    threads: int,
    output_dir: str,
    max_duration: float | None = None,
):
    concat_list_file = os.path.join(output_dir, "ffmpeg-concat-list.txt")
    with open(concat_list_file, "w", encoding="utf-8") as fp:
        for clip_file in clip_files:
            fp.write(f"file '{_format_ffmpeg_concat_path(clip_file)}'\n")

    def build_command(codec: str) -> list[str]:
        command = [
            utils.get_ffmpeg_binary(),
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_list_file,
            "-c:v",
            codec,
        ]
        # copy 模式直接复制帧，不重新编码，也不需要指定像素格式/线程数。
        if codec != "copy":
            command.extend(["-threads", str(threads or 2), "-pix_fmt", "yuv420p"])
            if codec == _DEFAULT_VIDEO_CODEC:
                encode_args = _get_video_encode_args()
                command.extend(["-preset", str(encode_args.get("preset", _DEFAULT_VIDEO_PRESET))])
                command.extend(["-crf", str(encode_args.get("crf", _DEFAULT_VIDEO_CRF))])
                if encode_args.get("bitrate"):
                    command.extend(["-b:v", str(encode_args["bitrate"])])
        if max_duration is not None and max_duration > 0:
            command.extend(["-t", f"{max_duration:.3f}"])
        command.append(output_file)
        return command

    def run_concat(codec: str):
        command = build_command(codec)
        # 使用 ffmpeg 只做一次串联与编码，避免 MoviePy 逐段合并时反复重编码，
        # 从而降低画质劣化与颜色偏移风险。
        # 卡死的 ffmpeg（损坏的 concat 输入、挂起的网络盘）会让整个任务线程
        # 无限阻塞；这里给子进程一个硬超时，超时后结束进程树并抛出明确错误。
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=_FFMPEG_CONCAT_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"ffmpeg concat timed out after {_FFMPEG_CONCAT_TIMEOUT_SECONDS}s"
            ) from exc
        if result.returncode != 0:
            error_message = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(error_message or "ffmpeg concat failed")
        return codec

    try:
        # 临时片段都由同一段代码编码，分辨率/编码器/像素格式一致，先尝试
        # 流复制（-c copy）串联，秒级完成；一旦参数不一致或损坏，ffmpeg
        # 会失败，再退回完整的逐帧编码路径。
        try:
            return run_concat("copy")
        except Exception:
            pass

        effective_codec = _get_effective_video_codec()
        try:
            return run_concat(effective_codec)
        except Exception as exc:
            if effective_codec == _DEFAULT_VIDEO_CODEC:
                raise
            result_codec = run_concat(_DEFAULT_VIDEO_CODEC)
            _disable_runtime_video_codec(effective_codec, str(exc))
            return result_codec
    finally:
        delete_files(concat_list_file)


def _sanitize_image_file(image_path: str) -> str:
    # 某些本地图片虽然能被 Pillow 打开，但会因为损坏的 EXIF/eXIf 元数据导致
    # ImageClip 在解析阶段直接抛异常。这里重新导出一份“干净图片”，把坏元数据剥离掉。
    image_root, _ = os.path.splitext(image_path)
    sanitized_path = f"{image_root}.sanitized.png"

    with Image.open(image_path) as image:
        # 统一导出为 PNG，避免 JPEG/PNG 不同元数据路径继续把坏块带过去。
        # 直接走 PIL 原生解码→编码路径并显式剥离 EXIF，避免
        # Image.new + putdata(list(image.getdata()))：那会为每个像素构造
        # Python 元组/列表，4K 图片（约 830 万像素）可瞬时占用数百 MB 内存。
        image.save(sanitized_path, exif=b"")

    return sanitized_path


def _open_image_clip_with_fallback(image_path: str):
    # 优先直接打开原始图片；如果因为损坏元数据失败，再尝试生成无元数据副本。
    try:
        return ImageClip(image_path), image_path
    except Exception as exc:
        logger.warning(
            f"failed to open image directly, trying sanitized copy: {image_path}, error: {str(exc)}"
        )
        sanitized_path = _sanitize_image_file(image_path)
        return ImageClip(sanitized_path), sanitized_path


def _kenburns_image_to_video(
    image_path: str,
    clip_duration: int = 4,
    fps: int = 30,
    effect: str = "kenburns",
) -> str:
    """
    把一张静态图片转成带运镜效果的短视频片段。

    生成式供应商（Gemini / Pollinations）返回的是静态图片，而合成引擎按
    视频片段处理素材。这里复用了本地素材预处理阶段的运镜逻辑，输出
    ``<image>.mp4``，让图片素材在 combine_videos 阶段与普通视频走同一条
    流水线。任何失败都返回空字符串，由调用方决定降级策略。

    ``effect`` 支持 kenburns（默认慢速推镜）、zoom_in、zoom_out、
    slide_left、slide_right、fade、random（每次随机选一个非静态效果）。
    """
    effect = (effect or "kenburns").strip().lower()
    if effect == "random":
        effect = random.choice(
            ["kenburns", "zoom_in", "zoom_out", "slide_left", "slide_right", "fade"]
        )

    clip = None
    final_clip = None
    try:
        clip = (
            ImageClip(image_path)
            .with_duration(clip_duration)
            .with_position("center")
        )
        from app.services.utils import video_effects as _effects

        if effect == "zoom_in":
            final_clip = _effects.zoomin_transition(clip, clip_duration)
        elif effect == "zoom_out":
            final_clip = _effects.zoomout_transition(clip, clip_duration)
        elif effect == "slide_left":
            final_clip = _effects.slidein_transition(clip, clip_duration, "left")
        elif effect == "slide_right":
            final_clip = _effects.slidein_transition(clip, clip_duration, "right")
        elif effect == "fade":
            final_clip = _effects.crossfade_transition(clip, clip_duration)
        else:
            # 默认 Ken Burns：从原尺寸缓慢放大到 112%（clip_duration 秒片段），
            # 模拟摄像机推近镜头。
            zoom_clip = clip.resized(
                lambda t: 1 + (clip_duration * 0.03) * (t / max(clip.duration, 0.01))
            )
            final_clip = CompositeVideoClip([zoom_clip])
        video_file = f"{image_path}.mp4"
        final_clip.write_videofile(video_file, fps=fps, logger=None)
        logger.success(f"image processed: {video_file} (effect={effect})")
        return video_file
    except Exception as exc:
        logger.warning(
            f"failed to convert image material to motion clip: "
            f"file={image_path}, effect={effect}, "
            f"error={type(exc).__name__}, detail={exc}"
        )
        return ""
    finally:
        close_clip(clip)
        close_clip(final_clip)


def delete_image_material_clips(video_paths: List[str]) -> None:
    """删除由图片素材生成的 Ken Burns 中间 .mp4，保留原始图片文件。

    生成路径形如 ``<image_path>.mp4``。只删除“去掉 .mp4 后仍是图片扩展名”
    的条目，避免误删用户上传的真实视频；文件不存在时静默跳过。
    """
    intermediates = []
    for video_path in video_paths:
        if not video_path or not video_path.endswith(".mp4"):
            continue
        source_path = video_path[: -len(".mp4")]
        if utils.parse_extension(source_path) in const.FILE_TYPE_IMAGES:
            intermediates.append(video_path)
    if intermediates:
        logger.debug(f"cleaning image material clips: {intermediates}")
        delete_files(intermediates)


def convert_image_materials_to_videos(
    video_paths: List[str], clip_duration: int = 4, image_effect: str = "kenburns"
) -> List[str]:
    """
    把素材列表里的图片（生成式 AI 提供）批量转成带运镜效果的视频片段。

    只转换图片扩展名的条目，其余路径原样保留；单个图片转换失败时保留
    原路径，避免因为一张图导致整条任务失败，后续合成阶段会按普通视频
    探测处理。
    """
    converted: List[str] = []
    for path in video_paths:
        ext = utils.parse_extension(path)
        if ext in const.FILE_TYPE_IMAGES:
            video_file = _kenburns_image_to_video(path, clip_duration, effect=image_effect)
            if video_file:
                converted.append(video_file)
                continue
            logger.warning(
                f"keep unconverted image material in pipeline: {path}"
            )
        converted.append(path)
    return converted


def _open_video_clip_quietly(video_path: str, audio: bool = False) -> VideoFileClip:
    """
    安静地打开视频文件，避免 MoviePy 2.1.x 把 ffmpeg 探测信息直接打印到 stdout。

    背景：
    当前依赖版本的 `FFMPEG_VideoReader` 内部存在 `print(self.infos)` 和
    `print(ffmpeg command)`，读取无音轨的中间视频时会输出
    `audio_found: False`。这只是输入素材 metadata，不代表最终成片没有音频，
    但会误导 WebUI/终端用户以为生成失败。

    实现：
    1. 只在打开 VideoFileClip 的短窗口内重定向 stdout；
    2. 默认 `audio=False`，因为项目视频素材阶段不需要保留素材原声，
       最终音频会在 `generate_video()` 阶段统一挂载；
    3. 如果依赖库确实输出了内容，降级为 debug 日志，便于必要时排查。
    """
    captured_stdout = io.StringIO()
    with redirect_stdout(captured_stdout):
        clip = VideoFileClip(video_path, audio=audio)

    moviepy_stdout = captured_stdout.getvalue().strip()
    if moviepy_stdout:
        logger.debug(
            "suppressed MoviePy video reader stdout for "
            f"{video_path}, chars: {len(moviepy_stdout)}"
        )

    return clip


def _probe_video_metadata(video_path: str) -> tuple[float, int, int] | None:
    """轻量读取视频时长与尺寸，避免每个素材都被完整打开两次。

    探测阶段只需要时长和宽高，却用 VideoFileClip 打开会拉起一条 FFmpeg
    帧读取管线（每个片段两次，任务里动辄十几条）。这里优先用 ffprobe 只读
    容器元数据；ffprobe 不可用或探测失败时回退到 ``_open_video_clip_quietly``，
    保证正确性不依赖外部工具。
    """
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        try:
            result = subprocess.run(
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height,duration:format=duration",
                    "-of",
                    "json",
                    video_path,
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=_FFPROBE_TIMEOUT_SECONDS,
            )
            if result.returncode == 0 and result.stdout.strip():
                data = json.loads(result.stdout)
                stream = (data.get("streams") or [{}])[0]
                width = int(stream.get("width") or 0)
                height = int(stream.get("height") or 0)
                duration = float(
                    stream.get("duration")
                    or (data.get("format") or {}).get("duration")
                    or 0
                )
                if width > 0 and height > 0 and duration > 0:
                    return duration, width, height
        except Exception as probe_error:
            logger.debug(
                f"ffprobe metadata read failed, falling back to MoviePy: "
                f"{os.path.basename(video_path)}, error={type(probe_error).__name__}"
            )

    clip = None
    try:
        clip = _open_video_clip_quietly(video_path)
        return clip.duration, int(clip.size[0]), int(clip.size[1])
    except Exception:
        return None
    finally:
        if clip is not None:
            close_clip(clip)


def close_clip(clip):
    if clip is None:
        return

    try:
        # close main resources
        if hasattr(clip, "reader") and clip.reader is not None:
            clip.reader.close()

        # close audio resources
        if hasattr(clip, "audio") and clip.audio is not None:
            if hasattr(clip.audio, "reader") and clip.audio.reader is not None:
                clip.audio.reader.close()
            del clip.audio

        # close mask resources
        if hasattr(clip, "mask") and clip.mask is not None:
            if hasattr(clip.mask, "reader") and clip.mask.reader is not None:
                clip.mask.reader.close()
            del clip.mask

        # handle child clips in composite clips
        if hasattr(clip, "clips") and clip.clips:
            for child_clip in clip.clips:
                if child_clip is not clip:  # avoid possible circular references
                    close_clip(child_clip)

        # clear clip list
        if hasattr(clip, "clips"):
            clip.clips = []

    except Exception as e:
        logger.error(f"failed to close clip: {str(e)}")


def delete_files(files: List[str] | str):
    import time

    if isinstance(files, str):
        files = [files]

    unique_files = dict.fromkeys(file for file in files if file)
    for file in unique_files:
        for attempt in range(3):
            try:
                os.remove(file)
                break
            except FileNotFoundError:
                break
            except PermissionError:
                if attempt < 2:
                    time.sleep(0.1 * (attempt + 1))
                else:
                    logger.warning(f"failed to delete temporary file {file} after retries: permission denied")
            except OSError as e:
                logger.warning(f"failed to delete temporary file {file}: {str(e)}")
                break


def get_bgm_file(bgm_type: str = "random", bgm_file: str = ""):
    if not bgm_type:
        return ""

    if bgm_file:
        try:
            resolved_bgm_file = bgm_service.resolve_bgm_file(bgm_file)
        except ValueError as exc:
            # API 请求里的 bgm_file 来自用户输入，只允许解析到用户 BGM 或内置
            # 歌曲目录，阻止 MoviePy 读取配置、密钥等任意服务器文件。
            logger.warning(f"reject unsafe bgm file: {bgm_file}, error: {str(exc)}")
            return ""
        return resolved_bgm_file

    if bgm_type == "random":
        files = bgm_service.list_bgm_files()
        # 当背景音乐目录为空时，直接回退为“不使用 BGM”，避免 random.choice([]) 抛异常。
        if not files:
            logger.warning("no background music files found")
            return ""
        return random.choice(files)

    return ""


def _cover_size(
    source_w: int, source_h: int, target_w: int, target_h: int
) -> tuple[int, int]:
    """Scale-to-cover size: 最小边长恰好盖住目标画幅（允许超出后被裁切）。"""
    scale = max(target_w / source_w, target_h / source_h)
    return max(int(round(source_w * scale)), target_w), max(
        int(round(source_h * scale)), target_h
    )


def _build_blurred_background(clip, target_width: int, target_height: int):
    """Build a blurred, darkened, scale-to-cover background layer for a clip.

    用于画幅不匹配的素材：把素材放大到完整覆盖目标画幅，再通过“剧烈缩小后
    放大”做廉价高斯模糊，并整体压暗，形成短视频常见的模糊填充背景，替代
    原来的纯黑底。返回与 clip 等长的背景层。
    """
    cover_w, cover_h = _cover_size(clip.w, clip.h, target_width, target_height)
    cover = clip.resized(new_size=(cover_w, cover_h))
    thumb_w = max(cover_w // 8, 1)
    thumb_h = max(cover_h // 8, 1)
    blurred = cover.resized(new_size=(thumb_w, thumb_h)).resized(
        new_size=(cover_w, cover_h)
    )
    return blurred.with_effects([vfx.MultiplyColor(0.5)]).with_position("center")


# MoviePy 的 concatenate_videoclips(method="compose") 返回的 CompositeVideoClip
# 会持有全部子片段的引用，渲染时按需从各子片段 reader（ffmpeg 进程）逐帧读取。
# 因此“折叠式拼接”（把上一次结果继续作为输入再拼下一段）无法真正释放已拼片段：
# 上一轮 composite 被嵌套引用，其子片段 reader 必须一直打开到最终渲染完成，
# 内存仍会随片段总数线性增长。这里改为“分块成片”：每组至多
# _MIX_CONCAT_CHUNK_SIZE 个片段，混合拼接后立即写出临时文件并关闭全部 clip，
# 下一轮再对临时文件分组拼接。任意时刻内存里只有一组片段驻留。
_MIX_CONCAT_CHUNK_SIZE = 3


def _mix_concat_group(
    clip_files: List[str],
    overlap: float,
    codec: str,
    fps: int,
    threads: int,
    output_file: str,
) -> str:
    """把一组片段做 Mix 交叉溶解拼接并写出视频文件，返回 output_file。

    与旧实现等价的视觉行为：除第一个片段外，其余片段先应用 CrossFadeIn
    交叉溶解，再用 concatenate_videoclips(..., padding=-overlap,
    method="compose") 叠加。组内片段数即同时打开的句柄峰值（调用方保证
    不超过 _MIX_CONCAT_CHUNK_SIZE）。无论成功或失败，本函数打开的所有
    clip 都会在 finally 中关闭，句柄不会泄漏到异常路径。
    """
    group_clips = []
    final_video = None
    try:
        for i, clip_file in enumerate(clip_files):
            clip = _open_video_clip_quietly(clip_file)
            if i > 0:
                clip = video_effects.crossfade_transition(clip, overlap)
            group_clips.append(clip)

        final_video = concatenate_videoclips(
            group_clips, padding=-overlap, method="compose"
        )
        _write_videofile_with_codec_fallback(
            final_video,
            output_file,
            codec=codec,
            logger=None,
            fps=fps,
            threads=threads,
            **_get_video_encode_args(),
        )
    finally:
        # composite 递归引用组内片段；先关 composite 再关一遍显式列表，
        # 与旧 mix 分支的双重清理一致，确保 reader 一定被释放。
        close_clip(final_video)
        for clip in group_clips:
            close_clip(clip)
    return output_file


def combine_videos(
    combined_video_path: str,
    video_paths: List[str],
    audio_file: str,
    video_aspect: VideoAspect = VideoAspect.portrait,
    video_concat_mode: VideoConcatMode = VideoConcatMode.random,
    video_transition_mode: VideoTransitionMode = None,
    max_clip_duration: int = 5,
    threads: int = 2,
    clip_speed: float = 1.0,
    mix_overlap_duration: float = 1.0,
) -> str:
    audio_clip = AudioFileClip(audio_file)
    try:
        # 这里只需要读取旁白音频时长来决定素材视频拼接长度；后续不会再使用
        # audio_clip。读取完成后立即关闭，避免早退或异常路径泄漏文件句柄。
        audio_duration = audio_clip.duration
    finally:
        close_clip(audio_clip)
    logger.info(f"audio duration: {audio_duration} seconds")
    logger.info(f"maximum clip duration: {max_clip_duration} seconds")
    required_video_duration = _get_required_video_duration(audio_duration)
    logger.info(
        f"required video duration: {required_video_duration:.2f} seconds "
        f"(audio duration + {_VIDEO_DURATION_SAFETY_MARGIN:.2f}s safety margin)"
    )

    # 兼容 API 直接调用时未传转场模式的情况，避免后续访问 .value 时崩溃。
    transition_value = getattr(video_transition_mode, "value", video_transition_mode)
    normalized_clip_speed = utils.normalize_clip_speed(clip_speed)
    if normalized_clip_speed != 1.0:
        # 只记录一次最终生效值，既方便定位 API 越界参数被归一化的问题，
        # 也避免在逐片段热路径中重复输出相同日志。
        logger.info(f"clip playback speed: {normalized_clip_speed:.2f}x")
    # max_clip_duration 约束的是成片里的最终播放时长，而不是源视频读取时长。
    # MoviePy 以 0.5 倍速播放 1.5 秒源画面会得到 3 秒片段，以 2 倍速播放
    # 6 秒源画面同样会得到 3 秒片段。因此切片前必须按速度反推源时长；如果
    # 仍固定读取 3 秒再慢放、裁剪，下一段却从源视频第 3 秒开始，会跳过中间
    # 1.5 秒画面。该计算同时保证不同速度下的源时间线连续且无重叠。
    source_clip_duration = max_clip_duration * normalized_clip_speed
    # Mix 模式的交叉溶解重叠时长必须严格小于单片段时长，否则累计进度
    # 为零，循环补足素材时长时会陷入无限循环。这里统一收敛到安全上限，
    # 所有后续时长计算、转场效果和拼接 padding 都使用同一个收敛值。
    effective_mix_overlap = min(
        max(float(mix_overlap_duration or 0.0), 0.0),
        max(float(max_clip_duration) - 0.05, 0.0),
    )
    if (
        transition_value == VideoTransitionMode.mix.value
        and effective_mix_overlap != float(mix_overlap_duration or 0.0)
    ):
        logger.warning(
            f"mix overlap duration clamped from {mix_overlap_duration}s "
            f"to {effective_mix_overlap:.2f}s (must be smaller than clip duration)"
        )
    output_dir = os.path.dirname(combined_video_path)

    aspect = VideoAspect(video_aspect)
    video_width, video_height = aspect.to_resolution()

    processed_clips = []
    subclipped_items = []
    video_duration = 0
    for video_path in video_paths:
        # 损坏或无法探测的素材只跳过该文件，不中断整个合成（与后面逐片段
        # 处理时的容错语义保持一致）。探测走 ffprobe 轻量路径，避免每个
        # 素材都被完整打开两次。
        try:
            metadata = _probe_video_metadata(video_path)
            if metadata is None:
                logger.warning(
                    f"failed to probe material video, skipping: {os.path.basename(video_path)}"
                )
                continue
            clip_duration, clip_w, clip_h = metadata
        except Exception as probe_error:
            logger.warning(
                f"failed to probe material video, skipping: "
                f"{os.path.basename(video_path)}, "
                f"error={type(probe_error).__name__}: {probe_error}"
            )
            continue

        start_time = 0

        while start_time < clip_duration:
            end_time = min(start_time + source_clip_duration, clip_duration)

            # 保留所有有效分段。
            # 这样既不会丢掉“整段视频本身就短于 max_clip_duration”的素材，
            # 也不会吞掉长视频最后剩下的一小段尾部内容。
            if end_time > start_time:
                subclipped_items.append(
                    SubClippedVideoClip(
                        file_path=video_path,
                        start_time=start_time,
                        end_time=end_time,
                        width=clip_w,
                        height=clip_h,
                        source_file_path=video_path,
                    )
                )

            start_time = end_time
            if video_concat_mode.value == VideoConcatMode.sequential.value:
                break

    subclipped_items = _prioritize_unique_source_clips(
        subclipped_items=subclipped_items,
        concat_mode=video_concat_mode,
    )

    logger.debug(f"total subclipped items: {len(subclipped_items)}")

    video_duration = 0.0
    transition_times = []

    # Add downloaded clips over and over until the duration of the audio (max_duration) has been reached
    for i, subclipped_item in enumerate(subclipped_items):
        if video_duration >= required_video_duration:
            break

        logger.debug(
            f"processing clip {i + 1}: {subclipped_item.width}x{subclipped_item.height}, "
            f"source: {os.path.basename(subclipped_item.source_file_path)}, "
            f"current duration: {video_duration:.2f}s, "
            f"remaining: {required_video_duration - video_duration:.2f}s"
        )

        try:
            clip = None
            clip = _open_video_clip_quietly(subclipped_item.file_path)
            clip = clip.subclipped(
                subclipped_item.start_time, subclipped_item.end_time
            )
            # 播放速度属于素材本身属性，应在转场前应用。这样 Fade/Slide 等一秒转场
            # 不会跟随素材速度变成 0.5 秒或 2 秒；后续最大时长裁剪继续作为
            # 浮点误差或异常素材时长的安全兜底，保证最终片段不突破配置上限。
            if normalized_clip_speed != 1.0:
                clip = clip.with_speed_scaled(normalized_clip_speed)
            clip_duration = clip.duration
            # Not all videos are same size, so we need to resize them
            clip_w, clip_h = clip.size
            if clip_w != video_width or clip_h != video_height:
                clip_ratio = clip.w / clip.h
                video_ratio = video_width / video_height
                logger.debug(
                    f"resizing clip, source: {clip_w}x{clip_h}, ratio: {clip_ratio:.2f}, target: {video_width}x{video_height}, ratio: {video_ratio:.2f}"
                )

                if clip_ratio == video_ratio:
                    clip = clip.resized(new_size=(video_width, video_height))
                else:
                    if clip_ratio > video_ratio:
                        scale_factor = video_width / clip_w
                    else:
                        scale_factor = video_height / clip_h

                    new_width = int(clip_w * scale_factor)
                    new_height = int(clip_h * scale_factor)

                    # 画幅不匹配时用“模糊填充背景”替代纯黑底：背景放大到完整
                    # 覆盖目标画幅后模糊压暗，前景保持原有的缩放居中。模糊背景
                    # 构建失败时回退到黑底，绝不阻塞合成。
                    try:
                        background = _build_blurred_background(
                            clip, video_width, video_height
                        )
                    except Exception as blur_error:
                        logger.warning(
                            "failed to build blurred background, falling back to black: "
                            f"{type(blur_error).__name__}: {blur_error}"
                        )
                        background = ColorClip(
                            size=(video_width, video_height), color=(0, 0, 0)
                        ).with_duration(clip_duration)
                    clip_resized = clip.resized(
                        new_size=(new_width, new_height)
                    ).with_position("center")
                    # 背景 cover 尺寸会大于目标画幅（超出部分被裁切），必须显式
                    # 传入 size，否则 CompositeVideoClip 会自动膨胀到最大子片尺寸。
                    clip = CompositeVideoClip(
                        [background, clip_resized], size=(video_width, video_height)
                    )

            shuffle_side = random.choice(["left", "right", "top", "bottom"])
            if transition_value in (
                None,
                VideoTransitionMode.none.value,
                VideoTransitionMode.mix.value,
            ):
                clip = clip
            elif transition_value == VideoTransitionMode.fade_in.value:
                clip = video_effects.fadein_transition(clip, 1)
            elif transition_value == VideoTransitionMode.fade_out.value:
                clip = video_effects.fadeout_transition(clip, 1)
            elif transition_value == VideoTransitionMode.slide_in.value:
                clip = video_effects.slidein_transition(clip, 1, shuffle_side)
            elif transition_value == VideoTransitionMode.slide_out.value:
                clip = video_effects.slideout_transition(clip, 1, shuffle_side)
            elif transition_value == VideoTransitionMode.zoom_in.value:
                clip = video_effects.zoomin_transition(clip, 1)
            elif transition_value == VideoTransitionMode.zoom_out.value:
                clip = video_effects.zoomout_transition(clip, 1)
            elif transition_value == VideoTransitionMode.shuffle.value:
                transition_funcs = [
                    lambda c: video_effects.fadein_transition(c, 1),
                    lambda c: video_effects.fadeout_transition(c, 1),
                    lambda c: video_effects.slidein_transition(c, 1, shuffle_side),
                    lambda c: video_effects.slideout_transition(c, 1, shuffle_side),
                    lambda c: video_effects.zoomin_transition(c, 1),
                    lambda c: video_effects.zoomout_transition(c, 1),
                ]
                shuffle_transition = random.choice(transition_funcs)
                clip = shuffle_transition(clip)
            elif transition_value == VideoTransitionMode.auto.value:
                transition_sequence = [
                    lambda c: video_effects.fadein_transition(c, 1),
                    lambda c: video_effects.zoomin_transition(c, 1),
                    lambda c: video_effects.fadeout_transition(c, 1),
                    lambda c: video_effects.slidein_transition(c, 1, shuffle_side),
                ]
                auto_transition = transition_sequence[i % len(transition_sequence)]
                clip = auto_transition(clip)

            if clip.duration > max_clip_duration:
                clip = clip.subclipped(0, max_clip_duration)

            # wirte clip to temp file
            clip_file = f"{output_dir}/temp-clip-{i + 1}.mp4"
            _write_videofile_with_codec_fallback(
                clip,
                clip_file,
                codec=_get_configured_video_codec(),
                logger=None,
                fps=fps,
                threads=threads,
                **_get_video_encode_args(),
            )

            # Store clip duration before closing
            clip_duration_saved = clip.duration
            close_clip(clip)

            processed_clips.append(
                SubClippedVideoClip(
                    file_path=clip_file,
                    duration=clip_duration_saved,
                    width=clip_w,
                    height=clip_h,
                    source_file_path=subclipped_item.source_file_path,
                )
            )
            if (
                transition_value != VideoTransitionMode.none.value
                and transition_value is not None
                and i > 0
            ):
                transition_times.append(video_duration)

            if transition_value == VideoTransitionMode.mix.value and i > 0:
                video_duration += clip_duration_saved - effective_mix_overlap
            else:
                video_duration += clip_duration_saved

        except Exception as e:
            if clip is not None:
                # 失败路径也关闭本次打开的 MoviePy clip，避免文件句柄泄漏；
                # close_clip 本身容错，不会抛出。
                close_clip(clip)
            logger.error(f"failed to process clip: {str(e)}")

    # loop processed clips until the video duration covers the audio duration and the small safety margin.
    if video_duration < required_video_duration:
        logger.warning(
            f"video duration ({video_duration:.2f}s) is shorter than required duration "
            f"({required_video_duration:.2f}s), looping clips to match audio length."
        )
        base_clips = processed_clips.copy()
        for clip in itertools.cycle(base_clips):
            if video_duration >= required_video_duration:
                break

            # Transition happens exactly at the current accumulated video_duration
            if (
                transition_value != VideoTransitionMode.none.value
                and transition_value is not None
            ):
                transition_times.append(video_duration)

            processed_clips.append(clip)
            if transition_value == VideoTransitionMode.mix.value:
                # 重叠时长已在上方收敛到严格小于片段时长；这里再兜底一次，
                # 防止极端输入（如异常短的片段）让累计进度停滞导致死循环。
                progress = clip.duration - effective_mix_overlap
                if progress <= 0:
                    logger.warning(
                        f"mix overlap {effective_mix_overlap:.2f}s is not smaller than "
                        f"clip duration {clip.duration:.2f}s, stop looping clips"
                    )
                    break
                video_duration += progress
            else:
                video_duration += clip.duration
        logger.info(
            f"video duration: {video_duration:.2f}s, audio duration: {audio_duration:.2f}s, "
            f"required duration: {required_video_duration:.2f}s, "
            f"looped {len(processed_clips) - len(base_clips)} clips"
        )

    # merge video clips progressively, avoid loading all videos at once to avoid memory overflow
    logger.info("starting clip merging process")
    if not processed_clips:
        logger.warning("no clips available for merging")
        raise RuntimeError("no clips available for merging")

    clip_files = [clip.file_path for clip in processed_clips]

    intermediate_files: List[str] = []
    try:
        if transition_value == VideoTransitionMode.mix.value:
            logger.info(
                f"concatenating {len(clip_files)} clips using MoviePy cross-dissolve overlap"
            )
            # 分块混合拼接，避免长视频把所有片段一次性加载进内存。每组
            # （<= _MIX_CONCAT_CHUNK_SIZE）拼接后立即写出临时文件并关闭句柄，
            # 下一轮再把这些临时文件分组继续拼，直到只剩一个文件写出最终成片。
            current_files = clip_files
            chunk_counter = 0
            while len(current_files) > _MIX_CONCAT_CHUNK_SIZE:
                next_files: List[str] = []
                for i in range(0, len(current_files), _MIX_CONCAT_CHUNK_SIZE):
                    group = current_files[i : i + _MIX_CONCAT_CHUNK_SIZE]
                    chunk_counter += 1
                    group_output = os.path.join(
                        output_dir, f"mix-chunk-{chunk_counter}.mp4"
                    )
                    _mix_concat_group(
                        group,
                        effective_mix_overlap,
                        _get_configured_video_codec(),
                        fps,
                        threads,
                        group_output,
                    )
                    intermediate_files.append(group_output)
                    next_files.append(group_output)
                current_files = next_files

            _mix_concat_group(
                current_files,
                effective_mix_overlap,
                _get_configured_video_codec(),
                fps,
                threads,
                combined_video_path,
            )
        else:
            logger.info(f"concatenating {len(clip_files)} clips with ffmpeg")
            concat_video_clips_with_ffmpeg(
                clip_files=clip_files,
                output_file=combined_video_path,
                threads=threads,
                output_dir=output_dir,
                max_duration=audio_duration,
            )
    except Exception:
        # 任何合并/编码异常都先清理中间分块文件与素材临时文件，避免失败任务
        # 留下 mix-chunk-*.mp4 与 temp-clip-*.mp4 垃圾；clip 句柄由
        # _mix_concat_group 的 finally 统一释放。异常继续上抛给调用方。
        delete_files(intermediate_files)
        delete_files(clip_files)
        raise

    # clean temp files
    delete_files(intermediate_files)
    delete_files(clip_files)

    if transition_times:
        try:
            with open(combined_video_path + ".transitions.json", "w") as f:
                json.dump(transition_times, f)
        except Exception as e:
            logger.warning(f"failed to save transition times: {e}")

    logger.info("video combining completed")
    return combined_video_path


def wrap_text(text, max_width, font="Arial", fontsize=60):
    # 字幕换行必须在真正创建 TextClip 前完成，否则 MoviePy 只会按原始文本
    # 计算渲染区域。这里用 PIL 按当前字体和字号测量宽度，确保每一行都尽量
    # 控制在视频可用宽度内，避免大字号或中文长句直接溢出画面。
    # 实现委托给 subtitle_engine（同一算法，逐点保持历史行为）。
    from app.services.subtitle_engine.text import wrap_text as _wrap_text

    return _wrap_text(
        text,
        max_width=max_width,
        font=font,
        fontsize=fontsize,
    )


# ---------------------------------------------------------------------------
# 字幕动画（动态字号 / 弹出弹跳 / 漂浮运动）
#
# 这些纯函数的实现全部委托给 subtitle_engine 模块，曲线与历史行为逐点
# 一致（弹出弹跳 0 -> 1.1 -> 1.0 分段线性、漂浮正弦、动态字号按真实宽度
# 缩放）。保留在 video.py 的包装层是为了兼容既有调用方与测试；下划线常量
# 是引擎常量的别名，历史测试与调用方仍按旧名字引用。
# ---------------------------------------------------------------------------
from app.services.subtitle_engine.animation import (  # noqa: E402
    DYNAMIC_MAX_LINES as _SUBTITLE_DYNAMIC_MAX_LINES,  # noqa: F401
    DYNAMIC_SCALE_MAX_RATIO as _SUBTITLE_DYNAMIC_MAX_RATIO,  # noqa: F401
    DYNAMIC_SCALE_MIN_RATIO as _SUBTITLE_DYNAMIC_MIN_RATIO,  # noqa: F401
    KINETIC_FLOAT_AMPLITUDE_RATIO as _SUBTITLE_FLOAT_AMPLITUDE_RATIO,  # noqa: F401
    KINETIC_FLOAT_MIN_AMPLITUDE as _SUBTITLE_FLOAT_MIN_AMPLITUDE,  # noqa: F401
    POP_IN_WINDOW_SECONDS as _SUBTITLE_POP_IN_WINDOW_SECONDS,  # noqa: F401
)

# 字幕超采样：内部以 2 倍分辨率渲染字幕层，再 LANCZOS 降采样回目标尺寸，
# 消除 PIL 直接绘制小字号时的锯齿边缘。MoviePy 的 Resize 用 PIL LANCZOS
# 重采样，transform 后 clip.size 会按实际帧自动更新，位置计算不受影响。
_SUBTITLE_SUPERSAMPLE_SCALE = 2


def subtitle_pop_in_scale(t: float, duration: float) -> float:
    """返回时刻 ``t`` 的弹出弹跳缩放系数（0.0 -> 1.1 -> 1.0）。

    曲线是确定性的分段线性函数：前半程从 0 放大到 1+overshoot，
    后半程回落并稳定到 1.0。字幕时长短于动画窗口时，把整个曲线压缩到
    字幕时长内，避免短字幕来不及完成动画；动画窗口之后的时刻恒为 1.0。
    """
    from app.services.subtitle_engine.animation import pop_in_scale as _pop

    return _pop(t, duration)


def subtitle_float_offset(t: float, duration: float, video_height: int) -> float:
    """返回时刻 ``t`` 的漂浮垂直偏移（像素），正值表示向下。

    使用正弦曲线在 +amp / -amp 之间平滑往返一次，起点和终点都在 0，
    视觉上是缓慢的“呼吸式”浮动而不是漂移出画面。幅度限制在视频高度
    的一小部分，保证任何字幕位置（顶/底/自定义）都不会飘出安全区。
    """
    from app.services.subtitle_engine.animation import (
        kinetic_float_offset as _kinetic,
    )

    return _kinetic(t, duration, video_height)[0]


def dynamic_subtitle_font_size(
    phrase: str,
    base_size: int,
    max_width: int,
    font_path: str,
) -> int:
    """按实际文本宽度计算动态字号。

    单行文本剩余空间充足（宽度小于可用宽度的 60%）时按比例放大，上限
    1.5 倍（"STOP!" 这类短语会得到最大冲击力）；需要换行的长文本则逐步
    缩小字号，直到换行后不超过两行或触及 0.6 倍下限。测量用 PIL 按当前
    字体真实测量，与 ``wrap_text`` 的换行基准一致。
    """
    from app.services.subtitle_engine.animation import (
        dynamic_font_size as _dynamic,
    )

    return _dynamic(phrase, base_size, max_width, font_path)


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    # 字幕背景色来自 API/WebUI 参数，可能为空或格式不规范。这里统一只接受
    # #RRGGBB 形式，非法值回退为黑色，避免 PIL 渲染阶段抛出异常中断任务。
    from app.services.subtitle_engine.renderer import (
        _hex_to_rgb as _engine_hex_to_rgb,
    )

    return _engine_hex_to_rgb(color)


def _rounded_subtitle_background_clip(
    width: int,
    height: int,
    color: str,
    alpha: int = 140,
    radius: int = 16,
) -> ImageClip:
    # 新字幕背景仅在用户显式开启时使用：通过 RGBA 图片绘制圆角半透明底板，
    # 再交给 MoviePy 作为透明 ImageClip 参与合成。这样默认路径完全不变，
    # 同时可以低成本试验更柔和的字幕视觉效果。
    from app.services.subtitle_engine.renderer import (
        _rounded_subtitle_background_clip as _engine_bg_clip,
    )

    return _engine_bg_clip(
        width=width,
        height=height,
        color=color,
        alpha=alpha,
        radius=radius,
    )


def _get_visible_center_position(
    text_clip: TextClip,
    container_width: int,
    container_height: int,
) -> tuple[int, int]:
    """
    按文字真实可见像素把 TextClip 放到背景容器中心。

    MoviePy 的 TextClip 会按字体行高和 baseline 创建透明画布。很多字体的
    可见字形并不在这个画布的几何中心，直接 `with_position("center")`
    会把整块透明画布居中，导致字幕看起来偏上或偏下。这里读取 TextClip
    的透明 mask，只根据实际有像素的 bbox 计算偏移，让用户看到的文字
    在字幕背景里视觉居中。
    """
    from app.services.subtitle_engine.renderer import (
        _get_visible_center_position as _engine_center,
    )

    return _engine_center(text_clip, container_width, container_height)


def subtitle_colors_are_indistinguishable(params: VideoParams) -> bool:
    """判断字幕文字和背景是否同色，提醒用户可能无法看清字幕。"""
    if not params.subtitle_enabled or not params.text_background_color:
        return False

    def normalize_color(value):
        if isinstance(value, bool):
            return "#000000" if value else ""
        return str(value or "").strip().lower()

    text_color = normalize_color(params.text_fore_color)
    background_color = normalize_color(params.text_background_color)
    return bool(text_color and text_color == background_color)


@lru_cache(maxsize=64)
def _subtitle_font_supports_sample(font_path: str, sample: str) -> bool:
    """检查字体是否包含样本文字需要的字形，并缓存重复检查结果。"""
    try:
        font = ImageFont.truetype(font_path, 30)
        missing_mask = font.getmask("\U0010ffff")
        missing_signature = (
            missing_mask.size,
            missing_mask.getbbox(),
            bytes(missing_mask),
        )
        for char in sample:
            char_mask = font.getmask(char)
            char_signature = (
                char_mask.size,
                char_mask.getbbox(),
                bytes(char_mask),
            )
            if char_mask.getbbox() is None or char_signature == missing_signature:
                return False
        return True
    except Exception as e:
        # 字体探测失败不应阻止用户生成；保留日志供环境兼容问题排查。
        logger.warning(f"failed to inspect subtitle font glyphs: {font_path}, {e}")
        return True


def subtitle_font_supports_text(font_path: str, text: str) -> bool:
    """检查字体能否绘制文本中的字母和数字，忽略空白及标点符号。"""
    sample = "".join(
        dict.fromkeys(
            char
            for char in str(text or "")
            if unicodedata.category(char)[0] in {"L", "N"}
        )
    )[:64]
    if not sample:
        return True
    return _subtitle_font_supports_sample(font_path, sample)


SUBTITLE_CASING_ORIGINAL = "original"
SUBTITLE_CASING_UPPER = "upper"
SUBTITLE_CASING_TITLE = "title"
SUBTITLE_CASING_LOWER = "lower"


def apply_subtitle_casing(text: str, mode: str | None) -> str:
    """在字幕渲染前应用文本大小写转换。

    支持 ``original``（保持原样）、``upper``（全大写）、``title``（标题式）
    和 ``lower``（全小写）等历史模式，以及新式枚举模式（sentence_case /
    title_case / as_spoken / uppercase / lowercase）。转换只作用于文本本身，
    不改变 SRT 时间轴、字体、描边或换行逻辑——phrase 宽度仍由 wrap_text
    按转换后的文本重新测量，因此描边轮廓和文本框会自动适配新的大小写。
    """
    from app.services.subtitle_engine.styles import (
        apply_subtitle_casing as _apply_casing,
    )

    return _apply_casing(text, mode)


def _wrap_overlay_text(
    text: str, font_path: str, font_size: int, max_width: int
) -> str:
    """按实际字体宽度把叠加层文案换行，供标题卡/事实卡复用。"""
    from app.services.subtitle_engine.text import wrap_text

    wrapped, _ = wrap_text(
        text,
        max_width=max_width,
        font=font_path,
        fontsize=int(font_size),
    )
    return wrapped or text


def _build_overlay_card_clip(
    item,
    font_path: str,
    video_width: int,
    video_height: int,
    params: VideoParams,
    overlay_enabled: bool,
) -> ImageClip | None:
    """
    把单个 OverlayItem 渲染成透明 ImageClip（圆角底板 + 文字）。

    - title / callout 放顶部居中，事实卡放下三分之一处。
    - 每个卡片只在 start~end 时间窗内可见；opacity 由
      overlay_image_opacity 与 item 类型决定，异常时整个卡片跳过。
    """
    from app.services.subtitle_engine.renderer import _rounded_subtitle_background_clip as _bg

    from app.services.overlay import (
        CALLOUT_POSITION,
        FACT_POSITION,
        TITLE_POSITION,
    )

    if not overlay_enabled:
        return None

    font_size = int(video_width * 0.05) if item.kind == "title" else int(video_width * 0.038)
    max_width = int(video_width * 0.82)
    wrapped = _wrap_overlay_text(item.text, font_path, font_size, max_width)
    if not wrapped.strip():
        return None

    pad_x = int(font_size * 0.4)
    pad_y = int(font_size * 0.3)

    try:
        txt = TextClip(
            text=wrapped,
            font=font_path,
            font_size=font_size,
            color=getattr(params, "overlay_text_color", "#FFFFFF"),
        )
        text_w, text_h = txt.size
        bg = _bg(
            width=text_w + 2 * pad_x,
            height=text_h + 2 * pad_y,
            color=getattr(params, "overlay_bg_color", "#000000"),
            alpha=160,
            radius=max(8, int(font_size * 0.3)),
        )
        centered = CompositeVideoClip([bg, txt.with_position("center")])
        centered = centered.with_duration(max(0.1, item.end - item.start))
        centered = centered.with_start(item.start)

        if item.position == TITLE_POSITION:
            centered = centered.with_position(("center", int(video_height * 0.06)))
        elif item.position == CALLOUT_POSITION:
            centered = centered.with_position(("center", int(video_height * 0.1)))
        elif item.position == FACT_POSITION:
            centered = centered.with_position(("center", int(video_height * 0.72)))
        else:
            centered = centered.with_position(("center", int(video_height * 0.72)))
        centered.close = centered.close  # keep refcount balanced under ExitStack

        opacity = float(
            getattr(params, "overlay_image_opacity", 0.85) or 0.85
        )
        if opacity < 1.0:
            centered = centered.with_opacity(opacity)
        return centered
    except Exception as e:
        logger.warning(f"failed to build overlay card '{item.text}': {e}")
        return None


def build_overlay_clips(
    overlay_items: List,
    font_path: str,
    video_width: int,
    video_height: int,
    params: VideoParams,
) -> List:
    """把所有 OverlayItem 渲染成可合成的透明片段列表。"""
    overlay_enabled = getattr(params, "overlay_enabled", False)
    if not overlay_enabled or not overlay_items:
        return []
    clips = []
    for item in overlay_items:
        clip = _build_overlay_card_clip(
            item,
            font_path,
            video_width,
            video_height,
            params,
            overlay_enabled,
        )
        if clip is not None:
            clips.append(clip)
    return clips


def build_overlay_image_clip(
    image_path: str,
    video_width: int,
    video_height: int,
    params: VideoParams,
    total_duration: float,
) -> ImageClip | None:
    """
    把角落叠加图（logo/水印/装饰图）缩放后放到右上角，覆盖整个成片时长。

    图片不存在或加载失败时返回 None，绝不中断视频合成。
    """
    if not image_path or not os.path.exists(image_path):
        return None
    try:
        img = ImageClip(image_path, transparent=True)
        img_w, img_h = img.size
        target_w = int(video_width * 0.16)
        scale = max(target_w / max(img_w, 1), 1e-6)
        img = img.resize(width=int(img_w * scale), height=int(img_h * scale))
        img = img.with_duration(max(total_duration, 0.1))
        img = img.with_position(("right", int(video_height * 0.05)))
        opacity = float(getattr(params, "overlay_image_opacity", 0.85) or 0.85)
        if opacity < 1.0:
            img = img.with_opacity(opacity)
        return img
    except Exception as e:
        logger.warning(f"failed to load overlay image {image_path}: {e}")
        return None


def generate_video(
    video_path: str,
    audio_path: str,
    subtitle_path: str,
    output_file: str,
    params: VideoParams,
    bgm_file_override: str | None = None,
) -> bool:
    """
    合成最终视频，并返回本次背景音乐处理是否成功。

    返回值只描述 BGM 处理状态：没有请求 BGM 或成功混合时返回 True；请求了
    BGM 但加载、特效或混合失败时返回 False。即使 BGM 失败仍会继续输出只有
    旁白的视频，让任务编排层决定是否向用户展示降级警告。
    """
    aspect = VideoAspect(params.video_aspect)
    video_width, video_height = aspect.to_resolution()

    logger.info(f"generating video: {video_width} x {video_height}")
    logger.info(f"  ① video: {video_path}")
    logger.info(f"  ② audio: {audio_path}")
    logger.info(f"  ③ subtitle: {subtitle_path}")
    logger.info(f"  ④ output: {output_file}")

    # https://github.com/Papiwrld/reelsync/issues/217
    # PermissionError: [WinError 32] The process cannot access the file because it is being used by another process: 'final-1.mp4.tempTEMP_MPY_wvf_snd.mp3'
    # write into the same directory as the output file
    output_dir = os.path.dirname(output_file)

    font_path = ""
    renderer = None
    word_timings = []
    if params.subtitle_enabled:
        # 字幕样式与字体解析统一交给字幕引擎：注册表自动回退缺失字体
        # （旧配置引用的 STHeiti 系列已从仓库移除），永不因字体崩溃。
        from app.services.subtitle_engine.fonts import get_font_registry
        from app.services.subtitle_engine.renderer import SubtitleRenderer
        from app.services.subtitle_engine.styles import SubtitleStyleResolver
        from app.services.subtitle_engine.timing import (
            load_word_timings_from_json,
        )

        style_resolver = SubtitleStyleResolver(font_registry=get_font_registry())
        style = style_resolver.resolve(params)
        font_path = style.font_path
        if not font_path:
            font_path = os.path.join(
                utils.font_dir(), "Montserrat-Bold.ttf"
            )
        if os.name == "nt":
            font_path = font_path.replace("\\", "/")

        logger.info(f"  ⑤ font: {font_path}")

        renderer = SubtitleRenderer(
            style=style,
            video_width=video_width,
            video_height=video_height,
            font_registry=style_resolver.font_registry,
            text_clip_factory=TextClip,
            composite_clip_factory=CompositeVideoClip,
            image_clip_factory=ImageClip,
        )
        if subtitle_path:
            word_timings = load_word_timings_from_json(subtitle_path + ".words.json")


    # MoviePy 的 CompositeAudioClip.close() 不会关闭子 AudioFileClip。这里用
    # ExitStack 显式持有所有原始文件 reader，确保成功、字幕异常、混音失败和
    # 视频写入失败等路径都能释放 FFmpeg 子进程，尤其避免 Windows 文件被占用。
    with ExitStack() as clip_stack:
        source_video_clip = clip_stack.enter_context(
            _open_video_clip_quietly(video_path)
        )
        voice_source_clip = clip_stack.enter_context(AudioFileClip(audio_path))
        video_clip = source_video_clip
        audio_clip = voice_source_clip.with_effects(
            [afx.MultiplyVolume(params.voice_volume)]
        )

        def make_textclip(text):
            return TextClip(
                text=apply_subtitle_casing(
                    text, getattr(params, "subtitle_casing", None)
                ),
                font=font_path,
                font_size=params.font_size,
            )

        if subtitle_path and os.path.exists(subtitle_path) and renderer is not None:
            sub = clip_stack.enter_context(
                SubtitlesClip(
                    subtitles=subtitle_path,
                    encoding="utf-8",
                    make_textclip=make_textclip,
                )
            )
            text_clips = []
            for item in sub.subtitles:
                # 每个字幕帧交给字幕引擎渲染（含预设样式、定位、动画与
                # 逐词高亮）。缺失/损坏的词语时间轴只影响高亮，不阻断渲染。
                clip = renderer.render(item, word_timings)
                if clip is None:
                    continue
                if isinstance(clip, list):
                    text_clips.extend(clip)
                else:
                    text_clips.append(clip)
            video_clip = CompositeVideoClip([video_clip, *text_clips])
            clip_stack.callback(video_clip.close)

        # 图文叠加层：标题卡 / 事实卡 / callout 按字幕时间轴对齐渲染。
        if getattr(params, "overlay_enabled", False):
            try:
                from app.services.overlay import (
                    build_overlay_plan,
                    parse_subtitle_phrases,
                )

                overlay_font_path = font_path
                if not overlay_font_path:
                    from app.services.subtitle_engine.fonts import get_font_registry
                    from app.services.subtitle_engine.styles import SubtitleStyleResolver

                    style_resolver = SubtitleStyleResolver(
                        font_registry=get_font_registry()
                    )
                    overlay_font_path = style_resolver.resolve(params).font_path
                    if not overlay_font_path:
                        overlay_font_path = os.path.join(
                            utils.font_dir(), "Montserrat-Bold.ttf"
                        )
                    if os.name == "nt":
                        overlay_font_path = overlay_font_path.replace("\\", "/")

                subtitle_lines = []
                if subtitle_path and os.path.exists(subtitle_path):
                    from app.services.subtitle import file_to_subtitles

                    subtitle_lines = file_to_subtitles(subtitle_path)
                overlay_items = build_overlay_plan(
                    params,
                    subject=params.video_subject or "",
                    script=params.video_script or "",
                    subtitle_phrases=parse_subtitle_phrases(subtitle_lines),
                    video_duration=audio_clip.duration,
                )
                overlay_clips = build_overlay_clips(
                    overlay_items,
                    font_path=overlay_font_path,
                    video_width=video_width,
                    video_height=video_height,
                    params=params,
                )
                overlay_image = getattr(params, "overlay_image", None)
                if overlay_image:
                    image_clip = build_overlay_image_clip(
                        overlay_image,
                        video_width,
                        video_height,
                        params,
                        total_duration=audio_clip.duration,
                    )
                    if image_clip is not None:
                        overlay_clips.append(image_clip)
                if overlay_clips:
                    video_clip = CompositeVideoClip([video_clip, *overlay_clips])
                    clip_stack.callback(video_clip.close)
            except Exception as e:
                logger.warning(f"failed to composite overlays: {e}")

        ducking_timestamps = []
        if subtitle_path and os.path.exists(subtitle_path):
            # 复用上面已解析的 SubtitlesClip（带 make_textclip），不要再新建一个
            # 不带 make_textclip 的实例——moviepy 会因 “Argument font is required
            # if make_textclip is None” 拒绝解析，导致 ducking 时间轴永远为空。
            try:
                for item in sub.subtitles:
                    ducking_timestamps.append((item[0][0], item[0][1]))
            except Exception as e:
                logger.warning(f"failed to load subtitle timings for ducking: {e}")

        ducking_effects = []
        if params.audio_ducking_enabled and ducking_timestamps:
            for start, end in ducking_timestamps:
                ducking_effects.append(
                    afx.MultiplyVolume(
                        params.audio_ducking_intensity,
                        start_time=max(0, start - 0.2),
                        end_time=end + 0.2,
                    )
                )

        sfx_clips_to_merge = []
        transition_times = []
        transitions_file = video_path + ".transitions.json"
        try:
            with open(transitions_file, "r") as f:
                transition_times = json.load(f)
            os.remove(transitions_file)
        except Exception:
            pass

        # Load Transition SFX
        transitions_dir = os.path.join(utils.resource_dir(), "sfx", "transitions")
        if os.path.exists(transitions_dir):
            sfx_files = [
                os.path.join(transitions_dir, f)
                for f in os.listdir(transitions_dir)
                if f.endswith((".mp3", ".wav", ".ogg"))
            ]
            if sfx_files and transition_times:
                for t_time in transition_times:
                    sfx_path = random.choice(sfx_files)
                    try:
                        sfx_clip = clip_stack.enter_context(AudioFileClip(sfx_path))
                        # SFX starts slightly before transition overlap
                        sfx_clip = sfx_clip.with_start(
                            max(0, t_time - 0.5)
                        ).with_effects([afx.MultiplyVolume(params.sfx_volume)])
                        sfx_clips_to_merge.append(sfx_clip)
                    except Exception as e:
                        logger.warning(f"failed to load transition sfx {sfx_path}: {e}")

        # Load Atmosphere
        atmosphere_dir = os.path.join(utils.resource_dir(), "sfx", "atmosphere")
        if params.atmosphere_enabled and os.path.exists(atmosphere_dir):
            atmo_files = [
                os.path.join(atmosphere_dir, f)
                for f in os.listdir(atmosphere_dir)
                if f.endswith((".mp3", ".wav", ".ogg"))
            ]
            if atmo_files:
                atmo_path = random.choice(atmo_files)
                try:
                    atmo_clip = clip_stack.enter_context(AudioFileClip(atmo_path))
                    atmo_effects = [
                        afx.MultiplyVolume(params.atmosphere_volume),
                        afx.AudioLoop(duration=video_clip.duration),
                    ]
                    atmo_effects.extend(ducking_effects)
                    atmo_clip = atmo_clip.with_effects(atmo_effects)
                    sfx_clips_to_merge.append(atmo_clip)
                except Exception as e:
                    logger.warning(f"failed to load atmosphere {atmo_path}: {e}")

        bgm_enabled = bgm_service.should_use_bgm(params.bgm_type, params.bgm_volume)
        if not bgm_enabled and params.bgm_type:
            logger.info(
                f"skipping background music because volume is not positive: "
                f"type={params.bgm_type}, volume={params.bgm_volume}"
            )

        bgm_file = ""
        if bgm_enabled:
            bgm_file = (
                bgm_file_override
                if bgm_file_override is not None
                else get_bgm_file(
                    bgm_type=params.bgm_type,
                    bgm_file=params.bgm_file,
                )
            )
        bgm_mix_succeeded = True
        bgm_clip = None
        if bgm_file:
            try:
                bgm_effects = [
                    afx.MultiplyVolume(params.bgm_volume),
                    afx.AudioFadeOut(3),
                ]
                bgm_effects.extend(ducking_effects)

                if bgm_file_override is None:
                    bgm_effects.append(afx.AudioLoop(duration=video_clip.duration))
                bgm_source_clip = clip_stack.enter_context(AudioFileClip(bgm_file))
                bgm_clip = bgm_source_clip.with_effects(bgm_effects)
            except Exception:
                bgm_mix_succeeded = False
                logger.exception(
                    f"failed to mix background music: type={params.bgm_type}, "
                    f"file={bgm_file}"
                )

        # Build final audio bus
        master_audio_list = [audio_clip]
        if bgm_clip:
            master_audio_list.append(bgm_clip)
        master_audio_list.extend(sfx_clips_to_merge)
        if len(master_audio_list) == 1:
            # 只有旁白时直接使用原 clip，避免无意义的混音包装引入额外的
            # 重采样和复合开销；也保证 BGM/SFX 全部失败时输出保持纯净。
            audio_clip = master_audio_list[0]
        else:
            audio_clip = CompositeAudioClip(master_audio_list)

        final_video_clip = video_clip.with_audio(audio_clip)
        clip_stack.callback(final_video_clip.close)
        # 显式沿用输入音频的采样率；如果取不到，再回退 MoviePy 默认的 44100Hz。
        # 这样可以减少不同环境，尤其 Docker 中再次重采样带来的音质波动。
        output_audio_fps = int(getattr(audio_clip, "fps", 0) or 44100)

        # 先写入临时文件再原子改名：编码失败时不会在最终路径留下半成品，
        # 任务编排层不会把损坏的 final-*.mp4 当成有效产物。临时文件名以
        # ".tmp" 开头但保留真实扩展名结尾，ffmpeg 才能正确推断封装格式；
        # WebUI 的 final-<index>.<ext> 全匹配规则也不会把它误判成成片。
        temp_output_file = output_file + ".tmp.mp4"
        try:
            _write_videofile_with_codec_fallback(
                final_video_clip,
                output_file=temp_output_file,
                codec=_get_configured_video_codec(),
                audio_codec=audio_codec,
                audio_fps=output_audio_fps,
                audio_bitrate=audio_bitrate,
                temp_audiofile_path=_get_temp_audio_dir(output_dir),
                threads=params.n_threads or 2,
                logger=None,
                fps=fps,
                **_get_video_encode_args(),
            )
            os.replace(temp_output_file, output_file)
        except Exception:
            delete_files([temp_output_file])
            raise
        return bgm_mix_succeeded


def preprocess_video(materials: List[MaterialInfo], clip_duration=4, image_effect: str = "kenburns"):
    # WebUI 在某些二次生成场景下可能传入空素材列表，这里直接返回空结果，避免抛出 NoneType 异常。
    if not materials:
        return []

    # 仅返回通过预处理校验的素材，避免低分辨率图片继续进入后续的视频合成流程。
    valid_materials = []
    local_videos_dir = utils.storage_dir("local_videos", create=True)

    for material in materials:
        if not material.url:
            continue

        try:
            material_source_path = file_security.resolve_path_within_directory(
                local_videos_dir, material.url
            )
        except ValueError as exc:
            # local video_source 的素材路径来自 API 参数，必须限制在专用素材目录。
            # 允许用户传文件名，也兼容历史返回的绝对路径，但不允许逃逸到系统
            # 其他目录，避免任意文件读取或通过 MoviePy 探测本地敏感文件。
            logger.warning(
                f"skip unsafe local material: {material.url}, "
                f"local_videos_dir: {local_videos_dir}, error: {str(exc)}"
            )
            continue

        ext = utils.parse_extension(material_source_path)
        try:
            # 图片素材直接按图片方式读取，避免先走 VideoFileClip 误判后触发不稳定的回退分支。
            if ext in const.FILE_TYPE_IMAGES:
                clip, material_source_path = _open_image_clip_with_fallback(
                    material_source_path
                )
            else:
                clip = _open_video_clip_quietly(material_source_path)
        except Exception:
            # 非标准扩展名或探测失败时再回退到图片模式，兼容历史上直接传本地图片路径的情况。
            try:
                clip, material_source_path = _open_image_clip_with_fallback(
                    material_source_path
                )
            except Exception as exc:
                logger.warning(
                    f"skip unreadable local material: {material.url}, error: {str(exc)}"
                )
                continue
        try:
            width = clip.size[0]
            height = clip.size[1]
            if not is_material_resolution_acceptable(width, height):
                logger.warning(
                    f"low resolution material: {width}x{height}, minimum "
                    f"{_MIN_MATERIAL_DIMENSION}x{_MIN_MATERIAL_DIMENSION} required "
                    f"(tolerance {_MIN_DIMENSION_TOLERANCE}px)"
                )
                # 探测到低分辨率素材后立即关闭资源，并且不要把该素材返回给后续流程。
                close_clip(clip)
                continue

            if ext in const.FILE_TYPE_IMAGES:
                logger.info(f"processing image: {material_source_path}")
                # 探测尺寸时已经打开过一次素材，这里先释放探测句柄，再重新创建用于导出的图片 clip。
                close_clip(clip)
                video_file = _kenburns_image_to_video(
                    material_source_path, clip_duration, effect=image_effect
                )
                if not video_file:
                    raise RuntimeError("Ken Burns conversion failed")
                material.url = video_file
                logger.success(f"image processed: {video_file}")
            else:
                # 普通视频素材只需要读取尺寸做校验，校验完成后立即释放句柄即可。
                close_clip(clip)
                # Update url to the resolved absolute path so that downstream
                # stages (combine_videos) can open the file without re-resolving.
                material.url = material_source_path
        except Exception:
            close_clip(clip)
            raise

        valid_materials.append(material)

    return valid_materials
