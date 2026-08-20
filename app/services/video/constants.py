"""Video module constants and pure helpers (separated from behavior).

Extracted from the original monolithic ``video`` module. This file holds
read-only configuration values and pure functions that depend only on their
arguments (no shared mutable state, no I/O). Keeping them here:
  * gives a single source of truth for encoding constants,
  * removes the module-load side-effect surface,
  * lets these helpers be unit-tested without importing MoviePy.

The ``video`` package facade (``__init__``) re-exports these names so all
existing ``app.services.video.<name>`` call-sites keep working unchanged.
"""

# ---------------------------------------------------------------------------
# Encoding defaults
# ---------------------------------------------------------------------------
audio_codec = "aac"
# Docker 里的 ffmpeg/AAC 组合在默认配置下更容易出现音频质量波动，
# 这里显式抬高音频码率，避免成片阶段因为默认值过低而引入明显失真。
audio_bitrate = "192k"
fps = 30

# FFmpeg 按帧率拼接/转码时，最终时长可能比 MoviePy 读到的理论时长短几十毫秒。
# 这里给视频素材多留一个很小的安全余量，避免音频末尾因为帧舍入出现黑屏、
# 卡顿或最后一小段旁白没有画面的情况。
_VIDEO_DURATION_SAFETY_MARGIN = 0.1

_MIN_MATERIAL_DIMENSION = 480
# 消息类应用和部分编码器会把画面尺寸向下取整，例如 WhatsApp 会把 9:16 的
# 素材压成 478x850，比 480 少两个像素。直接按 480 硬卡会让这类素材全部被
# 丢弃，最终以 "no valid materials found" 整体失败。这里留一个很小的容差，
# 既能放行仅仅因为取整而略低于阈值的素材，也仍然能挡住真正的低清素材。
_MIN_DIMENSION_TOLERANCE = 10

_DEFAULT_VIDEO_CODEC = "libx264"
_SUPPORTED_VIDEO_CODECS = (
    "libx264",
    "h264_nvenc",
    "h264_amf",
    "h264_qsv",
    "h264_mf",
    "h264_videotoolbox",
)

# ffmpeg 单次 concat/编码的硬超时。损坏的输入或挂起的存储会让 ffmpeg 无限
# 卡住；超时后抛出明确错误，任务失败而非整个队列阻塞。
_FFMPEG_CONCAT_TIMEOUT_SECONDS = 600

# ffprobe 元数据探测的硬超时。探测只是拿时长/尺寸，正常应毫秒级返回；
# 超时说明容器损坏或存储挂起，直接回退到 MoviePy 打开路径。
_FFPROBE_TIMEOUT_SECONDS = 30


def _get_required_video_duration(audio_duration: float) -> float:
    """
    返回视频素材拼接的目标时长。

    使用场景：合成视频时需要素材时长覆盖旁白音频。只做到“刚好等于”
    音频时长时，FFmpeg 可能因为帧率舍入让最终视频略短，因此统一加一个
    轻量余量。函数独立出来，便于测试和后续按实际反馈调整余量大小。
    """
    return max(0.0, float(audio_duration) + _VIDEO_DURATION_SAFETY_MARGIN)


def is_material_resolution_acceptable(width: int, height: int) -> bool:
    """
    判断素材分辨率是否足够用于合成。

    标称最小值是 480x480，但允许比它低 `_MIN_DIMENSION_TOLERANCE` 个像素，
    以兼容编码器/消息应用向下取整导致的尺寸（例如 WhatsApp 的 478x850）。
    """
    min_dimension = _MIN_MATERIAL_DIMENSION - _MIN_DIMENSION_TOLERANCE
    return width >= min_dimension and height >= min_dimension