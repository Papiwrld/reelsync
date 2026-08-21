import warnings
from enum import Enum
from typing import Any, List, Literal, Optional, Union

import pydantic
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.config import config

# 忽略 Pydantic 的特定警告
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Field name.*shadows an attribute in parent.*",
)


class VideoConcatMode(str, Enum):
    random = "random"
    sequential = "sequential"


class VideoTransitionMode(str, Enum):
    none = None
    auto = "Auto"
    shuffle = "Shuffle"
    mix = "Mix"
    fade_in = "FadeIn"
    fade_out = "FadeOut"
    slide_in = "SlideIn"
    slide_out = "SlideOut"
    zoom_in = "ZoomIn"
    zoom_out = "ZoomOut"


class VideoAspect(str, Enum):
    landscape = "16:9"
    portrait = "9:16"
    square = "1:1"

    def to_resolution(self):
        """返回该画幅对应的输出分辨率（宽, 高）。"""
        try:
            if _is_4k_requested():
                return ASPECT_RESOLUTIONS_4K[self]
            return ASPECT_RESOLUTIONS[self]
        except (KeyError, TypeError):
            raise ValueError(f"unsupported video aspect: {self}") from None


class SubtitleCasing(str, Enum):
    """字幕大小写风格。值采用新式规范名；旧配置字符串在字段校验时迁移。"""

    UPPERCASE = "uppercase"
    LOWERCASE = "lowercase"
    SENTENCE_CASE = "sentence_case"
    TITLE_CASE = "title_case"
    AS_SPOKEN = "as_spoken"


class SubtitlePosition(str, Enum):
    """字幕锚点位置。``CUSTOM`` 保留历史百分比定位，``DYNAMIC`` 是智能避让。"""

    BOTTOM = "bottom"
    CENTER = "center"
    TOP = "top"
    DYNAMIC = "dynamic"
    CUSTOM = "custom"


class SubtitleAnimation(str, Enum):
    """字幕入场/运动动画。"""

    NONE = "none"
    POP_BOUNCE = "pop_bounce"
    KINETIC_FLOAT = "kinetic_float"
    DYNAMIC_SCALE = "dynamic_scale"


class SubtitlePreset(str, Enum):
    """字幕样式预设：Hormozi / TikTok / CapCut / Cinematic / Minimal / Neon / Custom。"""

    HORMOZI = "hormozi"
    TIKTOK = "tiktok"
    CAPCUT = "capcut"
    CINEMATIC = "cinematic"
    MINIMAL = "minimal"
    NEON = "neon"
    CUSTOM = "custom"


# 旧配置（config.toml / script.json / API 请求）里的历史字符串到新枚举的迁移。
_LEGACY_CASING_MAP: dict[str, SubtitleCasing] = {
    "original": SubtitleCasing.AS_SPOKEN,
    "upper": SubtitleCasing.UPPERCASE,
    "title": SubtitleCasing.TITLE_CASE,
    "lower": SubtitleCasing.LOWERCASE,
}
_LEGACY_POSITION_MAP: dict[str, SubtitlePosition] = {
    "auto": SubtitlePosition.DYNAMIC,
}


def coerce_subtitle_casing(value) -> SubtitleCasing:
    """把任意合法输入解析为 ``SubtitleCasing``（含旧配置字符串迁移）。"""
    if isinstance(value, SubtitleCasing):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        mapped = _LEGACY_CASING_MAP.get(normalized)
        if mapped is not None:
            return mapped
        try:
            return SubtitleCasing(normalized)
        except ValueError:
            pass
    return SubtitleCasing.AS_SPOKEN


def coerce_subtitle_position(value) -> SubtitlePosition:
    """把任意合法输入解析为 ``SubtitlePosition``（含旧配置字符串迁移）。"""
    if isinstance(value, SubtitlePosition):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        mapped = _LEGACY_POSITION_MAP.get(normalized, normalized)
        try:
            return SubtitlePosition(mapped)
        except ValueError:
            pass
    return SubtitlePosition.BOTTOM


# 画幅到输出分辨率的唯一事实来源：WebUI、CLI、API 和素材供应商
# （Gemini/Pollinations 生成尺寸）都通过这里取分辨率，避免各处各自维护
# 一套宽高常量导致生成尺寸与成片尺寸不一致。
ASPECT_RESOLUTIONS: dict[VideoAspect, tuple[int, int]] = {
    VideoAspect.landscape: (1920, 1080),
    VideoAspect.portrait: (1080, 1920),
    VideoAspect.square: (1080, 1080),
}

ASPECT_RESOLUTIONS_4K: dict[VideoAspect, tuple[int, int]] = {
    VideoAspect.landscape: (3840, 2160),
    VideoAspect.portrait: (2160, 3840),
    VideoAspect.square: (2160, 2160),
}


def _is_4k_requested() -> bool:
    try:
        return str(config.app.get("video_resolution", "1080p") or "1080p").strip().lower() in (
            "4k",
            "2160p",
            "uhd",
            "3840x2160",
            "2160x3840",
        )
    except Exception:
        return False

# 支持直接解析的字符串形式，兼容 UI 值、CLI 参数和上游返回的宽松格式。
_ASPECT_STRING_VARIANTS: dict[str, VideoAspect] = {
    "16:9": VideoAspect.landscape,
    "1920x1080": VideoAspect.landscape,
    "landscape": VideoAspect.landscape,
    "9:16": VideoAspect.portrait,
    "1080x1920": VideoAspect.portrait,
    "portrait": VideoAspect.portrait,
    "1:1": VideoAspect.square,
    "1080x1080": VideoAspect.square,
    "square": VideoAspect.square,
}


def video_aspect_from_string(value) -> VideoAspect:
    """把任意合法输入解析为 ``VideoAspect``，非法输入抛出 ``ValueError``。

    接受 ``VideoAspect`` 成员、比率字符串（"16:9"/"9:16"/"1:1"）、
    常见别名（"landscape"/"portrait"/"square"）和分辨率字符串
    （"1920x1080" 等）。统一从这里解析可以保证所有入口对画幅的校验一致。
    """
    if isinstance(value, VideoAspect):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        parsed = _ASPECT_STRING_VARIANTS.get(normalized)
        if parsed is not None:
            return parsed
    raise ValueError(f"unsupported video aspect: {value!r}")


def validate_video_aspect(value) -> bool:
    """宽松校验：返回该值能否解析为受支持的画幅。"""
    try:
        video_aspect_from_string(value)
        return True
    except (TypeError, ValueError):
        return False


_MATERIAL_DATACLASS_CONFIG: ConfigDict = ConfigDict(arbitrary_types_allowed=True)


@pydantic.dataclasses.dataclass(config=_MATERIAL_DATACLASS_CONFIG)
class MaterialInfo:
    provider: str = "pexels"
    url: str = ""
    duration: int = 0
    # 可选的分辨率描述（如 "1920x1080"）。Web 抓取素材在搜索阶段就能拿到
    # 尺寸，直接存到字段上可以避免在排名阶段依赖 source_info 的 rendition。
    resolution: Optional[str] = ""
    # 在线素材搜索会附带经过筛选的公开来源信息，供搜索缓存和任务记录复用。
    # 本地上传素材不需要填写；写入任务文件前仍会按字段白名单重新构造，
    # 避免外部请求传入的签名 URL、凭据或无关字段进入持久化数据。
    source_info: Optional[dict[str, Any]] = None


class VideoParams(BaseModel):
    """
    {
      "video_subject": "",
      "video_aspect": "横屏 16:9（西瓜视频）",
      "voice_name": "女生-晓晓",
      "bgm_name": "random",
      "font_name": "Montserrat-Bold.ttf",
      "text_color": "#FFFFFF",
      "font_size": 60,
      "stroke_color": "#000000",
      "stroke_width": 1.5
    }
    """

    video_subject: str
    video_script: str = ""  # Script used to generate the video
    video_terms: Optional[str | list] = None  # Keywords used to generate the video
    video_aspect: Optional[VideoAspect] = VideoAspect.portrait.value
    video_concat_mode: Optional[VideoConcatMode] = VideoConcatMode.random.value
    video_transition_mode: Optional[VideoTransitionMode] = None
    # 0 = 自动：根据目标时长与素材数量推导每段素材时长。
    video_clip_duration: int = Field(default=5, ge=0)
    video_clip_speed: Optional[float] = 1.0
    match_materials_to_script: bool = False
    video_count: int = Field(default=1, ge=1)

    video_source: Optional[str] = "pexels"
    video_materials: Optional[List[MaterialInfo]] = (
        None  # Materials used to generate the video
    )
    # 素材媒体类型筛选：images_videos（默认，允许图片+视频混合）|
    # videos_only（只用真实视频素材，丢弃生成式 AI 静态图片）。
    material_media_type: str = "images_videos"
    # 静态图片转视频时的运镜效果：kenburns | zoom_in | zoom_out |
    # slide_left | slide_right | fade | random。空值回落为 kenburns。
    image_motion_effect: str = "kenburns"

    custom_audio_file: Optional[str] = (
        None  # Custom audio file path, will ignore TTS and can still use Whisper subtitles
    )
    video_language: Optional[str] = ""  # auto detect

    voice_name: Optional[str] = ""
    voice_volume: Optional[float] = 1.0
    voice_rate: Optional[float] = 1.0
    bgm_type: Optional[str] = "random"
    bgm_file: Optional[str] = ""
    bgm_volume: Optional[float] = 0.2
    mix_overlap_duration: Optional[float] = 1.0
    audio_ducking_enabled: Optional[bool] = True
    audio_ducking_intensity: Optional[float] = 0.3
    atmosphere_enabled: Optional[bool] = False
    atmosphere_volume: Optional[float] = 0.3
    sfx_volume: Optional[float] = 0.8
    # 视频配乐供应商共用提示词，WebUI 新任务统一写入该字段。保留下面的
    # Sonilo 专用字段以兼容旧任务记录和现有 CLI 参数。
    video_music_prompt: str = Field(default="", max_length=2000)
    sonilo_bgm_prompt: str = Field(default="", max_length=2000)

    subtitle_enabled: Optional[bool] = True
    subtitle_position: Optional[SubtitlePosition] = coerce_subtitle_position(
        config.ui.get("subtitle_position", "bottom")
    )  # top, center, bottom, dynamic(原 auto), custom
    subtitle_casing: Optional[SubtitleCasing] = coerce_subtitle_casing(
        config.ui.get("subtitle_casing", "original")
    )  # 新式: uppercase/lowercase/sentence_case/title_case/as_spoken
    custom_position: float = config.ui.get("custom_position", 70.0)
    font_name: Optional[str] = "Montserrat-Bold.ttf"
    text_fore_color: Optional[str] = "#FFFFFF"
    text_background_color: Union[bool, str] = False
    rounded_subtitle_background: bool = False

    font_size: int = 60
    stroke_color: Optional[str] = "#000000"
    stroke_width: float = 1.5
    # 字幕动画开关：默认全部关闭以保持历史渲染行为，WebUI 可单独开启。
    subtitle_dynamic_sizing: bool = False
    subtitle_pop_in_bounce: bool = False
    subtitle_floating_motion: bool = False

    # ---- 字幕工作室（Subtitle Studio）新增配置 ----
    # 新式字段全部带安全默认值；旧配置只填 legacy 字段时行为完全不变。
    subtitle_style_preset: SubtitlePreset = SubtitlePreset.CUSTOM
    subtitle_font: Optional[str] = None  # 覆盖 font_name；None 时沿用 font_name
    subtitle_color: Optional[str] = None  # 覆盖 text_fore_color；None 时沿用
    subtitle_highlight_color: str = "#FFD60A"
    subtitle_outline_color: Optional[str] = None  # 覆盖 stroke_color
    subtitle_outline_width: Optional[float] = None  # 覆盖 stroke_width
    subtitle_background: Optional[str] = None  # 覆盖 text_background_color
    subtitle_background_opacity: float = Field(default=0.55, ge=0.0, le=1.0)
    subtitle_vertical_offset: int = Field(default=0, ge=-200, le=200)
    subtitle_animation: SubtitleAnimation = SubtitleAnimation.NONE
    subtitle_pop_bounce: Optional[bool] = None  # 覆盖 subtitle_pop_in_bounce
    subtitle_kinetic_float: Optional[bool] = None  # 覆盖 subtitle_floating_motion
    subtitle_dynamic_scaling: Optional[bool] = None  # 覆盖 subtitle_dynamic_sizing
    subtitle_active_word_highlight: bool = False
    subtitle_dynamic_auto_avoidance: bool = True

    # 自定义成片输出目录（API 每请求覆盖 config.app.output_dir）。
    # 空值表示仅使用默认 storage/tasks/<task_id>/。
    output_dir: str = ""

    # 按场景分组的搜索词（agentic 场景规划产出），用于脚本-画面精准匹配。
    # None 表示使用扁平 video_terms 传统路径；有值时按场景顺序分配素材。
    scene_search_terms: Optional[List[List[str]]] = None
    scene_narrations: Optional[List[str]] = None
    scene_durations: Optional[List[float]] = None

    # ---- 图文叠加层（Overlay）----
    # 素材时长自动模式时，video_clip_duration=0 由服务层推导，这里记录推导后的值。
    # overlay_enabled 总开关；叠加层文本卡与图片卡均仅在开启时合成。
    overlay_enabled: bool = False
    # 叠加层式样：title_fact | title_only | facts_only | callouts_only | full
    overlay_style: str = "title_fact"
    # 顶部标题卡开关（subject 作为标题卡，仅在成片开头显示）。
    overlay_title_card: bool = True
    # 数据/事实卡：脚本中含数字/百分比/年份/引用等的句子显示为左下角事实卡。
    overlay_fact_cards: bool = True
    # 关键句 callout：脚本中短小的结论性/转折性句子显示为顶部 callout。
    overlay_callouts: bool = False
    # 图片叠加层：每场景一张角标图（logo/水印/装饰图），空则跳过。
    overlay_image: Optional[str] = None
    overlay_image_opacity: float = Field(default=0.85, ge=0.0, le=1.0)
    # 叠加层文字与底板颜色（默认白字黑底）。
    overlay_text_color: str = "#FFFFFF"
    overlay_bg_color: str = "#000000"

    n_threads: Optional[int] = 2
    # 视频帧率：24-60，默认 30。由 config.video_fps 或请求参数覆盖
    video_fps: int = Field(default=30, ge=24, le=60)
    # 音频响度归一化：开启后对最终混音做 -14 LUFS 归一化，防止多轨削波
    audio_loudnorm: bool = False
    paragraph_number: int = Field(default=1, ge=1, le=10)
    video_script_prompt: str = Field(default="", max_length=2000)
    custom_system_prompt: str = Field(default="", max_length=8000)
    # 文案措辞风格：simple_humanized | field_expert | storytelling |
    # persuasive | educational | casual。空值在服务层回落为默认人性化简单风格。
    script_style: str = ""

    # 目标成片时长（秒）。0 = 自动（跟随内容画像的 preferred_video_length）。
    # 服务层会把它注入脚本生成提示词，让文案长度贴近目标时长。
    video_duration_seconds: int = Field(default=0, ge=0, le=600)

    # Agentic content planning: when enabled, the script is produced by the
    # strategy graph (topic analysis -> strategy -> hooks -> narrative ->
    # script -> critic) under the selected content profile instead of the
    # single generic prompt. Defaults keep the legacy linear pipeline.
    agentic_planning: bool = False
    content_profile: str = ""

    # Content Intelligence context (Phase 2A): when any of these are set, the
    # agentic pipeline composes a ContentIntelligence contract and (per the
    # automation level) runs the research orchestrator. All optional: the
    # classic agentic flow behaves exactly as before when they are empty.
    niche: str = ""
    sub_niche: str = ""
    audience: str = ""
    platform: str = ""  # youtube | youtube_shorts | tiktok | instagram_reels | bilibili | x
    content_format: str = ""  # documentary | explainer | tutorial | news_analysis | storytelling | list | case_study
    content_goal: str = ""  # growth | engagement | education | awareness | monetization
    automation_level: str = ""  # manual | assisted | automatic | autopilot
    trend_preference: str = ""
    sources: Optional[List[str]] = None  # user-provided research notes/URLs
    fact_check_level: str = ""  # optional override: normal | strong | very_strong
    research_depth: str = ""  # optional override: low | medium | high | very_high

    # Phase 2C-2E optional strategy inputs. All empty by default so the legacy
    # and classic-agentic flows are unaffected.
    tone: str = ""  # user override for the voice/tone
    reference_channels: Optional[List[str]] = None  # competitor/peer channels
    narrative_strategy: str = ""  # optional explicit override (else selected)
    topic_discovery_mode: str = ""  # trending | evergreen | opportunity | ...

    @field_validator("subtitle_position", mode="before")
    @classmethod
    def _migrate_subtitle_position(cls, value):
        # 旧任务记录 / 旧 config.toml 里的 "auto" 迁移为 DYNAMIC，其余按枚举校验。
        return coerce_subtitle_position(value)

    @field_validator("subtitle_casing", mode="before")
    @classmethod
    def _migrate_subtitle_casing(cls, value):
        # 旧配置里的 original/upper/title/lower 迁移到新式枚举值。
        return coerce_subtitle_casing(value)


class SubtitleRequest(BaseModel):
    video_script: str
    video_language: Optional[str] = ""
    voice_name: Optional[str] = "zh-CN-XiaoxiaoNeural-Female"
    voice_volume: Optional[float] = 1.0
    voice_rate: Optional[float] = 1.2
    bgm_type: Optional[str] = "random"
    bgm_file: Optional[str] = ""
    bgm_volume: Optional[float] = 0.2
    subtitle_position: Optional[SubtitlePosition] = coerce_subtitle_position(
        config.ui.get("subtitle_position", "bottom")
    )
    subtitle_casing: Optional[SubtitleCasing] = coerce_subtitle_casing(
        config.ui.get("subtitle_casing", "original")
    )
    font_name: Optional[str] = "Montserrat-Bold.ttf"
    text_fore_color: Optional[str] = "#FFFFFF"
    text_background_color: Union[bool, str] = False
    rounded_subtitle_background: bool = False
    font_size: int = 60
    stroke_color: Optional[str] = "#000000"
    stroke_width: float = 1.5
    subtitle_dynamic_sizing: bool = False
    subtitle_pop_in_bounce: bool = False
    subtitle_floating_motion: bool = False
    # 字幕工作室新式字段（与 VideoParams 保持一致的安全默认值）。
    subtitle_style_preset: SubtitlePreset = SubtitlePreset.CUSTOM
    subtitle_font: Optional[str] = None
    subtitle_color: Optional[str] = None
    subtitle_highlight_color: str = "#FFD60A"
    subtitle_outline_color: Optional[str] = None
    subtitle_outline_width: Optional[float] = None
    subtitle_background: Optional[str] = None
    subtitle_background_opacity: float = Field(default=0.55, ge=0.0, le=1.0)
    subtitle_vertical_offset: int = Field(default=0, ge=-200, le=200)
    subtitle_animation: SubtitleAnimation = SubtitleAnimation.NONE
    subtitle_pop_bounce: Optional[bool] = None
    subtitle_kinetic_float: Optional[bool] = None
    subtitle_dynamic_scaling: Optional[bool] = None
    subtitle_active_word_highlight: bool = False
    subtitle_dynamic_auto_avoidance: bool = True
    video_source: Optional[str] = "local"
    subtitle_enabled: Optional[str] = "true"

    @field_validator("subtitle_position", mode="before")
    @classmethod
    def _migrate_subtitle_position(cls, value):
        return coerce_subtitle_position(value)

    @field_validator("subtitle_casing", mode="before")
    @classmethod
    def _migrate_subtitle_casing(cls, value):
        return coerce_subtitle_casing(value)


class AudioRequest(BaseModel):
    video_script: str
    video_language: Optional[str] = ""
    voice_name: Optional[str] = "zh-CN-XiaoxiaoNeural-Female"
    voice_volume: Optional[float] = 1.0
    voice_rate: Optional[float] = 1.2
    bgm_type: Optional[str] = "random"
    bgm_file: Optional[str] = ""
    bgm_volume: Optional[float] = 0.2
    video_source: Optional[str] = "local"


class ClipRequest(BaseModel):
    """
    Clip Generator 请求：从一段长视频提取多条竖屏高光片段。

    source_video 可以是本地上传视频（storage/local_videos、resource 等受信任
    目录内的文件），也可以是 http(s) 直链/平台链接，服务端会先用 yt-dlp 下载
    到任务目录后再排队处理。
    """

    source_video: str
    count: int = Field(default=3, ge=1, le=10)
    min_duration: float = Field(default=15.0, ge=5, le=120)
    max_duration: float = Field(default=60.0, ge=10, le=300)
    target_width: int = 1080
    target_height: int = 1920
    burn_subtitles: bool = False
    face_track: bool = True


class VideoScriptParams:
    """
    {
      "video_subject": "春天的花海",
      "video_language": "",
      "paragraph_number": 1,
      "video_script_prompt": "",
      "custom_system_prompt": ""
    }
    """

    video_subject: Optional[str] = "春天的花海"
    video_language: Optional[str] = ""
    paragraph_number: int = Field(default=1, ge=1, le=10)
    video_script_prompt: str = Field(default="", max_length=2000)
    custom_system_prompt: str = Field(default="", max_length=8000)
    # 文案措辞风格：simple_humanized | field_expert | storytelling |
    # persuasive | educational | casual。默认人性化简单风格。
    script_style: str = ""
    # 目标成片时长（秒）。0 = 自动。注入脚本提示词以控制文案长度。
    target_duration_seconds: int = Field(default=0, ge=0, le=600)


class VideoTermsParams:
    """
    {
      "video_subject": "",
      "video_script": "",
      "amount": 5,
      "match_materials_to_script": false
    }
    """

    video_subject: Optional[str] = "春天的花海"
    video_script: Optional[str] = (
        "春天的花海，如诗如画般展现在眼前。万物复苏的季节里，大地披上了一袭绚丽多彩的盛装。金黄的迎春、粉嫩的樱花、洁白的梨花、艳丽的郁金香……"
    )
    amount: Optional[int] = 5
    match_materials_to_script: bool = False


class VideoSocialMetadataParams:
    """
    {
      "video_subject": "A day in Shanghai",
      "video_script": "",
      "language": "auto",
      "platform": "tiktok"
    }
    """

    video_subject: Optional[str] = Field(default="A day in Shanghai", max_length=500)
    video_script: Optional[str] = Field(default="", max_length=8000)
    language: Optional[str] = Field(default="auto", max_length=64)
    platform: Optional[str] = Field(default="tiktok", max_length=64)


class BaseResponse(BaseModel):
    status: int = 200
    message: Optional[str] = "success"
    data: Any = None


class TaskVideoRequest(VideoParams, BaseModel):
    pass


class TaskQueryRequest(BaseModel):
    pass


class VideoScriptRequest(VideoScriptParams, BaseModel):
    pass


class VideoTermsRequest(VideoTermsParams, BaseModel):
    pass


class VideoSocialMetadataRequest(VideoSocialMetadataParams, BaseModel):
    pass


######################################################################################################
######################################################################################################
######################################################################################################
######################################################################################################
class TaskResponse(BaseResponse):
    class TaskResponseData(BaseModel):
        task_id: str

    data: TaskResponseData

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": 200,
                "message": "success",
                "data": {"task_id": "6c85c8cc-a77a-42b9-bc30-947815aa0558"},
            },
        }
    )


class TaskStatusData(BaseModel):
    """任务查询对外保证的稳定字段；历史和扩展字段继续原样透传。"""

    model_config = ConfigDict(extra="allow")

    task_id: str
    state: int
    progress: int = 0
    videos: Optional[List[str]] = None
    combined_videos: Optional[List[str]] = None
    failed_stage: Optional[str] = None
    error: Optional[str] = None
    cross_post_state: Optional[
        Literal["pending", "processing", "complete", "failed"]
    ] = None
    cross_post_results: Optional[List[dict[str, Any]]] = None
    cross_post_error: Optional[str] = None


class TaskListData(BaseModel):
    """分页任务列表结构。"""

    tasks: List[TaskStatusData]
    total: int
    page: int
    page_size: int


class TaskQueryResponse(BaseResponse):
    """
    任务查询会返回生成状态和可选的跨平台发布状态。

    生成失败时包含 `failed_stage` 和 `error`；生成完成后如果启用了自动发布，
    `cross_post_state` 会依次进入 pending、processing、complete 或 failed。
    """

    data: TaskStatusData

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "status": 200,
                    "message": "success",
                    "data": {
                        "task_id": "6c85c8cc-a77a-42b9-bc30-947815aa0558",
                        "state": 1,
                        "progress": 100,
                        "videos": ["/tasks/example/final-1.mp4"],
                        "cross_post_state": "complete",
                        "cross_post_results": [{"success": True}],
                    },
                },
                {
                    "status": 200,
                    "message": "success",
                    "data": {
                        "task_id": "6c85c8cc-a77a-42b9-bc30-947815aa0558",
                        "state": -1,
                        "progress": 30,
                        "failed_stage": "audio",
                        "error": "TTS request timed out",
                    },
                },
            ],
        }
    )


class TaskListResponse(BaseResponse):
    """任务列表使用独立响应模型，避免与单任务查询混用文档结构。"""

    data: TaskListData

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": 200,
                "message": "success",
                "data": {
                    "tasks": [
                        {
                            "task_id": "6c85c8cc-a77a-42b9-bc30-947815aa0558",
                            "state": 4,
                            "progress": 50,
                        }
                    ],
                    "total": 1,
                    "page": 1,
                    "page_size": 10,
                },
            }
        }
    )


class TaskDeletionResponse(BaseResponse):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": 200,
                "message": "success",
                "data": {
                    "state": 1,
                    "progress": 100,
                    "videos": [
                        "http://127.0.0.1:8080/tasks/6c85c8cc-a77a-42b9-bc30-947815aa0558/final-1.mp4"
                    ],
                    "combined_videos": [
                        "http://127.0.0.1:8080/tasks/6c85c8cc-a77a-42b9-bc30-947815aa0558/combined-1.mp4"
                    ],
                },
            },
        }
    )


class VideoScriptResponse(BaseResponse):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": 200,
                "message": "success",
                "data": {
                    "video_script": "春天的花海，是大自然的一幅美丽画卷。在这个季节里，大地复苏，万物生长，花朵争相绽放，形成了一片五彩斑斓的花海..."
                },
            },
        }
    )


class VideoTermsResponse(BaseResponse):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": 200,
                "message": "success",
                "data": {"video_terms": ["sky", "tree"]},
            },
        }
    )


class VideoSocialMetadataResponse(BaseResponse):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": 200,
                "message": "success",
                "data": {
                    "title": "A Day in Shanghai You Should Not Miss",
                    "caption": "Save this quick Shanghai inspiration and follow for more short travel ideas.",
                    "hashtags": ["#shorts", "#travel", "#shanghai", "#viral", "#fyp"],
                },
            },
        }
    )


class BgmRetrieveResponse(BaseResponse):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": 200,
                "message": "success",
                "data": {
                    "files": [
                        {
                            "name": "4fca18fce7344f3aa824777a40d45c8c.mp3",
                            "size": 1891269,
                            "file": "4fca18fce7344f3aa824777a40d45c8c.mp3",
                        }
                    ]
                },
            },
        }
    )


class BgmUploadResponse(BaseResponse):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": 200,
                "message": "success",
                "data": {"file": "4fca18fce7344f3aa824777a40d45c8c.mp3"},
            },
        }
    )


class VideoMaterialRetrieveResponse(BaseResponse):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": 200,
                "message": "success",
                "data": {
                    "files": [
                        {
                            "name": "example.mp4",
                            "size": 12345678,
                            "file": "/ReelSync/resource/videos/example.mp4",
                        }
                    ]
                },
            },
        }
    )


class VideoMaterialUploadResponse(BaseResponse):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": 200,
                "message": "success",
                "data": {
                    "file": "/ReelSync/resource/videos/example.mp4",
                },
            },
        }
    )
