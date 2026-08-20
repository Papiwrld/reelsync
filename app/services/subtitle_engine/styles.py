"""Subtitle style configuration and resolution.

``SubtitleStyleResolver`` merges, in order:

1. legacy fields (``font_name``, ``text_fore_color``, ``stroke_color``, ...);
2. new-style fields (``subtitle_font``, ``subtitle_color``, ...) which override
   the legacy ones when explicitly set;
3. the selected preset (unless ``CUSTOM``) — applied to the *controls* values;
4. explicit per-field toggles (``subtitle_pop_bounce`` ...) on top of the
   preset, so the WebUI can keep single-feature switches meaningful.

The result is a plain ``SubtitleStyleConfig`` consumed by the renderer. Old
configurations that only set legacy fields produce exactly the historical
behavior (the default preset is ``CUSTOM`` which never overrides anything).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


from app.models.schema import (
    SubtitleAnimation,
    SubtitleCasing,
    SubtitlePosition,
    SubtitlePreset,
    VideoParams,
    coerce_subtitle_casing,
    coerce_subtitle_position,
)
from app.services.subtitle_engine.fonts import FontRegistry, get_font_registry
from app.services.subtitle_engine.presets import (
    PRESET_FONT_HORMOZI,
    PRESET_FONT_HORMOZI_ALT,
    SubtitleStylePresetRegistry,
    get_preset_registry,
)

# 垂直偏移的取值范围与默认值（-200 上移 / 0 保持历史位置 / +200 下移）。
VERTICAL_OFFSET_MIN = -200
VERTICAL_OFFSET_MAX = 200


@dataclass
class SubtitleStyleConfig:
    """渲染器消费的、已完全解析的字幕样式。所有字段都有确定性默认值。"""

    font_size: int = 60
    font_name: str = "Montserrat-Bold.ttf"
    font_path: str = ""
    casing: SubtitleCasing = SubtitleCasing.AS_SPOKEN
    color: str = "#FFFFFF"
    highlight_color: str = "#FFD60A"
    outline_color: str = "#000000"
    outline_width: float = 1.5
    background: Optional[str] = None
    background_opacity: float = 0.55
    rounded_background: bool = False
    position: SubtitlePosition = SubtitlePosition.BOTTOM
    custom_position: float = 70.0
    vertical_offset: int = 0
    animation: SubtitleAnimation = SubtitleAnimation.NONE
    pop_bounce: bool = False
    kinetic_float: bool = False
    dynamic_scaling: bool = False
    active_word_highlight: bool = False
    dynamic_auto_avoidance: bool = True
    preset: SubtitlePreset = SubtitlePreset.CUSTOM


class SubtitleStyleResolver:
    """把 ``VideoParams``（含 legacy 字段与预设）解析为 ``SubtitleStyleConfig``。"""

    def __init__(
        self,
        font_registry: Optional[FontRegistry] = None,
        preset_registry: Optional[SubtitleStylePresetRegistry] = None,
    ):
        self.font_registry = font_registry or get_font_registry()
        self.preset_registry = preset_registry or get_preset_registry()

    # ------------------------------------------------------------------
    def resolve(self, params: VideoParams) -> SubtitleStyleConfig:
        preset_key = getattr(params, "subtitle_style_preset", SubtitlePreset.CUSTOM)
        preset = self.preset_registry.get_required(preset_key)

        config = SubtitleStyleConfig()

        # 1. legacy 字段：历史渲染行为的基准。
        config.font_name = str(getattr(params, "font_name", None) or config.font_name)
        config.font_size = int(getattr(params, "font_size", None) or config.font_size)
        config.color = str(
            getattr(params, "text_fore_color", None) or config.color
        )
        config.outline_color = str(
            getattr(params, "stroke_color", None) or config.outline_color
        )
        config.outline_width = float(
            getattr(params, "stroke_width", None) or config.outline_width
        )
        legacy_background = getattr(params, "text_background_color", False)
        config.background = _resolve_background(legacy_background)
        config.rounded_background = bool(
            getattr(params, "rounded_subtitle_background", False)
        )
        config.custom_position = float(
            getattr(params, "custom_position", None) or config.custom_position
        )
        config.position = coerce_subtitle_position(
            getattr(params, "subtitle_position", None) or config.position
        )
        config.casing = coerce_subtitle_casing(
            getattr(params, "subtitle_casing", None) or config.casing
        )
        config.dynamic_scaling = bool(
            getattr(params, "subtitle_dynamic_sizing", False)
        )
        config.pop_bounce = bool(getattr(params, "subtitle_pop_in_bounce", False))
        config.kinetic_float = bool(
            getattr(params, "subtitle_floating_motion", False)
        )

        # 2. 新式字段覆盖 legacy（显式设置才生效）。
        _apply_if_set(params, "subtitle_font", config, "font_name")
        _apply_if_set(params, "subtitle_color", config, "color")
        _apply_if_set(params, "subtitle_outline_color", config, "outline_color")
        _apply_if_set(params, "subtitle_outline_width", config, "outline_width")
        _apply_if_set(params, "subtitle_background", config, "background")
        if getattr(params, "subtitle_background_opacity", None) is not None:
            config.background_opacity = float(
                max(0.0, min(1.0, params.subtitle_background_opacity))
            )
        config.vertical_offset = int(
            max(
                VERTICAL_OFFSET_MIN,
                min(VERTICAL_OFFSET_MAX, int(getattr(params, "subtitle_vertical_offset", 0) or 0)),
            )
        )
        config.animation = _coerce_animation(getattr(params, "subtitle_animation", None))
        config.active_word_highlight = bool(
            getattr(params, "subtitle_active_word_highlight", False)
        )
        config.dynamic_auto_avoidance = bool(
            getattr(params, "subtitle_dynamic_auto_avoidance", True)
        )
        config.preset = coerce_preset(preset_key)

        # 3. 预设覆盖（CUSTOM 预设为空，什么都不改）。
        config = _apply_preset(config, preset, self.font_registry)

        # 4. 单功能开关在预设之上显式生效。
        if getattr(params, "subtitle_pop_bounce", None) is not None:
            config.pop_bounce = bool(params.subtitle_pop_bounce)
        if getattr(params, "subtitle_kinetic_float", None) is not None:
            config.kinetic_float = bool(params.subtitle_kinetic_float)
        if getattr(params, "subtitle_dynamic_scaling", None) is not None:
            config.dynamic_scaling = bool(params.subtitle_dynamic_scaling)

        # 5. 动画枚举与单功能开关对齐：枚举是主开关，开关是细化闸门。
        _align_animation(config)

        # 6. 字体解析（注册表回退，永不抛异常）。
        config.font_path = self.font_registry.resolve(config.font_name)

        return config


# ---------------------------------------------------------------------------
# 纯函数辅助
# ---------------------------------------------------------------------------


def _apply_if_set(params, source_field, config, target_field) -> None:
    value = getattr(params, source_field, None)
    if value is None:
        return
    if source_field == "subtitle_background":
        setattr(config, target_field, _resolve_background(value))
    else:
        setattr(config, target_field, value)


def _resolve_background(value) -> Optional[str]:
    """兼容布尔与字符串两种历史背景写法：True -> 黑色，False/空 -> 无背景。"""
    if isinstance(value, bool):
        return "#000000" if value else None
    text = str(value or "").strip()
    return text or None


def coerce_preset(value) -> SubtitlePreset:
    if isinstance(value, SubtitlePreset):
        return value
    try:
        return SubtitlePreset(str(value or "").strip().lower())
    except ValueError:
        return SubtitlePreset.CUSTOM


def _coerce_animation(value) -> SubtitleAnimation:
    if isinstance(value, SubtitleAnimation):
        return value
    try:
        return SubtitleAnimation(str(value or "").strip().lower())
    except ValueError:
        return SubtitleAnimation.NONE


def _apply_preset(
    config: SubtitleStyleConfig,
    preset,
    font_registry: FontRegistry,
) -> SubtitleStyleConfig:
    """把预设值应用到一个已经装好 legacy/新式字段的配置上。"""
    if preset is None or preset.key == SubtitlePreset.CUSTOM.value:
        return config

    mapping = preset.as_mapping()
    if "font" in mapping:
        # TheBoldFont 是商业字体，只能作为可选外部字体：用户自行放入
        # resource/fonts 时优先使用，否则回退到 Anton（风格相近且已随
        # 仓库分发），注册表再兜底到仓库默认字体。
        requested = mapping["font"]
        if requested == PRESET_FONT_HORMOZI and font_registry.path_for_name(
            PRESET_FONT_HORMOZI_ALT
        ):
            requested = PRESET_FONT_HORMOZI_ALT
        config.font_name = requested
        config.font_path = ""
    for key in (
        "casing",
        "color",
        "highlight_color",
        "outline_color",
        "outline_width",
        "background",
        "background_opacity",
        "rounded_background",
        "position",
        "vertical_offset",
        "animation",
        "pop_bounce",
        "kinetic_float",
        "dynamic_scaling",
        "active_word_highlight",
        "dynamic_auto_avoidance",
        "font_size",
    ):
        if key in mapping:
            setattr(config, key, mapping[key])
    return config


def _align_animation(config: SubtitleStyleConfig) -> None:
    """动画枚举与细化开关保持自洽，且绝不破坏历史行为。

    - 枚举 NONE（默认值）时完全尊重 legacy 开关：历史配置里 pop-in/漂浮
      打开就继续生效，只是把枚举升级为对应动画类型用于上报。
    - 枚举为具体动画时，对应开关开启、其它动画开关关闭，避免预设/手动
      组合出现多个动画同时作用于同一段文字。
    """
    if config.animation == SubtitleAnimation.NONE:
        if config.pop_bounce:
            config.animation = SubtitleAnimation.POP_BOUNCE
        elif config.kinetic_float:
            config.animation = SubtitleAnimation.KINETIC_FLOAT
        elif config.dynamic_scaling:
            config.animation = SubtitleAnimation.DYNAMIC_SCALE
        return
    if config.animation == SubtitleAnimation.POP_BOUNCE:
        config.pop_bounce = True
        config.kinetic_float = False
    elif config.animation == SubtitleAnimation.KINETIC_FLOAT:
        config.pop_bounce = False
        config.kinetic_float = True
    elif config.animation == SubtitleAnimation.DYNAMIC_SCALE:
        config.dynamic_scaling = True


def apply_subtitle_casing(text: str, mode) -> str:
    """在字幕渲染前应用大小写转换（含旧式模式名迁移）。

    支持：uppercase / lowercase / sentence_case / title_case / as_spoken，
    以及历史值 original / upper / title / lower。
    """
    if not text:
        return text
    casing = coerce_subtitle_casing(mode)
    if casing == SubtitleCasing.UPPERCASE:
        return text.upper()
    if casing == SubtitleCasing.LOWERCASE:
        return text.lower()
    if casing == SubtitleCasing.TITLE_CASE:
        return text.title()
    if casing == SubtitleCasing.SENTENCE_CASE:
        return _to_sentence_case(text)
    return text


_SENTENCE_TERMINATORS = tuple(".!?。！？")


def _to_sentence_case(text: str) -> str:
    """把文本转换为句子式大小写：每个句子首字母大写，其余保持小写。

    只识别常见中英文句末标点，避免把缩写（如 U.S.）拆成多个句子。
    """
    if not text:
        return text
    lowered = text.lower()
    chars = list(lowered)
    new_sentence = True
    for index, char in enumerate(chars):
        if new_sentence and char.isalpha():
            chars[index] = char.upper()
            new_sentence = False
        elif char in _SENTENCE_TERMINATORS:
            new_sentence = True
    return "".join(chars)


def resolve_font_for_text(
    font_registry: FontRegistry, requested_font: str, text: str
) -> tuple[str, list]:
    """解析首选字体，并按字形支持把文本切成多语言 run。

    返回 ``(首选字体路径, [TextRun, ...])``。全部字符被首选字体支持时
    只有一个 run；否则按字符级支持情况分段，每段使用能够覆盖它的字体。
    """
    primary = font_registry.resolve(requested_font)
    runs = font_registry.split_runs(text, primary)
    return primary, runs