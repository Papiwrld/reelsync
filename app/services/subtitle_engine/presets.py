"""Subtitle style presets.

Presets are configuration objects, not scattered if/else logic in the renderer.
``SubtitleStylePresetRegistry`` holds the canonical preset definitions; the style
resolver merges a preset with manual/legacy overrides into a single resolved
``SubtitleStyleConfig``.

All preset values are deterministic constants so two renders with the same
preset produce identical styling. Presets reference fonts by *requested* name;
the ``FontRegistry`` resolves them and falls back gracefully when a font is not
bundled (e.g. TheBoldFont, which is commercial and therefore optional).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from app.models.schema import (
    SubtitleAnimation,
    SubtitleCasing,
    SubtitlePosition,
    SubtitlePreset,
)

# 预设中引用的字体。TheBoldFont 是商业字体，不能随仓库分发；当用户在
# resource/fonts 里自行放置该文件时，Hormozi 预设会优先使用它，
# 否则回退到 Anton（预设值本来就是“Anton 或 TheBoldFont 回退”）。
PRESET_FONT_HORMOZI = "Anton-Regular.ttf"
PRESET_FONT_HORMOZI_ALT = "TheBoldFont.ttf"
PRESET_FONT_CINEMATIC = "Montserrat-ExtraBold.ttf"
PRESET_FONT_MINIMAL = "Poppins-Bold.ttf"
PRESET_FONT_NEON = "BebasNeue-Regular.ttf"
# TikTok 风格的圆润粗体；CapCut 风格的中性高对比粗体。
PRESET_FONT_TIKTOK = "BeVietnamPro-Bold.ttf"
PRESET_FONT_CAPCUT = "Roboto-Bold.ttf"

# 预设共享的默认高亮色：鲜艳的黄/绿，保证在黑白对比之外有高辨识度。
DEFAULT_HIGHLIGHT_COLOR = "#FFD60A"


@dataclass(frozen=True)
class SubtitleStylePreset:
    """一个完整、确定性的字幕样式预设。"""

    key: str
    display_name: str
    # 显式控制的字段；为 None 的字段不参与覆盖（沿用当前值）。
    font: Optional[str] = None
    casing: Optional[SubtitleCasing] = None
    color: Optional[str] = None
    highlight_color: Optional[str] = None
    outline_color: Optional[str] = None
    outline_width: Optional[float] = None
    background: Optional[str] = None
    background_opacity: Optional[float] = None
    rounded_background: Optional[bool] = None
    position: Optional[SubtitlePosition] = None
    vertical_offset: Optional[int] = None
    animation: Optional[SubtitleAnimation] = None
    pop_bounce: Optional[bool] = None
    kinetic_float: Optional[bool] = None
    dynamic_scaling: Optional[bool] = None
    active_word_highlight: Optional[bool] = None
    dynamic_auto_avoidance: Optional[bool] = None
    font_size: Optional[int] = None

    def as_mapping(self) -> dict[str, Any]:
        return {key: value for key, value in self.__dict__.items() if value is not None}


class SubtitleStylePresetRegistry:
    """注册与查询字幕样式预设。"""

    def __init__(self) -> None:
        self._presets: dict[str, SubtitleStylePreset] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        self.register(
            SubtitleStylePreset(
                key=SubtitlePreset.HORMOZI.value,
                display_name="Hormozi / Viral Bold",
                font=PRESET_FONT_HORMOZI,
                casing=SubtitleCasing.UPPERCASE,
                color="#FFFFFF",
                highlight_color=DEFAULT_HIGHLIGHT_COLOR,
                outline_color="#000000",
                outline_width=4.0,
                background=None,
                background_opacity=0.55,
                rounded_background=False,
                position=SubtitlePosition.DYNAMIC,
                vertical_offset=0,
                animation=SubtitleAnimation.POP_BOUNCE,
                pop_bounce=True,
                kinetic_float=False,
                dynamic_scaling=True,
                active_word_highlight=True,
                dynamic_auto_avoidance=True,
                font_size=68,
            )
        )
        self.register(
            SubtitleStylePreset(
                key=SubtitlePreset.TIKTOK.value,
                display_name="TikTok Viral",
                font=PRESET_FONT_TIKTOK,
                casing=SubtitleCasing.UPPERCASE,
                color="#FFFFFF",
                highlight_color=DEFAULT_HIGHLIGHT_COLOR,
                outline_color="#000000",
                outline_width=3.5,
                background=None,
                background_opacity=0.4,
                rounded_background=False,
                position=SubtitlePosition.DYNAMIC,
                vertical_offset=0,
                animation=SubtitleAnimation.POP_BOUNCE,
                pop_bounce=True,
                kinetic_float=False,
                dynamic_scaling=True,
                active_word_highlight=True,
                dynamic_auto_avoidance=True,
                font_size=66,
            )
        )
        self.register(
            SubtitleStylePreset(
                key=SubtitlePreset.CAPCUT.value,
                display_name="CapCut Clean",
                font=PRESET_FONT_CAPCUT,
                casing=SubtitleCasing.AS_SPOKEN,
                color="#FFFFFF",
                highlight_color=DEFAULT_HIGHLIGHT_COLOR,
                outline_color="#000000",
                outline_width=2.0,
                background="#000000",
                background_opacity=0.5,
                rounded_background=True,
                position=SubtitlePosition.BOTTOM,
                vertical_offset=0,
                animation=SubtitleAnimation.POP_BOUNCE,
                pop_bounce=True,
                kinetic_float=False,
                dynamic_scaling=True,
                active_word_highlight=True,
                dynamic_auto_avoidance=True,
                font_size=56,
            )
        )
        self.register(
            SubtitleStylePreset(
                key=SubtitlePreset.CINEMATIC.value,
                display_name="Cinematic Documentary",
                font=PRESET_FONT_CINEMATIC,
                casing=SubtitleCasing.TITLE_CASE,
                color="#FFFFFF",
                highlight_color="#FFE9A8",
                outline_color="#1A1A1A",
                outline_width=1.5,
                background=None,
                background_opacity=0.35,
                rounded_background=False,
                position=SubtitlePosition.DYNAMIC,
                vertical_offset=0,
                animation=SubtitleAnimation.KINETIC_FLOAT,
                pop_bounce=False,
                kinetic_float=True,
                dynamic_scaling=True,
                active_word_highlight=False,
                dynamic_auto_avoidance=True,
                font_size=60,
            )
        )
        self.register(
            SubtitleStylePreset(
                key=SubtitlePreset.MINIMAL.value,
                display_name="Minimalist Clean",
                font=PRESET_FONT_MINIMAL,
                casing=SubtitleCasing.SENTENCE_CASE,
                color="#FFFFFF",
                highlight_color="#FFFFFF",
                outline_color="#000000",
                outline_width=0.5,
                background="#000000",
                background_opacity=0.45,
                rounded_background=True,
                position=SubtitlePosition.BOTTOM,
                vertical_offset=0,
                animation=SubtitleAnimation.NONE,
                pop_bounce=False,
                kinetic_float=False,
                dynamic_scaling=False,
                active_word_highlight=False,
                dynamic_auto_avoidance=True,
                font_size=52,
            )
        )
        self.register(
            SubtitleStylePreset(
                key=SubtitlePreset.NEON.value,
                display_name="Neon Glow",
                font=PRESET_FONT_NEON,
                casing=SubtitleCasing.UPPERCASE,
                color="#00F5FF",
                highlight_color="#FF00E5",
                outline_color="#00F5FF",
                outline_width=2.0,
                background=None,
                background_opacity=0.3,
                rounded_background=False,
                position=SubtitlePosition.CENTER,
                vertical_offset=0,
                animation=SubtitleAnimation.KINETIC_FLOAT,
                pop_bounce=False,
                kinetic_float=True,
                dynamic_scaling=True,
                active_word_highlight=True,
                dynamic_auto_avoidance=True,
                font_size=66,
            )
        )
        # Custom 预设：不覆盖任何字段，手动选择的值全部保留。
        self.register(
            SubtitleStylePreset(
                key=SubtitlePreset.CUSTOM.value,
                display_name="Custom",
            )
        )

    def register(self, preset: SubtitleStylePreset) -> None:
        self._presets[preset.key] = preset

    def get(self, key: str | SubtitlePreset | None) -> Optional[SubtitleStylePreset]:
        normalized = getattr(key, "value", key)
        return self._presets.get(str(normalized or "").lower())

    def get_required(self, key: str | SubtitlePreset | None) -> SubtitleStylePreset:
        preset = self.get(key)
        if preset is None:
            return self.get(SubtitlePreset.CUSTOM)  # type: ignore[return-value]
        return preset

    def list_presets(self) -> list[SubtitleStylePreset]:
        # 按“预设、然后自定义”的顺序展示；自定义永远在最后。
        ordered = [p for p in self._presets.values() if p.key != SubtitlePreset.CUSTOM.value]
        ordered.append(self._presets[SubtitlePreset.CUSTOM.value])
        return ordered

    def keys(self) -> list[str]:
        return [preset.key for preset in self.list_presets()]


_registry: Optional[SubtitleStylePresetRegistry] = None


def get_preset_registry() -> SubtitleStylePresetRegistry:
    """返回进程级共享预设注册表。"""
    global _registry
    if _registry is None:
        _registry = SubtitleStylePresetRegistry()
    return _registry


def reset_preset_registry() -> SubtitleStylePresetRegistry:
    """重建共享预设注册表（测试隔离用）。"""
    global _registry
    _registry = SubtitleStylePresetRegistry()
    return _registry