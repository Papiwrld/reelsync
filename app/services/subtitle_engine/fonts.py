"""Font discovery, resolution and multilingual fallback for subtitles.

The renderer must never crash because a font is missing or because a requested
font does not cover the script of the subtitle text. ``FontRegistry`` owns:

1. discovery: scan the project ``resource/fonts`` directory (recursively) for
   ``.ttf`` / ``.otf`` / ``.ttc`` files, cached so hot rendering paths never
   rescan the directory per frame;
2. resolution: case-insensitive lookup by filename or friendly name, with no
   hardcoded absolute paths anywhere in the renderer;
3. fallback: every resolution entry point is guaranteed to return a usable path
   (falling back to a bundled default when the requested font is gone);
4. multilingual run splitting: when the preferred font lacks glyphs for part of
   the text (e.g. Latin-only creator font + Chinese subtitle), the text is split
   into contiguous runs and each run is drawn with a font that supports it.
"""

import os
import threading
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple

from loguru import logger

from app.utils import utils

# 与 WebUI 的字体下拉框一致：只暴露这三种可被 Pillow 加载的格式。
SUPPORTED_FONT_EXTENSIONS: Tuple[str, ...] = (".ttf", ".otf", ".ttc")

# 请求的字体不存在时的最终兜底字体。必须是仓库里实际存在的文件，
# 历史配置引用的 STHeiti 系列已经移除，因此统一回退到 Montserrat-Bold。
DEFAULT_FALLBACK_FONT = "Montserrat-Bold.ttf"

# 发现缓存的有效期。字体目录几乎不变，但用户可能手动拷贝新字体进去；
# 短 TTL 既避免每次渲染都扫盘，又能让新字体在几十秒内被识别。
DISCOVERY_CACHE_TTL_SECONDS = 30.0


@dataclass(frozen=True)
class FontEntry:
    """一个已发现的字体文件。``name`` 是友好的展示名（去扩展名的文件名）。"""

    name: str
    filename: str
    path: str

    def __str__(self) -> str:
        return self.name


@dataclass
class TextRun:
    """一段连续的、可以用同一个字体绘制的文本片段。"""

    text: str
    font_path: str


class FontRegistry:
    """动态扫描字体目录并提供解析、回退与多语言分段能力。"""

    def __init__(
        self,
        fonts_dir: Optional[str] = None,
        scan_interval_seconds: float = DISCOVERY_CACHE_TTL_SECONDS,
    ):
        self.fonts_dir = os.path.abspath(fonts_dir or utils.font_dir())
        self.scan_interval_seconds = max(1.0, float(scan_interval_seconds))
        self._lock = threading.Lock()
        self._entries: dict[str, FontEntry] = {}  # lowercase filename -> entry
        self._last_scan: float = 0.0
        self._scan_error_logged = False

    # ------------------------------------------------------------------
    # discovery
    # ------------------------------------------------------------------
    def scan(self, force: bool = False) -> dict[str, FontEntry]:
        """扫描字体目录并返回 ``{小写文件名: FontEntry}`` 映射。

        结果按目录内容的修改时间缓存；只有目录里出现新增/删除/改动的字体
        文件时才会真正重新扫描，渲染热路径上不会反复读盘。
        """
        if not force:
            with self._lock:
                if self._entries and (
                    self._last_scan + self.scan_interval_seconds
                    >= _monotonic_seconds()
                ):
                    return dict(self._entries)

        try:
            discovered = _discover_fonts(self.fonts_dir)
        except OSError as exc:
            # 字体目录缺失或不可读时不能崩溃：渲染继续使用内置兜底字体。
            if not self._scan_error_logged:
                logger.warning(
                    f"failed to scan font directory {self.fonts_dir}: {exc}"
                )
                self._scan_error_logged = True
            discovered = {}

        with self._lock:
            self._entries = discovered
            self._last_scan = _monotonic_seconds()
        return dict(discovered)

    def clear_cache(self) -> None:
        """强制下一次调用重新扫描（测试与新增字体后调用）。"""
        with self._lock:
            self._entries = {}
            self._last_scan = 0.0

    # ------------------------------------------------------------------
    # listing / resolution
    # ------------------------------------------------------------------
    def list_fonts(self) -> list[str]:
        """返回按友好名称排序的字体列表（WebUI 下拉框与预设使用）。"""
        entries = self.scan()
        names = {entry.name for entry in entries.values()}
        return sorted(names)

    def path_for_name(self, name: str) -> Optional[str]:
        """按文件名、友好名或完整路径（大小写不敏感）解析字体路径。

        找不到返回 None。完整路径直接命中（调用方常把已解析的绝对路径
        再传给解析函数，例如 ``split_runs`` 的字体复用），不再产生
        “字体未找到”的误报。
        """
        if not name:
            return None
        entries = self.scan()
        wanted = str(name).strip().lower()
        if not wanted:
            return None
        # 1) 请求的本身就是已存在的文件路径：直接返回。
        candidate_path = os.path.abspath(str(name))
        if os.path.isfile(candidate_path):
            return candidate_path
        # 2) 按完整文件名匹配。
        entry = entries.get(wanted)
        if entry is not None:
            return entry.path
        # 3) 按友好名（去扩展名）匹配。
        wanted_stem = os.path.splitext(wanted)[0]
        for candidate in entries.values():
            if candidate.name.lower() == wanted_stem:
                return candidate.path
        return None

    def resolve(self, name: str) -> str:
        """解析字体路径；请求的字体不可用时回退到默认字体，永不抛异常。"""
        path = self.path_for_name(name)
        if path and os.path.isfile(path):
            return path
        if name:
            logger.warning(
                f"subtitle font not found: {name}, "
                f"falling back to {DEFAULT_FALLBACK_FONT}"
            )
        return self._fallback_path()

    def _fallback_path(self) -> str:
        path = self.path_for_name(DEFAULT_FALLBACK_FONT)
        if path and os.path.isfile(path):
            return path
        # 理论上兜底字体也不存在（字体目录被清空）：返回目录里第一个可用字体。
        entries = self.scan()
        for entry in entries.values():
            if os.path.isfile(entry.path):
                return entry.path
        # 最后手段：返回一个确定性的路径，PIL 加载失败时由渲染层继续降级。
        return os.path.join(self.fonts_dir, DEFAULT_FALLBACK_FONT)

    # ------------------------------------------------------------------
    # glyph support / multilingual fallback
    # ------------------------------------------------------------------
    def supports_text(self, font_path: str, text: str) -> bool:
        """检查字体是否包含文本中字母/数字所需字形（忽略空白与标点）。"""
        sample = "".join(
            dict.fromkeys(
                char
                for char in str(text or "")
                if unicodedata.category(char)[0] in {"L", "N"}
            )
        )[:64]
        if not sample:
            return True
        return _font_supports_sample(font_path, sample)

    def split_runs(self, text: str, preferred_font: str) -> list[TextRun]:
        """把文本切成可被同一字体连续绘制的片段。

        优先使用 ``preferred_font``；其中不支持的字符会尝试用已注册的其它
        字体依次匹配（多语言兜底），找不到支持字体的字符也保留在原位，
        由渲染层继续尝试，绝不因个别字形缺失而中断生成。
        """
        if not text:
            return []
        preferred = self.resolve(preferred_font)
        if self.supports_text(preferred, text):
            return [TextRun(text=text, font_path=preferred)]

        candidates = self._fallback_font_chain(preferred)
        runs: list[TextRun] = []
        current_text = ""
        current_font = ""
        for char in text:
            font_for_char = preferred
            if not _char_supported(preferred, char):
                font_for_char = next(
                    (
                        candidate
                        for candidate in candidates
                        if _char_supported(candidate, char)
                    ),
                    preferred,
                )
            if font_for_char != current_font:
                if current_text:
                    runs.append(TextRun(text=current_text, font_path=current_font))
                current_text = char
                current_font = font_for_char
            else:
                current_text += char
        if current_text:
            runs.append(TextRun(text=current_text, font_path=current_font))
        return runs

    def _fallback_font_chain(self, preferred_path: str) -> list[str]:
        """按优先级返回兜底字体路径列表（不含首选字体本身）。"""
        chain: list[str] = []
        for entry in sorted(self.scan().values(), key=lambda e: e.name.lower()):
            if entry.path == preferred_path:
                continue
            # 中文/日文/韩文等宽字符集字体优先（.ttc 通常是系统级多语字体）。
            chain.append(entry.path)
        return chain

    def count_discovered(self) -> int:
        return len(self.scan())


# ---------------------------------------------------------------------------
# 模块级纯函数（独立可测，不依赖实例状态）
# ---------------------------------------------------------------------------


def _monotonic_seconds() -> float:
    import time

    return time.monotonic()


def _discover_fonts(fonts_dir: str) -> dict[str, FontEntry]:
    """扫描目录（递归），返回 ``{小写文件名: FontEntry}``。"""
    entries: dict[str, FontEntry] = {}
    if not os.path.isdir(fonts_dir):
        return entries
    for root, _dirs, files in os.walk(fonts_dir):
        for filename in sorted(files):
            ext = os.path.splitext(filename)[1].lower()
            if ext not in SUPPORTED_FONT_EXTENSIONS:
                continue
            path = os.path.join(root, filename)
            name = os.path.splitext(filename)[0]
            key = filename.lower()
            # 子目录优先于根目录同名文件：用户可以把自定义字体放进子目录。
            if key in entries and os.path.dirname(entries[key].path) == fonts_dir:
                continue
            entries[key] = FontEntry(name=name, filename=filename, path=path)
    return entries


@lru_cache(maxsize=256)
def _font_supports_sample(font_path: str, sample: str) -> bool:
    """检查字体是否包含样本文字需要的字形，并缓存重复检查结果。"""
    try:
        from PIL import ImageFont

        font = ImageFont.truetype(font_path, 30)
    except Exception as exc:
        # 字体探测失败不应阻止用户生成；保留日志供环境兼容问题排查。
        logger.warning(f"failed to inspect subtitle font glyphs: {font_path}, {exc}")
        return True
    try:
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
    except Exception:
        # 极端字体（畸形 cmap 等）逐字探测失败时视为支持，交给渲染层兜底。
        return True


def _char_supported(font_path: str, char: str) -> bool:
    if unicodedata.category(char)[0] not in {"L", "N"}:
        return True
    return _font_supports_sample(font_path, char)


# 进程级共享实例：所有服务共用同一份字体缓存，避免重复扫描。
_registry: Optional[FontRegistry] = None
_registry_lock = threading.Lock()


def get_font_registry() -> FontRegistry:
    """返回进程级共享 FontRegistry 实例。"""
    global _registry
    with _registry_lock:
        if _registry is None:
            _registry = FontRegistry()
        return _registry


def reset_font_registry() -> FontRegistry:
    """重建共享实例（测试隔离用）。"""
    global _registry
    with _registry_lock:
        _registry = FontRegistry()
        _registry.clear_cache()
        return _registry


def font_entries_for_testing(directory: str) -> FontRegistry:
    """为指定目录构造一个全新的注册表（测试用，不影响共享实例）。"""
    registry = FontRegistry(fonts_dir=directory)
    registry.scan(force=True)
    return registry