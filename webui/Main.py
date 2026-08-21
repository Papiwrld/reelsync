import hashlib
import html
import json
import math
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import time
import webbrowser
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from uuid import UUID, uuid4

import requests
import streamlit as st
from PIL import Image
from loguru import logger
from streamlit_tour import Tour

# WebUI 作为独立入口运行时，需要让项目根目录优先于第三方依赖，
# 避免依赖中的同名 app 包遮蔽 ReelSync 自己的 app 包。
root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if root_dir in sys.path:
    sys.path.remove(root_dir)
sys.path.insert(0, root_dir)

from app.config import config
from app.models import const
from app.models.llm_provider import (
    DEFAULT_LLM_PROVIDER_ID,
    LLM_PROVIDER_REGISTRY,
    get_llm_provider,
    normalize_provider_override,
)
from app.models.schema import (
    MaterialInfo,
    SubtitleCasing,
    SubtitlePosition,
    SubtitlePreset,
    VideoAspect,
    VideoConcatMode,
    VideoParams,
    VideoTransitionMode,
)
from app.services import bgm as bgm_service
from app.services import cache_manager, llm, video, voice, webui_task
from app.services import agent_llm, agentic
from app.services import research as research_service
from app.services import content_profile
from app.services import custom_media as custom_media_service
from app.services import elevenlabs_music as elevenlabs_music_service
from app.services import sonilo as sonilo_service
from app.services import trends as trends_service
from app.services import state as sm
from app.services import task as tm
from app.utils.logging_utils import configure_terminal_logger
from app.utils import utils
from app.utils.secrets import get_secret

st.set_page_config(
    page_title="ReelSync",
    # PNG favicon：JPEG 不支持透明且 1024px 原图过重，浏览器标签页与书签
    # 都需要小而清晰的图标。icon.png 是 128px 深色圆角 + 品牌红播放标记。
    page_icon=Image.open(utils.resource_dir("icon.png")),
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        "Report a bug": "https://github.com/Papiwrld/reelsync/issues",
        "About": "# ReelSync\nSimply provide a topic or keyword for a video, and it will "
        "automatically generate the video copy, video materials, video subtitles, "
        "and video background music before synthesizing a high-definition short "
        "video.\n\nhttps://github.com/Papiwrld/reelsync",
    },
)


# Streamlit 1.59 会在页面右上角默认展示 Deploy、skills nudge 等平台入口。
# ReelSync 是面向终端用户的本地工具，这些入口会造成顶部大块空白，
# 也会让新用户误以为需要安装额外组件。这里统一隐藏 Streamlit 平台工具栏，
# 并压缩主容器顶部留白，只保留项目自己的标题、语言选择和业务设置区域。
style_file = Path(__file__).with_name("styles.css")
streamlit_style = f"<style>{style_file.read_text(encoding='utf-8')}</style>"
st.markdown(streamlit_style, unsafe_allow_html=True)
# 定义资源目录
font_dir = os.path.join(root_dir, "resource", "fonts")
song_dir = os.path.join(root_dir, "resource", "songs")
i18n_dir = os.path.join(root_dir, "webui", "i18n")
config_file = os.path.join(root_dir, "webui", ".streamlit", "webui.toml")
# 语言列表必须在会话状态初始化前可用，首次访问时才能把浏览器 locale 映射到
# 项目真正支持的语言；自动识别结果只进入当前会话，不修改全局配置。
locales = utils.load_locales(i18n_dir)
DEFAULT_CHATTERBOX_BASE_URL = "http://127.0.0.1:4123/v1"
DEFAULT_CHATTERBOX_MODEL = "chatterbox"
DEFAULT_CHATTERBOX_VOICES = ["default-Female"]
ONBOARDING_TOUR_KEY = "mpt-onboarding-v1"
VOICE_MODE_TTS = "tts"
VOICE_MODE_UPLOAD = "upload"
VOICE_MODE_NONE = "none"
# “默认”是 WebUI 专用哨兵，不会写入 config.toml，也不会传给 FFmpeg。
# 后端在 video_codec 未配置时继续采用稳定的 libx264；单独保留该哨兵可以区分
# “跟随项目默认策略”和“用户明确固定 libx264”，便于未来安全调整默认策略。
DEFAULT_VIDEO_CODEC_OPTION = "__default__"
DEFAULT_SUBTITLE_SETTINGS = {
    "subtitle_enabled": True,
    "font_name": "MicrosoftYaHeiBold.ttc",
    "subtitle_position": "bottom",
    "subtitle_casing": "original",
    "custom_position": 70.0,
    "text_fore_color": "#FFFFFF",
    "font_size": 60,
    "stroke_color": "#000000",
    "stroke_width": 1.5,
    "subtitle_background_enabled": False,
    "subtitle_background_color": "#000000",
    "rounded_subtitle_background": False,
    "subtitle_dynamic_sizing": False,
    "subtitle_pop_in_bounce": False,
    "subtitle_floating_motion": False,
    "subtitle_style_preset": "custom",
    "subtitle_highlight_color": "#FFD60A",
    "subtitle_background_opacity": 0.55,
    "subtitle_vertical_offset": 0,
    "subtitle_active_word_highlight": False,
    "subtitle_dynamic_auto_avoidance": True,
}
LOCAL_MATERIAL_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".flv",
    ".mkv",
    ".jpg",
    ".jpeg",
    ".png",
}
CUSTOM_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}
_FINAL_VIDEO_PATTERN = re.compile(
    r"^final-(?P<index>\d+)\.(?P<extension>mp4|mov|mkv|webm)$",
    re.IGNORECASE,
)
_DOWNLOAD_FILENAME_INVALID_PATTERN = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RUNTIME_CONFIG_SECTIONS = {
    "app": config.app,
    "azure": config.azure,
    "chatterbox": config.chatterbox,
    "elevenlabs": config.elevenlabs,
    "minimax_tts": config.minimax_tts,
    "siliconflow": config.siliconflow,
    "ui": config.ui,
}


# -----------------------------------------------------------------------------
# 启动配置、会话状态与本地化
# -----------------------------------------------------------------------------


def _set_runtime_config(section_name, key, value):
    """
    更新 WebUI 配置，但不等待正在生成视频的后台任务。

    后台任务结束前，配置层只保留同一配置项的最新值；任务释放配置锁时会自动
    应用并保存。页面控件值仍由 Streamlit session_state 维护，因此暂存期间的
    rerun 不会把用户刚输入的内容重置为旧配置。
    """
    config_section = _RUNTIME_CONFIG_SECTIONS[section_name]
    updated = config.update_config_nonblocking(config_section, key, value)
    if not updated:
        logger.debug(f"deferred WebUI config update: section={section_name}, key={key}")
    return updated


def _delete_runtime_config(section_name, key):
    """删除 WebUI 配置项；后台任务占用配置时延后执行。"""
    config_section = _RUNTIME_CONFIG_SECTIONS[section_name]
    deleted = config.delete_config_nonblocking(config_section, key)
    if not deleted:
        logger.debug(f"deferred WebUI config delete: section={section_name}, key={key}")
    return deleted


def _save_runtime_config():
    """请求保存 WebUI 配置；后台任务占用配置时立即返回。"""
    saved = config.try_save_config()
    if not saved:
        logger.debug("deferred WebUI config save until active task completes")
    return saved


def _run_llm_read_operation(operation_name, operation):
    """
    使用稳定的当前 LLM 配置执行只读请求，并避免等待视频生成任务。

    能立即取得配置锁时继续沿用原来的互斥保护；锁已被后台视频任务持有时，
    全局配置在任务结束前不会发生变化，因此可以安全复制当前配置，并叠加页面
    尚未落盘的 Provider、模型和密钥。这样新文案使用界面中的最新选择，同时
    不会改变正在生成的视频任务。
    """
    with config.try_runtime_config_lock() as lock_acquired:
        # 配置层在复制全局值和叠加待更新值期间持有队列锁，因此快照只能看到
        # 更新前或更新后的完整状态，不会混用两组 Provider 参数。
        app_config_snapshot = config.snapshot_config_with_pending(config.app)
        if lock_acquired:
            return operation(app_config_snapshot)

    logger.info(
        f"run read-only LLM operation with active task configuration: "
        f"operation={operation_name}"
    )
    return operation(app_config_snapshot)


def _parse_chatterbox_voices(voices):
    # Chatterbox 是自托管服务，音色列表由用户在 WebUI 中手动输入。
    # 这里统一兼容 TOML 数组和输入框里的逗号分隔字符串，避免下拉框、
    # 试听按钮和后续生成流程使用不同格式导致状态不一致。
    if isinstance(voices, str):
        return [v.strip() for v in voices.split(",") if v.strip()]
    return [str(v).strip() for v in voices or [] if str(v).strip()]


def _sync_chatterbox_config_from_session_state():
    # Streamlit 的按钮会触发整页 rerun，而 Chatterbox 配置输入框位于
    # “试听语音合成”按钮之后。如果试听时只读取 config.chatterbox，可能拿不到
    # 用户刚在输入框里填入的 base_url/model/voices。先从 session_state 同步一次，
    # 可以保证按钮逻辑和输入框显示逻辑使用同一份最新配置。
    _set_runtime_config(
        "chatterbox",
        "base_url",
        (
            st.session_state.get(
                "chatterbox_base_url_input",
                config.chatterbox.get("base_url") or DEFAULT_CHATTERBOX_BASE_URL,
            )
            or ""
        ).strip(),
    )
    _set_runtime_config(
        "chatterbox",
        "api_key",
        st.session_state.get(
            "chatterbox_api_key_input", config.chatterbox.get("api_key", "")
        ),
    )
    _set_runtime_config(
        "chatterbox",
        "model_id",
        (
            st.session_state.get(
                "chatterbox_model_input",
                config.chatterbox.get("model_id") or DEFAULT_CHATTERBOX_MODEL,
            )
            or DEFAULT_CHATTERBOX_MODEL
        ).strip(),
    )
    _set_runtime_config(
        "chatterbox",
        "voices",
        _parse_chatterbox_voices(
            st.session_state.get(
                "chatterbox_voices_input",
                config.chatterbox.get("voices") or DEFAULT_CHATTERBOX_VOICES,
            )
        ),
    )


def _detect_audio_mime(audio_file: str, audio_bytes: bytes) -> str:
    # 有些 OpenAI-compatible TTS 服务，例如 travisvn/chatterbox-tts-api，
    # 即使请求 response_format=mp3，也会返回 WAV 内容。WebUI 试听如果固定
    # 使用 audio/mp3，浏览器可能无法播放，因此这里按文件头识别真实格式。
    header = audio_bytes[:12]
    if header.startswith(b"RIFF") and header[8:12] == b"WAVE":
        return "audio/wav"
    if header.startswith(b"ID3") or header[:2] in (
        b"\xff\xfb",
        b"\xff\xf3",
        b"\xff\xf2",
    ):
        return "audio/mp3"
    if header.startswith(b"OggS"):
        return "audio/ogg"
    ext = os.path.splitext(audio_file)[1].lower()
    return {
        ".wav": "audio/wav",
        ".m4a": "audio/mp4",
        ".aac": "audio/aac",
        ".ogg": "audio/ogg",
        ".flac": "audio/flac",
    }.get(ext, "audio/mp3")


def _build_uploaded_file_path(uploaded_file, target_dir, allowed_extensions, prefix):
    """为浏览器上传文件生成受控的服务端保存路径。"""
    original_name = os.path.basename(str(uploaded_file.name or ""))
    extension = os.path.splitext(original_name)[1].lower()
    if extension not in allowed_extensions:
        logger.warning(
            f"reject unsupported uploaded file extension: {original_name or '<empty>'}"
        )
        raise ValueError("unsupported uploaded file type")

    normalized_target_dir = os.path.realpath(target_dir)
    os.makedirs(normalized_target_dir, exist_ok=True)
    # 不复用浏览器传入的文件名，避免路径分隔符、控制字符或同名覆盖。UUID 只用于
    # 服务端落盘，不改变用户在上传控件中看到的原始名称。
    file_path = os.path.realpath(
        os.path.join(normalized_target_dir, f"{prefix}-{uuid4().hex}{extension}")
    )
    if os.path.commonpath([normalized_target_dir, file_path]) != normalized_target_dir:
        logger.warning(f"invalid uploaded file path: {file_path}")
        raise ValueError("invalid uploaded file path")
    return file_path


def _initialize_session_state():
    """集中初始化跨 rerun 保留的页面状态。"""
    if not st.session_state.get("cross_post_recovery_checked"):
        # WebUI 可以不经过 FastAPI 独立运行，因此也需要在首次会话初始化时处理
        # 进程重启留下的发布状态。恢复失败时不写标记，后续 rerun 会再次尝试。
        recovered = tm.recover_interrupted_cross_posts()
        if recovered is not None:
            st.session_state["cross_post_recovery_checked"] = True

    saved_ui_language = config.ui.get("language", "")
    browser_locale = st.context.locale
    initial_ui_language = utils.resolve_ui_language(
        saved_language=saved_ui_language,
        browser_locale=browser_locale,
        supported_languages=locales.keys(),
    )

    defaults = {
        "video_subject": "",
        "video_script": "",
        "video_terms": "",
        "video_script_prompt": "",
        "custom_system_prompt": llm.DEFAULT_SCRIPT_SYSTEM_PROMPT,
        "match_materials_to_script": bool(
            config.app.get("match_materials_to_script", False)
        ),
        "ui_language": initial_ui_language,
        # 已落盘的本地素材允许用户只修改文案后继续复用。
        "local_video_materials": [],
        # 生成按钮回调先登记任务，使顶部入口能立即显示运行中数量。
        "active_generation_tasks": {},
        # 最近一次从当前页面提交的任务。生成改为后台执行后，页面 Fragment
        # 通过这个 ID 查询状态；刷新时不再依赖正在执行的旧页面脚本。
        "current_generation_task_id": "",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


_initialize_session_state()


def tr(key):
    loc = locales.get(st.session_state["ui_language"], {})
    return loc.get("Translation", {}).get(key, key)


# -----------------------------------------------------------------------------
# 任务管理：历史扫描、运行状态、参数恢复与列表交互
# -----------------------------------------------------------------------------


def _format_task_time(timestamp):
    if not timestamp:
        return "-"
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")


def _format_task_subject(subject, max_length=30):
    subject = str(subject or "").replace("\n", " ").strip()
    if len(subject) <= max_length:
        return subject or "-"
    return f"{subject[:max_length]}..."


# 任务管理 fragment 每 2 秒轮询一次历史任务。script.json 与成片探测按
# (目录, 目录 mtime) 缓存：目录未变化时直接复用，避免空闲时每 2 秒产生
# 数十次文件读取。超上限时整体清空（键含 mtime，天然去旧）。
_TASK_FILE_CACHE_MAX = 256
_task_script_cache = {}
_task_final_video_cache = {}


def _safe_load_task_script(task_path, mtime=None):
    # 以 script.json 自身的 mtime 作为缓存键：文件被原地覆写时目录 mtime
    # 不一定变化，文件级 mtime 才能保证恢复/重生成后读到新内容。
    script_file = os.path.join(task_path, "script.json")
    try:
        file_mtime = os.stat(script_file).st_mtime
    except OSError:
        return {}

    cache_key = (script_file, file_mtime)
    if cache_key in _task_script_cache:
        return _task_script_cache[cache_key]

    try:
        with open(script_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.warning(f"failed to read task script data: {script_file}, {e}")
        return {}

    if len(_task_script_cache) >= _TASK_FILE_CACHE_MAX:
        _task_script_cache.clear()
    _task_script_cache[cache_key] = data
    return data


def _find_final_task_video(task_path: str, mtime=None) -> str:
    """
    返回任务目录中序号最小的最终成片。

    合成流程还会产生 combined、temp-clip 和 MoviePy 临时文件，这些文件不能
    表示任务已成功完成，因此这里只接受 ``final-<序号>.<扩展名>``。
    """
    cache_key = (task_path, mtime)
    if mtime is not None and cache_key in _task_final_video_cache:
        return _task_final_video_cache[cache_key]

    try:
        files = os.listdir(task_path)
    except OSError:
        return ""

    candidates = []
    for file_name in files:
        match = _FINAL_VIDEO_PATTERN.fullmatch(file_name)
        if match:
            candidates.append((int(match.group("index")), file_name))

    if not candidates:
        result = ""
    else:
        _, file_name = min(candidates, key=lambda item: item[0])
        result = os.path.join(task_path, file_name)

    if mtime is not None:
        if len(_task_final_video_cache) >= _TASK_FILE_CACHE_MAX:
            _task_final_video_cache.clear()
        _task_final_video_cache[cache_key] = result
    return result


def _build_restore_upload_requirements(params: Mapping) -> dict:
    """
    记录历史任务中无法由 Streamlit 自动恢复的上传文件依赖。

    浏览器不允许程序重新填充 file_uploader，因此恢复任务时需要单独记录本地
    素材和自定义音频依赖，并在用户重新生成前检查是否已经主动补充或替换。
    """
    return {
        "local_materials": params.get("video_source") == "local",
        "custom_audio": bool(params.get("custom_audio_file")),
        "original_voice_name": params.get("voice_name") or "",
    }


def _get_unmet_restore_upload_requirements(
    requirements: Mapping | None,
    *,
    video_source: str,
    voice_name: str,
    has_local_materials: bool,
    has_custom_audio: bool,
    voice_mode: str | None = None,
) -> set[str]:
    """返回当前表单仍未满足的历史上传文件依赖。"""
    requirements = requirements or {}
    unmet = set()

    if (
        requirements.get("local_materials")
        and video_source == "local"
        and not has_local_materials
    ):
        unmet.add("local_materials")

    if requirements.get("custom_audio") and not has_custom_audio:
        if voice_mode is not None:
            # 新版 WebUI 使用显式配音方式。用户切换到自动配音或无配音，表示
            # 已主动替换历史上传音频；只有继续选择上传模式时才要求重新上传。
            if voice_mode == VOICE_MODE_UPLOAD:
                unmet.add("custom_audio")
        elif voice_name == requirements.get("original_voice_name", ""):
            # 保留旧调用方按音色判断的兼容行为，避免影响 API 和已有测试工具。
            unmet.add("custom_audio")

    return unmet


def _queue_task_restore(task_id):
    # 任务列表运行在 fragment 中，不能直接修改已经创建的主表单控件状态。
    # 这里只记录候选任务并触发整页 rerun，确认和参数恢复由主页面统一处理。
    st.session_state["task_restore_candidate_id"] = task_id
    st.session_state["task_manager_popover_nonce"] = (
        st.session_state.get("task_manager_popover_nonce", 0) + 1
    )
    st.rerun(scope="app")


def _normalize_task_state(state):
    if state in (
        const.TASK_STATE_COMPLETE,
        const.TASK_STATE_FAILED,
        const.TASK_STATE_PROCESSING,
    ):
        return state
    try:
        return int(state)
    except (TypeError, ValueError):
        return state


def _active_generation_tasks():
    tasks = st.session_state.setdefault("active_generation_tasks", {})
    if not isinstance(tasks, dict):
        tasks = {}
        st.session_state["active_generation_tasks"] = tasks
    return tasks


def _add_active_generation_task(task_id, subject=None):
    tasks = _active_generation_tasks()
    task = tasks.setdefault(task_id, {})
    task["subject"] = subject or task.get("subject") or task_id
    task["mtime"] = task.get("mtime") or datetime.now().timestamp()


def _remove_active_generation_task(task_id):
    tasks = _active_generation_tasks()
    if task_id in tasks:
        del tasks[task_id]
    if st.session_state.get("pending_generation_task_id") == task_id:
        del st.session_state["pending_generation_task_id"]


def _prepare_generation_task():
    # st.button 的 on_click 会在页面脚本重新执行前触发。这里提前生成任务 ID，
    # 顶部任务管理入口就能在同一次 rerun 中显示“生成中”数量。
    task_id = str(uuid4())
    st.session_state["pending_generation_task_id"] = task_id
    subject = st.session_state.get("video_subject") or st.session_state.get(
        "video_script"
    )
    _add_active_generation_task(task_id, subject=subject)


def _task_state_label(state, has_video):
    normalized_state = _normalize_task_state(state)
    if normalized_state == const.TASK_STATE_COMPLETE:
        return tr("Task Status Complete")
    if normalized_state == const.TASK_STATE_FAILED:
        return tr("Task Status Failed")
    if normalized_state == const.TASK_STATE_PROCESSING:
        return tr("Task Status Processing")
    if has_video:
        return tr("Task Status Complete")
    return tr("Task Status History")


def _task_state_filter_key(task):
    normalized_state = _normalize_task_state(task.get("state"))
    if normalized_state == const.TASK_STATE_PROCESSING:
        return "processing"
    if normalized_state == const.TASK_STATE_FAILED:
        return "failed"
    if normalized_state == const.TASK_STATE_COMPLETE or task["video_file"]:
        return "complete"
    return "history"


def _scan_history_tasks(limit=30):
    tasks_root = utils.task_dir()
    if not os.path.isdir(tasks_root):
        return []

    # 任务管理 fragment 每两秒刷新一次。先只读取低成本的目录元数据并截取最近
    # 的任务，再解析 script.json 和视频列表，避免历史任务很多时反复扫描全部内容。
    task_entries = []
    try:
        with os.scandir(tasks_root) as entries:
            for entry in entries:
                try:
                    if entry.name.startswith(".") or not entry.is_dir(
                        follow_symlinks=False
                    ):
                        continue
                    task_entries.append(
                        (
                            entry.stat(follow_symlinks=False).st_mtime,
                            entry.name,
                            entry.path,
                        )
                    )
                except OSError as e:
                    # 单个任务目录可能正在被删除，不应因此让整个任务面板失效。
                    logger.debug(f"skip unavailable task directory: {entry.path}, {e}")
    except OSError as e:
        logger.warning(f"failed to scan task directory: {tasks_root}, {e}")
        return []

    task_entries.sort(key=lambda item: item[0], reverse=True)
    tasks = []
    for mtime, name, task_path in task_entries[:limit]:
        script_data = _safe_load_task_script(task_path, mtime)
        params_data = script_data.get("params", {}) if script_data else {}
        video_file = _find_final_task_video(task_path, mtime)
        subject = (
            params_data.get("video_subject")
            or script_data.get("script", "")[:40]
            or name
        )
        tasks.append(
            {
                "task_id": name,
                "subject": subject,
                "state": const.TASK_STATE_COMPLETE if video_file else None,
                "progress": 100 if video_file else 0,
                "mtime": mtime,
                "task_path": task_path,
                "video_file": video_file,
                "source": "history",
            }
        )

    return tasks


def _collect_task_summaries(limit=20):
    history_tasks = {task["task_id"]: task for task in _scan_history_tasks(limit=50)}

    try:
        runtime_tasks, _ = sm.state.get_all_tasks(1, 50)
    except Exception as e:
        logger.warning(f"failed to load runtime tasks: {e}")
        runtime_tasks = []

    for task in runtime_tasks:
        task_id = task.get("task_id", "")
        if not task_id:
            continue

        task_path = os.path.join(utils.task_dir(), task_id)
        history_task = history_tasks.get(task_id, {})
        video_files = task.get("videos") or []
        video_file = (
            video_files[0] if video_files else history_task.get("video_file", "")
        )
        subject = (
            task.get("video_subject")
            or history_task.get("subject")
            or (task.get("script", "")[:40] if task.get("script") else "")
            or task_id
        )

        history_tasks[task_id] = {
            "task_id": task_id,
            "subject": subject,
            "state": task.get("state"),
            "cross_post_state": task.get("cross_post_state"),
            "progress": int(task.get("progress", 0) or 0),
            "mtime": os.path.getmtime(task_path)
            if os.path.isdir(task_path)
            else history_task.get("mtime", 0),
            "task_path": task_path,
            "video_file": video_file,
            "source": "runtime",
        }

    for task_id, active_task in _active_generation_tasks().items():
        history_task = history_tasks.get(task_id, {})
        if history_task and _task_state_filter_key(history_task) in {
            "complete",
            "failed",
        }:
            # 会话中的 active 标记只负责覆盖任务刚提交到状态存储前的极短窗口。
            # 后台任务结束后必须以真实终态为准，不能把失败任务重新显示为生成中。
            continue

        task_path = os.path.join(utils.task_dir(), task_id)
        history_tasks[task_id] = {
            "task_id": task_id,
            "subject": active_task.get("subject")
            or history_task.get("subject")
            or task_id,
            "state": const.TASK_STATE_PROCESSING,
            "progress": history_task.get("progress", 0),
            "mtime": active_task.get("mtime")
            or history_task.get("mtime", datetime.now().timestamp()),
            "task_path": task_path,
            "video_file": history_task.get("video_file", ""),
            "source": "active",
        }

    tasks = list(history_tasks.values())
    return sorted(tasks, key=lambda item: item["mtime"], reverse=True)[:limit]


def _open_task_path(task_path):
    tasks_root = os.path.abspath(utils.task_dir())
    normalized_path = os.path.abspath(task_path)
    if not normalized_path.startswith(tasks_root + os.sep):
        logger.warning(f"invalid task folder path: {normalized_path}")
        return
    if os.path.isdir(normalized_path):
        webbrowser.open(f"file://{normalized_path}")


def _open_task_video(video_file):
    tasks_root = os.path.abspath(utils.task_dir())
    normalized_file = os.path.abspath(video_file)

    # 视频路径来自任务目录扫描或运行期状态。这里仍然限制只能打开任务目录
    # 内的文件，避免 UI 操作被异常路径扩展成任意本地文件打开能力。
    if not normalized_file.startswith(tasks_root + os.sep):
        logger.warning(f"invalid task video path: {normalized_file}")
        return
    if not os.path.isfile(normalized_file):
        logger.warning(f"task video does not exist: {normalized_file}")
        return

    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", normalized_file])
        elif sys.platform.startswith("win"):
            os.startfile(normalized_file)  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", normalized_file])
    except Exception as e:
        logger.error(f"failed to open task video: {normalized_file}, {e}")


def _delete_task(task_id, task_path, task_state=None):
    # 页面展示的状态可能落后于后台任务。删除前同时检查传入状态、当前会话的
    # 活跃任务和最新状态，避免任务刚开始或已产出中间视频时被误删。
    current_task = None
    try:
        current_task = sm.state.get_task(task_id)
    except Exception as e:
        logger.exception(f"failed to verify task state before deletion: {task_id}, {e}")
        return False

    task_snapshot = dict(current_task or {})
    task_snapshot.setdefault("state", task_state)
    if task_id in _active_generation_tasks():
        task_snapshot["state"] = const.TASK_STATE_PROCESSING

    if tm.is_task_busy(task_snapshot):
        logger.warning(f"refused to delete running task: {task_id}")
        return False

    tasks_root = os.path.abspath(utils.task_dir())
    normalized_path = os.path.abspath(task_path)

    # 删除任务会移除任务状态和本地生成文件。这里必须限定在 storage/tasks
    # 下，避免异常 task_path 造成误删其它本地目录。
    if not normalized_path.startswith(tasks_root + os.sep):
        logger.warning(f"invalid task folder path for deletion: {normalized_path}")
        return False

    try:
        if hasattr(sm.state, "delete_task"):
            sm.state.delete_task(task_id)
        if os.path.isdir(normalized_path):
            shutil.rmtree(normalized_path)
        logger.info(f"deleted task: {task_id}")
        return True
    except Exception as e:
        logger.exception(f"failed to delete task: {task_id}, {e}")
        return False


def _count_processing_tasks(tasks):
    # 顶部任务管理入口只需要展示“生成中”任务数量。
    # 这里复用内部状态 key 判断，避免依赖多语言展示文案导致不同语言下统计不一致。
    processing_task_ids = {
        task["task_id"]
        for task in tasks
        if _task_state_filter_key(task) == "processing"
    }
    return len(processing_task_ids)


def _task_manager_label(processing_count):
    label = tr("Task Manager")
    if processing_count <= 0:
        return label
    return f"{label} · {processing_count}"


def _build_video_download_name(subject, index, total):
    """根据视频主题生成跨平台安全的下载文件名。"""
    safe_subject = _DOWNLOAD_FILENAME_INVALID_PATTERN.sub(" ", str(subject or ""))
    safe_subject = re.sub(r"\s+", " ", safe_subject).strip(" .")[:80].rstrip(" .")
    if not safe_subject:
        safe_subject = "video"

    suffix = f"-{index}" if total > 1 else ""
    return f"{safe_subject}{suffix}.mp4"


def _render_task_table(filtered_tasks, key_prefix):
    if not filtered_tasks:
        st.info(tr("No Tasks Match Filter"))
        return

    visible_tasks = filtered_tasks[:12]
    with st.container(border=False):
        for task in visible_tasks:
            task_id = task["task_id"]
            has_video = bool(task["video_file"] and os.path.isfile(task["video_file"]))
            is_processing = _task_state_filter_key(task) == "processing"
            is_busy = is_processing or tm.is_task_busy(task)
            has_restore_data = os.path.isfile(
                os.path.join(task["task_path"], "script.json")
            )
            safe_task_key = "".join(ch if ch.isalnum() else "_" for ch in task_id)[:40]

            with st.container(
                key=f"task_row_{key_prefix}_{safe_task_key}", border=True
            ):
                st.markdown("<div class='mpt-task-card'></div>", unsafe_allow_html=True)
                row_cols = st.columns(
                    [3.5, 1.5],
                    vertical_alignment="center",
                )
                
                with row_cols[0]:
                    status_lbl = _task_state_label(task["state"], has_video)
                    time_lbl = _format_task_time(task["mtime"])
                    subject = _format_task_subject(task["subject"])
                    
                    st.markdown(f"**{subject}**")
                    st.caption(f"{status_lbl} &nbsp;•&nbsp; {time_lbl} &nbsp;•&nbsp; Progress: **{task['progress']}%**")

                action_cols = row_cols[1].columns(
                    4,
                    vertical_alignment="center",
                    gap="small",
                )
                with action_cols[0]:
                    st.markdown("<div class='mpt-action-play'></div>", unsafe_allow_html=True)
                    if st.button(
                        " ", # Empty label
                        icon=":material/play_arrow:",
                        key=f"play_task_{key_prefix}_{task_id}",
                        use_container_width=True,
                        help=tr("Play"),
                        disabled=not has_video,
                    ):
                        _open_task_video(task["video_file"])

                with action_cols[1]:
                    st.markdown("<div class='mpt-action-folder'></div>", unsafe_allow_html=True)
                    if st.button(
                        " ", # Empty label
                        icon=":material/folder_open:",
                        key=f"open_task_{key_prefix}_{task_id}",
                        use_container_width=True,
                        help=tr("Open Task Folder"),
                    ):
                        _open_task_path(task["task_path"])

                with action_cols[2]:
                    st.markdown("<div class='mpt-action-retry'></div>", unsafe_allow_html=True)
                    if st.button(
                        " ", # Empty label
                        icon=":material/refresh:",
                        key=f"restore_task_{key_prefix}_{task_id}",
                        use_container_width=True,
                        help=tr("Regenerate Task"),
                        disabled=is_processing or not has_restore_data,
                    ):
                        _queue_task_restore(task_id)

                with action_cols[3]:
                    delete_help = (
                        f"{tr('Delete Task')} ({tr('Task Status Processing')})"
                        if is_busy
                        else tr("Delete Task")
                    )
                    st.markdown("<div class='mpt-action-delete'></div>", unsafe_allow_html=True)
                    if st.button(
                        " ", # Empty label
                        icon=":material/delete:",
                        key=f"delete_task_{key_prefix}_{task_id}",
                        use_container_width=True,
                        help=delete_help,
                        disabled=is_busy,
                    ):
                        if _delete_task(task_id, task["task_path"], task["state"]):
                            st.toast(tr("Task Deleted"))
                            st.rerun()
                        else:
                            st.error(tr("Task Delete Failed"))


def _delete_all_tasks(tasks):
    """批量删除所有非运行中的任务，返回 (成功数, 失败数, 跳过数)。

    运行中/忙碌的任务与单个删除的行为一致：跳过不删，避免破坏正在进行的生成。
    """
    deleted = failed = skipped = 0
    for task in tasks:
        task_id = task.get("task_id", "")
        if not task_id:
            continue
        if _task_state_filter_key(task) == "processing":
            skipped += 1
            continue
        if _delete_task(task_id, task["task_path"], task.get("state")):
            deleted += 1
        else:
            failed += 1
    return deleted, failed, skipped


def _render_delete_all_row(tasks):
    """任务列表顶部的“全部删除”按钮：两段式确认，避免误删。"""
    deletable_count = sum(
        1 for task in tasks if _task_state_filter_key(task) != "processing"
    )
    if not deletable_count:
        return

    confirm_key = "task_manager_delete_all_arm"
    armed = bool(st.session_state.get(confirm_key))
    label = (
        tr("Confirm Delete All Tasks").format(count=deletable_count)
        if armed
        else tr("Delete All Tasks")
    )
    if st.button(
        label,
        icon=":material/delete_sweep:",
        type="primary" if armed else "secondary",
        key="task_manager_delete_all_button",
        use_container_width=True,
    ):
        if not armed:
            st.session_state[confirm_key] = True
            st.rerun(scope="fragment")
        else:
            st.session_state[confirm_key] = False
            deleted, failed, skipped = _delete_all_tasks(tasks)
            if deleted:
                st.toast(tr("Tasks Deleted Summary").format(count=deleted), icon="🗑️")
            if skipped:
                st.toast(tr("Tasks Skipped Processing").format(count=skipped))
            if failed:
                st.error(tr("Task Delete Failed"))
            st.rerun(scope="fragment")


def _render_task_manager_panel(tasks=None):
    tasks = tasks if tasks is not None else _collect_task_summaries()
    if not tasks:
        st.info(tr("No Tasks Yet"))
        return

    _render_delete_all_row(tasks)

    # Streamlit 1.59 支持有状态 Tabs 的惰性渲染。切换时只重新构建当前列表，
    # 避免定时 Fragment 每两秒重复创建四套任务行和操作按钮。
    status_tabs = [
        ("all", tr("All Tasks")),
        ("processing", tr("Task Status Processing")),
        ("complete", tr("Task Status Complete")),
        ("failed", tr("Task Status Failed")),
    ]
    tabs = st.tabs(
        [label for _, label in status_tabs],
        key="task_manager_status_tabs",
        on_change="rerun",
    )
    for (status_key, _), tab in zip(status_tabs, tabs):
        if not tab.open:
            continue
        with tab:
            filtered_tasks = [
                task
                for task in tasks
                if status_key == "all" or _task_state_filter_key(task) == status_key
            ]
            _render_task_table(filtered_tasks, status_key)


@st.fragment(run_every="2s")
def _render_task_manager_entry():
    # 任务可能由当前页面或其它页面触发生成。入口单独用 fragment 定时刷新，
    # 只更新任务数量和 popover 内容，不打断主页面表单输入。
    task_summaries = _collect_task_summaries()
    processing_task_count = _count_processing_tasks(task_summaries)
    with st.container(key="task_manager_entry", width="content"):
        with st.popover(
            _task_manager_label(processing_task_count),
            width="content",
            icon=":material/task:",
            key=(
                "task_manager_popover_"
                f"{st.session_state.get('task_manager_popover_nonce', 0)}"
            ),
        ):
            _render_task_manager_panel(task_summaries)


def _load_task_restore_payload(task_id):
    tasks_root = os.path.realpath(utils.task_dir())
    task_path = os.path.realpath(os.path.join(tasks_root, str(task_id)))
    try:
        if os.path.commonpath([tasks_root, task_path]) != tasks_root:
            raise ValueError("task path is outside the task directory")
    except ValueError as e:
        logger.warning(f"invalid task restore path: {task_id}, {e}")
        return None

    script_data = _safe_load_task_script(task_path)
    raw_params = script_data.get("params")
    if not isinstance(raw_params, dict):
        logger.warning(f"task has no restorable parameters: {task_id}")
        return None

    params_input = dict(raw_params)
    if script_data.get("script"):
        params_input["video_script"] = script_data["script"]
    if script_data.get("search_terms"):
        params_input["video_terms"] = script_data["search_terms"]

    try:
        params = VideoParams.model_validate(params_input).model_dump(mode="json")
    except Exception as e:
        logger.warning(f"failed to validate task restore parameters: {task_id}, {e}")
        return None

    return {
        "task_id": str(task_id),
        "subject": params.get("video_subject") or script_data.get("script") or task_id,
        "params": params,
    }


def _infer_tts_server_from_voice(voice_name):
    if voice.is_no_voice(voice_name):
        return voice.NO_VOICE_NAME
    if voice.is_siliconflow_voice(voice_name):
        return "siliconflow"
    if voice.is_gemini_voice(voice_name):
        return "gemini-tts"
    if voice.is_mimo_voice(voice_name):
        return "mimo-tts"
    if voice.is_minimax_voice(voice_name):
        return "minimax-tts"
    if voice.is_elevenlabs_voice(voice_name):
        return "elevenlabs"
    if voice.is_chatterbox_voice(voice_name):
        return "chatterbox"
    if voice.is_azure_v2_voice(voice_name):
        return "azure-tts-v2"
    return "azure-tts-v1"


def _set_stable_widget_value(key, value):
    if value is not None:
        st.session_state[localized_widget_key(key)] = value


def _apply_pending_task_restore():
    payload = st.session_state.pop("task_restore_payload", None)
    if not payload:
        return False

    params = payload["params"]
    video_terms = params.get("video_terms") or ""
    if isinstance(video_terms, list):
        video_terms = ", ".join(str(term) for term in video_terms)

    # 文案与高级脚本设置。
    st.session_state["video_subject"] = params.get("video_subject") or ""
    st.session_state["video_script"] = params.get("video_script") or ""
    st.session_state["video_terms"] = str(video_terms)
    _set_stable_widget_value(
        "script_language_select", params.get("video_language") or ""
    )
    st.session_state["paragraph_number_input"] = params.get("paragraph_number", 1)
    _set_stable_widget_value(
        "target_duration_select", params.get("video_duration_seconds") or 0
    )
    st.session_state["video_script_prompt"] = params.get("video_script_prompt") or ""
    st.session_state["custom_system_prompt"] = (
        params.get("custom_system_prompt") or llm.DEFAULT_SCRIPT_SYSTEM_PROMPT
    )

    # 视频设置。素材上传控件不能由服务端写入，因此本地素材需要用户重新选择。
    video_source = params.get("video_source") or "pexels"
    _set_stable_widget_value("video_source_select", video_source)
    _set_stable_widget_value(
        "material_media_type_select", params.get("material_media_type") or "images_videos"
    )
    _set_stable_widget_value(
        "image_motion_effect_select", params.get("image_motion_effect") or "kenburns"
    )
    _set_stable_widget_value(
        "video_concat_mode_select", params.get("video_concat_mode") or "random"
    )
    _set_stable_widget_value(
        "video_transition_mode_select",
        params.get("video_transition_mode") or VideoTransitionMode.none.value,
    )
    _set_stable_widget_value(
        f"video_aspect_for_{video_source}",
        params.get("video_aspect") or VideoAspect.portrait.value,
    )
    _set_stable_widget_value(
        "video_clip_duration_select",
        int(params.get("video_clip_duration") or 0),
    )
    _set_stable_widget_value(
        "video_clip_speed_slider",
        # API 可以写入超过 WebUI 范围的速度，任务生成阶段会安全归一化，但
        # 历史记录仍可能保留原值。恢复任务前再次归一化，避免给 Streamlit
        # slider 注入越界值、NaN 或无穷值导致控件状态异常。
        utils.normalize_clip_speed(params.get("video_clip_speed", 1.0)),
    )
    _set_stable_widget_value("video_count_select", params.get("video_count", 1))
    st.session_state["match_materials_to_script"] = bool(
        params.get("match_materials_to_script", False)
    )

    # 音频设置。TTS server 未写入旧任务，根据历史 voice_name 推断。
    voice_name = params.get("voice_name") or voice.NO_VOICE_NAME
    tts_server = _infer_tts_server_from_voice(voice_name)
    if params.get("custom_audio_file"):
        voice_mode = VOICE_MODE_UPLOAD
    elif voice.is_no_voice(voice_name):
        voice_mode = VOICE_MODE_NONE
    else:
        voice_mode = VOICE_MODE_TTS
    _set_stable_widget_value("voice_mode_control", voice_mode)
    if tts_server != voice.NO_VOICE_NAME:
        _set_stable_widget_value("tts_server_select", tts_server)
        _set_stable_widget_value(f"speech_synthesis_select_{tts_server}", voice_name)
    _set_stable_widget_value("voice_volume_select", params.get("voice_volume", 1.0))
    _set_stable_widget_value("voice_rate_select", params.get("voice_rate", 1.0))
    bgm_type = params.get("bgm_type") or ""
    _set_stable_widget_value("bgm_type_select", bgm_type)
    _set_stable_widget_value("bgm_volume_select", params.get("bgm_volume", 0.2))
    st.session_state["custom_bgm_file_input"] = params.get("bgm_file") or ""
    st.session_state["sonilo_bgm_prompt_input"] = (
        params.get("video_music_prompt") or params.get("sonilo_bgm_prompt") or ""
    )
    st.session_state["elevenlabs_music_prompt_input"] = (
        params.get("video_music_prompt") or ""
    )

    # 字幕设置。对旧任务中的越界数值做最小限幅，避免 Slider 无法初始化。
    st.session_state["subtitle_enabled_checkbox"] = bool(
        params.get("subtitle_enabled", True)
    )
    _set_stable_widget_value("font_name_select", params.get("font_name") or "")
    _set_stable_widget_value(
        "subtitle_position_select", params.get("subtitle_position") or "bottom"
    )
    _set_stable_widget_value(
        "subtitle_casing_select", params.get("subtitle_casing") or "original"
    )
    custom_position = min(100.0, max(0.0, float(params.get("custom_position", 70.0))))
    st.session_state["custom_position_input"] = str(custom_position)
    st.session_state["font_color_picker"] = params.get("text_fore_color") or "#FFFFFF"
    st.session_state["font_size_slider"] = min(
        100, max(30, int(params.get("font_size", 60)))
    )
    st.session_state["stroke_color_picker"] = params.get("stroke_color") or "#000000"
    st.session_state["stroke_width_slider"] = min(
        10.0, max(0.0, float(params.get("stroke_width", 1.5)))
    )
    background_color = params.get("text_background_color")
    background_enabled = bool(background_color)
    st.session_state["subtitle_background_enabled_checkbox"] = background_enabled
    if isinstance(background_color, str):
        st.session_state["subtitle_background_color_picker"] = background_color
    st.session_state["rounded_subtitle_background_checkbox"] = bool(
        params.get("rounded_subtitle_background", False) and background_enabled
    )
    st.session_state["subtitle_dynamic_sizing_checkbox"] = bool(
        params.get("subtitle_dynamic_sizing", False)
    )
    st.session_state["subtitle_pop_in_bounce_checkbox"] = bool(
        params.get("subtitle_pop_in_bounce", False)
    )
    st.session_state["subtitle_floating_motion_checkbox"] = bool(
        params.get("subtitle_floating_motion", False)
    )
    _set_stable_widget_value(
        "subtitle_style_preset_select",
        params.get("subtitle_style_preset")
        or DEFAULT_SUBTITLE_SETTINGS["subtitle_style_preset"],
    )
    st.session_state["subtitle_highlight_color_picker"] = params.get(
        "subtitle_highlight_color", DEFAULT_SUBTITLE_SETTINGS["subtitle_highlight_color"]
    )
    st.session_state["subtitle_background_opacity_slider"] = min(
        1.0,
        max(0.05, float(params.get("subtitle_background_opacity", 0.55))),
    )
    st.session_state["subtitle_vertical_offset_slider"] = min(
        200,
        max(-200, int(params.get("subtitle_vertical_offset", 0))),
    )
    st.session_state["subtitle_active_word_highlight_checkbox"] = bool(
        params.get("subtitle_active_word_highlight", False)
    )
    st.session_state["subtitle_dynamic_auto_avoidance_checkbox"] = bool(
        params.get("subtitle_dynamic_auto_avoidance", True)
    )

    st.session_state["overlay_enabled_checkbox"] = bool(
        params.get("overlay_enabled", False)
    )
    _set_stable_widget_value(
        "overlay_style_select",
        params.get("overlay_style") or "title_fact",
    )
    st.session_state["overlay_title_card_checkbox"] = bool(
        params.get("overlay_title_card", True)
    )
    st.session_state["overlay_fact_cards_checkbox"] = bool(
        params.get("overlay_fact_cards", True)
    )
    st.session_state["overlay_callouts_checkbox"] = bool(
        params.get("overlay_callouts", False)
    )
    st.session_state["overlay_text_color_picker"] = params.get(
        "overlay_text_color", "#FFFFFF"
    )
    st.session_state["overlay_bg_color_picker"] = params.get(
        "overlay_bg_color", "#000000"
    )
    st.session_state["overlay_image_opacity_slider"] = min(
        1.0,
        max(0.0, float(params.get("overlay_image_opacity", 0.85))),
    )

    st.session_state.pop("local_video_materials_uploader", None)
    # 历史任务只保存素材路径，不能保证这些文件在当前环境仍然存在。
    # 同时清空当前页面已缓存的上传素材，避免恢复后误用另一个任务的文件。
    st.session_state["local_video_materials"] = []
    st.session_state.pop("custom_audio_file_uploader", None)
    st.session_state.pop("custom_bgm_uploader", None)
    st.session_state.pop("custom_bgm_validation", None)
    st.session_state["task_restore_upload_requirements"] = (
        _build_restore_upload_requirements(params)
    )

    st.session_state["task_restore_succeeded"] = True
    logger.info(f"restored task configuration: {payload['task_id']}")
    return True


def _dismiss_task_restore_dialog():
    st.session_state.pop("task_restore_candidate_id", None)


@st.dialog(
    tr("Regenerate Task"),
    width="small",
    on_dismiss=_dismiss_task_restore_dialog,
)
def _render_task_restore_dialog(task_id):
    payload = _load_task_restore_payload(task_id)
    if payload is None:
        st.error(tr("Task Restore Failed"))
        if st.button(tr("Cancel"), key="cancel_invalid_task_restore"):
            st.session_state.pop("task_restore_candidate_id", None)
            st.rerun(scope="app")
        return

    st.write(tr("Regenerate Task Confirmation"))
    st.caption(_format_task_subject(payload["subject"], max_length=80))
    cancel_col, load_col = st.columns(2)
    if cancel_col.button(
        tr("Cancel"),
        key="cancel_task_restore",
        use_container_width=True,
    ):
        st.session_state.pop("task_restore_candidate_id", None)
        st.rerun(scope="app")
    if load_col.button(
        tr("Load Task Configuration"),
        key="confirm_task_restore",
        type="primary",
        use_container_width=True,
    ):
        st.session_state["task_restore_payload"] = payload
        st.session_state.pop("task_restore_candidate_id", None)
        st.rerun(scope="app")


def _dismiss_settings_dialog():
    """关闭设置弹窗，并确保下一次整页 rerun 不会再次自动打开。"""
    st.session_state["settings_dialog_open"] = False


def _render_brand():
    """渲染项目名称和当前版本。"""
    st.markdown(
        f"""
        <h1 class="mpt-brand">
            <span class="mpt-brand__name">ReelSync</span>
            <a class="mpt-brand__version"
               href="https://github.com/Papiwrld/reelsync"
               target="_blank"
               rel="noopener noreferrer"
               aria-label="Open ReelSync on GitHub"
               title="Open project on GitHub">v{html.escape(str(config.project_version))}</a>
        </h1>
        """,
        unsafe_allow_html=True,
    )


def _render_top_bar():
    """渲染品牌、任务管理、设置和语言切换组成的页面顶部栏。"""
    # 顶部栏分为品牌区和操作区两个独立区域。窄屏下由 Streamlit
    # 将两个区域整体换行，操作区内部再根据剩余宽度自动换行。
    with st.container(key="top_bar"):
        brand_col, actions_col = st.columns(
            # 品牌区只占内容宽度的一小部分（文字本身很窄），操作区需要更多空间
            # 容纳任务管理、设置、语言三个控件。原先 3.5:2.0 的比例让品牌区
            # 占掉大半宽度，窄屏下语言下拉框会被挤到第二行。
            [1.0, 2.0],
            vertical_alignment="center",
            gap="small",
        )

    with brand_col:
        _render_brand()

    with actions_col:
        with st.container(
            key="top_bar_actions",
            horizontal=True,
            horizontal_alignment="right",
            vertical_alignment="center",
            gap="small",
            width="stretch",
        ):
            _render_task_manager_entry()

            # 用户手册：工作流程 + 功能说明。独立弹层，带书本文档图标，
            # 只读展示，不打断用户正在填写的表单。
            with st.popover(
                tr("User Manual"),
                # st.popover 只接受正整数像素、"stretch" 或 "content"（"medium"
                # 是 st.dialog 的参数）。全局 CSS 会把弹层内容拉伸到最多 860px，
                # 因此 content 宽度对手册来说仍然足够宽。
                width="content",
                icon=":material/menu_book:",
                key="user_manual_popover",
            ):
                st.markdown(f"**{tr('Manual Workflow Title')}**")
                for step_index in range(1, 9):
                    st.markdown(
                        f"{step_index}. {tr(f'Manual Step {step_index}')}"
                    )
                st.markdown("---")
                st.markdown(f"**{tr('Manual Features Title')}**")
                for feature_index in range(1, 7):
                    st.markdown(
                        f"- {tr(f'Manual Feature {feature_index}')}"
                    )

            if st.button(
                tr("Settings"),
                key="open_settings_dialog_button",
                type="secondary",
                icon=":material/settings:",
                width="content",
            ):
                st.session_state["settings_dialog_open"] = True

            language_codes = list(locales.keys())
            selected_index = 0
            for i, code in enumerate(language_codes):
                if code == st.session_state.get("ui_language", ""):
                    selected_index = i

            selected_language_code = st.selectbox(
                "Language / 语言",
                options=language_codes,
                index=selected_index,
                format_func=lambda code: locales[code].get("Language", code),
                key="top_language_code_selector",
                # collapsed 会为标签保留垂直占位，导致下拉框比相邻按钮更高、
                # 顶部操作区三个控件无法在同一中线上对齐；hidden 完全不占位。
                label_visibility="hidden",
                width=180,
            )
            if selected_language_code:
                previous_language = st.session_state.get("ui_language", "")
                if selected_language_code != previous_language:
                    logger.info(
                        "UI language changed by user: "
                        f"previous_language={previous_language or '<empty>'}, "
                        f"selected_language={selected_language_code}"
                    )
                    st.session_state["ui_language"] = selected_language_code
                    # 浏览器自动识别只影响当前会话；只有用户主动切换下拉框时才
                    # 写入 config.toml，后续新会话将优先使用该明确选择。
                    _set_runtime_config("ui", "language", selected_language_code)
                    _save_runtime_config()
                    # 切换语言后强制刷新，避免 selectbox 继续展示旧语言文案。
                    st.rerun()


support_locales = [
    "zh-CN",
    "zh-HK",
    "zh-TW",
    "de-DE",
    "en-US",
    "es-ES",
    "fr-FR",
    "ru-RU",
    "vi-VN",
    "th-TH",
    "tr-TR",
]


# -----------------------------------------------------------------------------
# 通用 UI 组件、资源缓存与日志
# -----------------------------------------------------------------------------


@st.cache_data(ttl=30, show_spinner=False)
def get_all_fonts():
    # 字体目录很少变化，但 Streamlit 每次控件交互都会 rerun 页面。短周期缓存
    # 可以避免连续重复 os.walk，同时保证新增字体后最多 30 秒即可被发现。
    fonts = []
    for root, dirs, files in os.walk(font_dir):
        for file in files:
            if file.endswith(".ttf") or file.endswith(".ttc"):
                fonts.append(file)
    fonts.sort()
    return fonts


@st.cache_data(ttl=30, show_spinner=False)
def get_all_songs():
    # 背景音乐与字体使用相同的短周期策略，不做永久缓存，兼顾 rerun 性能和
    # 用户运行期间手动添加音乐文件的场景。
    songs = []
    for root, dirs, files in os.walk(song_dir):
        for file in files:
            if file.endswith(".mp3"):
                songs.append(file)
    return songs


def open_task_folder(task_id):
    try:
        # task_id 应始终是服务端生成的 UUID。这里先做格式校验，避免异常值
        # 通过路径拼接访问任务目录之外的位置，也避免后续打开目录时触发
        # 平台 shell 对特殊字符的解释。
        normalized_task_id = str(UUID(str(task_id)))
        tasks_root = os.path.abspath(os.path.join(root_dir, "storage", "tasks"))
        path = os.path.abspath(os.path.join(tasks_root, normalized_task_id))

        # 即使 UUID 校验通过，也再次确认最终路径仍在任务根目录内，避免
        # 未来调用方调整 task_id 来源时引入路径穿越风险。
        if not path.startswith(tasks_root + os.sep):
            logger.warning(f"invalid task folder path: {path}")
            return

        if os.path.isdir(path):
            webbrowser.open(f"file://{path}")
    except Exception as e:
        logger.exception(f"failed to open task folder: task_id={task_id}, error={e}")


@st.cache_resource
def init_log():
    # 基础日志 Handler 属于进程级资源，而不是页面会话状态。Streamlit 每次组件
    # 交互都会 rerun 页面脚本，代码热重载也可能让缓存失效。日志初始化只能
    # 精确替换终端 Handler，不能清空正在生成任务使用的 WebUI 临时 Handler。
    _lvl = "DEBUG"

    return configure_terminal_logger(
        sys.stdout,
        level=_lvl,
        colorize=True,
    )


init_log()


def tr_optional(key, fallback_language=""):
    loc = locales.get(st.session_state["ui_language"], {})
    value = loc.get("Translation", {}).get(key, "")
    if not value and fallback_language:
        fallback_loc = locales.get(fallback_language, {})
        value = fallback_loc.get("Translation", {}).get(key, "")
    return value if value else ""


def render_onboarding_tour():
    # 引导只覆盖三个稳定入口，不尝试控制 Dialog、Tabs 或业务表单。这样既能让
    # 新用户理解完整流程，也不会把引导状态与 Streamlit 的动态组件生命周期耦合。
    steps = [
        Tour.bind(
            "open_settings_dialog_button",
            title=tr("Onboarding Model Settings Title"),
            desc=tr("Onboarding Model Settings Description"),
            side="bottom",
            align="end",
        ),
        Tour.bind(
            "main_settings_grid",
            title=tr("Onboarding Creation Settings Title"),
            desc=tr("Onboarding Creation Settings Description"),
            side="top",
            align="center",
        ),
        Tour.bind(
            "generate_video_button",
            title=tr("Onboarding Generate Video Title"),
            desc=tr("Onboarding Generate Video Description"),
            side="top",
            align="center",
        ),
    ]

    # streamlit-tour 1.1.0 没有在 Python 构造参数中暴露导航文案，但底层
    # Driver.js 支持在每一步的 popover 配置中覆盖按钮文本。这里统一注入本地化
    # 文案，并对内容做 HTML 转义，因为组件会通过 innerHTML 渲染这些字段。
    previous_text = html.escape(tr("Onboarding Previous"))
    next_text = html.escape(tr("Onboarding Next"))
    done_text = html.escape(tr("Onboarding Done"))
    for index, step in enumerate(steps):
        step.popover["prevBtnText"] = f"&larr; {previous_text}"
        # Driver.js 会在合并单步配置时覆盖已经替换过变量的进度模板，因此直接
        # 写入当前步骤和总步骤数，避免页面显示未解析的 {{current}} 占位符。
        step.popover["progressText"] = f"{index + 1} / {len(steps)}"
        if index == len(steps) - 1:
            step.popover["doneBtnText"] = done_text
        else:
            step.popover["nextBtnText"] = f"{next_text} &rarr;"

    tour = Tour(
        steps=steps,
        key=ONBOARDING_TOUR_KEY,
        show_progress=True,
        animate=True,
        overlay_opacity=0.55,
        one_time_tour=True,
    )

    # 每个 Streamlit 会话只主动启动一次。是否已经完成则由组件通过浏览器
    # localStorage 判断，避免页面 rerun 或普通控件交互反复弹出引导。
    auto_start_key = f"{ONBOARDING_TOUR_KEY}-auto-started"
    if not st.session_state.get(auto_start_key, False):
        st.session_state[auto_start_key] = True
        tour.start()


def _render_generation_logs(task_id):
    """渲染后台任务日志快照，不从工作线程访问 Streamlit 会话状态。"""
    if config.ui.get("hide_log", False):
        return

    log_records = webui_task.get_task_logs(task_id)
    if not log_records:
        return

    st.code("\n".join(log_records))


def _render_generation_task_snapshot(task_id, task):
    """根据状态存储中的快照渲染进度、失败原因或最终成片。"""
    if not task:
        st.info(tr("Generating Video"))
        _render_generation_logs(task_id)
        return

    state = _normalize_task_state(task.get("state"))
    progress = max(0, min(100, int(task.get("progress", 0) or 0)))
    if state == const.TASK_STATE_PROCESSING:
        st.info(tr("Generating Video"))
        st.progress(
            progress,
            text=f"{tr('Task Progress')}: {progress}%",
        )
        _render_generation_logs(task_id)
        return

    if state == const.TASK_STATE_FAILED:
        error = str(task.get("error") or "").strip()
        message = tr("Video Generation Failed")
        st.error(f"{message}: {error}" if error else message)
        _render_generation_logs(task_id)
        return

    video_files = task.get("videos") or []
    if state != const.TASK_STATE_COMPLETE or not video_files:
        st.error(tr("Video Generation Failed"))
        _render_generation_logs(task_id)
        return

    st.success(tr("Video Generation Completed"))
    for warning in task.get("warnings") or []:
        if isinstance(warning, Mapping) and warning.get("code") == "sonilo_bgm_failed":
            st.warning(
                tr("Sonilo BGM Fallback Warning").format(
                    index=warning.get("video_index", "")
                )
            )
        elif (
            isinstance(warning, Mapping)
            and warning.get("code") == "elevenlabs_bgm_failed"
        ):
            st.warning(
                tr("ElevenLabs BGM Fallback Warning").format(
                    index=warning.get("video_index", "")
                )
            )
        else:
            st.warning(str(warning))

    try:
        player_cols = st.columns(len(video_files) * 2 + 1)
        for i, url in enumerate(video_files):
            with player_cols[i * 2 + 1]:
                st.video(url)
                if not os.path.isfile(url):
                    logger.warning(
                        f"generated video is unavailable for download: "
                        f"task_id={task_id}, video_file={url}"
                    )
                    continue

                download_label = tr("Download Video")
                if len(video_files) > 1:
                    download_label = f"{download_label} {i + 1}"
                download_name = _build_video_download_name(
                    task.get("video_subject"),
                    i + 1,
                    len(video_files),
                )
                with open(url, "rb") as video_file:
                    st.download_button(
                        download_label,
                        data=video_file,
                        file_name=download_name,
                        mime=mimetypes.guess_type(url)[0] or "video/mp4",
                        key=f"download_generated_video_{task_id}_{i}",
                        icon=":material/download:",
                        on_click="ignore",
                        use_container_width=True,
                    )
    except Exception as exc:
        logger.exception(
            f"failed to render generated video preview: task_id={task_id}, "
            f"video_files={video_files}, error={exc}"
        )

    _render_generation_logs(task_id)
    # Phase 2E.4: advanced users can inspect the agentic plan (strategy,
    # story brief, scene plan, titles, QA, decision log) when available.
    _render_agentic_plan_inspection(task_id)
    if st.session_state.get("handled_generation_task_id") != task_id:
        # Fragment 可能重复渲染同一个完成任务。无论是否开启自动打开目录，
        # 每个任务都只处理一次完成事件，避免重复弹出资源管理器或重复写入日志。
        st.session_state["handled_generation_task_id"] = task_id
        st.toast(tr("Video Generation Completed"), icon="🎉")
        st.balloons()
        if config.ui.get("open_task_folder_on_completion", True):
            open_task_folder(task_id)
        logger.info(f"{tr('Video Generation Completed')}: task_id={task_id}")


@st.fragment(run_every=webui_task.TASK_LOG_REFRESH_INTERVAL_SECONDS)
def _render_running_generation_task(task_id):
    """只在任务运行期间轮询；结束后切回静态结果，停止不必要的定时刷新。"""
    try:
        task = sm.state.get_task(task_id)
    except Exception as exc:
        logger.exception(
            f"failed to query WebUI generation task: task_id={task_id}, error={exc}"
        )
        st.error(tr("Video Generation Failed"))
        return

    state = _normalize_task_state((task or {}).get("state"))
    if state in {const.TASK_STATE_COMPLETE, const.TASK_STATE_FAILED}:
        _remove_active_generation_task(task_id)
        # 完整页面脚本现在没有耗时生成逻辑，可以安全 rerun 并把结果改为静态
        # 渲染。这样任务结束后不会让浏览器永久保留一个两秒轮询的 Fragment。
        st.rerun(scope="app")

    _render_generation_task_snapshot(task_id, task)


def _latest_running_generation_task():
    """从服务端状态存储恢复最近仍在运行的任务。

    页面刷新或会话重建后 session_state 丢失，``current_generation_task_id``
    为空，日志面板会跟着消失。任务状态与日志快照都保存在服务端进程里，
    这里直接查询状态存储兜底，让刷新后的页面继续显示运行中任务与日志。
    """
    try:
        tasks, _ = sm.state.get_all_tasks(1, 50)
    except Exception as exc:
        logger.exception(
            f"failed to query running tasks for log restore: error={exc}"
        )
        return "", None
    running = [
        task
        for task in tasks
        if _normalize_task_state(task.get("state"))
        == const.TASK_STATE_PROCESSING
    ]
    if not running:
        return "", None
    # MemoryState 按提交顺序保存任务，最后一个即最近提交的运行中任务。
    task = running[-1]
    return task.get("task_id") or "", task


def _render_current_generation_task():
    """在生成按钮下方恢复当前页面最近提交任务的可查询 UI。"""
    task_id = st.session_state.get("current_generation_task_id", "")
    task = None
    if task_id:
        try:
            task = sm.state.get_task(task_id)
        except Exception as exc:
            logger.exception(
                f"failed to query current WebUI task: task_id={task_id}, error={exc}"
            )
            st.error(tr("Video Generation Failed"))
            return

    if not task:
        # 会话丢失后根据服务端状态恢复运行中任务的日志面板。
        task_id, task = _latest_running_generation_task()
        if task_id:
            st.session_state["current_generation_task_id"] = task_id
        if not task:
            return

    state = _normalize_task_state(task.get("state"))
    if state in {const.TASK_STATE_COMPLETE, const.TASK_STATE_FAILED}:
        _remove_active_generation_task(task_id)
        _render_generation_task_snapshot(task_id, task)
        return

    _render_running_generation_task(task_id)


def get_llm_provider_tips(provider_id, **kwargs):
    # LLM provider 说明文案统一使用 `llm_provider_tips.<provider_id>` 规则。
    # 这样新增 provider 时只需要在 locale 中补文案；没有文案时不展示提示块，
    # 避免 Main.py 里继续堆叠大量中英文硬编码说明。
    provider = get_llm_provider(provider_id)
    if provider is None:
        return ""

    # Provider 配置说明目前统一维护中文和英文两套规范模板；其它界面语言
    # 统一使用英文，避免在 locale 中复制英文后长期不同步。后续某个语种完成
    # 全量翻译后，再将它加入这里的独立维护范围。
    ui_language = st.session_state.get("ui_language", "en")
    tips_language = ui_language if ui_language in {"zh", "en"} else "en"
    tips = (
        locales.get(tips_language, {}).get("Translation", {}).get(provider.tips_key, "")
    )
    if not tips:
        return tips

    format_context = {
        "api_key_url": (
            provider.international_api_key_url
            if tips_language == "en" and provider.international_api_key_url
            else provider.api_key_url
        ),
        "default_model": provider.default_model,
        "default_base_url": provider.default_base_url,
        **{
            f"default_{field.config_suffix}": field.default_value
            for field in provider.extra_fields
        },
        **kwargs,
    }
    try:
        return tips.format(**format_context)
    except Exception as e:
        logger.warning(f"format llm provider tips failed: {provider_id}, {e}")
        return tips


def get_llm_provider_label(provider):
    return tr_optional(provider.label_key) or provider.default_label


def get_tts_provider_tips(provider_id):
    # TTS 配置说明与 LLM Provider 采用相同维护策略：只维护中英文，
    # 其它界面语言统一回退英文，避免复制后长期不同步。
    ui_language = st.session_state.get("ui_language", "en")
    tips_language = ui_language if ui_language in {"zh", "en"} else "en"
    return (
        locales.get(tips_language, {})
        .get("Translation", {})
        .get(f"tts_provider_tips.{provider_id}", "")
    )


def localized_widget_key(name, *parts):
    # 部分 Streamlit selectbox 使用稳定 key 记住选择状态，但展示文本来自 locale。
    # 语言切换时把语言也放进 key，可以强制重建控件，避免选中项仍显示旧语言。
    language = st.session_state.get("ui_language", config.ui.get("language", ""))
    suffix_parts = [name, language, *[str(part) for part in parts if part]]
    return "_".join(suffix_parts)


def stable_selectbox(label, options, default_value, key, format_func=None, **kwargs):
    # Streamlit 1.59 对 selectbox 的状态复用更敏感：如果控件没有固定 key，
    # 或者真实选项只是一组临时下标，页面 rerun 后容易被重新计算的 index 覆盖，
    # 表现为用户第一次选择不生效、需要再选一次。这个 helper 统一用稳定业务值
    # 作为真实选项，并在 session_state 里保存该值；展示文案只通过 format_func
    # 转换，避免翻译文案、选项顺序或上游配置变化影响选择状态。
    options = list(options)
    if not options:
        raise ValueError(f"selectbox options cannot be empty: {key}")

    if default_value not in options:
        default_value = options[0]

    widget_key = localized_widget_key(key)
    selected_value = st.session_state.get(widget_key)
    accepts_custom_value = bool(kwargs.get("accept_new_options"))
    has_valid_custom_value = (
        accepts_custom_value
        and isinstance(selected_value, str)
        and bool(selected_value.strip())
    )
    if selected_value not in options and not has_valid_custom_value:
        # 如果上游选项发生变化（例如切换 TTS provider 后声音列表变了），
        # 旧值已经不合法。控件创建前直接初始化 session_state，之后只让 key
        # 管理状态，不再同时传入 index。这样可以避免 Streamlit 在 rerun 时
        # 用重新计算的 index 覆盖用户刚选择的值，导致第一次选择不生效。
        st.session_state[widget_key] = default_value

    if format_func is None:
        format_func = str

    return st.selectbox(
        label,
        options=options,
        format_func=format_func,
        key=widget_key,
        **kwargs,
    )


def sync_script_order_concat_mode():
    """在文案顺序匹配开启时固定使用顺序拼接，并在关闭后恢复原选择。"""
    widget_key = localized_widget_key("video_concat_mode_select")
    previous_key = "video_concat_mode_before_script_order_match"
    match_script_order = bool(st.session_state.get("match_materials_to_script", False))

    if match_script_order:
        current_mode = st.session_state.get(widget_key, VideoConcatMode.random.value)
        if current_mode != VideoConcatMode.sequential.value:
            st.session_state[previous_key] = current_mode
        st.session_state[widget_key] = VideoConcatMode.sequential.value
        return

    previous_mode = st.session_state.pop(previous_key, None)
    if previous_mode in {
        VideoConcatMode.sequential.value,
        VideoConcatMode.random.value,
    }:
        st.session_state[widget_key] = previous_mode


def reset_script_system_prompt():
    """将高级脚本设置中的系统提示词恢复为当前版本的默认内容。"""
    st.session_state["custom_system_prompt"] = llm.DEFAULT_SCRIPT_SYSTEM_PROMPT


def reset_subtitle_settings():
    """恢复 WebUI 字幕控件和持久化配置中的默认值。"""
    defaults = DEFAULT_SUBTITLE_SETTINGS
    st.session_state["subtitle_enabled_checkbox"] = defaults["subtitle_enabled"]
    _set_stable_widget_value("font_name_select", defaults["font_name"])
    _set_stable_widget_value("subtitle_position_select", defaults["subtitle_position"])
    _set_stable_widget_value("subtitle_casing_select", defaults["subtitle_casing"])
    _set_stable_widget_value(
        "subtitle_style_preset_select", defaults["subtitle_style_preset"]
    )
    st.session_state["custom_position_input"] = str(defaults["custom_position"])
    st.session_state["font_color_picker"] = defaults["text_fore_color"]
    st.session_state["font_size_slider"] = defaults["font_size"]
    st.session_state["stroke_color_picker"] = defaults["stroke_color"]
    st.session_state["stroke_width_slider"] = defaults["stroke_width"]
    st.session_state["subtitle_background_enabled_checkbox"] = defaults[
        "subtitle_background_enabled"
    ]
    st.session_state["subtitle_background_color_picker"] = defaults[
        "subtitle_background_color"
    ]
    st.session_state["rounded_subtitle_background_checkbox"] = defaults[
        "rounded_subtitle_background"
    ]
    st.session_state["subtitle_dynamic_sizing_checkbox"] = defaults[
        "subtitle_dynamic_sizing"
    ]
    st.session_state["subtitle_pop_in_bounce_checkbox"] = defaults[
        "subtitle_pop_in_bounce"
    ]
    st.session_state["subtitle_floating_motion_checkbox"] = defaults[
        "subtitle_floating_motion"
    ]
    st.session_state["subtitle_highlight_color_picker"] = defaults[
        "subtitle_highlight_color"
    ]
    st.session_state["subtitle_background_opacity_slider"] = defaults[
        "subtitle_background_opacity"
    ]
    st.session_state["subtitle_vertical_offset_slider"] = defaults[
        "subtitle_vertical_offset"
    ]
    st.session_state["subtitle_active_word_highlight_checkbox"] = defaults[
        "subtitle_active_word_highlight"
    ]
    st.session_state["subtitle_dynamic_auto_avoidance_checkbox"] = defaults[
        "subtitle_dynamic_auto_avoidance"
    ]

    # 同步会持久化的 UI 选项，确保恢复后刷新页面仍保持默认设置。
    for key in (
        "font_name",
        "subtitle_position",
        "subtitle_casing",
        "custom_position",
        "text_fore_color",
        "font_size",
        "subtitle_background_enabled",
        "subtitle_background_color",
        "rounded_subtitle_background",
        "subtitle_dynamic_sizing",
        "subtitle_pop_in_bounce",
        "subtitle_floating_motion",
        "subtitle_style_preset",
        "subtitle_highlight_color",
        "subtitle_background_opacity",
        "subtitle_vertical_offset",
        "subtitle_active_word_highlight",
        "subtitle_dynamic_auto_avoidance",
    ):
        _set_runtime_config("ui", key, defaults[key])


@st.dialog(tr("Final Prompt Preview"), width="large")
def render_script_prompt_preview(prompt):
    """展示将要发送给大模型的完整脚本生成提示词。"""
    st.code(prompt, language="markdown", wrap_lines=True)


def stable_segmented_control(
    label, options, default_value, key, format_func=None, **kwargs
):
    """使用稳定业务值创建单选分段控件，避免语言切换后状态被展示文案覆盖。"""
    options = list(options)
    if not options:
        raise ValueError(f"segmented control options cannot be empty: {key}")

    if default_value not in options:
        default_value = options[0]

    widget_key = localized_widget_key(key)
    if st.session_state.get(widget_key) not in options:
        st.session_state[widget_key] = default_value

    return st.segmented_control(
        label,
        options=options,
        selection_mode="single",
        required=True,
        format_func=format_func or str,
        key=widget_key,
        **kwargs,
    )


@st.cache_data(ttl=300, show_spinner=False)
def get_groq_model_ids(api_key: str, base_url: str) -> list[str]:
    if not api_key:
        return []

    normalized_base_url = (
        (base_url or "https://api.groq.com/openai/v1").strip().rstrip("/")
    )
    models_url = f"{normalized_base_url}/models"

    try:
        response = requests.get(
            models_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data", [])

        model_ids = []
        for item in data:
            if isinstance(item, dict):
                model_id = item.get("id")
                if isinstance(model_id, str) and model_id.strip():
                    model_ids.append(model_id.strip())

        return sorted(set(model_ids))
    except Exception as e:
        logger.warning(f"failed to fetch groq models: {e}")
        return []


def _get_material_api_keys(config_key):
    """将配置中的素材 API Key 统一转换为 WebUI 可编辑字符串。"""
    api_keys = config.app.get(config_key, [])
    if isinstance(api_keys, str):
        api_keys = [api_keys]
    return ", ".join(api_keys)


def _save_material_api_keys(config_key, value):
    """保存逗号分隔的素材 API Key，并允许用户显式清空旧配置。"""
    normalized_value = value.replace(" ", "")
    _set_runtime_config(
        "app",
        config_key,
        normalized_value.split(",") if normalized_value else [],
    )


def _format_file_size(size_bytes):
    """将字节数格式化为适合设置页展示的紧凑容量文本。"""
    size = float(max(0, size_bytes))
    units = ("B", "KB", "MB", "GB", "TB")
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.0f} {unit}" if unit in ("B", "KB") else f"{size:.2f} {unit}"
        size /= 1024
    return f"{size_bytes} B"


@st.cache_data(ttl=30, show_spinner=False)
def _get_video_cache_stats(max_age_days=None):
    """
    短周期缓存目录统计，避免设置弹窗内普通控件交互反复扫描大量文件。

    缓存键包含清理天数，因此切换范围只会为每个范围扫描一次；主动刷新或清理
    完成后会显式清空，最多 30 秒的缓存不会影响实际删除时的二次扫描。

    返回普通 dict 而不是 VideoCacheStats 对象：st.cache_data 会用 pickle
    序列化返回值，而热重载/进程残留可能导致内存中的类与磁盘不一致、无法
    序列化。纯 dict 永远可序列化，彻底消除这一类报错。
    """
    stats = cache_manager.get_video_cache_stats(max_age_days=max_age_days)
    return {
        "file_count": stats.file_count,
        "total_size": stats.total_size,
        "oldest_mtime": stats.oldest_mtime,
        "newest_mtime": stats.newest_mtime,
    }


def _render_cache_management_settings(panel):
    """渲染默认在线视频素材缓存的统计、预览和安全清理操作。"""
    with panel:
        cleanup_message = st.session_state.pop("video_cache_cleanup_message", None)
        if cleanup_message:
            message_type, message = cleanup_message
            if message_type == "success":
                st.success(message)
            else:
                st.warning(message)

        st.caption(tr("Video Cache Directory"))
        st.code(cache_manager.video_cache_dir(), language="text")

        total_stats = _get_video_cache_stats()
        metric_count, metric_size, metric_oldest = st.columns(3)
        metric_count.metric(tr("Cache File Count"), total_stats["file_count"])
        metric_size.metric(
            tr("Cache Total Size"), _format_file_size(total_stats["total_size"])
        )
        oldest_text = (
            datetime.fromtimestamp(total_stats["oldest_mtime"]).strftime("%Y-%m-%d")
            if total_stats["oldest_mtime"] is not None
            else "-"
        )
        metric_oldest.metric(tr("Oldest Cache Date"), oldest_text)

        st.caption(tr("Video Cache Management Help"))
        cleanup_options = (30, 7, 90, None)
        cleanup_labels = {
            30: tr("Cache Older Than 30 Days"),
            7: tr("Cache Older Than 7 Days"),
            90: tr("Cache Older Than 90 Days"),
            None: tr("All Video Cache"),
        }
        max_age_days = st.selectbox(
            tr("Cache Cleanup Range"),
            options=cleanup_options,
            format_func=lambda value: cleanup_labels[value],
            key="video_cache_cleanup_range",
        )
        cleanup_preview = _get_video_cache_stats(max_age_days=max_age_days)
        st.info(
            tr("Cache Cleanup Preview").format(
                count=cleanup_preview["file_count"],
                size=_format_file_size(cleanup_preview["total_size"]),
            )
        )

        confirm_nonce = st.session_state.get("video_cache_cleanup_confirm_nonce", 0)
        confirmed = st.checkbox(
            tr("Confirm Cache Cleanup"),
            key=f"video_cache_cleanup_confirm_{confirm_nonce}",
        )
        refresh_col, open_col, cleanup_col = st.columns(3)
        if refresh_col.button(
            tr("Refresh Cache Stats"),
            key="refresh_video_cache_stats",
            icon=":material/refresh:",
            use_container_width=True,
        ):
            _get_video_cache_stats.clear()
            st.rerun(scope="fragment")

        if open_col.button(
            tr("Open Cache Directory"),
            key="open_video_cache_directory",
            icon=":material/folder_open:",
            use_container_width=True,
        ):
            webbrowser.open(Path(cache_manager.video_cache_dir()).as_uri())

        cleanup_disabled = not confirmed or cleanup_preview["file_count"] == 0
        if cleanup_col.button(
            tr("Clean Cache Now"),
            key="clean_video_cache_now",
            type="primary",
            disabled=cleanup_disabled,
            icon=":material/delete_sweep:",
            use_container_width=True,
        ):
            result = cache_manager.clean_video_cache(max_age_days=max_age_days)
            message_key = (
                "Cache Cleanup Completed With Failures"
                if result.failed_count
                else "Cache Cleanup Completed"
            )
            st.session_state["video_cache_cleanup_message"] = (
                "warning" if result.failed_count else "success",
                tr(message_key).format(
                    count=result.deleted_count,
                    size=_format_file_size(result.deleted_size),
                    failed=result.failed_count,
                ),
            )
            # Streamlit 不允许在控件实例化后修改同名 session_state。通过递增
            # nonce 让下一次 fragment rerun 创建未勾选的新控件，避免清理完成后
            # 危险确认状态被继续保留。
            st.session_state["video_cache_cleanup_confirm_nonce"] = confirm_nonce + 1
            _get_video_cache_stats.clear()
            st.rerun(scope="fragment")


# -----------------------------------------------------------------------------
# 设置与提示词弹窗
# -----------------------------------------------------------------------------


# 设置属于低频操作，使用中等尺寸 Dialog 避免长期占用主页面纵向空间，
# 同时控制阅读行宽，避免弹窗在宽屏设备上显得过于松散。
# Dialog 继承 fragment 行为，内部控件交互只重绘弹窗；函数末尾单独保存配置，
# 关闭时通过回调触发整页同步，确保生成流程读取最新 Provider 和界面设置。
@st.dialog(
    tr("Settings"),
    width="medium",
    on_dismiss=_dismiss_settings_dialog,
)
def _render_settings_dialog():
    with st.container():
        # 历史 hide_config 只用于隐藏旧基础设置面板。改为固定设置入口后，该值
        # 不再有用户可见意义，统一迁移为 false，避免旧配置影响后续版本。
        _set_runtime_config("app", "hide_config", False)
        (
            middle_config_panel,
            right_config_panel,
            cache_config_panel,
            left_config_panel,
        ) = st.tabs(
            [
                tr("LLM Settings Tab"),
                tr("Material API Tab"),
                tr("Cache Management Tab"),
                tr("Interface Settings Tab"),
            ]
        )

        # 左侧面板 - 日志设置
        with left_config_panel:
            hide_log = st.checkbox(
                tr("Hide Log"),
                value=config.ui.get("hide_log", False),
                key="hide_log_checkbox",
            )
            _set_runtime_config("ui", "hide_log", hide_log)

        _render_cache_management_settings(cache_config_panel)

        # 中间面板 - LLM 设置

        with middle_config_panel:
            # 下拉顺序、默认 label 和稳定 provider id 全部来自 Registry；locale
            # 只覆盖展示文案，不再让 Main.py 维护第二份 Provider 列表。
            llm_provider_ids = [
                provider.provider_id for provider in LLM_PROVIDER_REGISTRY
            ]
            llm_provider_labels = {
                provider.provider_id: get_llm_provider_label(provider)
                for provider in LLM_PROVIDER_REGISTRY
            }
            saved_llm_provider = config.app.get(
                "llm_provider", DEFAULT_LLM_PROVIDER_ID
            ).lower()
            if saved_llm_provider not in llm_provider_ids:
                saved_llm_provider = DEFAULT_LLM_PROVIDER_ID

            llm_provider = stable_selectbox(
                tr("LLM Provider"),
                options=llm_provider_ids,
                default_value=saved_llm_provider,
                key="llm_provider_select",
                format_func=lambda provider_id: llm_provider_labels[provider_id],
            )
            # 配置表单和 Provider 说明并排展示，减少长说明在窄列中的换行，
            # 同时充分利用基础设置面板的横向空间。
            llm_form_panel, llm_help_panel = st.columns(
                [0.9, 1.1],
                gap="large",
                vertical_alignment="top",
            )
            llm_helper = llm_help_panel.container()
            _set_runtime_config("app", "llm_provider", llm_provider)
            llm_provider_spec = get_llm_provider(llm_provider)
            if llm_provider_spec is None:
                # 正常情况下下拉选项全部来自 Registry，不会进入该分支；保留
                # 明确错误用于诊断损坏的 session state 或后续接入遗漏。
                raise RuntimeError(f"unsupported llm provider: {llm_provider}")

            llm_api_key = config.app.get(llm_provider_spec.config_key("api_key"), "")
            llm_base_url = (
                config.app.get(llm_provider_spec.config_key("base_url"), "")
                or llm_provider_spec.default_base_url
            )
            llm_default_base_url = llm_provider_spec.default_base_url
            llm_model_name = llm_provider_spec.resolve_model_name(
                config.app.get(llm_provider_spec.config_key("model_name"), "")
            )

            provider_tip_context = {}
            if llm_provider == "ollama":
                llm_default_base_url = config.get_default_ollama_base_url()
                if not llm_base_url:
                    llm_base_url = llm_default_base_url
                docker_hint = ""
                if config.is_running_in_container():
                    docker_hint = tr_optional(
                        "llm_provider_tips.ollama.docker_hint",
                        fallback_language="en",
                    )
                provider_tip_context["docker_hint"] = docker_hint

            tips = get_llm_provider_tips(llm_provider, **provider_tip_context)
            if tips:
                with llm_helper:
                    st.info(tips)

            st_llm_api_key = llm_api_key
            if llm_provider_spec.show_api_key:
                st_llm_api_key = llm_form_panel.text_input(
                    tr("API Key"),
                    value=llm_api_key,
                    type="password",
                    key=f"{llm_provider}_api_key_input",
                )
                # 可选备用 Key：主 Key 失败（额度耗尽 / Key 无效 / 限流）时按
                # 顺序自动尝试。用 toggle 开关收纳，节省空间；关闭时清空备用
                # 配置（toggle 即权威开关），避免对话框中 expander 的渲染异常。
                has_fallback = bool(
                    config.app.get("llm_fallback_api_keys", "")
                    or config.app.get("llm_fallback_base_url", "")
                    or config.app.get("llm_fallback_model_name", "")
                )
                fallback_enabled = llm_form_panel.toggle(
                    tr("LLM Fallback Settings"),
                    value=has_fallback,
                    key="llm_fallback_enabled_toggle",
                )
                if fallback_enabled:
                    st_llm_fallback_api_keys = llm_form_panel.text_input(
                        tr("LLM Fallback API Keys"),
                        value=config.app.get("llm_fallback_api_keys", ""),
                        type="password",
                        key="llm_fallback_api_keys_input",
                        help=tr("LLM Fallback API Keys Hint"),
                        placeholder=tr("LLM Fallback API Keys Placeholder"),
                    )
                    _set_runtime_config(
                        "app",
                        "llm_fallback_api_keys",
                        st_llm_fallback_api_keys.strip(),
                    )

                    st_llm_fallback_base_url = llm_form_panel.text_input(
                        tr("LLM Fallback Base Url"),
                        value=config.app.get("llm_fallback_base_url", ""),
                        key="llm_fallback_base_url_input",
                        help=tr("LLM Fallback Base Url Hint"),
                        placeholder=tr("LLM Fallback Base Url Placeholder"),
                    )
                    _set_runtime_config(
                        "app",
                        "llm_fallback_base_url",
                        st_llm_fallback_base_url.strip(),
                    )

                    st_llm_fallback_model_name = llm_form_panel.text_input(
                        tr("LLM Fallback Model Name"),
                        value=config.app.get("llm_fallback_model_name", ""),
                        key="llm_fallback_model_name_input",
                        help=tr("LLM Fallback Model Name Hint"),
                        placeholder=tr("LLM Fallback Model Name Placeholder"),
                    )
                    _set_runtime_config(
                        "app",
                        "llm_fallback_model_name",
                        st_llm_fallback_model_name.strip(),
                    )
                elif has_fallback:
                    # 关闭开关时清空备用配置，让 LLM 调用不再尝试备用 Key。
                    _set_runtime_config("app", "llm_fallback_api_keys", "")
                    _set_runtime_config("app", "llm_fallback_base_url", "")
                    _set_runtime_config("app", "llm_fallback_model_name", "")

            st_llm_base_url = llm_base_url
            if llm_provider_spec.show_base_url:
                st_llm_base_url = llm_form_panel.text_input(
                    tr("Base Url"),
                    value=llm_base_url,
                    key=f"{llm_provider}_base_url_input",
                )
            st_llm_model_name = ""
            if llm_provider == "groq":
                effective_api_key = st_llm_api_key or llm_api_key
                effective_base_url = st_llm_base_url or llm_base_url
                groq_models = get_groq_model_ids(
                    api_key=effective_api_key,
                    base_url=effective_base_url,
                )

                if groq_models:
                    selected_index = 0
                    if llm_model_name in groq_models:
                        selected_index = groq_models.index(llm_model_name)

                    st_llm_model_name = llm_form_panel.selectbox(
                        tr("Model Name"),
                        options=groq_models,
                        index=selected_index,
                        key="groq_model_name_select",
                    )
                else:
                    st_llm_model_name = llm_form_panel.text_input(
                        tr("Model Name"),
                        value=llm_model_name,
                        key="groq_model_name_input",
                    )
                    if effective_api_key:
                        llm_form_panel.caption(tr("Groq Model List Load Failed"))
                    else:
                        llm_form_panel.caption(
                            tr("Groq API Key Required for Model List")
                        )
            else:
                st_llm_model_name = llm_form_panel.text_input(
                    tr("Model Name"),
                    value=llm_model_name,
                    key=f"{llm_provider}_model_name_input",
                )
            # 输入框展示 Registry 默认值，但配置只保存真实的用户覆盖值。
            # 这样默认模型、Base URL 更新后，未自定义的用户能够自动跟随。
            _set_runtime_config(
                "app",
                llm_provider_spec.config_key("api_key"),
                st_llm_api_key,
            )
            _set_runtime_config(
                "app",
                llm_provider_spec.config_key("base_url"),
                normalize_provider_override(
                    st_llm_base_url,
                    llm_default_base_url,
                ),
            )
            _set_runtime_config(
                "app",
                llm_provider_spec.config_key("model_name"),
                normalize_provider_override(
                    st_llm_model_name,
                    llm_provider_spec.default_model,
                ),
            )

            # Provider 专用字段也由 Registry 声明。例如 Cloudflare AI Gateway
            # 需要 Account ID；以后新增类似字段时无需再在 Main.py 增加判断。
            for field in llm_provider_spec.extra_fields:
                field_config_key = llm_provider_spec.config_key(field.config_suffix)
                field_value = llm_form_panel.text_input(
                    tr(field.label_key),
                    value=(config.app.get(field_config_key, "") or field.default_value),
                    type="password" if field.secret else "default",
                    key=f"{llm_provider}_{field.config_suffix}_input",
                )
                _set_runtime_config(
                    "app",
                    field_config_key,
                    normalize_provider_override(
                        field_value,
                        field.default_value,
                    ),
                )

            if _action_button_clicked(
                "test_llm_connection",
                llm_form_panel.button(
                    tr("Test LLM Connection"),
                    key="test_llm_connection_button",
                    use_container_width=True,
                    type="secondary",
                    icon=":material/network_check:",
                    disabled=not _action_ready("test_llm_connection"),
                ),
            ):
                with config.try_runtime_config_lock() as lock_acquired:
                    if not lock_acquired:
                        llm_form_panel.warning(tr("Runtime Configuration Busy"))
                    else:
                        with llm_form_panel.spinner(tr("Testing LLM Connection")):
                            connection_ok, connection_error, connection_elapsed = (
                                llm.test_connection()
                            )

                if not lock_acquired:
                    connection_ok = None
                elif connection_ok:
                    llm_form_panel.success(
                        tr("LLM Connection Test Succeeded").format(
                            provider=llm_provider_labels[llm_provider],
                            model=st_llm_model_name or "-",
                            elapsed=f"{connection_elapsed:.2f}",
                        )
                    )
                else:
                    llm_form_panel.error(
                        tr("LLM Connection Test Failed").format(error=connection_error)
                    )
            # 探测完成后才进入冷却窗口：Streamlit 排队的重复点击在操作结束后
            # 才处理，此时标记才能拦住它。
            _action_mark_triggered("test_llm_connection")

        # 右侧面板 - API 密钥设置
        with right_config_panel:
            # 免费模式徽章：没有配置任何付费供应商时，Pollinations 免费来源自动
            # 生效，零 API Key 也能生成视频素材。
            if custom_media_service.is_pollinations_enabled():
                st.info(
                    tr("Free Mode Active")
                    if hasattr(tr, "__call__")
                    else "Free Mode: Pollinations provides free image material, "
                    "no API key needed",
                    icon=":material/auto_awesome:",
                )
            pexels_api_key = _get_material_api_keys("pexels_api_keys")
            pexels_api_key = st.text_input(
                tr("Pexels API Key"),
                value=pexels_api_key,
                type="password",
                key="pexels_api_keys_input",
            )
            _save_material_api_keys("pexels_api_keys", pexels_api_key)

            pixabay_api_key = _get_material_api_keys("pixabay_api_keys")
            pixabay_api_key = st.text_input(
                tr("Pixabay API Key"),
                value=pixabay_api_key,
                type="password",
                key="pixabay_api_keys_input",
            )
            _save_material_api_keys("pixabay_api_keys", pixabay_api_key)

            coverr_api_key = _get_material_api_keys("coverr_api_keys")
            coverr_api_key = st.text_input(
                tr("Coverr API Key"),
                value=coverr_api_key,
                type="password",
                key="coverr_api_keys_input",
            )
            _save_material_api_keys("coverr_api_keys", coverr_api_key)

            st.divider()
            st.caption(
                tr("Custom Media API")
                if hasattr(tr, "__call__")
                else "Custom Media API & Web Scraping"
            )

            _CUSTOM_API_PRESETS = {
                "Generic (Standard)": {},
                "Google Gemini (Nano Banana / Veo)": {
                    "custom_api_video_url": "https://generativelanguage.googleapis.com/v1beta/interactions",
                    "custom_api_image_url": "https://generativelanguage.googleapis.com/v1beta/interactions",
                    "custom_api_response_format": "gemini",
                    "custom_api_video_model": "gemini-omni-flash-preview",
                    "custom_api_image_model": "gemini-3.1-flash-image",
                },
                "OpenAI (Images / Sora)": {
                    "custom_api_video_url": "https://api.openai.com/v1/videos",
                    "custom_api_image_url": "https://api.openai.com/v1/images/generations",
                    "custom_api_response_format": "openai",
                },
                "fal.ai (Nano Banana / Veo)": {
                    "custom_api_video_url": "https://queue.fal.run/fal-ai/veo",
                    "custom_api_image_url": "https://queue.fal.run/fal-ai/google/nano-banana",
                    "custom_api_response_format": "standard",
                },
            }
            preset_options = list(_CUSTOM_API_PRESETS.keys())
            current_preset = config.app.get("custom_api_provider_preset", "")
            selected_preset = st.selectbox(
                tr("Custom API Provider")
                if hasattr(tr, "__call__")
                else "Custom API Provider",
                options=preset_options,
                index=(
                    preset_options.index(current_preset)
                    if current_preset in preset_options
                    else 0
                ),
                key="custom_api_provider_select",
                help="Presets fill in the endpoints, response format, and models for "
                "common providers. Google Gemini is natively supported (Nano Banana "
                "images + Omni Flash / Veo videos, downloaded automatically). Other "
                "presets assume an adapter endpoint that returns the expected JSON.",
            )
            if selected_preset != current_preset:
                for key, value in _CUSTOM_API_PRESETS[selected_preset].items():
                    _set_runtime_config("app", key, value)
                _set_runtime_config(
                    "app", "custom_api_provider_preset", selected_preset
                )

            custom_api_url = config.app.get("custom_api_url", "")
            custom_api_url = st.text_input(
                tr("Custom API URL") if hasattr(tr, "__call__") else "Custom API URL",
                value=custom_api_url,
                placeholder="https://api.example.com/v1/videos/generate",
                key="custom_api_url_input",
            )
            _set_runtime_config("app", "custom_api_url", custom_api_url)

            custom_api_video_url = config.app.get("custom_api_video_url", "")
            custom_api_video_url = st.text_input(
                tr("Custom API Video URL (Optional)")
                if hasattr(tr, "__call__")
                else "Custom API Video URL (Optional)",
                value=custom_api_video_url,
                placeholder="https://api.example.com/v1/videos/generate",
                help="Optional. Overrides Custom API URL for the video endpoint. "
                "Queried first; images are used only when it returns nothing.",
                key="custom_api_video_url_input",
            )
            _set_runtime_config("app", "custom_api_video_url", custom_api_video_url)

            custom_api_image_url = config.app.get("custom_api_image_url", "")
            custom_api_image_url = st.text_input(
                tr("Custom API Image URL (Optional)")
                if hasattr(tr, "__call__")
                else "Custom API Image URL (Optional)",
                value=custom_api_image_url,
                placeholder="https://api.example.com/v1/images/generate",
                help="Optional. Overrides Custom API URL for the image endpoint. "
                "Used automatically only when the video endpoint returns no clips.",
                key="custom_api_image_url_input",
            )
            _set_runtime_config("app", "custom_api_image_url", custom_api_image_url)

            custom_api_key = config.app.get("custom_api_key", "")
            custom_api_key = st.text_input(
                tr("Custom API Key") if hasattr(tr, "__call__") else "Custom API Key",
                value=custom_api_key,
                type="password",
                placeholder="Bearer token or API key",
                key="custom_api_key_input",
            )
            _set_runtime_config("app", "custom_api_key", custom_api_key)

            with st.expander("📖 How to get your API key", expanded=False):
                st.markdown(
                    "Pick a provider preset above, then paste its API key.\n\n"
                    "**Google Gemini** (native): create a key at "
                    "https://aistudio.google.com/apikey. Nano Banana generates images, "
                    "Omni Flash / Veo generate videos; clips are downloaded automatically.\n\n"
                    "**Other providers** (OpenAI, fal.ai): the preset fills in the endpoint "
                    "and response format. Providers that return base64 data or async jobs "
                    "need a small adapter endpoint that returns the standard "
                    "`{\"videos\": [{\"url\": ...}]}` JSON.\n\n"
                    "For a custom backend, copy your endpoint URL and key, then configure "
                    "the HTTP method and response format in the advanced settings below."
                )

            with st.expander(
                ":material/settings: Advanced Custom Endpoint Settings", expanded=False
            ):
                custom_api_method = config.app.get("custom_api_method", "POST")
                custom_api_method = st.selectbox(
                    tr("Custom API Method")
                    if hasattr(tr, "__call__")
                    else "Custom API Method",
                    options=["POST", "GET"],
                    index=0 if custom_api_method.upper() == "POST" else 1,
                    key="custom_api_method_select",
                    help="HTTP method used for the request. Most generation APIs require POST.",
                )
                _set_runtime_config("app", "custom_api_method", custom_api_method)

                custom_api_response_format = config.app.get(
                    "custom_api_response_format", "standard"
                )
                fmt_options = ["standard", "openai", "url_list", "gemini"]
                fmt_index = (
                    fmt_options.index(custom_api_response_format)
                    if custom_api_response_format in fmt_options
                    else 0
                )
                custom_api_response_format = st.selectbox(
                    tr("Custom API Response Format")
                    if hasattr(tr, "__call__")
                    else "Custom API Response Format",
                    options=["standard", "openai", "url_list", "gemini"],
                    index=fmt_index,
                    help="standard: {videos:[...]}, openai: {data:[...]}, "
                    "url_list: [\"url\",...], gemini: Google Interactions API "
                    "(Nano Banana / Veo / Omni Flash, native support).",
                    key="custom_api_response_format_select",
                )
                _set_runtime_config(
                    "app", "custom_api_response_format", custom_api_response_format
                )

                custom_api_video_model = config.app.get(
                    "custom_api_video_model", "gemini-omni-flash-preview"
                )
                custom_api_video_model = st.text_input(
                    tr("Custom API Video Model")
                    if hasattr(tr, "__call__")
                    else "Custom API Video Model",
                    value=custom_api_video_model,
                    help="Model id for the video endpoint with the gemini format "
                    "(e.g. gemini-omni-flash-preview, veo-3.1-generate-001).",
                    key="custom_api_video_model_input",
                )
                _set_runtime_config(
                    "app", "custom_api_video_model", custom_api_video_model
                )

                custom_api_image_model = config.app.get(
                    "custom_api_image_model", "gemini-3.1-flash-image"
                )
                custom_api_image_model = st.text_input(
                    tr("Custom API Image Model")
                    if hasattr(tr, "__call__")
                    else "Custom API Image Model",
                    value=custom_api_image_model,
                    help="Model id for the image endpoint with the gemini format "
                    "(e.g. gemini-3.1-flash-image, gemini-3-pro-image).",
                    key="custom_api_image_model_input",
                )
                _set_runtime_config(
                    "app", "custom_api_image_model", custom_api_image_model
                )

                custom_api_extra_body = config.app.get("custom_api_extra_body", "")
                custom_api_extra_body = st.text_input(
                    tr("Custom API Extra Body (JSON)")
                    if hasattr(tr, "__call__")
                    else "Custom API Extra Body (JSON)",
                    value=custom_api_extra_body,
                    placeholder='{"model": "runway-gen3", "duration": 5}',
                    help="Additional JSON payload to send with the request.",
                    key="custom_api_extra_body_input",
                )
                _set_runtime_config(
                    "app", "custom_api_extra_body", custom_api_extra_body
                )

            hybrid_video_mode = config.app.get("hybrid_video_mode", True)
            hybrid_video_mode = st.toggle(
                tr("Hybrid Mode: Pexels Fallback")
                if hasattr(tr, "__call__")
                else "Hybrid Mode: Pexels Fallback",
                value=bool(hybrid_video_mode),
                help="When Custom API returns no results, automatically fall back to Pexels.",
                key="hybrid_video_mode_toggle",
            )
            _set_runtime_config("app", "hybrid_video_mode", hybrid_video_mode)

            enable_web_scraping = config.app.get("enable_web_scraping", False)
            enable_web_scraping = st.toggle(
                tr("Enable Web Video Scraping")
                if hasattr(tr, "__call__")
                else "Enable Web Video Scraping",
                value=bool(enable_web_scraping),
                help="Allow the system to fetch public web videos (via yt-dlp) alongside stock APIs when using Auto mode.",
                key="enable_web_scraping_toggle",
            )
            _set_runtime_config("app", "enable_web_scraping", enable_web_scraping)

    _save_runtime_config()


# -----------------------------------------------------------------------------
# 主生成表单：文案、视频、音频与字幕面板
# -----------------------------------------------------------------------------


def _render_state_status(key, running_label):
    """Render a persisted operation status from session state (state level).

    Long-running operations (Discover Topics, Import Profile, script
    generation) write a status record into session_state when they finish.
    This helper renders that record on EVERY rerun, so the outcome survives
    page interactions instead of flashing only during the click run.

    ``key`` is the session_state key; ``running_label`` is only a fallback
    when the record has no label. Records look like:
    {"state": "running"|"complete"|"error", "label": str, "detail": str,
     "log": [stage lines...]}
    """
    record = st.session_state.get(key)
    if not isinstance(record, dict) or not record.get("state"):
        return
    state = record["state"]
    label = str(record.get("label") or running_label)[:200]
    detail = str(record.get("detail") or "")[:400]
    log = record.get("log") or []
    if not isinstance(log, list):
        log = []
    log_lines = [str(line)[:120] for line in log[-10:]]
    if state == "running":
        st.status(label, expanded=True, state="running")
    elif state == "error":
        st.status(label, expanded=True, state="error")
        if detail:
            st.caption(detail)
    else:
        st.status(label, expanded=False, state="complete")
        if detail:
            st.caption(detail)
    # 阶段日志：显示每个代理阶段（研究/策略/脚本/QA 等）的执行轨迹，
    # 运行中和结束后都保留，让用户清楚看到长操作进行到哪一步。
    if log_lines:
        st.markdown("  \n".join(f"- {line}" for line in log_lines))


# 操作去重窗口（秒）：按钮点击后在这个窗口内再次点击会被吞掉。防止用户
# 双击或连点同一个按钮时，Streamlit 的排队 rerun 把同一个 LLM 调用执行两遍。
_ACTION_COOLDOWN_SECONDS = 2.0


# 代理阶段键 -> i18n 键：agentic 规划图每个阶段的进度汇报映射到可读文案。
# 键名与 app/services/agentic.py 中 progress_cb 上报的阶段一致。
_AGENT_STAGE_KEYS = {
    "intelligence": "Agent Stage Intelligence",
    "research": "Agent Stage Research",
    "analysis": "Agent Stage Analysis",
    "strategy": "Agent Stage Strategy",
    "hooks": "Agent Stage Hooks",
    "narrative": "Agent Stage Narrative",
    "script": "Agent Stage Script",
    "script_revision": "Agent Stage Script Revision",
    "visuals": "Agent Stage Visuals",
    "titles": "Agent Stage Titles",
    "qa": "Agent Stage QA",
}


def _agent_stage_label(stage_key):
    """Translate an agentic stage key into a readable, localized label."""
    i18n_key = _AGENT_STAGE_KEYS.get(str(stage_key or "").strip().lower())
    return tr(i18n_key) if i18n_key else str(stage_key or "")


def _action_last_run(op_key):
    """操作最近一次触发的时间戳（session 级，跨 rerun 保留）。"""
    return float(st.session_state.get(f"_action_last_{op_key}", 0.0))


def _action_ready(op_key):
    """该操作是否允许再次触发（距上次触发已超过冷却窗口）。"""
    return time.time() - _action_last_run(op_key) >= _ACTION_COOLDOWN_SECONDS


def _action_mark_triggered(op_key):
    """记录操作触发时间，进入冷却窗口。"""
    st.session_state[f"_action_last_{op_key}"] = time.time()


def _action_button_clicked(op_key, clicked):
    """合并冷却窗口与按钮返回值：冷却期内即使按钮返回 True 也吞掉这次点击。

    Streamlit 会在页面 rerun 期间把用户的第二次点击排队，等当前脚本执行完
    后再以新的 rerun 处理；若不拦截，同一个 LLM 操作会被执行两遍。冷却时间
    由操作结束时（_run_with_state_status 的 finally / 调用方）统一标记，
    长操作结束后排队的重复点击必然落在冷却窗口内。
    """
    if not clicked:
        return False
    return _action_ready(op_key)


def _generation_in_progress():
    """是否有生成任务正在后台运行（页面刷新/会话重建后也能正确判断）。"""
    try:
        tasks, _ = sm.state.get_all_tasks(1, 50)
    except Exception as exc:  # noqa: BLE001 - 只是 UI 判断，失败按无任务处理
        logger.debug(f"failed to query tasks for button state: {exc}")
        return False
    for task in tasks:
        if _normalize_task_state(task.get("state")) == const.TASK_STATE_PROCESSING:
            return True
    return False


class UserFacingError(RuntimeError):
    """用户可直接阅读的操作失败（如 API Key 无效、限流）。

    抛出该异常时，_run_with_state_status 会以 st.error 呈现简洁信息并正常
    返回，而不是把原始堆栈抛给 Streamlit 展示。
    """


def _friendly_llm_error(message: str) -> str:
    """把 LLM 层返回的原始错误串压缩成一句用户能直接读懂的提示。"""
    raw = str(message or "").strip()
    lowered = raw.lower()
    if "invalid_api_key" in lowered or "incorrect api key" in lowered or " 401" in lowered:
        return tr("LLM API Key Invalid")
    if " 429" in lowered or "rate limit" in lowered:
        return tr("LLM Rate Limited")
    if "quota" in lowered or "billing" in lowered or "exceeded" in lowered:
        return tr("LLM Quota Exceeded")
    if "timeout" in lowered or "timed out" in lowered:
        return tr("LLM Timeout")
    if (
        "connection" in lowered
        or "network" in lowered
        or "unreachable" in lowered
        or "proxy" in lowered
        or "resolve" in lowered
    ):
        return tr("LLM Unreachable")
    # 兜底：去掉 "Error: " 前缀，只保留第一行有意义的文本。
    line = next(
        (ln.strip() for ln in raw.splitlines() if ln.strip()),
        "",
    )
    return line.removeprefix("Error:").strip() or tr("LLM Request Failed")


def _run_with_state_status(
    key,
    running_label,
    done_label,
    operation,
    success_detail="",
    error_detail=None,
    stage_reporter=False,
):
    """Run a long operation with a live spinner AND a persisted state record.

    - Shows ``st.status`` live while the operation runs (spinner).
    - Persists the outcome in ``session_state[key]`` so ``_render_state_status``
      keeps showing it on later reruns (state level, not just click run).
    - Re-raises failures so callers keep their existing error handling.

    ``error_detail`` may be a callable receiving the exception.

    ``stage_reporter``: when True, the operation is called as
    ``operation(report_stage)`` where ``report_stage(stage_key)`` appends a
    translated stage line to the persisted log AND renders it live inside the
    status widget — so users see which agent stage (research / strategy /
    script / QA) is running while it runs, and the trail survives reruns.
    """
    st.session_state[key] = {"state": "running", "label": running_label, "log": []}
    try:
        with st.status(running_label, expanded=True) as status:
            log_box = st.empty()

            def report_stage(stage_key):
                stage_label = _agent_stage_label(stage_key)
                log = list(st.session_state[key].get("log") or [])
                log.append(stage_label)
                st.session_state[key]["log"] = log
                log_box.markdown(
                    "  \n".join(f"- {line}" for line in log[-10:])
                )
                status.update(label=f"{running_label} · {stage_label}")

            if stage_reporter:
                result = operation(report_stage)
            else:
                result = operation()
            st.session_state[key] = {
                "state": "complete",
                "label": done_label,
                "detail": success_detail() if callable(success_detail) else str(success_detail or ""),
                "log": st.session_state[key].get("log") or [],
            }
            status.update(label=done_label, state="complete", expanded=False)
        return result
    except BaseException as exc:  # noqa: BLE001 - record outcome for BaseException too
        # 操作内部主动调用 st.rerun()（成功路径）时抛出的 RerunException
        # 继承自 BaseException：不能当作失败记录，要在 rerun 前把状态标记为
        # 完成，否则下一次渲染会一直显示陈旧的 "running"。
        if type(exc).__name__ == "RerunException":
            st.session_state[key] = {
                "state": "complete",
                "label": done_label,
                "detail": success_detail() if callable(success_detail) else str(success_detail or ""),
                "log": st.session_state[key].get("log") or [],
            }
            raise
        if isinstance(exc, UserFacingError):
            # 用户可读的失败：直接以简洁错误呈现，不抛堆栈。
            detail = str(exc)
            st.session_state[key] = {
                "state": "error",
                "label": done_label,
                "detail": detail[:400],
                "log": st.session_state[key].get("log") or [],
            }
            st.error(detail)
            return None
        detail = (
            error_detail(exc)
            if callable(error_detail)
            else (str(error_detail) if error_detail else str(exc))
        )
        st.session_state[key] = {
            "state": "error",
            "label": done_label,
            "detail": str(detail)[:400],
            "log": st.session_state[key].get("log") or [],
        }
        raise
    finally:
        # 冷却窗口在操作结束时才开始计时。Streamlit 会把操作执行期间的第二次
        # 点击排队到操作结束后处理；如果按点击时间标记，长操作（数十秒）结束后
        # 冷却早已过期，排队点击仍会把同一个操作再执行一遍。
        _action_mark_triggered(key)


def _discover_and_store_topics(params):
    """Phase 2E.4: discover scored topic candidates and cache them in the
    session so the user can adopt one into the subject field.

    Never raises: topic discovery is a convenience feature, so provider
    failures degrade to a session note instead of blocking the page.
    """
    try:
        profile_name = getattr(params, "content_profile", "") or "custom"
        profile = content_profile.get_content_profile(profile_name)
        intelligence = None
        # 用 tracker 观察 trend_analyst 是否因 LLM 失败而静默回落：
        # 模型推断失败（如 API 额度耗尽、Key 失效）不会抛出异常，而是返回
        # 空候选。只有同时看 tracker 才能把“LLM 失败”与“真的没有选题”区分开。
        tracker = agent_llm.AgentTracker()
        candidates = trends_service.discover_topics(
            profile,
            intelligence=intelligence,
            mode=getattr(params, "topic_discovery_mode", "") or trends_service.TOPIC_MODE_TRENDING,
            niche=getattr(params, "niche", "") or profile_name,
            max_candidates=6,
            tracker=tracker,
        )
        payload = [candidate.model_dump() for candidate in candidates]
        st.session_state["discovered_topics_candidates"] = payload
        st.session_state["discovered_topics_count"] = len(payload)
        if payload:
            st.session_state.pop("discovered_topics_error", None)
        else:
            # 空结果：区分“LLM 不可用”与“真的没有候选”。前者给用户可操作的
            # 反馈（额度/Key/网络），后者保持静默并显示提示语。
            llm_reason = tracker.reasons.get("trend_analyst", "")
            if tracker.statuses.get("trend_analyst") in ("fallback", "failed") or llm_reason:
                st.session_state["discovered_topics_error"] = llm_reason
    except Exception as exc:  # noqa: BLE001 - convenience feature must not break the page
        logger.warning(f"topic discovery failed: {exc}")
        st.session_state.pop("discovered_topics_candidates", None)
        st.session_state["discovered_topics_error"] = str(exc)


def _topic_discovery_summary():
    """One-line summary of the last topic discovery run for the status widget."""
    candidates = st.session_state.get("discovered_topics_candidates", [])
    count = len(candidates)
    if count:
        top = candidates[0].get("topic", "")
        return f"{count} topic(s) found — e.g. {str(top)[:60]}"
    error = st.session_state.get("discovered_topics_error", "")
    if error:
        return f"No topics found. {str(error)[:200]}"
    return "No topics found — try adjusting the niche or automation level."


def _generate_script_preview(params, progress=None):
    """Generate the script (agentic graph or linear) and record a summary.

    Stores the result into session state (so the script text area below picks
    it up) and ``script_generation_summary`` for the state-level status
    widget. Raises on failure so ``_run_with_state_status`` records an error
    state; the agentic graph falling back to linear is NOT a failure.

    ``progress`` (optional) is the stage reporter from
    ``_run_with_state_status(stage_reporter=True)``: it is forwarded into the
    agentic planning graph so each stage (research / strategy / script / QA)
    shows up in the live status widget.
    """
    if getattr(params, "agentic_planning", False):
        try:
            state = _run_llm_read_operation(
                "generate_agentic_script",
                lambda app_config_snapshot: agentic.plan_video_content_from_params(
                    params, app_config=app_config_snapshot, progress_cb=progress
                ),
            )
        except agentic.AgenticError as exc:
            logger.warning(f"agentic preview failed, falling back to linear: {exc}")
            state = None

        if state is None:
            script = llm.generate_script(
                video_subject=params.video_subject,
                language=params.video_language,
                paragraph_number=params.paragraph_number,
                video_script_prompt=params.video_script_prompt,
                custom_system_prompt=params.custom_system_prompt,
                script_style=params.script_style,
                target_duration_seconds=getattr(params, "video_duration_seconds", 0) or 0,
            )
            if "Error: " in script:
                raise UserFacingError(_friendly_llm_error(script))
            st.session_state["video_script"] = script
            st.session_state["script_generation_summary"] = (
                f"{len(script.split())} words (linear fallback)"
            )
            return script

        final_review = (state.final_review or {}).get("verdict", "approved")
        script = state.script or ""
        if final_review == "blocked":
            summary = (state.qa_report or {}).get("summary", "QA blocked")
            raise RuntimeError(f"{tr('Agentic Script Generation Failed')}: {summary}")
        if not script:
            raise RuntimeError(str(tr("Agentic Script Generation Failed")))
        st.session_state["video_script"] = script
        st.session_state["agentic_preview_state"] = state.model_dump()
        st.session_state["script_generation_summary"] = (
            f"{len(script.split())} words via the content strategy graph "
            f"(narrative: {(state.narrative_strategy or {}).get('strategy', {}).get('id', '?') if isinstance((state.narrative_strategy or {}).get('strategy'), dict) else '?'})"
        )
        return script

    def generate_script_and_terms(app_config_snapshot):
        script = llm.generate_script(
            video_subject=params.video_subject,
            language=params.video_language,
            paragraph_number=params.paragraph_number,
            video_script_prompt=params.video_script_prompt,
            custom_system_prompt=params.custom_system_prompt,
            script_style=params.script_style,
            target_duration_seconds=getattr(params, "video_duration_seconds", 0) or 0,
            app_config=app_config_snapshot,
        )
        terms = llm.generate_terms(
            params.video_subject,
            script,
            amount=8 if params.match_materials_to_script else 5,
            match_script_order=params.match_materials_to_script,
            app_config=app_config_snapshot,
        )
        return script, terms

    script, terms = _run_llm_read_operation(
        "generate_script_and_terms", generate_script_and_terms
    )
    if "Error: " in script:
        raise UserFacingError(_friendly_llm_error(script))
    if "Error: " in terms:
        raise UserFacingError(_friendly_llm_error(terms))
    st.session_state["video_script"] = script
    st.session_state["video_terms"] = ", ".join(terms)
    st.session_state["script_generation_summary"] = (
        f"{len(script.split())} words + {len(terms)} keywords"
    )
    return script


def _generate_terms_preview(params):
    """Generate video search keywords and record a summary (state-level status)."""
    terms = _run_llm_read_operation(
        "generate_terms",
        lambda app_config_snapshot: llm.generate_terms(
            params.video_subject,
            params.video_script,
            amount=8 if params.match_materials_to_script else 5,
            match_script_order=params.match_materials_to_script,
            app_config=app_config_snapshot,
        ),
    )
    if "Error: " in terms:
        raise UserFacingError(_friendly_llm_error(terms))
    st.session_state["video_terms"] = ", ".join(terms)
    st.session_state["terms_generation_summary"] = f"{len(terms)} keyword(s)"
    return terms


def _import_social_profile(profile_url):
    """Phase 2A convenience: infer niche/style from a social profile URL and
    pre-fill the Content Intelligence fields.

    Stores the inference for display plus a prefill payload; the prefill is
    applied on the next run before the widgets are created (setting widget
    state here would raise, because the widgets already exist in this run).
    Never raises: convenience failures degrade to a session note instead of
    blocking the page.
    """
    profile_url = (profile_url or "").strip()
    if not profile_url:
        return

    def _run(app_config_snapshot):
        from app.services.social_profile import infer_content_context

        return infer_content_context(profile_url, app_config=app_config_snapshot)

    try:
        inference = _run_llm_read_operation("social_profile", _run)
        st.session_state["social_profile_inference"] = inference.model_dump()
        st.session_state["social_profile_prefill"] = {
            "niche_input": inference.niche,
            "sub_niche_input": inference.sub_niche,
            "audience_input": inference.audience,
            "tone_input": inference.tone,
            "platform_select": inference.platform,
            # 主页链接同时追加到研究笔记，让研究层把它当作一条不受信任的
            # 用户来源（user_notes），与导入的赛道/风格一起参与事实核查。
            "profile_url": profile_url,
        }
        st.session_state.pop("social_profile_error", None)
        st.rerun()
    except Exception as exc:  # noqa: BLE001 - convenience feature must not break the page
        logger.warning(f"social profile import failed: {exc}")
        st.session_state["social_profile_error"] = str(exc)


def _social_profile_summary():
    """One-line summary of the last profile import for the status widget."""
    inference = st.session_state.get("social_profile_inference") or {}
    summary = str(inference.get("summary") or "")
    if summary:
        return summary[:300]
    error = st.session_state.get("social_profile_error", "")
    return f"Profile could not be imported. {str(error)[:200]}" if error else "Profile imported."


def _apply_social_profile_prefill():
    """Apply a stored profile-import prefill to the Content Intelligence
    widgets. Must run before those widgets are created, so direct
    session_state writes are legal."""
    prefill = st.session_state.pop("social_profile_prefill", None)
    if not isinstance(prefill, dict):
        return
    for key in (
        "niche_input",
        "sub_niche_input",
        "audience_input",
        "tone_input",
        "platform_select",
    ):
        value = prefill.get(key)
        if value is not None:
            st.session_state[key] = value
    # 主页链接追加到研究笔记（去重后），后续生成时随 sources 进入研究层。
    profile_url = str(prefill.get("profile_url") or "").strip()
    if profile_url:
        notes = str(st.session_state.get("research_notes_input", "") or "")
        if profile_url not in notes:
            st.session_state["research_notes_input"] = (
                notes.rstrip() + "\n" + profile_url if notes.strip() else profile_url
            )


def _render_agentic_plan_inspection(task_id):
    """Phase 2E.4: let advanced users inspect the agentic plan produced for a
    completed task: narrative strategy, story brief, scene plan, QA report,
    repurposing plan and the decision log.

    The state artifact is optional and diagnostic; when it is missing the
    panel simply does not render (legacy tasks keep the old UI unchanged).
    """
    from app.services import task_artifacts

    state_data = task_artifacts.load_agentic_state(task_id)
    if not state_data:
        return

    with st.expander(tr("Content Intelligence Plan"), expanded=False):
        narrative = state_data.get("narrative_strategy") or {}
        strategy = narrative.get("strategy") or {}
        if strategy:
            st.markdown(
                f"**{tr('Narrative Strategy')}:** {strategy.get('label', strategy.get('id', '?'))}"
            )
            if narrative.get("rationale"):
                st.caption(narrative["rationale"])

        story_brief = state_data.get("story_brief") or {}
        if story_brief:
            st.markdown(f"**{tr('Story Brief')}:**")
            brief_fields = [
                ("central_question", tr("Central Question")),
                ("hook", tr("Hook")),
                ("conflict", tr("Conflict")),
                ("stakes", tr("Stakes")),
                ("turning_point", tr("Turning Point")),
                ("payoff", tr("Payoff")),
                ("conclusion", tr("Conclusion")),
            ]
            for field, label in brief_fields:
                value = story_brief.get(field)
                if value:
                    st.markdown(f"- **{label}:** {value}")

        scene_plan = state_data.get("scene_plan") or {}
        scenes = scene_plan.get("scenes") or []
        if scenes:
            st.markdown(f"**{tr('Scene Plan')}:** ({len(scenes)} scenes)")
            for scene in scenes[:12]:
                material = scene.get("material_type", "?")
                terms = ", ".join(scene.get("search_terms") or [])[:80]
                st.markdown(f"- {material}: {terms or scene.get('narration', '')[:60]}")

        titles = state_data.get("title_candidates") or []
        if titles:
            st.markdown(f"**{tr('Title Candidates')}:**")
            for title in titles[:5]:
                score = title.get("overall", 0)
                st.markdown(f"- ({score}) {title.get('text', '')}")

        qa = state_data.get("qa_report") or {}
        issues = qa.get("issues") or []
        if qa:
            st.markdown(
                f"**{tr('Quality Assurance')}:** {qa.get('summary', '')}"
            )
            for issue in issues[:8]:
                st.markdown(f"- [{issue.get('severity', '')}] {issue.get('message', '')}")
            if qa.get("publication_blocked"):
                st.warning(tr("Publication Blocked By QA"))

        repurposing = state_data.get("repurposing_plan") or {}
        shorts = repurposing.get("shorts") or []
        if shorts:
            st.markdown(f"**{tr('Repurposing Plan')}:** {len(shorts)} short clips")

        decisions = state_data.get("decision_log") or []
        if decisions:
            with st.expander(tr("Decision Log"), expanded=False):
                for entry in decisions[-15:]:
                    st.markdown(
                        f"- **{entry.get('stage', '?')}:** {entry.get('decision', '')}"
                    )


def _render_script_settings(panel, params):
    """渲染文案设置并更新生成参数。"""
    with panel:
        with st.container(border=True):
            st.markdown(f"### {tr('Video Script Settings')}")
            # 内容智能（赛道/受众/平台等策略上下文）放在标题之前：先定义内容
            # 上下文，再填写视频主题，形成“先策略、后标题”的简单工作流。
            agentic_col, profile_col = st.columns(2)
            params.agentic_planning = agentic_col.checkbox(
                tr("Agentic Planning"),
                value=False,
                key="agentic_planning",
                help=tr("Agentic Planning Help"),
            )
            profile_names = content_profile.list_content_profiles()
            params.content_profile = profile_col.selectbox(
                tr("Content Profile"),
                options=profile_names,
                index=0,
                key="content_profile_select",
                format_func=lambda name: content_profile.get_content_profile(
                    name
                ).description
                or name,
                disabled=not params.agentic_planning,
            )

            if params.agentic_planning:
                with st.container(key="content_intelligence_settings"):
                    with st.expander(
                        tr("Content Intelligence Settings"), expanded=False
                    ):
                        # 上次点击“导入主页”产生的回填数据必须先于控件创建应用，
                        # 否则控件已实例化后无法写入 session_state。
                        _apply_social_profile_prefill()
                        # 快速导入：粘贴 TikTok / Instagram / Facebook / X /
                        # YouTube 主页链接，自动推断赛道与风格并回填下方字段。
                        profile_url = st.text_input(
                            tr("Social Profile URL"),
                            placeholder=tr("Social Profile URL Placeholder"),
                            key="social_profile_url_input",
                        ).strip()
                        _render_state_status(
                            "social_profile_status", tr("Importing Profile")
                        )
                        if _action_button_clicked(
                            "social_profile_status",
                            st.button(
                                tr("Import Profile"),
                                key="import_social_profile_button",
                                use_container_width=True,
                                disabled=not _action_ready("social_profile_status"),
                            ),
                        ):
                            _run_with_state_status(
                                "social_profile_status",
                                tr("Importing Profile"),
                                tr("Profile Imported"),
                                lambda: _import_social_profile(profile_url),
                                success_detail=lambda: _social_profile_summary(),
                                error_detail=lambda exc: _friendly_llm_error(exc) if isinstance(exc, UserFacingError) else str(exc),
                            )
                        st.caption(tr("Social Profile Import Hint"))
                        imported = st.session_state.get("social_profile_inference")
                        if imported:
                            summary = str(imported.get("summary") or "")
                            note = str(imported.get("note") or "")
                            st.caption(
                                f"**{tr('Profile Imported')}:** {summary}"
                                + (f" — {note}" if note else "")
                            )
                        elif st.session_state.get("social_profile_error"):
                            st.caption(tr("Profile Import Failed"))
                        # 分组布局：下拉框统一使用半行宽，避免 1/4 列把选项压成省略号、
                        # 展开菜单也过窄看不清。三个分组用弱化标题分隔，减轻拥挤感。
                        st.markdown(
                            "<div class='mpt-ci-section'>"
                            f"{html.escape(tr('Content Intelligence Strategy Section'))}"
                            "</div>",
                            unsafe_allow_html=True,
                        )
                        automation_col, platform_col = st.columns(2)
                        params.automation_level = automation_col.selectbox(
                            tr("Automation Level"),
                            options=["", "manual", "assisted", "automatic", "autopilot"],
                            index=2,
                            key="automation_level_select",
                            format_func=lambda value: {
                                "": tr("Automation Level Default"),
                                "manual": tr("Automation Level Manual"),
                                "assisted": tr("Automation Level Assisted"),
                                "automatic": tr("Automation Level Automatic"),
                                "autopilot": tr("Automation Level Autopilot"),
                            }.get(value, value),
                            help=tr("Automation Level Help"),
                        )
                        params.platform = platform_col.selectbox(
                            tr("Platform"),
                            options=["", "youtube", "youtube_shorts", "tiktok", "instagram_reels", "bilibili", "x"],
                            index=0,
                            key="platform_select",
                            format_func=lambda value: {
                                "": tr("Platform Default"),
                                "youtube": tr("Platform YouTube"),
                                "youtube_shorts": tr("Platform YouTube Shorts"),
                                "tiktok": tr("Platform TikTok"),
                                "instagram_reels": tr("Platform Instagram Reels"),
                                "bilibili": tr("Platform Bilibili"),
                                "x": tr("Platform X"),
                            }.get(value, value),
                        )
                        format_col, goal_col = st.columns(2)
                        params.content_format = format_col.selectbox(
                            tr("Content Format"),
                            options=["", "documentary", "explainer", "tutorial", "news_analysis", "storytelling", "list", "case_study"],
                            index=0,
                            key="content_format_select",
                            format_func=lambda value: {
                                "": tr("Content Format Default"),
                                "documentary": tr("Content Format Documentary"),
                                "explainer": tr("Content Format Explainer"),
                                "tutorial": tr("Content Format Tutorial"),
                                "news_analysis": tr("Content Format News Analysis"),
                                "storytelling": tr("Content Format Storytelling"),
                                "list": tr("Content Format List"),
                                "case_study": tr("Content Format Case Study"),
                            }.get(value, value),
                        )
                        params.content_goal = goal_col.selectbox(
                            tr("Content Goal"),
                            options=["", "growth", "engagement", "education", "awareness", "monetization"],
                            index=0,
                            key="content_goal_select",
                            format_func=lambda value: {
                                "": tr("Content Goal Default"),
                                "growth": tr("Content Goal Growth"),
                                "engagement": tr("Content Goal Engagement"),
                                "education": tr("Content Goal Education"),
                                "awareness": tr("Content Goal Awareness"),
                                "monetization": tr("Content Goal Monetization"),
                            }.get(value, value),
                        )
                        st.markdown(
                            "<div class='mpt-ci-section'>"
                            f"{html.escape(tr('Content Intelligence Audience Section'))}"
                            "</div>",
                            unsafe_allow_html=True,
                        )
                        niche_col, subniche_col = st.columns(2)
                        params.niche = niche_col.text_input(
                            tr("Niche"),
                            placeholder=tr("Niche Placeholder"),
                            key="niche_input",
                        ).strip()
                        params.sub_niche = subniche_col.text_input(
                            tr("Sub Niche"),
                            placeholder=tr("Sub Niche Placeholder"),
                            key="sub_niche_input",
                        ).strip()
                        audience_col, tone_col = st.columns(2)
                        params.audience = audience_col.text_input(
                            tr("Audience"),
                            placeholder=tr("Audience Placeholder"),
                            key="audience_input",
                        ).strip()
                        params.tone = tone_col.text_input(
                            tr("Tone"),
                            placeholder=tr("Tone Placeholder"),
                            key="tone_input",
                        ).strip()
                        st.markdown(
                            "<div class='mpt-ci-section'>"
                            f"{html.escape(tr('Content Intelligence Research Section'))}"
                            "</div>",
                            unsafe_allow_html=True,
                        )
                        depth_col, factcheck_col = st.columns(2)
                        params.research_depth = depth_col.selectbox(
                            tr("Research Depth"),
                            options=["", "low", "medium", "high", "very_high"],
                            index=0,
                            key="research_depth_select",
                            format_func=lambda value: {
                                "": tr("Research Depth Default"),
                                "low": tr("Research Depth Low"),
                                "medium": tr("Research Depth Medium"),
                                "high": tr("Research Depth High"),
                                "very_high": tr("Research Depth Very High"),
                            }.get(value, value),
                        )
                        params.fact_check_level = factcheck_col.selectbox(
                            tr("Fact Check Level"),
                            options=["", "normal", "strong", "very_strong"],
                            index=0,
                            key="fact_check_level_select",
                            format_func=lambda value: {
                                "": tr("Fact Check Level Default"),
                                "normal": tr("Fact Check Level Normal"),
                                "strong": tr("Fact Check Level Strong"),
                                "very_strong": tr("Fact Check Level Very Strong"),
                            }.get(value, value),
                        )
                        trendpref_col, channels_col = st.columns(2)
                        params.trend_preference = trendpref_col.selectbox(
                            tr("Trend Preference"),
                            options=["", "trending", "evergreen", "rising"],
                            index=0,
                            key="trend_preference_select",
                            format_func=lambda value: {
                                "": tr("Trend Preference Default"),
                                "trending": tr("Trend Preference Trending"),
                                "evergreen": tr("Trend Preference Evergreen"),
                                "rising": tr("Trend Preference Rising"),
                            }.get(value, value),
                        )
                        channels_text = channels_col.text_input(
                            tr("Reference Channels"),
                            placeholder=tr("Reference Channels Placeholder"),
                            key="reference_channels_input",
                        ).strip()
                        params.reference_channels = [
                            item.strip()
                            for item in channels_text.replace(",", " ").split()
                            if item.strip()
                        ]
                        sources_text = st.text_area(
                            tr("Research Notes"),
                            placeholder=tr("Research Notes Placeholder"),
                            height=90,
                            key="research_notes_input",
                        ).strip()
                        params.sources = [
                            line.strip()
                            for line in sources_text.splitlines()
                            if line.strip()
                        ]
                        st.caption(tr("Content Intelligence Hint"))

                        with st.expander(
                            tr("Zero-Key Research Metrics"), expanded=False
                        ):
                            research_metrics = research_service.zero_key_metrics()
                            if research_metrics is None:
                                st.caption(tr("Zero-Key Research Disabled Hint"))
                            else:
                                metric_cols = st.columns(3)
                                metric_cols[0].metric(
                                    tr("Zero-Key External Requests"),
                                    int(research_metrics.get("total_requests", 0)),
                                )
                                metric_cols[1].metric(
                                    tr("Zero-Key Requests Avoided"),
                                    int(research_metrics.get("requests_avoided", 0)),
                                )
                                metric_cols[2].metric(
                                    tr("Zero-Key Cache Hit Rate"),
                                    f"{100 * float(research_metrics.get('cache_hit_rate', 0.0)):.0f}%",
                                )
                                provider_requests = research_metrics.get(
                                    "requests_by_provider", {}
                                )
                                if provider_requests:
                                    st.caption(tr("Zero-Key Provider Requests"))
                                    st.json(
                                        provider_requests, expanded=False
                                    )

                        # Phase 2E.4: Topic Recommendation — generate scored
                        # topic candidates for the current niche and let the
                        # user adopt one into the subject field.
                        with st.container(key="topic_discovery_settings"):
                            _render_state_status(
                                "topic_discovery_status", tr("Discovering Topics")
                            )
                            if _action_button_clicked(
                                "topic_discovery_status",
                                st.button(
                                    tr("Discover Topics"),
                                    key="discover_topics_button",
                                    use_container_width=True,
                                    disabled=not _action_ready(
                                        "topic_discovery_status"
                                    ),
                                ),
                            ):
                                _run_with_state_status(
                                    "topic_discovery_status",
                                    tr("Discovering Topics"),
                                    tr("Discover Topics Done"),
                                    lambda: _discover_and_store_topics(params),
                                    success_detail=lambda: _topic_discovery_summary(),
                                )
                            candidates = st.session_state.get(
                                "discovered_topics_candidates", []
                            )
                            if candidates:
                                labels = [
                                    f"{item['total']:.1f} — {item['topic']}"
                                    for item in candidates
                                ]
                                picked = st.selectbox(
                                    tr("Recommended Topics"),
                                    options=range(len(candidates)),
                                    index=0,
                                    key="discovered_topic_pick",
                                    format_func=lambda index: labels[index],
                                )
                                if _action_button_clicked(
                                    "use_recommended_topic",
                                    st.button(
                                        tr("Use Recommended Topic"),
                                        key="use_recommended_topic_button",
                                        disabled=not _action_ready(
                                            "use_recommended_topic"
                                        ),
                                    ),
                                ):
                                    # 采纳选题是即时操作（设置 subject + 触发 rerun），
                                    # 直接在点击时进入冷却窗口，防止排队重复点击。
                                    _action_mark_triggered("use_recommended_topic")
                                    topic = candidates[picked]["topic"]
                                    st.session_state["video_subject"] = topic
                                    st.session_state.pop(
                                        "discovered_topics_candidates", None
                                    )
                                    # 采纳选题后自动生成文案：下一页渲染到
                                    # 文案区时看到这个标记会自动触发一次生成。
                                    st.session_state[
                                        "auto_generate_script_requested"
                                    ] = True
                                    st.rerun(scope="app")
                            else:
                                topic_error = st.session_state.get(
                                    "discovered_topics_error", ""
                                )
                                if topic_error:
                                    st.warning(
                                        f"{tr('Discover Topics Error')} — {topic_error}\n\n"
                                        + tr("Discover Topics Error Hint")
                                    )
                                else:
                                    st.caption(tr("Discover Topics Hint"))

            params.video_subject = st.text_area(
                tr("Video Subject"),
                placeholder=tr("Video Subject Placeholder"),
                height=96,
                key="video_subject",
            ).strip()

            video_languages = [
                (tr("Auto Detect"), ""),
            ]
            for code in support_locales:
                video_languages.append((code, code))

            selected_language_code = stable_selectbox(
                tr("Script Language"),
                options=[value for _, value in video_languages],
                default_value="",
                key="script_language_select",
                format_func=lambda value: dict(
                    (v, label) for label, v in video_languages
                )[value],
            )
            params.video_language = selected_language_code

            # 使用带 key 的局部容器限定折叠入口样式，保持 expander 的原生交互，
            # 同时避免样式误伤页面顶部的“基础设置”等其他折叠区域。
            with st.container(key="advanced_settings_script"):
                with st.expander(tr("Advanced Script Settings"), expanded=False):
                    st.session_state.setdefault("paragraph_number_input", 1)
                    paragraph_col, duration_col = st.columns(2, gap="small")
                    with paragraph_col:
                        params.paragraph_number = st.slider(
                            tr("Script Paragraph Number"),
                            min_value=llm.MIN_SCRIPT_PARAGRAPH_NUMBER,
                            max_value=llm.MAX_SCRIPT_PARAGRAPH_NUMBER,
                            key="paragraph_number_input",
                        )
                    with duration_col:
                        duration_options = [
                            (tr("Target Duration Auto"), 0),
                            (tr("Target Duration 30s"), 30),
                            (tr("Target Duration 45s"), 45),
                            (tr("Target Duration 60s"), 60),
                            (tr("Target Duration 90s"), 90),
                            (tr("Target Duration 120s"), 120),
                            (tr("Target Duration 150s"), 150),
                            (tr("Target Duration 180s"), 180),
                            (tr("Target Duration 240s"), 240),
                            (tr("Target Duration 300s"), 300),
                            (tr("Target Duration 450s"), 450),
                            (tr("Target Duration 600s"), 600),
                        ]
                        saved_duration = getattr(params, "video_duration_seconds", 0) or 0
                        if saved_duration not in {value for _, value in duration_options}:
                            saved_duration = 0
                        params.video_duration_seconds = stable_selectbox(
                            tr("Target Video Length"),
                            options=[value for _, value in duration_options],
                            default_value=saved_duration,
                            key="target_duration_select",
                            format_func=lambda value: dict(
                                (v, label) for label, v in duration_options
                            )[value],
                            help=tr("Target Video Length Help"),
                        )
                    params.video_script_prompt = st.text_area(
                        tr("Custom Script Requirements"),
                        height=100,
                        max_chars=llm.MAX_SCRIPT_PROMPT_LENGTH,
                        placeholder=tr("Custom Script Requirements Placeholder"),
                        key="video_script_prompt",
                    ).strip()

                    system_prompt = st.text_area(
                        tr("Custom System Prompt"),
                        height=240,
                        max_chars=llm.MAX_SCRIPT_SYSTEM_PROMPT_LENGTH,
                        key="custom_system_prompt",
                    ).strip()
                    # 默认内容由服务层统一维护。界面虽然直接展示默认提示词，但只有
                    # 用户实际修改后才随任务传递，避免历史任务固化旧版本默认规则。
                    params.custom_system_prompt = (
                        ""
                        if system_prompt == llm.DEFAULT_SCRIPT_SYSTEM_PROMPT.strip()
                        else system_prompt
                    )

                    restore_prompt_col, preview_prompt_col = st.columns(2)
                    if restore_prompt_col.button(
                        tr("Restore Default System Prompt"),
                        key="restore_default_system_prompt",
                        icon=":material/restart_alt:",
                        on_click=reset_script_system_prompt,
                        use_container_width=True,
                    ):
                        st.toast(tr("Default System Prompt Restored"))
                    if preview_prompt_col.button(
                        tr("Preview Final Prompt"),
                        key="preview_final_script_prompt",
                        icon=":material/preview:",
                        use_container_width=True,
                    ):
                        render_script_prompt_preview(
                            llm.build_script_prompt(
                                video_subject=params.video_subject,
                                language=params.video_language,
                                paragraph_number=params.paragraph_number,
                                video_script_prompt=params.video_script_prompt,
                                custom_system_prompt=params.custom_system_prompt,
                                target_duration_seconds=getattr(params, "video_duration_seconds", 0) or 0,
                            )
                        )

            # 采纳“推荐选题”后自动生成文案；标记在文案区渲染前被消费。
            auto_generate_script = st.session_state.pop(
                "auto_generate_script_requested", False
            )
            # 文案措辞风格：默认人性化简单风格，也可选专业/故事化/说服/教育/
            # 随性。作用于线性与智能体两条生成链路。
            script_styles = [
                (tr("Script Style Simple Humanized"), "simple_humanized"),
                (tr("Script Style Field Expert"), "field_expert"),
                (tr("Script Style Storytelling"), "storytelling"),
                (tr("Script Style Persuasive"), "persuasive"),
                (tr("Script Style Educational"), "educational"),
                (tr("Script Style Casual"), "casual"),
            ]
            script_style_labels = {value: label for label, value in script_styles}
            saved_style = (params.script_style or "simple_humanized").strip().lower()
            if saved_style not in {value for _, value in script_styles}:
                saved_style = "simple_humanized"
            params.script_style = stable_selectbox(
                tr("Script Style"),
                options=[value for _, value in script_styles],
                default_value=saved_style,
                key="script_style_select",
                help=tr("Script Style Help"),
                format_func=lambda value: script_style_labels[value],
            )
            if params.agentic_planning:
                st.caption(tr("Agentic Planning Enabled Hint"))
            # 状态级反馈：上一次生成的结果（成功/失败/字数）会跨 rerun 持续显示，
            # 而不只是点击那一瞬间的 spinner。
            _render_state_status(
                "script_generation_status", tr("Generating Video Script and Keywords")
            )
            script_button_clicked = _action_button_clicked(
                "script_generation_status",
                st.button(
                    tr("Generate Video Script and Keywords"),
                    key="auto_generate_script",
                    use_container_width=True,
                    type="secondary",
                    icon=":material/auto_awesome:",
                    disabled=not _action_ready("script_generation_status"),
                ),
            )
            if script_button_clicked or auto_generate_script:
                if not params.video_subject:
                    # 视频主题是脚本生成的必要输入，提前拦截可以避免无意义的模型调用。
                    st.toast(tr("Please Enter the Video Subject First"))
                    st.warning(tr("Please Enter the Video Subject First"))
                else:
                    _run_with_state_status(
                        "script_generation_status",
                        tr("Generating Video Script and Keywords"),
                        tr("Script Generated"),
                        lambda report_stage: _generate_script_preview(
                            params, progress=report_stage
                        ),
                        stage_reporter=True,
                        success_detail=lambda: st.session_state.get(
                            "script_generation_summary", ""
                        ),
                        error_detail=lambda exc: _friendly_llm_error(exc) if isinstance(exc, UserFacingError) else str(exc),
                    )
            params.video_script = st.text_area(
                tr("Video Script"),
                help=tr("Video Script Help"),
                height=180,
                key="video_script",
            )
            _render_state_status(
                "terms_generation_status", tr("Generating Video Keywords")
            )
            if _action_button_clicked(
                "terms_generation_status",
                st.button(
                    tr("Generate Video Keywords"),
                    key="auto_generate_terms",
                    use_container_width=True,
                    type="secondary",
                    icon=":material/auto_awesome:",
                    disabled=not _action_ready("terms_generation_status"),
                ),
            ):
                if not params.video_script:
                    # 视频关键词需要基于文案提取，文案为空时提前提示并跳过模型调用。
                    st.toast(tr("Please Enter the Video Subject"))
                    st.warning(tr("Please Enter the Video Subject"))
                else:
                    _run_with_state_status(
                        "terms_generation_status",
                        tr("Generating Video Keywords"),
                        tr("Video Keywords Generated"),
                        lambda: _generate_terms_preview(params),
                        success_detail=lambda: st.session_state.get(
                            "terms_generation_summary", ""
                        ),
                        error_detail=lambda exc: _friendly_llm_error(exc) if isinstance(exc, UserFacingError) else str(exc),
                    )

            params.video_terms = st.text_area(
                tr("Video Keywords"),
                help=tr("Video Keywords Help"),
                key="video_terms",
            )


def _render_video_settings(panel, params):
    """渲染视频设置并返回本次选择的本地素材。"""
    uploaded_files = []
    with panel:
        with st.container(border=True):
            st.markdown(f"### {tr('Video Settings')}")
            video_concat_modes = [
                (tr("Sequential"), "sequential"),
                (tr("Random"), "random"),
            ]
            video_sources = [
                (
                    tr("Auto (Best Across All Sources)")
                    if hasattr(tr, "__call__")
                    else "Auto (Best Across All Sources)",
                    "auto",
                ),
                ("Custom API (Hybrid)", "custom_api"),
                (tr("Gemini Image (Nano Banana)"), "gemini_image"),
                (tr("Pexels"), "pexels"),
                (tr("Pixabay"), "pixabay"),
                (tr("Coverr"), "coverr"),
                (tr("Web Scrape"), "web_scrape"),
                (tr("Pollinations (Free)"), "pollinations"),
                (tr("Local file"), "local"),
            ]

            saved_video_source_name = config.app.get("video_source", "pexels")

            params.video_source = stable_selectbox(
                tr("Video Source"),
                options=[value for _, value in video_sources],
                default_value=saved_video_source_name,
                key="video_source_select",
                format_func=lambda value: dict(
                    (v, label) for label, v in video_sources
                )[value],
            )
            _set_runtime_config("app", "video_source", params.video_source)

            # 素材媒体类型：默认允许图片+视频混合；选择“仅视频”时丢弃生成式
            # AI 静态图片素材（custom_api 图片兜底 / pollinations）。
            media_type_options = [
                (tr("Material Media Type Images Videos"), "images_videos"),
                (tr("Material Media Type Videos Only"), "videos_only"),
            ]
            saved_media_type = getattr(params, "material_media_type", "images_videos") or "images_videos"
            if saved_media_type not in {value for _, value in media_type_options}:
                saved_media_type = "images_videos"
            media_col, effect_col = st.columns(2, gap="small")
            with media_col:
                params.material_media_type = stable_selectbox(
                    tr("Material Media Type"),
                    options=[value for _, value in media_type_options],
                    default_value=saved_media_type,
                    key="material_media_type_select",
                    format_func=lambda value: dict(
                        (v, label) for label, v in media_type_options
                    )[value],
                    help=tr("Material Media Type Help"),
                )

            # 静态图片运镜效果：图片素材转成视频片段时使用的动画效果。仅当
            # 素材媒体类型允许图片时才有意义，videos_only 下隐藏避免歧义。
            if params.material_media_type == "videos_only":
                params.image_motion_effect = "kenburns"
            else:
                effect_options = [
                    (tr("Image Effect Ken Burns"), "kenburns"),
                    (tr("Image Effect Zoom In"), "zoom_in"),
                    (tr("Image Effect Zoom Out"), "zoom_out"),
                    (tr("Image Effect Slide Left"), "slide_left"),
                    (tr("Image Effect Slide Right"), "slide_right"),
                    (tr("Image Effect Fade"), "fade"),
                    (tr("Image Effect Random"), "random"),
                ]
                saved_effect = getattr(params, "image_motion_effect", "") or "kenburns"
                if saved_effect not in {value for _, value in effect_options}:
                    saved_effect = "kenburns"
                with effect_col:
                    params.image_motion_effect = stable_selectbox(
                        tr("Image Motion Effect"),
                        options=[value for _, value in effect_options],
                        default_value=saved_effect,
                        key="image_motion_effect_select",
                        format_func=lambda value: dict(
                            (v, label) for label, v in effect_options
                        )[value],
                        help=tr("Image Motion Effect Help"),
                    )

            if params.video_source == "auto":
                # F1 多来源选择：指定哪些供应商参与 auto 来源；顺序仍按内置
                # 优先级（custom_api → gemini_image → pexels → pixabay → coverr
                # → web_scrape → pollinations），多选框只决定“是否参与”。
                auto_provider_options = [
                    "custom_api",
                    "gemini_image",
                    "pexels",
                    "pixabay",
                    "coverr",
                    "web_scrape",
                    "pollinations",
                ]
                auto_provider_labels = {
                    "custom_api": "Custom API (Hybrid)",
                    "gemini_image": tr("Gemini Image (Nano Banana)"),
                    "pexels": tr("Pexels"),
                    "pixabay": tr("Pixabay"),
                    "coverr": tr("Coverr"),
                    "web_scrape": tr("Web Scrape"),
                    "pollinations": tr("Pollinations (Free)"),
                }
                saved_auto_providers = config.app.get("auto_providers")
                if not isinstance(saved_auto_providers, list) or not saved_auto_providers:
                    saved_auto_providers = auto_provider_options
                st.multiselect(
                    tr("Auto Video Sources"),
                    options=auto_provider_options,
                    default=saved_auto_providers,
                    key="auto_providers_select",
                    format_func=lambda value: auto_provider_labels.get(
                        value, value
                    ),
                    help=tr("Auto Video Sources Help"),
                )
                if st.session_state["auto_providers_select"]:
                    _set_runtime_config(
                        "app",
                        "auto_providers",
                        list(st.session_state["auto_providers_select"]),
                    )

            if params.video_source == "local":
                # Streamlit 的文件类型校验对扩展名大小写敏感，这里同时放行大小写两种形式。
                local_file_types = sorted(
                    extension.removeprefix(".")
                    for extension in LOCAL_MATERIAL_EXTENSIONS
                )
                uploaded_files = st.file_uploader(
                    tr("Upload Local Files"),
                    type=local_file_types
                    + [file_type.upper() for file_type in local_file_types],
                    accept_multiple_files=True,
                    key="local_video_materials_uploader",
                )

            # 文案顺序匹配会从关键词生成到最终合成全程保持叙事顺序，因此开启时
            # 顺序拼接是唯一符合实际执行逻辑的选项。同步控件值可避免界面仍显示
            # “随机拼接”，同时保留用户原选择，关闭后自动恢复。
            sync_script_order_concat_mode()
            selected_concat_mode = stable_selectbox(
                tr("Video Concat Mode"),
                options=[value for _, value in video_concat_modes],
                default_value=VideoConcatMode.random.value,
                key="video_concat_mode_select",
                format_func=lambda value: dict(
                    (v, label) for label, v in video_concat_modes
                )[value],
                disabled=bool(st.session_state.get("match_materials_to_script", False)),
            )
            params.video_concat_mode = VideoConcatMode(selected_concat_mode)

            params.match_materials_to_script = st.checkbox(
                tr("Match Materials to Script Order"),
                help=tr("Match Materials to Script Order Help"),
                key="match_materials_to_script",
                on_change=sync_script_order_concat_mode,
            )
            _set_runtime_config(
                "app",
                "match_materials_to_script",
                params.match_materials_to_script,
            )

            # 视频转场模式
            video_transition_modes = [
                (tr("None"), VideoTransitionMode.none.value),
                (tr("Auto"), VideoTransitionMode.auto.value),
                (tr("Mix"), VideoTransitionMode.mix.value),
                (tr("Shuffle"), VideoTransitionMode.shuffle.value),
                (tr("FadeIn"), VideoTransitionMode.fade_in.value),
                (tr("FadeOut"), VideoTransitionMode.fade_out.value),
                (tr("SlideIn"), VideoTransitionMode.slide_in.value),
                (tr("SlideOut"), VideoTransitionMode.slide_out.value),
                (tr("ZoomIn"), VideoTransitionMode.zoom_in.value),
                (tr("ZoomOut"), VideoTransitionMode.zoom_out.value),
            ]
            selected_transition_mode = stable_selectbox(
                tr("Video Transition Mode"),
                options=[value for _, value in video_transition_modes],
                default_value=VideoTransitionMode.none.value,
                key="video_transition_mode_select",
                format_func=lambda value: dict(
                    (v, label) for label, v in video_transition_modes
                )[value],
            )
            params.video_transition_mode = VideoTransitionMode(selected_transition_mode)

            if params.video_transition_mode == VideoTransitionMode.mix:
                params.mix_overlap_duration = st.slider(
                    tr("Mix Overlap Duration (Seconds)"),
                    min_value=0.1,
                    max_value=2.0,
                    value=getattr(params, "mix_overlap_duration", 1.0) or 1.0,
                    step=0.1,
                    key="mix_overlap_duration_slider",
                )
            video_aspect_ratios = [
                (tr("Portrait"), VideoAspect.portrait.value),
                (tr("Landscape"), VideoAspect.landscape.value),
            ]
            # Coverr 库 99% 是 16:9 横屏,默认竖屏会让画面被大量黑边包围。
            # 用 source-specific widget key 让每个 source 各自记忆 aspect 选择:
            #   - 首次切到 coverr → 默认 Landscape(index=1)
            #   - 其他 source 沿用 Portrait(index=0)
            #   - 用户在某 source 下手动改过 aspect,session_state 会记住,
            #     下次回到同一 source 时尊重用户选择,不会再被强制覆盖。
            default_aspect_index = 1 if params.video_source == "coverr" else 0
            selected_aspect_ratio = stable_selectbox(
                tr("Video Ratio"),
                options=[value for _, value in video_aspect_ratios],
                default_value=video_aspect_ratios[default_aspect_index][1],
                key=f"video_aspect_for_{params.video_source}",
                format_func=lambda value: dict(
                    (v, label) for label, v in video_aspect_ratios
                )[value],
            )
            params.video_aspect = VideoAspect(selected_aspect_ratio)

            clip_duration_options = [(tr("Auto"), 0)] + [
                (f"{seconds}s", seconds)
                for seconds in [2, 3, 4, 5, 6, 7, 8, 9, 10]
            ]
            params.video_clip_duration = stable_selectbox(
                tr("Clip Duration"),
                options=[value for _, value in clip_duration_options],
                default_value=0,
                key="video_clip_duration_select",
                format_func=lambda value: dict(
                    (v, label) for label, v in clip_duration_options
                )[value],
                help=tr("Clip Duration Help"),
            )
            clip_speed_key = localized_widget_key("video_clip_speed_slider")
            # session_state 可能来自旧任务、API 参数或旧版页面状态。控件创建前
            # 统一归一化，既保留合法选择，也确保 slider 始终收到 0.5～2.0
            # 范围内的有限浮点数。
            st.session_state[clip_speed_key] = utils.normalize_clip_speed(
                st.session_state.get(clip_speed_key, 1.0)
            )
            params.video_clip_speed = st.slider(
                tr("Clip Speed"),
                min_value=0.5,
                max_value=2.0,
                step=0.05,
                format="%.2fx",
                key=clip_speed_key,
                help=tr("Clip Speed Help"),
            )
            params.video_count = stable_selectbox(
                tr("Number of Videos Generated Simultaneously"),
                options=[1, 2, 3, 4, 5],
                default_value=1,
                key="video_count_select",
            )

            video_codec_options = [
                (tr("Default Video Encoder"), DEFAULT_VIDEO_CODEC_OPTION),
                ("libx264 (CPU)", "libx264"),
                ("NVIDIA NVENC (h264_nvenc)", "h264_nvenc"),
                ("AMD AMF (h264_amf)", "h264_amf"),
                ("Intel QSV (h264_qsv)", "h264_qsv"),
                ("Windows MediaFoundation (h264_mf)", "h264_mf"),
                ("macOS VideoToolbox (h264_videotoolbox)", "h264_videotoolbox"),
            ]
            saved_video_codec = config.app.get(
                "video_codec", DEFAULT_VIDEO_CODEC_OPTION
            )
            saved_video_codec_values = [item[1] for item in video_codec_options]
            if saved_video_codec not in saved_video_codec_values:
                # 旧版本或手工配置可能留下无效值。UI 回到“默认”而不是替用户
                # 固定某个编码器，后端仍会按稳定策略解析为 libx264。
                saved_video_codec = DEFAULT_VIDEO_CODEC_OPTION
            selected_video_codec = stable_selectbox(
                tr("Video Encoder"),
                options=saved_video_codec_values,
                default_value=saved_video_codec,
                key="video_encoder_select",
                format_func=lambda value: dict(
                    (v, label) for label, v in video_codec_options
                )[value],
                help=tr("Video Encoder Help"),
            )
            if selected_video_codec == DEFAULT_VIDEO_CODEC_OPTION:
                # 默认模式不持久化具体编码器，让配置表达“跟随项目默认值”。
                _delete_runtime_config("app", "video_codec")
            else:
                _set_runtime_config("app", "video_codec", selected_video_codec)

            with st.expander("Advanced Output Settings", expanded=False):
                saved_output_dir = config.app.get("output_dir", "") or ""
                new_output_dir = st.text_input(
                    "Custom Output Folder",
                    value=saved_output_dir,
                    key="output_dir_input",
                    help="Absolute or ~/ path. Empty = default storage/tasks/<task_id>/. Final videos are copied there with task-id prefix.",
                    placeholder="e.g. D:/Videos/MyProject or ~/Videos",
                )
                # 与 video_codec 相同的非阻塞保存逻辑：空值表示删除配置项
                if new_output_dir.strip() != saved_output_dir.strip():
                    if new_output_dir.strip() == "":
                        _delete_runtime_config("app", "output_dir")
                    else:
                        _set_runtime_config("app", "output_dir", new_output_dir.strip())
                if new_output_dir.strip() and not utils.resolve_custom_output_dir(new_output_dir.strip()):
                    st.warning("Custom output folder is not writable or not allowed")
                # 覆盖 VideoParams 的 per-request 输出目录，让本次生成立即生效
                params.output_dir = new_output_dir.strip()

                saved_keep = bool(config.app.get("keep_intermediate_clips", False))
                new_keep = st.checkbox(
                    "Keep Intermediate Clips (temp-clip / mix-chunk)",
                    value=saved_keep,
                    key="keep_intermediate_clips_checkbox",
                    help="When enabled, intermediate processed clips are kept in the task folder for debugging. Default deletes them to save space.",
                )
                if new_keep != saved_keep:
                    _set_runtime_config("app", "keep_intermediate_clips", new_keep)
    return uploaded_files


def _estimate_voiceover_duration_range(
    text: str, voice_rate: float
) -> tuple[float, float] | None:
    """
    在本地估算完整配音时长，返回保守的上下界秒数。

    该估算只用于帮助用户在调用付费 TTS 前判断文案量级，不参与任务执行。
    中文、日文和韩文按字符速度估算，其它使用空格分词的语言按单词速度估算，
    再计入常见标点停顿。不同 Provider、音色和语气会造成实际偏差，因此界面
    必须展示区间而不是伪精确的单一结果。
    """
    normalized_text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not normalized_text:
        return None

    script_chars = re.findall(
        r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]",
        normalized_text,
    )
    remaining_text = re.sub(
        r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]",
        " ",
        normalized_text,
    )
    words = re.findall(r"\b[\w]+(?:[-'’][\w]+)*\b", remaining_text, re.UNICODE)
    punctuation_count = len(re.findall(r"[,，.。!?！？;；:：]", normalized_text))

    # 4.2 字/秒和 2.6 词/秒接近日常解说语速；标点按 0.12 秒加入轻微停顿。
    # voice_rate 只作为估算修正项。部分生成式 TTS 不严格执行倍率，所以最终
    # 仍保留 ±15% 区间，避免让用户误以为该值等同于服务端真实结果。
    base_seconds = len(script_chars) / 4.2 + len(words) / 2.6 + punctuation_count * 0.12
    if base_seconds <= 0:
        return None

    normalized_rate = max(float(voice_rate or 1.0), 0.1)
    estimated_seconds = base_seconds / normalized_rate
    return (
        round(max(estimated_seconds * 0.85, 1.0), 1),
        round(max(estimated_seconds * 1.15, 1.0), 1),
    )


def _get_voice_preview_sample(voice_name: str) -> str:
    """返回适合当前音色的短试听文案，不使用用户的完整视频文案。"""
    # ElevenLabs 音色缺少明确语言字段时，根据展示名称中的越南语字符选择
    # 试听文案，避免用明显不匹配的语言判断音色效果。
    if voice.is_elevenlabs_voice(voice_name):
        parts = voice_name.split(":", 2)
        display = parts[2] if len(parts) >= 3 else ""
        vietnamese_chars = set("àáâãèéêìíòóôõùúýăđơưÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂĐƠƯ")
        if any(char in vietnamese_chars for char in display):
            return "Xin chào, đây là đoạn âm thanh thử nghiệm giọng nói."
    return tr("Voice Example")


def _voice_preview_fingerprint(
    *,
    preview_type: str,
    content: str,
    tts_server: str,
    voice_name: str,
    voice_rate: float,
    voice_volume: float,
    provider_signature: dict,
) -> str:
    """生成试听缓存指纹，任一配音参数变化后自动让旧试听结果失效。"""
    payload = {
        "preview_type": preview_type,
        "content": content,
        "tts_server": tts_server,
        "voice_name": voice_name,
        "voice_rate": voice_rate,
        "voice_volume": voice_volume,
        "provider_signature": provider_signature,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _credential_signature(value: str) -> str:
    """
    生成只用于缓存失效判断的凭证摘要。

    摘要不会写入配置、日志或任务文件。用户修改 API Key 后摘要会变化，从而
    强制重新调用当前配音服务，避免旧试听缓存让无效的新凭证看起来可用。
    """
    normalized_value = str(value or "")
    if not normalized_value:
        return ""
    return hashlib.sha256(normalized_value.encode("utf-8")).hexdigest()


def _get_voice_preview_provider_signature(tts_server: str) -> dict:
    """
    返回会影响试听结果的非敏感 Provider 配置。

    API Key 只以单向摘要参与缓存指纹，原始凭证不会进入缓存或日志。模型、
    服务地址、区域或凭证发生变化时都必须重新生成试听，否则界面可能继续播放
    旧 Provider 配置下的音频，让用户误判当前设置已经生效。
    """
    if tts_server == "azure-tts-v2":
        return {
            "speech_region": config.azure.get("speech_region", ""),
            "credential": _credential_signature(config.azure.get("speech_key", "")),
        }
    if tts_server == "siliconflow":
        return {
            "credential": _credential_signature(config.siliconflow.get("api_key", ""))
        }
    if tts_server == "gemini-tts":
        return {
            "credential": _credential_signature(config.app.get("gemini_api_key", ""))
        }
    if tts_server == "mimo-tts":
        return {"credential": _credential_signature(config.app.get("mimo_api_key", ""))}
    if tts_server == "minimax-tts":
        return {
            "base_url": voice.get_minimax_tts_endpoint(),
            "model_id": config.minimax_tts.get("model_id", ""),
            "voice_id": config.minimax_tts.get("voice_id", ""),
            "credential": _credential_signature(voice.get_minimax_tts_api_key()),
        }
    if tts_server == "elevenlabs":
        return {
            "model_id": config.elevenlabs.get("model_id", ""),
            "credential": _credential_signature(config.elevenlabs.get("api_key", "")),
        }
    if tts_server == "chatterbox":
        return {
            "base_url": config.chatterbox.get("base_url", ""),
            "model_id": config.chatterbox.get("model_id", ""),
            "credential": _credential_signature(config.chatterbox.get("api_key", "")),
        }
    return {}


def _synthesize_voice_preview(
    *,
    content: str,
    preview_type: str,
    selected_tts_server: str,
    voice_name: str,
    voice_rate: float,
    voice_volume: float,
) -> dict | None:
    """生成一次试听并转为内存缓存，临时文件不会跨会话长期保留。"""
    if selected_tts_server == "chatterbox":
        _sync_chatterbox_config_from_session_state()

    temp_dir = utils.storage_dir("temp", create=True)
    audio_file = os.path.join(temp_dir, f"tmp-voice-{str(uuid4())}.mp3")
    logger.info(
        f"generating {preview_type} voice preview: "
        f"voice={voice_name}, rate={voice_rate}, volume={voice_volume}, "
        f"text_length={len(content)}"
    )
    try:
        with config.try_runtime_config_lock() as lock_acquired:
            if not lock_acquired:
                return {"busy": True}
            sub_maker = voice.tts(
                text=content,
                voice_name=voice_name,
                voice_rate=voice_rate,
                voice_file=audio_file,
                voice_volume=voice_volume,
            )
        if not sub_maker or not os.path.exists(audio_file):
            logger.error(f"{preview_type} voice preview did not produce an audio file")
            return None

        with open(audio_file, "rb") as file:
            audio_bytes = file.read()
        if not audio_bytes:
            logger.error(f"voice preview audio file is empty: {audio_file}")
            return None

        duration = voice.get_audio_duration(audio_file)
        if (
            not isinstance(duration, (int, float))
            or not math.isfinite(duration)
            or duration <= 0
        ):
            logger.warning(
                f"voice preview duration is unavailable: "
                f"preview_type={preview_type}, voice={voice_name}"
            )
            duration = None

        return {
            "audio_bytes": audio_bytes,
            "mime_type": _detect_audio_mime(audio_file, audio_bytes),
            "duration": duration,
            "preview_type": preview_type,
            "sub_maker": sub_maker,
        }
    finally:
        # 浏览器播放器使用内存字节，文件读取完即可清理，避免频繁试听积累临时文件。
        try:
            os.remove(audio_file)
        except FileNotFoundError:
            pass
        except OSError as exc:
            # 清理失败不应覆盖真正的 TTS 响应或异常，但需要保留路径和系统错误，
            # 方便排查权限、只读文件系统等环境问题。
            logger.warning(
                f"failed to delete voice preview file {audio_file}: {str(exc)}"
            )


def _render_voice_preview(params, friendly_names, selected_tts_server, voice_name):
    """渲染低成本短试听、完整文案时长估算和按需完整配音预览。"""
    if not friendly_names:
        return

    script_content = str(params.video_script or "").strip()
    estimated_range = _estimate_voiceover_duration_range(
        script_content,
        params.voice_rate,
    )
    if estimated_range:
        st.caption(
            tr("Estimated Voiceover Duration").format(
                min=estimated_range[0],
                max=estimated_range[1],
            )
        )
    else:
        st.caption(tr("Voiceover Script Required"))

    sample_content = _get_voice_preview_sample(voice_name)
    provider_signature = _get_voice_preview_provider_signature(selected_tts_server)
    preview_columns = st.columns(2)
    short_preview_requested = preview_columns[0].button(
        tr("Play Voice"),
        key="play_voice_button",
        icon=":material/graphic_eq:",
        use_container_width=True,
    )
    full_preview_requested = preview_columns[1].button(
        tr("Generate Full Voiceover Preview"),
        key="generate_full_voiceover_preview_button",
        icon=":material/article:",
        help=tr("Full Voiceover Preview Cost Hint"),
        use_container_width=True,
        disabled=not bool(script_content),
    )

    preview_type = ""
    preview_content = ""
    if short_preview_requested:
        preview_type = "sample"
        preview_content = sample_content
    elif full_preview_requested:
        preview_type = "full"
        preview_content = script_content

    sample_fingerprint = _voice_preview_fingerprint(
        preview_type="sample",
        content=sample_content,
        tts_server=selected_tts_server,
        voice_name=voice_name,
        voice_rate=params.voice_rate,
        voice_volume=params.voice_volume,
        provider_signature=provider_signature,
    )
    full_fingerprint = (
        _voice_preview_fingerprint(
            preview_type="full",
            content=script_content,
            tts_server=selected_tts_server,
            voice_name=voice_name,
            voice_rate=params.voice_rate,
            voice_volume=params.voice_volume,
            provider_signature=provider_signature,
        )
        if script_content
        else ""
    )

    if preview_type:
        requested_fingerprint = (
            sample_fingerprint if preview_type == "sample" else full_fingerprint
        )
        cached_preview = st.session_state.get("voice_preview_audio")
        if (
            not cached_preview
            or cached_preview.get("fingerprint") != requested_fingerprint
        ):
            try:
                with st.spinner(tr("Synthesizing Voice")):
                    preview_result = _synthesize_voice_preview(
                        content=preview_content,
                        preview_type=preview_type,
                        selected_tts_server=selected_tts_server,
                        voice_name=voice_name,
                        voice_rate=params.voice_rate,
                        voice_volume=params.voice_volume,
                    )
            except Exception as exc:
                logger.exception(f"failed to generate {preview_type} voice preview")
                st.error(tr("Voice Preview Failed").format(error=str(exc)))
            else:
                if preview_result and preview_result.get("busy"):
                    st.warning(tr("Voice Preview Busy"))
                elif preview_result:
                    preview_result["fingerprint"] = requested_fingerprint
                    st.session_state["voice_preview_audio"] = preview_result
                else:
                    st.error(tr("Voice Preview No Audio"))

    cached_preview = st.session_state.get("voice_preview_audio")
    valid_fingerprints = {sample_fingerprint, full_fingerprint}
    if (
        cached_preview
        and cached_preview.get("fingerprint") in valid_fingerprints
        and cached_preview.get("audio_bytes")
    ):
        st.audio(
            cached_preview["audio_bytes"],
            format=cached_preview.get("mime_type", "audio/mp3"),
        )
        if cached_preview.get("preview_type") == "full":
            duration = cached_preview.get("duration")
            if isinstance(duration, (int, float)) and duration > 0:
                st.caption(
                    tr("Actual Voiceover Duration").format(duration=f"{duration:.1f}")
                )
            else:
                st.warning(tr("Voice Preview Duration Unavailable"))


def _get_reusable_full_voice_preview(params, voice_mode: str) -> dict | None:
    """
    返回与当前生成参数完全匹配的完整试听缓存。

    只复用完整文案试听，短音色样例永远不能进入正式任务。指纹统一覆盖文案、
    Provider、音色、语速、音量和非敏感配置摘要；任何参数变化都会自然回退到
    正常 TTS 流程。字幕时间轴和有效时长同样是必需条件，避免只复用音频后让
    Edge 字幕链路失去 SubMaker。
    """
    if voice_mode != VOICE_MODE_TTS:
        return None

    script_content = str(params.video_script or "").strip()
    selected_tts_server = config.ui.get("tts_server", "azure-tts-v1")
    if (
        not script_content
        or not params.voice_name
        # 正式视频会在 MoviePy 合成阶段统一应用配音音量；部分 Provider 又会
        # 在 TTS 阶段直接写入音量增益。非默认音量下复用试听可能造成二次增益，
        # 因此先保守回退原流程，避免为少量场景引入 Provider 特判。
        or not math.isclose(float(params.voice_volume), 1.0)
    ):
        return None

    expected_fingerprint = _voice_preview_fingerprint(
        preview_type="full",
        content=script_content,
        tts_server=selected_tts_server,
        voice_name=params.voice_name,
        voice_rate=params.voice_rate,
        voice_volume=params.voice_volume,
        provider_signature=_get_voice_preview_provider_signature(selected_tts_server),
    )
    cached_preview = st.session_state.get("voice_preview_audio")
    if (
        not cached_preview
        or cached_preview.get("fingerprint") != expected_fingerprint
        or cached_preview.get("preview_type") != "full"
        or not cached_preview.get("audio_bytes")
        or cached_preview.get("sub_maker") is None
    ):
        return None

    duration = cached_preview.get("duration")
    if (
        not isinstance(duration, (int, float))
        or not math.isfinite(duration)
        or duration <= 0
    ):
        return None

    return {
        "audio_bytes": bytes(cached_preview["audio_bytes"]),
        "duration": float(duration),
        "sub_maker": cached_preview["sub_maker"],
        "script": script_content,
        "voice_name": params.voice_name,
        "voice_rate": float(params.voice_rate),
        "voice_volume": float(params.voice_volume),
    }


def _sync_minimax_tts_api_key_input():
    """
    同步 MiniMax TTS 密码控件，并返回当前有效 Key。

    TTS 专用 Key 为空时允许复用 MiniMax LLM Key。共享 Key 只用于当前控件和
    请求，不自动复制到 [minimax_tts]，避免同一凭证在配置文件中重复维护。
    """
    widget_key = "minimax_tts_api_key_input"
    configured_key = str(config.minimax_tts.get("api_key", "") or "").strip()
    shared_key = str(
        config.app.get("minimax_api_key", "") or get_secret("MINIMAX_API_KEY") or ""
    ).strip()
    effective_key = configured_key or shared_key
    had_widget_state = widget_key in st.session_state
    entered_key = str(st.session_state.get(widget_key, "") or "").strip()

    if not entered_key and effective_key:
        # 浏览器重连可能重放空密码状态。恢复已配置凭证，防止空值覆盖配置，
        # 同时确保当前 rerun 的试听请求可以直接使用有效 Key。
        st.session_state[widget_key] = effective_key
        entered_key = effective_key
        if had_widget_state:
            logger.debug("restored MiniMax TTS API key after empty session replay")
    elif not had_widget_state:
        st.session_state[widget_key] = effective_key
        entered_key = effective_key

    if entered_key and entered_key != effective_key:
        _set_runtime_config("minimax_tts", "api_key", entered_key)

    return entered_key


def _get_cached_minimax_voices(api_key: str, endpoint: str) -> list[dict[str, str]]:
    """按站点和凭证摘要读取当前会话中的 MiniMax 音色查询结果。"""
    cache = st.session_state.get("minimax_tts_voice_catalog_cache", {})
    cache_key = f"{endpoint}|{_credential_signature(api_key)}"
    cached_voices = cache.get(cache_key, [])
    return cached_voices if isinstance(cached_voices, list) else []


def _cache_minimax_voices(
    api_key: str,
    endpoint: str,
    voices: list[dict[str, str]],
):
    """缓存主动查询到的音色，避免普通控件 rerun 后重复请求 MiniMax。"""
    cache = st.session_state.setdefault("minimax_tts_voice_catalog_cache", {})
    cache_key = f"{endpoint}|{_credential_signature(api_key)}"
    cache[cache_key] = voices


def _render_minimax_tts_settings() -> tuple[list[str], dict[str, str]]:
    """渲染 MiniMax TTS 配置，并返回统一音色选择器使用的选项和文案。"""
    effective_api_key = _sync_minimax_tts_api_key_input()
    effective_api_key = st.text_input(
        tr("MiniMax TTS API Key"),
        type="password",
        key="minimax_tts_api_key_input",
    ).strip()

    dedicated_key = str(config.minimax_tts.get("api_key", "") or "").strip()
    minimax_tts_endpoints = [voice.MINIMAX_TTS_GLOBAL_URL, voice.MINIMAX_TTS_CN_URL]
    effective_endpoint = voice.get_minimax_tts_endpoint()
    if effective_endpoint not in minimax_tts_endpoints:
        effective_endpoint = voice.MINIMAX_TTS_GLOBAL_URL
    minimax_tts_base_url = stable_selectbox(
        tr("MiniMax TTS Endpoint"),
        options=minimax_tts_endpoints,
        default_value=effective_endpoint,
        key="minimax_tts_endpoint_select",
        # 复用 LLM Key 时必须跟随 LLM 所在区域，避免界面允许选择一个实际
        # 不会生效的地址；填写独立 TTS Key 后即可单独选择站点。
        disabled=not dedicated_key,
    )
    if dedicated_key:
        _set_runtime_config("minimax_tts", "base_url", minimax_tts_base_url)

    configured_model = config.minimax_tts.get(
        "model_id", voice.MINIMAX_TTS_DEFAULT_MODEL
    )
    if configured_model not in voice.MINIMAX_TTS_MODELS:
        configured_model = voice.MINIMAX_TTS_DEFAULT_MODEL
    minimax_tts_model = stable_selectbox(
        tr("MiniMax TTS Model"),
        options=list(voice.MINIMAX_TTS_MODELS),
        default_value=configured_model,
        key="minimax_tts_model_select",
    )
    _set_runtime_config("minimax_tts", "model_id", minimax_tts_model)

    if _action_button_clicked(
        "load_minimax_voices",
        st.button(
            tr("Load MiniMax Voices"),
            key="load_minimax_voices_button",
            icon=":material/refresh:",
            use_container_width=True,
            disabled=not _action_ready("load_minimax_voices"),
        ),
    ):
        try:
            available_voices = voice.get_minimax_voice_catalog(
                api_key=effective_api_key,
                endpoint=minimax_tts_base_url,
                voice_type="all",
            )
        except Exception as exc:
            # 这里必须把异常暴露给用户并记录日志。账号区域不匹配、Key 权限不足
            # 或网络失败都很常见，静默返回空列表会让用户误以为账号没有音色。
            logger.warning(f"load MiniMax voices failed: {exc}")
            st.error(tr("MiniMax Voices Load Failed").format(error=str(exc)))
        else:
            _cache_minimax_voices(
                effective_api_key,
                minimax_tts_base_url,
                available_voices,
            )
            st.success(tr("MiniMax Voices Loaded").format(count=len(available_voices)))
        # 加载完成后才进入冷却窗口，排队重复点击不会重复拉取音色列表。
        _action_mark_triggered("load_minimax_voices")

    available_voices = _get_cached_minimax_voices(
        effective_api_key,
        minimax_tts_base_url,
    )
    voice_labels = {
        f"minimax:{item['voice_id']}": (
            f"{item['voice_name']} ({item['voice_id']})"
            if item["voice_name"] != item["voice_id"]
            else item["voice_id"]
        )
        for item in available_voices
    }
    configured_voice_id = str(
        config.minimax_tts.get("voice_id", voice.MINIMAX_TTS_DEFAULT_VOICE)
        or voice.MINIMAX_TTS_DEFAULT_VOICE
    ).strip()
    configured_voice = f"minimax:{configured_voice_id}"
    # 尚未点击获取音色、接口暂时不可用或配置使用列表外克隆音色时，仍保留
    # 当前 Voice ID，确保原有生成流程不依赖远端音色查询结果。
    voice_labels.setdefault(configured_voice, configured_voice_id)
    return list(voice_labels), voice_labels


def _sync_elevenlabs_api_key_input():
    """
    同步 ElevenLabs 密码控件、持久化配置和环境变量，并返回当前有效 Key。

    Streamlit 在浏览器标签页连接到重启后的服务时，可能重放一个空的密码控件
    状态。这个空值无法与用户主动清空可靠区分，因此当配置文件或环境变量仍有
    Key 时，优先恢复有效值，防止空状态覆盖配置并确保本次 rerun 能立即加载
    音色。需要彻底删除 Key 时应修改配置文件或环境变量，避免重连误判。
    """
    widget_key = "elevenlabs_api_key_input"
    configured_key = str(config.elevenlabs.get("api_key", "") or "").strip()
    env_key = (get_secret("ELEVENLABS_API_KEY") or "").strip()
    effective_key = configured_key or env_key
    had_widget_state = widget_key in st.session_state
    entered_key = str(st.session_state.get(widget_key, "") or "").strip()

    if not entered_key and effective_key:
        # 重连后的空状态不能覆盖有效凭证，同时必须在渲染音色列表之前恢复，
        # 否则配置文件虽然没有被清空，当前页面仍会使用空 Key 请求 ElevenLabs。
        st.session_state[widget_key] = effective_key
        entered_key = effective_key
        if had_widget_state:
            logger.debug("restored ElevenLabs API key after empty session replay")
    elif not had_widget_state:
        # 先初始化再创建控件，避免同时传 value 和 session_state 触发 Streamlit
        # 的默认值冲突警告；没有任何 Key 时初始化为空即可。
        st.session_state[widget_key] = entered_key

    if entered_key and entered_key != effective_key:
        # 用户主动输入的新值才落入 config.toml。环境变量作为有效值回填时不会
        # 被复制到文件，容器或部署平台注入的密钥仍只保留在运行环境中。
        for cache_key in list(st.session_state.keys()):
            if str(cache_key).startswith("elevenlabs_voices_"):
                del st.session_state[cache_key]
        _set_runtime_config("elevenlabs", "api_key", entered_key)

    return entered_key


def _render_elevenlabs_api_key_input(label_key):
    """
    渲染 ElevenLabs TTS 与配乐共用的唯一 API Key 输入状态。

    同一页面若为 TTS 和配乐分别使用两个 widget key，Streamlit 会各自保留旧值，
    后渲染的输入框还会覆盖共享配置。这里统一使用一个 key，并集中处理环境变量
    回填、配置更新和音色缓存失效，确保界面显示与后台任务始终读取同一个值。
    """
    _sync_elevenlabs_api_key_input()
    return st.text_input(
        tr(label_key),
        type="password",
        key="elevenlabs_api_key_input",
    ).strip()


def _render_background_music_settings(params, elevenlabs_api_key_rendered=False):
    """渲染背景音乐来源与音量设置，并返回本次待保存的上传文件。"""
    uploaded_bgm_file = None
    st.divider()
    bgm_options = [
        (tr("No Background Music"), ""),
        (tr("Random Background Music"), "random"),
        (tr("Custom Background Music"), "custom"),
        (tr("Sonilo Background Music"), "sonilo"),
        (tr("ElevenLabs Background Music"), "elevenlabs"),
    ]
    selected_bgm_type = stable_selectbox(
        tr("Background Music Source"),
        options=[value for _, value in bgm_options],
        default_value="random",
        key="bgm_type_select",
        format_func=lambda value: dict((v, label) for label, v in bgm_options)[value],
    )
    params.bgm_type = selected_bgm_type
    if params.bgm_type == "sonilo":
        configured_key = str(config.app.get("sonilo_api_key", "") or "").strip()
        effective_key = configured_key or (get_secret("SONILO_API_KEY") or "").strip()
        entered_key = st.text_input(
            tr("Sonilo API Key"),
            value=effective_key,
            type="password",
            key="sonilo_api_key_input",
        ).strip()
        # 用户要求已配置的 Key 直接回填到密码输入框。配置值优先于环境变量；
        # 仅当用户确实修改输入或本来就使用配置时写回，避免把环境变量中的 Key
        # 在无操作的情况下复制进 config.toml。
        if configured_key or entered_key != effective_key:
            _set_runtime_config("app", "sonilo_api_key", entered_key)
    elif params.bgm_type == "elevenlabs":
        if elevenlabs_api_key_rendered:
            # TTS 区域已经渲染共享输入框时不再创建第二个 widget，避免两个独立
            # session_state 值互相覆盖。说明文字帮助用户定位上方的共用配置。
            st.caption(tr("ElevenLabs API Key Help"))
        else:
            _render_elevenlabs_api_key_input("ElevenLabs Music API Key")

    params.bgm_volume = stable_selectbox(
        tr("Background Music Volume"),
        options=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        default_value=0.2,
        key="bgm_volume_select",
        format_func=lambda value: f"{int(value * 100)}%",
        disabled=not params.bgm_type,
    )

    # 音效套件属于进阶选项，默认折叠成一个干净的下拉入口，避免在窄面板里
    # 挤占主设置区域。使用带 key 的局部容器复用现有 advanced_settings_ 样式。
    with st.container(key="advanced_settings_audio_fx"):
        with st.expander(tr("Audio FX & Atmosphere Suite"), expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                params.audio_ducking_enabled = st.checkbox(
                    tr("Enable Smart Audio Ducking"),
                    value=getattr(params, "audio_ducking_enabled", True),
                    help="Automatically lowers background and atmospheric audio when voiceover speaks.",
                )
                params.audio_ducking_intensity = st.slider(
                    tr("Ducking Intensity"),
                    min_value=0.1,
                    max_value=1.0,
                    value=getattr(params, "audio_ducking_intensity", 0.3) or 0.3,
                    step=0.1,
                    help="How quiet BGM gets during voiceover. 0.3 = 30% volume.",
                    disabled=not params.audio_ducking_enabled,
                )
            with col2:
                params.atmosphere_enabled = st.checkbox(
                    tr("Enable Atmospheric Soundscape"),
                    value=getattr(params, "atmosphere_enabled", False),
                    help="Loads random ambient soundscapes from resource/sfx/atmosphere/",
                )
                params.atmosphere_volume = st.slider(
                    tr("Atmosphere Volume"),
                    min_value=0.0,
                    max_value=1.0,
                    value=getattr(params, "atmosphere_volume", 0.3) or 0.3,
                    step=0.1,
                    disabled=not params.atmosphere_enabled,
                )

            params.sfx_volume = st.slider(
                tr("Transition SFX Volume"),
                min_value=0.0,
                max_value=1.5,
                value=getattr(params, "sfx_volume", 0.8) or 0.8,
                step=0.1,
                help="Volume for whooshes/risers loaded from resource/sfx/transitions/",
            )
    bgm_enabled = bgm_service.should_use_bgm(params.bgm_type, params.bgm_volume)

    if params.bgm_type == "custom":
        uploaded_bgm_file = st.file_uploader(
            tr("Upload Background Music"),
            type=[
                extension.removeprefix(".")
                for extension in bgm_service.SUPPORTED_BGM_EXTENSIONS
            ],
            accept_multiple_files=False,
            key="custom_bgm_uploader",
            help=tr("Upload Background Music Help"),
            # Streamlit 默认会在控件上展示全局 200MB 上限。这里必须与服务层
            # 30MB 硬限制保持一致，避免界面允许选择、提交时才被服务端拒绝。
            max_upload_size=bgm_service.MAX_BGM_UPLOAD_BYTES // (1024 * 1024),
        )
        if uploaded_bgm_file is not None and bgm_enabled:
            try:
                safe_name = bgm_service.sanitize_upload_filename(uploaded_bgm_file.name)
                # Streamlit 在调整音量等任意控件后都会重新执行页面。使用内容哈希
                # 区分上传文件，并在当前会话内缓存完整解码结果，既不能只凭同名、
                # 同大小文件误用旧结果，也避免每次 rerun 都重复调用 FFmpeg。
                validation_key = (
                    safe_name,
                    uploaded_bgm_file.size,
                    hashlib.sha256(uploaded_bgm_file.getbuffer()).hexdigest(),
                )
                cached_validation = st.session_state.get("custom_bgm_validation")
                if (
                    not cached_validation
                    or cached_validation.get("key") != validation_key
                ):
                    try:
                        bgm_service.validate_bgm_upload(
                            uploaded_bgm_file.name, uploaded_bgm_file
                        )
                    except bgm_service.BgmUploadError as exc:
                        cached_validation = {
                            "key": validation_key,
                            "error": str(exc),
                            "error_type": "upload",
                        }
                        # 同一个文件指纹的失败结果会进入会话缓存，因此这里只在
                        # 首次真实执行校验时记录一次，避免普通控件 rerun 刷屏。
                        logger.warning(
                            "WebUI background music validation rejected: "
                            f"name={safe_name}, error={str(exc)}"
                        )
                    except bgm_service.BgmServiceError as exc:
                        cached_validation = {
                            "key": validation_key,
                            "error": str(exc),
                            "error_type": "service",
                        }
                        logger.error(
                            "WebUI background music validation failed: "
                            f"name={safe_name}, error={str(exc)}"
                        )
                    else:
                        cached_validation = {
                            "key": validation_key,
                            "error": "",
                            "error_type": "",
                        }
                    st.session_state["custom_bgm_validation"] = cached_validation

                if cached_validation.get("error"):
                    if cached_validation.get("error_type") == "service":
                        raise bgm_service.BgmServiceError(cached_validation["error"])
                    raise bgm_service.BgmUploadError(cached_validation["error"])
            except bgm_service.BgmUploadError:
                # 非法文件不能沿用上一次有效上传的名称，否则任务参数可能仍指向
                # 历史 BGM。保留 UploadedFile 返回值，让用户点击生成时仍会被最终
                # 服务端校验拦截，而不是静默生成一条没有背景音乐的视频。
                params.bgm_file = ""
                st.error(tr("Invalid Background Music"))
            except bgm_service.BgmServiceError:
                params.bgm_file = ""
                st.error(tr("Background Music Validation Failed"))
            else:
                # 完整解码校验通过后才展示播放器和“已就绪”。文件仍只在点击
                # 生成时持久化，用户仅预览或随后移除文件不会污染 storage/bgm。
                uploaded_mime_type = str(getattr(uploaded_bgm_file, "type", "") or "")
                preview_mime_type = (
                    uploaded_mime_type
                    if uploaded_mime_type.startswith("audio/")
                    else mimetypes.guess_type(safe_name)[0] or "audio/mpeg"
                )
                st.audio(uploaded_bgm_file, format=preview_mime_type)
                st.info(f"{tr('Background Music Ready')}: {safe_name}")
                params.bgm_file = safe_name

        custom_bgm_file = st.text_input(
            tr("Custom Background Music File"),
            key="custom_bgm_file_input",
            disabled=uploaded_bgm_file is not None,
        )
        if uploaded_bgm_file is None and custom_bgm_file and bgm_enabled:
            # 文件名由服务层映射到 storage/bgm 或 resource/songs 后校验，
            # UI 不接受两个白名单目录之外的任意路径。
            params.bgm_file = custom_bgm_file.strip()
        elif not bgm_enabled:
            # 上传控件继续保留用户已选择的文件，调高音量后的下一次 rerun 会自动
            # 完整校验；当前任务参数必须清空，避免 0 音量任务保存或解析该文件。
            params.bgm_file = ""

    if params.bgm_type == "sonilo":
        params.video_music_prompt = st.text_input(
            tr("Sonilo Music Prompt"),
            key="sonilo_bgm_prompt_input",
            max_chars=sonilo_service.MAX_PROMPT_LENGTH,
            help=tr("Sonilo Music Prompt Help"),
        ).strip()
        if params.video_count > 1:
            st.warning(tr("Sonilo Multiple Videos Warning"))
        if _action_button_clicked(
            "test_sonilo_connection",
            st.button(
                tr("Test Sonilo Connection"),
                key="test_sonilo_connection_button",
                use_container_width=True,
                disabled=not _action_ready("test_sonilo_connection"),
            ),
        ):
            try:
                sonilo_service.test_connection()
            except sonilo_service.SoniloError as exc:
                logger.warning(f"Sonilo connection test failed: {exc}")
                st.error(tr("Sonilo Connection Test Failed").format(error=str(exc)))
            else:
                st.success(tr("Sonilo Connection Test Succeeded"))
            # 探测完成后才进入冷却窗口，排队重复点击不会重复探测。
            _action_mark_triggered("test_sonilo_connection")
    elif params.bgm_type == "elevenlabs":
        params.video_music_prompt = st.text_input(
            tr("ElevenLabs Music Prompt"),
            key="elevenlabs_music_prompt_input",
            max_chars=elevenlabs_music_service.MAX_PROMPT_LENGTH,
            help=tr("ElevenLabs Music Prompt Help"),
        ).strip()
        if params.video_count > 1:
            st.warning(tr("ElevenLabs Multiple Videos Warning"))
        if _action_button_clicked(
            "test_elevenlabs_connection",
            st.button(
                tr("Test ElevenLabs Connection"),
                key="test_elevenlabs_music_connection_button",
                use_container_width=True,
                disabled=not _action_ready("test_elevenlabs_connection"),
            ),
        ):
            try:
                elevenlabs_music_service.test_connection()
            except elevenlabs_music_service.ElevenLabsPaidPlanRequiredError:
                st.error(tr("ElevenLabs Paid Plan Required"))
            except elevenlabs_music_service.ElevenLabsMusicError as exc:
                logger.warning(f"ElevenLabs connection test failed: {exc}")
                st.error(tr("ElevenLabs Connection Test Failed").format(error=str(exc)))
            else:
                st.success(tr("ElevenLabs Connection Test Succeeded"))
            # 探测完成后才进入冷却窗口，排队重复点击不会重复探测。
            _action_mark_triggered("test_elevenlabs_connection")
    if params.bgm_type == "sonilo" and bgm_enabled and not sonilo_service.is_enabled():
        # 音量为 0 时任务层不会生成或混合 Sonilo 配乐，因此无需提示 Key；
        # 该判断与任务入口共用服务层规则，避免界面提示和实际执行条件分叉。
        st.warning(tr("Sonilo API Key Required"))
    elif (
        params.bgm_type == "elevenlabs"
        and bgm_enabled
        and not elevenlabs_music_service.is_enabled()
    ):
        st.warning(tr("ElevenLabs API Key Required"))
    return uploaded_bgm_file


def _render_audio_settings(panel, params):
    """渲染音频设置并返回上传音频与当前配音模式。"""
    with panel:
        with st.container(border=True):
            st.markdown(f"### {tr('Audio Settings')}")

            # 配音方式是音频设置的一级状态，负责明确区分自动配音、用户上传和无配音。
            # 旧配置没有 voice_mode 时，根据原 tts_server 的无配音哨兵保持兼容。
            saved_tts_server = config.ui.get("tts_server", "azure-tts-v1")
            saved_voice_mode = config.ui.get("voice_mode")
            if saved_voice_mode not in {
                VOICE_MODE_TTS,
                VOICE_MODE_UPLOAD,
                VOICE_MODE_NONE,
            }:
                saved_voice_mode = (
                    VOICE_MODE_NONE
                    if saved_tts_server == voice.NO_VOICE_NAME
                    else VOICE_MODE_TTS
                )
            voice_mode_options = [VOICE_MODE_TTS, VOICE_MODE_UPLOAD, VOICE_MODE_NONE]
            voice_mode_labels = {
                VOICE_MODE_TTS: tr("Automatic Voiceover"),
                VOICE_MODE_UPLOAD: tr("Upload Voiceover"),
                VOICE_MODE_NONE: tr("No Voiceover"),
            }
            voice_mode = stable_segmented_control(
                tr("Voiceover Mode"),
                options=voice_mode_options,
                default_value=saved_voice_mode,
                key="voice_mode_control",
                format_func=lambda value: voice_mode_labels[value],
                width="stretch",
            )
            _set_runtime_config("ui", "voice_mode", voice_mode)
            tts_mode_enabled = voice_mode == VOICE_MODE_TTS

            # Provider 下拉只负责选择自动配音服务；无配音已经由上方模式控制，
            # 不再作为 TTS Provider 混入列表，避免两个入口表达同一状态。
            tts_servers = [
                ("azure-tts-v1", "Azure TTS V1"),
                ("azure-tts-v2", "Azure TTS V2"),
                ("siliconflow", "SiliconFlow TTS"),
                ("gemini-tts", "Google Gemini TTS"),
                ("mimo-tts", "Xiaomi MiMo TTS"),
                ("minimax-tts", "MiniMax TTS"),
                ("elevenlabs", "ElevenLabs TTS"),
                ("chatterbox", "Chatterbox TTS"),
            ]

            tts_server_values = [server_value for server_value, _ in tts_servers]
            if saved_tts_server not in tts_server_values:
                saved_tts_server = "azure-tts-v1"

            if tts_mode_enabled:
                selected_tts_server = stable_selectbox(
                    tr("Voiceover Service"),
                    options=tts_server_values,
                    default_value=saved_tts_server,
                    key="tts_server_select",
                    format_func=lambda value: dict(
                        (v, label) for v, label in tts_servers
                    )[value],
                )
            else:
                # 非自动配音模式不渲染 TTS 控件，但保留上次选择，切回后可以继续使用。
                selected_tts_server = saved_tts_server

            _set_runtime_config("ui", "tts_server", selected_tts_server)

            # 服务说明紧跟 Provider 选择，先告诉用户需要准备什么，再进入音色和
            # 凭证配置。没有说明的 Provider 不渲染空提示块。
            if tts_mode_enabled:
                provider_tips = get_tts_provider_tips(selected_tts_server)
                if provider_tips:
                    st.info(provider_tips)

            # MiniMax 只复用下方通用“配音声音”选择器。Provider 配置函数负责
            # 刷新远端音色并返回友好文案，不再额外渲染 Voice ID 和音色下拉框。
            minimax_voices = []
            minimax_voice_labels = {}
            if tts_mode_enabled and selected_tts_server == "minimax-tts":
                minimax_voices, minimax_voice_labels = _render_minimax_tts_settings()

            # 根据选择的TTS服务器获取声音列表
            filtered_voices = []
            saved_voice_name = config.ui.get("voice_name", "")
            if saved_tts_server == "azure-tts-v1":
                if (
                    not saved_voice_name
                    or "Natasha" in saved_voice_name
                    or saved_voice_name == "en-US-ChristopherNeural"
                ):
                    saved_voice_name = "en-US-ChristopherNeural-Male"
            elevenlabs_api_key_rendered = False

            if not tts_mode_enabled:
                # 上传音频和无配音模式不加载远程音色，减少无意义的网络请求和界面噪音。
                filtered_voices = []
            elif selected_tts_server == "siliconflow":
                # 获取硅基流动的声音列表
                filtered_voices = voice.get_siliconflow_voices()
            elif selected_tts_server == "gemini-tts":
                # 获取Gemini TTS的声音列表
                filtered_voices = voice.get_gemini_voices()
            elif selected_tts_server == "mimo-tts":
                # 获取 Xiaomi MiMo TTS 的预置音色列表
                filtered_voices = voice.get_mimo_voices()
            elif selected_tts_server == "minimax-tts":
                filtered_voices = minimax_voices
            elif selected_tts_server == "elevenlabs":
                # 音色列表位于 Key 输入框之前渲染，必须先统一恢复重连状态并读取
                # 配置/环境变量，否则页面会用空 Key 加载并缓存空音色列表。
                saved_elevenlabs_api_key = _sync_elevenlabs_api_key_input()
                cache_key = f"elevenlabs_voices_{saved_elevenlabs_api_key}"
                if cache_key not in st.session_state:
                    st.session_state[cache_key] = voice.get_elevenlabs_voices(
                        saved_elevenlabs_api_key
                    )
                filtered_voices = st.session_state[cache_key]
            elif selected_tts_server == "chatterbox":
                # 自托管 Chatterbox 服务的预置音色（来自 [chatterbox] voices 配置）
                _sync_chatterbox_config_from_session_state()
                filtered_voices = voice.get_chatterbox_voices()
            else:
                # 获取Azure的声音列表
                all_voices = voice.get_all_azure_voices(filter_locals=None)

                # 根据选择的TTS服务器筛选声音
                for v in all_voices:
                    if selected_tts_server == "azure-tts-v2":
                        # V2版本的声音名称中包含"v2"
                        if "V2" in v:
                            filtered_voices.append(v)
                    else:
                        # V1版本的声音名称中不包含"v2"
                        if "V2" not in v:
                            filtered_voices.append(v)

            def _friendly(v):
                if voice.is_no_voice(v):
                    return tr("No Voice Selected")
                if voice.is_elevenlabs_voice(v):
                    parts = v.split(":", 2)
                    return parts[2] if len(parts) >= 3 else v
                if voice.is_chatterbox_voice(v):
                    name = v.split(":", 1)[1] if ":" in v else v
                    return name.replace("-Female", "").replace("-Male", "")
                if voice.is_minimax_voice(v):
                    return minimax_voice_labels.get(v, v.split(":", 1)[1])
                return (
                    v.replace("Female", tr("Female"))
                    .replace("Male", tr("Male"))
                    .replace("Neural", "")
                )

            friendly_names = {v: _friendly(v) for v in filtered_voices}

            saved_voice_name_index = 0

            # 检查保存的声音是否在当前筛选的声音列表中
            if saved_voice_name in friendly_names:
                saved_voice_name_index = list(friendly_names.keys()).index(
                    saved_voice_name
                )
            else:
                # 如果不在，则根据当前UI语言选择一个默认声音
                for i, v in enumerate(filtered_voices):
                    if v.lower().startswith(st.session_state["ui_language"].lower()):
                        saved_voice_name_index = i
                        break

            # 如果没有找到匹配的声音，使用第一个声音
            if saved_voice_name_index >= len(friendly_names) and friendly_names:
                saved_voice_name_index = 0

            # 确保有声音可选
            if tts_mode_enabled and friendly_names:
                voice_name = stable_selectbox(
                    tr("Voiceover Voice"),
                    options=list(friendly_names.keys()),
                    default_value=list(friendly_names.keys())[saved_voice_name_index],
                    key=f"speech_synthesis_select_{selected_tts_server}",
                    format_func=lambda value: friendly_names.get(
                        value,
                        str(value).removeprefix("minimax:"),
                    ),
                    # MiniMax 支持用户直接输入列表外的克隆或生成音色 ID；其它
                    # Provider 维持原选择器行为，不扩大本次修改的影响范围。
                    accept_new_options=selected_tts_server == "minimax-tts",
                )

                if selected_tts_server == "minimax-tts":
                    custom_voice_id = str(voice_name or "").strip()
                    if custom_voice_id and not voice.is_minimax_voice(custom_voice_id):
                        voice_name = f"minimax:{custom_voice_id}"
                    if voice.is_minimax_voice(voice_name):
                        _set_runtime_config(
                            "minimax_tts",
                            "voice_id",
                            voice_name.split(":", 1)[1],
                        )

                params.voice_name = voice_name
                if not voice.is_no_voice(voice_name):
                    # 占位 sentinel 仅用于非自动模式的禁用展示，不覆盖用户上一次
                    # 真正选择的音色，切回自动配音后可以恢复原设置。
                    _set_runtime_config("ui", "voice_name", voice_name)
            elif tts_mode_enabled:
                # 如果没有声音可选，显示提示信息
                st.warning(
                    tr(
                        "No voices available for the selected TTS server. Please select another server."
                    )
                )
                voice_name = ""
                params.voice_name = ""
                _set_runtime_config("ui", "voice_name", "")
            else:
                # 非自动配音模式不显示音色控件，只复用保存值维持参数结构稳定。
                voice_name = saved_voice_name or voice.NO_VOICE_NAME
                params.voice_name = voice_name

            # 当选择V2版本或者声音是V2声音时，显示服务区域和API key输入框
            if tts_mode_enabled and (
                selected_tts_server == "azure-tts-v2"
                or (voice_name and voice.is_azure_v2_voice(voice_name))
            ):
                saved_azure_speech_region = config.azure.get("speech_region", "")
                saved_azure_speech_key = config.azure.get("speech_key", "")
                azure_speech_region = st.text_input(
                    tr("Speech Region"),
                    value=saved_azure_speech_region,
                    key="azure_speech_region_input",
                )
                azure_speech_key = st.text_input(
                    tr("Speech Key"),
                    value=saved_azure_speech_key,
                    type="password",
                    key="azure_speech_key_input",
                )
                _set_runtime_config("azure", "speech_region", azure_speech_region)
                _set_runtime_config("azure", "speech_key", azure_speech_key)

            if tts_mode_enabled and selected_tts_server == "gemini-tts":
                # Gemini TTS 与 Gemini LLM 共用同一份密钥；在音频面板提供直接入口，
                # 用户无需先切换 LLM Provider 才能完成语音配置。
                gemini_tts_api_key = st.text_input(
                    tr("Gemini API Key"),
                    value=config.app.get("gemini_api_key", ""),
                    type="password",
                    key="gemini_tts_api_key_input",
                )
                _set_runtime_config("app", "gemini_api_key", gemini_tts_api_key)

            # 当选择硅基流动时，显示API key输入框和说明信息
            if tts_mode_enabled and (
                selected_tts_server == "siliconflow"
                or (voice_name and voice.is_siliconflow_voice(voice_name))
            ):
                saved_siliconflow_api_key = config.siliconflow.get("api_key", "")

                siliconflow_api_key = st.text_input(
                    tr("SiliconFlow API Key"),
                    value=saved_siliconflow_api_key,
                    type="password",
                    key="siliconflow_api_key_input",
                )

                _set_runtime_config("siliconflow", "api_key", siliconflow_api_key)

            # 当选择 Xiaomi MiMo TTS 时，复用 MiMo LLM provider 的 API Key。
            # 这样用户如果同时使用 MiMo 生成文案和语音，只需要维护一份密钥。
            if tts_mode_enabled and (
                selected_tts_server == "mimo-tts"
                or (voice_name and voice.is_mimo_voice(voice_name))
            ):
                saved_mimo_api_key = config.app.get("mimo_api_key", "")

                mimo_api_key = st.text_input(
                    tr("MiMo API Key"),
                    value=saved_mimo_api_key,
                    type="password",
                    key="mimo_tts_api_key_input",
                )

                _set_runtime_config("app", "mimo_api_key", mimo_api_key)

            # ElevenLabs API key section
            if tts_mode_enabled and (
                selected_tts_server == "elevenlabs"
                or (voice_name and voice.is_elevenlabs_voice(voice_name))
            ):
                _render_elevenlabs_api_key_input(
                    "ElevenLabs API Key",
                )
                elevenlabs_api_key_rendered = True

                _elevenlabs_models = [
                    "eleven_multilingual_v2",
                    "eleven_flash_v2_5",
                    "eleven_v3",
                ]
                saved_elevenlabs_model = config.elevenlabs.get(
                    "model_id", "eleven_multilingual_v2"
                )
                if saved_elevenlabs_model not in _elevenlabs_models:
                    saved_elevenlabs_model = "eleven_multilingual_v2"
                elevenlabs_model = stable_selectbox(
                    tr("ElevenLabs Model"),
                    options=_elevenlabs_models,
                    default_value=saved_elevenlabs_model,
                    key="elevenlabs_model_select",
                )
                _set_runtime_config("elevenlabs", "model_id", elevenlabs_model)

            # Chatterbox API settings section (self-hosted, OpenAI-compatible)
            if tts_mode_enabled and (
                selected_tts_server == "chatterbox"
                or (voice_name and voice.is_chatterbox_voice(voice_name))
            ):
                chatterbox_base_url = st.text_input(
                    tr("Chatterbox Base URL"),
                    value=config.chatterbox.get("base_url")
                    or DEFAULT_CHATTERBOX_BASE_URL,
                    key="chatterbox_base_url_input",
                    placeholder=tr("Chatterbox Base URL Placeholder"),
                )
                _set_runtime_config(
                    "chatterbox", "base_url", (chatterbox_base_url or "").strip()
                )

                chatterbox_api_key = st.text_input(
                    tr("Chatterbox API Key"),
                    value=config.chatterbox.get("api_key", ""),
                    type="password",
                    key="chatterbox_api_key_input",
                )
                _set_runtime_config("chatterbox", "api_key", chatterbox_api_key)

                chatterbox_model = st.text_input(
                    tr("Chatterbox Model"),
                    value=config.chatterbox.get("model_id") or DEFAULT_CHATTERBOX_MODEL,
                    key="chatterbox_model_input",
                )
                _set_runtime_config(
                    "chatterbox",
                    "model_id",
                    (chatterbox_model or DEFAULT_CHATTERBOX_MODEL).strip(),
                )

                _saved_chatterbox_voices = (
                    _parse_chatterbox_voices(config.chatterbox.get("voices"))
                    or DEFAULT_CHATTERBOX_VOICES
                )
                if isinstance(_saved_chatterbox_voices, list):
                    _saved_chatterbox_voices = ", ".join(_saved_chatterbox_voices)
                chatterbox_voices = st.text_input(
                    tr("Chatterbox Voices"),
                    value=str(_saved_chatterbox_voices or ""),
                    key="chatterbox_voices_input",
                    placeholder=tr("Chatterbox Voices Placeholder"),
                )
                _set_runtime_config(
                    "chatterbox",
                    "voices",
                    _parse_chatterbox_voices(chatterbox_voices),
                )

            # 三种模式只渲染当前任务真正需要的控件。自动配音可调音量和语速；
            # 上传音频只需要文件和音量；无配音不再展示无效设置。
            params.voice_name = (
                voice.NO_VOICE_NAME if voice_mode == VOICE_MODE_NONE else voice_name
            )
            params.voice_volume = 1.0
            params.voice_rate = 1.0
            uploaded_audio_file = None

            if tts_mode_enabled:
                voice_control_cols = st.columns(2)
                with voice_control_cols[0]:
                    params.voice_volume = stable_selectbox(
                        tr("Voiceover Volume"),
                        options=[0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 4.0, 5.0],
                        default_value=1.0,
                        key="voice_volume_select",
                        format_func=lambda value: f"{int(value * 100)}%",
                        help=tr("Voiceover Volume Help"),
                    )

                with voice_control_cols[1]:
                    params.voice_rate = stable_selectbox(
                        tr("Voiceover Speed"),
                        options=[0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.8, 2.0],
                        default_value=1.0,
                        key="voice_rate_select",
                        format_func=lambda value: f"{value:.1f}×",
                        help=tr("Voiceover Speed Help"),
                    )

                # 试听必须位于音量和语速控件之后，确保调用使用当前控件值。
                _render_voice_preview(
                    params,
                    friendly_names,
                    selected_tts_server,
                    voice_name,
                )
            elif voice_mode == VOICE_MODE_UPLOAD:
                custom_audio_file_types = sorted(
                    extension.removeprefix(".") for extension in CUSTOM_AUDIO_EXTENSIONS
                )
                uploaded_audio_file = st.file_uploader(
                    tr("Upload Voiceover File"),
                    type=custom_audio_file_types
                    + [file_type.upper() for file_type in custom_audio_file_types],
                    accept_multiple_files=False,
                    key="custom_audio_file_uploader",
                    help=tr("Upload Voiceover File Help"),
                )
                params.voice_volume = stable_selectbox(
                    tr("Voiceover Volume"),
                    options=[0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 4.0, 5.0],
                    default_value=1.0,
                    key="voice_volume_select",
                    format_func=lambda value: f"{int(value * 100)}%",
                    help=tr("Voiceover Volume Help"),
                )
                if uploaded_audio_file:
                    st.audio(uploaded_audio_file, format="audio/mp3")
                    st.info(
                        tr(
                            "Custom audio will be used directly. TTS synthesis will be skipped for this task."
                        )
                    )
            uploaded_bgm_file = _render_background_music_settings(
                params,
                elevenlabs_api_key_rendered=elevenlabs_api_key_rendered,
            )
    return uploaded_audio_file, uploaded_bgm_file, voice_mode


_SUBTITLE_PRESET_LEGACY_CASING = {
    SubtitleCasing.AS_SPOKEN: "original",
    SubtitleCasing.UPPERCASE: "upper",
    SubtitleCasing.TITLE_CASE: "title",
    SubtitleCasing.LOWERCASE: "lower",
    SubtitleCasing.SENTENCE_CASE: "sentence_case",
}
_SUBTITLE_PRESET_LEGACY_POSITION = {
    SubtitlePosition.DYNAMIC: "auto",
    SubtitlePosition.BOTTOM: "bottom",
    SubtitlePosition.TOP: "top",
    SubtitlePosition.CENTER: "center",
    SubtitlePosition.CUSTOM: "custom",
}


def _subtitle_preset_widget_values(preset_key):
    """把预设的规范化值映射为 WebUI 控件的业务值。

    返回 {控件 key: 业务值}，用于把预设填充到控件初始状态；未覆盖的字段
    返回 None，调用方保留控件当前值。预设不写 session_state 以外的内容。
    """
    from app.services.subtitle_engine.presets import get_preset_registry

    mapping = get_preset_registry().get_required(preset_key).as_mapping()
    casing = mapping.get("casing")
    position = mapping.get("position")
    background = mapping.get("background")
    return {
        "font_name_select": mapping.get("font"),
        "subtitle_position_select": _SUBTITLE_PRESET_LEGACY_POSITION.get(position)
        if isinstance(position, SubtitlePosition)
        else position,
        "subtitle_casing_select": _SUBTITLE_PRESET_LEGACY_CASING.get(casing)
        if isinstance(casing, SubtitleCasing)
        else casing,
        "font_color_picker": mapping.get("color"),
        "font_size_slider": mapping.get("font_size"),
        "stroke_color_picker": mapping.get("outline_color"),
        "stroke_width_slider": mapping.get("outline_width"),
        "subtitle_background_enabled_checkbox": (
            bool(background) if background is not None else None
        ),
        "subtitle_background_color_picker": background,
        "rounded_subtitle_background_checkbox": mapping.get("rounded_background"),
        "subtitle_dynamic_sizing_checkbox": mapping.get("dynamic_scaling"),
        "subtitle_pop_in_bounce_checkbox": mapping.get("pop_bounce"),
        "subtitle_floating_motion_checkbox": mapping.get("kinetic_float"),
        "subtitle_highlight_color_picker": mapping.get("highlight_color"),
        "subtitle_background_opacity_slider": mapping.get("background_opacity"),
        "subtitle_vertical_offset_slider": mapping.get("vertical_offset"),
        "subtitle_active_word_highlight_checkbox": mapping.get("active_word_highlight"),
        "subtitle_dynamic_auto_avoidance_checkbox": mapping.get(
            "dynamic_auto_avoidance"
        ),
    }


def _preview_hex_to_rgb(hex_color, fallback=(255, 255, 255)):
    try:
        value = str(hex_color).lstrip("#")
        if len(value) == 6:
            return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))
        if len(value) == 3:
            return tuple(int(char * 2, 16) for char in value)
    except ValueError:
        pass
    return fallback


def _render_subtitle_preview(params, preview_text):
    """用 PIL 按当前样式渲染一张字幕预览图，不调用 MoviePy。

    预览只反映字体、字号、颜色、描边和背景等静态样式；位置与动画需要成片
    才能体现。字体缺失或渲染异常时静默返回 None，不影响页面其余功能。
    """
    try:
        from PIL import Image, ImageDraw, ImageFont

        from app.services.subtitle_engine.fonts import get_font_registry
        from app.services.subtitle_engine.styles import (
            SubtitleStyleResolver,
            apply_subtitle_casing,
        )
        from app.services.subtitle_engine.text import (
            estimate_line_height,
            wrap_text,
        )

        style = SubtitleStyleResolver().resolve(params)
        font_path = style.font_path or get_font_registry().resolve(style.font_name)
        cased_text = apply_subtitle_casing(preview_text, style.casing)

        preview_width, preview_height = 640, 220
        preview_font_size = 42
        max_text_width = int(preview_width * 0.9)
        wrapped, _ = wrap_text(
            cased_text,
            max_width=max_text_width,
            font=font_path,
            fontsize=preview_font_size,
        )
        lines = [line for line in wrapped.split("\n") if line.strip()] or [
            cased_text
        ]
        font = ImageFont.truetype(font_path, preview_font_size)
        line_height = estimate_line_height(font_path, preview_font_size)

        canvas = Image.new("RGB", (preview_width, preview_height), (16, 18, 26))
        draw = ImageDraw.Draw(canvas)
        top_color = (28, 32, 44)
        for y in range(preview_height):
            ratio = y / max(1, preview_height - 1)
            color = tuple(
                int(top_color[i] * (1 - ratio) + (10, 10, 14)[i] * ratio)
                for i in range(3)
            )
            draw.line([(0, y), (preview_width, y)], fill=color)

        pad_x = int(preview_font_size * 0.4)
        pad_y = int(preview_font_size * 0.35)
        interline = int(preview_font_size * 0.25)
        line_widths = []
        for line in lines:
            left, top, right, bottom = font.getbbox(line)
            line_widths.append(right - left)
        box_width = max(line_widths) + 2 * pad_x
        box_height = (
            len(lines) * line_height + 2 * pad_y + interline * (len(lines) - 1)
        )
        box_left = (preview_width - box_width) // 2
        box_top = (preview_height - box_height) // 2

        if style.background:
            background_rgb = _preview_hex_to_rgb(style.background, (0, 0, 0))
            alpha = int(255 * max(0.0, min(1.0, style.background_opacity or 1.0)))
            radius = max(8, int(preview_font_size * 0.4))
            overlay = Image.new("RGBA", (preview_width, preview_height), (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            overlay_draw.rounded_rectangle(
                [box_left, box_top, box_left + box_width, box_top + box_height],
                radius=radius,
                fill=background_rgb + (alpha,),
            )
            canvas = Image.alpha_composite(
                canvas.convert("RGBA"), overlay
            ).convert("RGB")
            draw = ImageDraw.Draw(canvas)

        stroke_color = _preview_hex_to_rgb(style.outline_color or "#000000", (0, 0, 0))
        text_color = _preview_hex_to_rgb(style.color, (255, 255, 255))
        stroke_width = max(0, int(round(style.outline_width or 0)))
        cursor_y = box_top + pad_y
        for index, line in enumerate(lines):
            left, top, right, bottom = font.getbbox(line)
            line_width = right - left
            text_x = box_left + (box_width - line_width) // 2 - left
            baseline = cursor_y + (line_height - (bottom - top)) // 2 - top
            draw.text(
                (text_x, baseline),
                line,
                font=font,
                fill=text_color,
                stroke_width=stroke_width,
                stroke_fill=stroke_color,
            )
            cursor_y += line_height + interline
        return canvas
    except Exception as exc:
        logger.warning(f"failed to render subtitle preview: {exc}")
        return None


def _render_subtitle_settings(panel, params):
    """渲染字幕设置并更新生成参数。"""
    with panel:
        with st.container(border=True):
            st.markdown(f"### {tr('Subtitle Settings')}")
            st.session_state.setdefault(
                "subtitle_enabled_checkbox",
                DEFAULT_SUBTITLE_SETTINGS["subtitle_enabled"],
            )
            params.subtitle_enabled = st.checkbox(
                tr("Enable Subtitles"),
                key="subtitle_enabled_checkbox",
            )
            subtitle_settings_disabled = not params.subtitle_enabled
            subtitle_presets = [
                (tr("Subtitle Preset Hormozi"), SubtitlePreset.HORMOZI.value),
                (tr("Subtitle Preset TikTok"), SubtitlePreset.TIKTOK.value),
                (tr("Subtitle Preset CapCut"), SubtitlePreset.CAPCUT.value),
                (tr("Subtitle Preset Cinematic"), SubtitlePreset.CINEMATIC.value),
                (tr("Subtitle Preset Minimal"), SubtitlePreset.MINIMAL.value),
                (tr("Subtitle Preset Neon"), SubtitlePreset.NEON.value),
                (tr("Subtitle Preset Custom"), SubtitlePreset.CUSTOM.value),
            ]
            saved_subtitle_preset = config.ui.get(
                "subtitle_style_preset",
                DEFAULT_SUBTITLE_SETTINGS["subtitle_style_preset"],
            )
            saved_preset_index = len(subtitle_presets) - 1
            for i, (_, preset_value) in enumerate(subtitle_presets):
                if preset_value == saved_subtitle_preset:
                    saved_preset_index = i
                    break
            selected_subtitle_preset = stable_selectbox(
                tr("Subtitle Preset"),
                options=[value for _, value in subtitle_presets],
                default_value=subtitle_presets[saved_preset_index][1],
                key="subtitle_style_preset_select",
                format_func=lambda value: dict(
                    (v, label) for label, v in subtitle_presets
                )[value],
                disabled=subtitle_settings_disabled,
                help=tr("Subtitle Preset Help"),
            )
            params.subtitle_style_preset = selected_subtitle_preset
            _set_runtime_config(
                "ui", "subtitle_style_preset", params.subtitle_style_preset
            )

            # 非自定义预设把样式值填充到控件初始状态；控件仍可编辑，任何手动
            # 修改都会在页面底部自动把预设切回 Custom 并给出提示。
            selected_preset_widget_values = None
            if selected_subtitle_preset != SubtitlePreset.CUSTOM.value:
                try:
                    selected_preset_widget_values = _subtitle_preset_widget_values(
                        selected_subtitle_preset
                    )
                    for widget_key, widget_value in (
                        selected_preset_widget_values.items()
                    ):
                        if widget_value is not None:
                            st.session_state[localized_widget_key(widget_key)] = (
                                widget_value
                            )
                except Exception as exc:
                    logger.warning(
                        f"failed to apply subtitle preset values: "
                        f"preset={selected_subtitle_preset}, {exc}"
                    )
                    selected_preset_widget_values = None

            font_names = get_all_fonts()
            saved_font_name = config.ui.get(
                "font_name", DEFAULT_SUBTITLE_SETTINGS["font_name"]
            )
            saved_font_name_index = 0
            if saved_font_name in font_names:
                saved_font_name_index = font_names.index(saved_font_name)
            params.font_name = stable_selectbox(
                tr("Font"),
                options=font_names,
                default_value=font_names[saved_font_name_index] if font_names else "",
                key="font_name_select",
                disabled=subtitle_settings_disabled,
            )
            _set_runtime_config("ui", "font_name", params.font_name)

            subtitle_positions = [
                (tr("Auto"), "auto"),
                (tr("Top"), "top"),
                (tr("Center"), "center"),
                (tr("Bottom"), "bottom"),
                (tr("Custom"), "custom"),
            ]
            saved_subtitle_position = config.ui.get(
                "subtitle_position", DEFAULT_SUBTITLE_SETTINGS["subtitle_position"]
            )
            saved_position_index = 3
            for i, (_, pos_value) in enumerate(subtitle_positions):
                if pos_value == saved_subtitle_position:
                    saved_position_index = i
                    break
            selected_subtitle_position = stable_selectbox(
                tr("Position"),
                options=[value for _, value in subtitle_positions],
                default_value=subtitle_positions[saved_position_index][1],
                key="subtitle_position_select",
                format_func=lambda value: dict(
                    (v, label) for label, v in subtitle_positions
                )[value],
                disabled=subtitle_settings_disabled,
            )
            params.subtitle_position = selected_subtitle_position
            _set_runtime_config("ui", "subtitle_position", params.subtitle_position)

            subtitle_casings = [
                (tr("Subtitle Casing As Spoken"), "as_spoken"),
                (tr("Subtitle Casing Uppercase"), "uppercase"),
                (tr("Subtitle Casing Lowercase"), "lowercase"),
                (tr("Subtitle Casing Sentence"), "sentence_case"),
                (tr("Subtitle Casing Title Case"), "title_case"),
                (tr("Subtitle Casing Original"), "original"),
                (tr("Subtitle Casing Upper"), "upper"),
                (tr("Subtitle Casing Title"), "title"),
                (tr("Subtitle Casing Lower"), "lower"),
            ]
            saved_subtitle_casing = config.ui.get(
                "subtitle_casing", DEFAULT_SUBTITLE_SETTINGS["subtitle_casing"]
            )
            saved_casing_index = 0
            for i, (_, casing_value) in enumerate(subtitle_casings):
                if casing_value == saved_subtitle_casing:
                    saved_casing_index = i
                    break
            params.subtitle_casing = stable_selectbox(
                tr("Subtitle Casing"),
                options=[value for _, value in subtitle_casings],
                default_value=subtitle_casings[saved_casing_index][1],
                key="subtitle_casing_select",
                format_func=lambda value: dict(
                    (v, label) for label, v in subtitle_casings
                )[value],
                disabled=subtitle_settings_disabled,
            )
            _set_runtime_config("ui", "subtitle_casing", params.subtitle_casing)

            if params.subtitle_position == "custom":
                saved_custom_position = config.ui.get(
                    "custom_position", DEFAULT_SUBTITLE_SETTINGS["custom_position"]
                )
                st.session_state.setdefault(
                    "custom_position_input", str(saved_custom_position)
                )
                custom_position = st.text_input(
                    tr("Custom Position (% from top)"),
                    key="custom_position_input",
                    disabled=subtitle_settings_disabled,
                )
                try:
                    params.custom_position = float(custom_position)
                    if params.custom_position < 0 or params.custom_position > 100:
                        st.error(tr("Please enter a value between 0 and 100"))
                    else:
                        _set_runtime_config(
                            "ui", "custom_position", params.custom_position
                        )
                except ValueError:
                    st.error(tr("Please enter a valid number"))

            saved_vertical_offset = config.ui.get(
                "subtitle_vertical_offset",
                DEFAULT_SUBTITLE_SETTINGS["subtitle_vertical_offset"],
            )
            st.session_state.setdefault(
                "subtitle_vertical_offset_slider", saved_vertical_offset
            )
            params.subtitle_vertical_offset = st.slider(
                tr("Subtitle Vertical Offset"),
                -200,
                200,
                key="subtitle_vertical_offset_slider",
                disabled=subtitle_settings_disabled,
                help=tr("Subtitle Vertical Offset Help"),
            )
            _set_runtime_config(
                "ui", "subtitle_vertical_offset", params.subtitle_vertical_offset
            )

            # 非中文语言的颜色标签通常比中文更长。为颜色选择器保留适当宽度，
            # 避免标签换行，同时仍给字号滑块保留足够的可操作空间。
            font_cols = st.columns([0.42, 0.58])
            with font_cols[0]:
                saved_text_fore_color = config.ui.get(
                    "text_fore_color", DEFAULT_SUBTITLE_SETTINGS["text_fore_color"]
                )
                st.session_state.setdefault("font_color_picker", saved_text_fore_color)
                params.text_fore_color = st.color_picker(
                    tr("Font Color"),
                    key="font_color_picker",
                    disabled=subtitle_settings_disabled,
                )
                _set_runtime_config("ui", "text_fore_color", params.text_fore_color)

            with font_cols[1]:
                saved_font_size = config.ui.get(
                    "font_size", DEFAULT_SUBTITLE_SETTINGS["font_size"]
                )
                st.session_state.setdefault("font_size_slider", saved_font_size)
                params.font_size = st.slider(
                    tr("Font Size"),
                    30,
                    100,
                    key="font_size_slider",
                    disabled=subtitle_settings_disabled,
                )
                _set_runtime_config("ui", "font_size", params.font_size)

            stroke_cols = st.columns([0.42, 0.58])
            with stroke_cols[0]:
                st.session_state.setdefault(
                    "stroke_color_picker", DEFAULT_SUBTITLE_SETTINGS["stroke_color"]
                )
                params.stroke_color = st.color_picker(
                    tr("Stroke Color"),
                    key="stroke_color_picker",
                    disabled=subtitle_settings_disabled,
                )
            with stroke_cols[1]:
                st.session_state.setdefault(
                    "stroke_width_slider", DEFAULT_SUBTITLE_SETTINGS["stroke_width"]
                )
                params.stroke_width = st.slider(
                    tr("Stroke Width"),
                    0.0,
                    10.0,
                    key="stroke_width_slider",
                    disabled=subtitle_settings_disabled,
                )

            style_cols = st.columns([0.42, 0.58])
            with style_cols[0]:
                saved_highlight_color = config.ui.get(
                    "subtitle_highlight_color",
                    DEFAULT_SUBTITLE_SETTINGS["subtitle_highlight_color"],
                )
                st.session_state.setdefault(
                    "subtitle_highlight_color_picker", saved_highlight_color
                )
                params.subtitle_highlight_color = st.color_picker(
                    tr("Subtitle Highlight Color"),
                    key="subtitle_highlight_color_picker",
                    disabled=subtitle_settings_disabled,
                    help=tr("Subtitle Highlight Color Help"),
                )
                _set_runtime_config(
                    "ui", "subtitle_highlight_color", params.subtitle_highlight_color
                )
            with style_cols[1]:
                saved_background_opacity = config.ui.get(
                    "subtitle_background_opacity",
                    DEFAULT_SUBTITLE_SETTINGS["subtitle_background_opacity"],
                )
                st.session_state.setdefault(
                    "subtitle_background_opacity_slider", saved_background_opacity
                )
                params.subtitle_background_opacity = st.slider(
                    tr("Subtitle Background Opacity"),
                    0.05,
                    1.0,
                    key="subtitle_background_opacity_slider",
                    disabled=subtitle_settings_disabled
                    or not st.session_state.get(
                        "subtitle_background_enabled_checkbox", False
                    ),
                    help=tr("Subtitle Background Opacity Help"),
                )
                _set_runtime_config(
                    "ui",
                    "subtitle_background_opacity",
                    params.subtitle_background_opacity,
                )

            # 背景开关的本地化名称普遍比颜色标签更长，因此让开关占据略多空间。
            subtitle_bg_cols = st.columns([0.55, 0.45])
            saved_subtitle_background_enabled = config.ui.get(
                "subtitle_background_enabled",
                DEFAULT_SUBTITLE_SETTINGS["subtitle_background_enabled"],
            )
            st.session_state.setdefault(
                "subtitle_background_enabled_checkbox",
                saved_subtitle_background_enabled,
            )
            with subtitle_bg_cols[0]:
                subtitle_background_enabled = st.checkbox(
                    tr("Enable Subtitle Background"),
                    key="subtitle_background_enabled_checkbox",
                    disabled=subtitle_settings_disabled,
                )
            _set_runtime_config(
                "ui",
                "subtitle_background_enabled",
                subtitle_background_enabled,
            )

            # 背景颜色和圆角样式都从属于字幕背景开关。子控件始终保留在页面中，
            # 父开关关闭时统一禁用，避免一个控件消失而另一个控件禁用造成布局跳动。
            # 颜色值仍保存在 UI 配置中，重新启用背景后可以恢复用户之前的选择；
            # 传给生成服务的参数则设为 False，确保关闭状态不会实际渲染背景。
            saved_subtitle_background_color = config.ui.get(
                "subtitle_background_color",
                DEFAULT_SUBTITLE_SETTINGS["subtitle_background_color"],
            )
            st.session_state.setdefault(
                "subtitle_background_color_picker",
                saved_subtitle_background_color,
            )
            with subtitle_bg_cols[1]:
                selected_subtitle_background_color = st.color_picker(
                    tr("Subtitle Background Color"),
                    key="subtitle_background_color_picker",
                    disabled=subtitle_settings_disabled
                    or not subtitle_background_enabled,
                )
            _set_runtime_config(
                "ui",
                "subtitle_background_color",
                selected_subtitle_background_color,
            )
            params.text_background_color = (
                selected_subtitle_background_color
                if subtitle_background_enabled
                else False
            )

            saved_rounded_subtitle_background = config.ui.get(
                "rounded_subtitle_background",
                DEFAULT_SUBTITLE_SETTINGS["rounded_subtitle_background"],
            )
            # 背景关闭时，圆角背景没有可渲染的底色。这里禁用控件但保留原配置，
            # 用户下次重新开启字幕背景后，可以继续使用之前保存的圆角偏好。
            rounded_background_disabled = (
                subtitle_settings_disabled or not subtitle_background_enabled
            )
            st.session_state.setdefault(
                "rounded_subtitle_background_checkbox",
                saved_rounded_subtitle_background,
            )
            selected_rounded_subtitle_background = st.checkbox(
                tr("Rounded Subtitle Background"),
                help=tr("Rounded Subtitle Background Help"),
                disabled=rounded_background_disabled,
                key="rounded_subtitle_background_checkbox",
            )
            params.rounded_subtitle_background = (
                selected_rounded_subtitle_background
                if subtitle_background_enabled
                else False
            )
            if not subtitle_settings_disabled and subtitle_background_enabled:
                _set_runtime_config(
                    "ui",
                    "rounded_subtitle_background",
                    selected_rounded_subtitle_background,
                )

            st.markdown(f"### {tr('Subtitle Animation')}")
            # 字幕动画开关默认全部关闭，保持历史渲染行为；用户可单独开启。
            saved_dynamic_sizing = config.ui.get(
                "subtitle_dynamic_sizing",
                DEFAULT_SUBTITLE_SETTINGS["subtitle_dynamic_sizing"],
            )
            st.session_state.setdefault(
                "subtitle_dynamic_sizing_checkbox", saved_dynamic_sizing
            )
            params.subtitle_dynamic_sizing = st.checkbox(
                tr("Dynamic Auto-Sizing"),
                help=tr("Dynamic Auto-Sizing Help"),
                key="subtitle_dynamic_sizing_checkbox",
                disabled=subtitle_settings_disabled,
            )
            _set_runtime_config(
                "ui", "subtitle_dynamic_sizing", params.subtitle_dynamic_sizing
            )

            saved_pop_in_bounce = config.ui.get(
                "subtitle_pop_in_bounce",
                DEFAULT_SUBTITLE_SETTINGS["subtitle_pop_in_bounce"],
            )
            st.session_state.setdefault(
                "subtitle_pop_in_bounce_checkbox", saved_pop_in_bounce
            )
            params.subtitle_pop_in_bounce = st.checkbox(
                tr("Enable Pop-in Bounce"),
                help=tr("Enable Pop-in Bounce Help"),
                key="subtitle_pop_in_bounce_checkbox",
                disabled=subtitle_settings_disabled,
            )
            _set_runtime_config(
                "ui", "subtitle_pop_in_bounce", params.subtitle_pop_in_bounce
            )

            saved_floating_motion = config.ui.get(
                "subtitle_floating_motion",
                DEFAULT_SUBTITLE_SETTINGS["subtitle_floating_motion"],
            )
            st.session_state.setdefault(
                "subtitle_floating_motion_checkbox", saved_floating_motion
            )
            params.subtitle_floating_motion = st.checkbox(
                tr("Enable Floating Motion"),
                help=tr("Enable Floating Motion Help"),
                key="subtitle_floating_motion_checkbox",
                disabled=subtitle_settings_disabled,
            )
            _set_runtime_config(
                "ui", "subtitle_floating_motion", params.subtitle_floating_motion
            )

            saved_active_word_highlight = config.ui.get(
                "subtitle_active_word_highlight",
                DEFAULT_SUBTITLE_SETTINGS["subtitle_active_word_highlight"],
            )
            st.session_state.setdefault(
                "subtitle_active_word_highlight_checkbox",
                saved_active_word_highlight,
            )
            params.subtitle_active_word_highlight = st.checkbox(
                tr("Subtitle Active Word Highlight"),
                help=tr("Subtitle Active Word Highlight Help"),
                key="subtitle_active_word_highlight_checkbox",
                disabled=subtitle_settings_disabled,
            )
            _set_runtime_config(
                "ui",
                "subtitle_active_word_highlight",
                params.subtitle_active_word_highlight,
            )

            saved_dynamic_auto_avoidance = config.ui.get(
                "subtitle_dynamic_auto_avoidance",
                DEFAULT_SUBTITLE_SETTINGS["subtitle_dynamic_auto_avoidance"],
            )
            st.session_state.setdefault(
                "subtitle_dynamic_auto_avoidance_checkbox",
                saved_dynamic_auto_avoidance,
            )
            params.subtitle_dynamic_auto_avoidance = st.checkbox(
                tr("Subtitle Dynamic Auto Avoidance"),
                help=tr("Subtitle Dynamic Auto Avoidance Help"),
                key="subtitle_dynamic_auto_avoidance_checkbox",
                disabled=subtitle_settings_disabled,
            )
            _set_runtime_config(
                "ui",
                "subtitle_dynamic_auto_avoidance",
                params.subtitle_dynamic_auto_avoidance,
            )

            if video.subtitle_colors_are_indistinguishable(params):
                # 同色配置仍然是合法的用户选择，因此只在字幕设置区域就近提示，
                # 不阻止生成。用户可以根据实际视觉需求决定是否继续。
                st.warning(tr("Subtitle Colors Are Indistinguishable"))

            # 预设模式下任何手动修改（与预设值不一致）都会把预设切回 Custom，
            # 保证用户看到的值和实际渲染一致，不会被预设静默覆盖。
            if (
                selected_subtitle_preset != SubtitlePreset.CUSTOM.value
                and selected_preset_widget_values is not None
            ):
                manual_changed = False
                for widget_key, preset_value in (
                    selected_preset_widget_values.items()
                ):
                    if preset_value is None:
                        continue
                    widget_value = st.session_state.get(
                        localized_widget_key(widget_key)
                    )
                    if widget_value != preset_value:
                        manual_changed = True
                        break
                if manual_changed:
                    params.subtitle_style_preset = SubtitlePreset.CUSTOM.value
                    st.session_state[
                        localized_widget_key("subtitle_style_preset_select")
                    ] = SubtitlePreset.CUSTOM.value
                    _set_runtime_config(
                        "ui", "subtitle_style_preset", SubtitlePreset.CUSTOM.value
                    )
                    st.info(tr("Subtitle Preset Switched To Custom Notice"))

            subtitle_preview_text = params.video_script or params.video_subject
            selected_font_path = os.path.join(font_dir, params.font_name)
            if (
                params.subtitle_enabled
                and subtitle_preview_text
                and not video.subtitle_font_supports_text(
                    selected_font_path, subtitle_preview_text
                )
            ):
                st.warning(tr("Subtitle Font Does Not Support Text"))

            if params.subtitle_enabled and subtitle_preview_text:
                subtitle_preview = _render_subtitle_preview(
                    params, subtitle_preview_text
                )
                if subtitle_preview is not None:
                    st.image(
                        subtitle_preview,
                        caption=tr("Subtitle Preview"),
                        use_container_width=True,
                    )

            if st.button(
                tr("Restore Default Subtitle Settings"),
                key="restore_default_subtitle_settings",
                icon=":material/restart_alt:",
                on_click=reset_subtitle_settings,
                use_container_width=True,
            ):
                st.toast(tr("Default Subtitle Settings Restored"))

        _render_overlay_settings(panel, params)


def _render_overlay_settings(panel, params):
    """渲染图文叠加层设置并更新生成参数（默认关闭，不影响历史行为）。"""
    with panel:
        with st.expander(tr("Overlay Studio"), expanded=False):
            st.markdown(f"### {tr('Overlay Settings')}")
            st.session_state.setdefault("overlay_enabled_checkbox", False)
            params.overlay_enabled = st.checkbox(
                tr("Enable Overlays"),
                help=tr("Enable Overlays Help"),
                key="overlay_enabled_checkbox",
            )
            overlay_disabled = not params.overlay_enabled

            overlay_styles = [
                (tr("Overlay Style Title Fact"), "title_fact"),
                (tr("Overlay Style Title Only"), "title_only"),
                (tr("Overlay Style Facts Only"), "facts_only"),
                (tr("Overlay Style Callouts Only"), "callouts_only"),
                (tr("Overlay Style Full"), "full"),
            ]
            saved_style = getattr(params, "overlay_style", "title_fact") or "title_fact"
            if saved_style not in {value for _, value in overlay_styles}:
                saved_style = "title_fact"
            params.overlay_style = stable_selectbox(
                tr("Overlay Style"),
                options=[value for _, value in overlay_styles],
                default_value=saved_style,
                key="overlay_style_select",
                format_func=lambda value: dict(
                    (v, label) for label, v in overlay_styles
                )[value],
                help=tr("Overlay Style Help"),
                disabled=overlay_disabled,
            )

            title_col, fact_col, callout_col = st.columns(3, gap="small")
            with title_col:
                st.session_state.setdefault("overlay_title_card_checkbox", True)
                params.overlay_title_card = st.checkbox(
                    tr("Title Card"),
                    key="overlay_title_card_checkbox",
                    disabled=overlay_disabled,
                )
            with fact_col:
                st.session_state.setdefault("overlay_fact_cards_checkbox", True)
                params.overlay_fact_cards = st.checkbox(
                    tr("Fact Cards"),
                    key="overlay_fact_cards_checkbox",
                    disabled=overlay_disabled,
                )
            with callout_col:
                st.session_state.setdefault("overlay_callouts_checkbox", False)
                params.overlay_callouts = st.checkbox(
                    tr("Callouts"),
                    key="overlay_callouts_checkbox",
                    disabled=overlay_disabled,
                )

            params.overlay_text_color = st.color_picker(
                tr("Overlay Text Color"),
                value=getattr(params, "overlay_text_color", "#FFFFFF"),
                key="overlay_text_color_picker",
                disabled=overlay_disabled,
            )
            params.overlay_bg_color = st.color_picker(
                tr("Overlay Background Color"),
                value=getattr(params, "overlay_bg_color", "#000000"),
                key="overlay_bg_color_picker",
                disabled=overlay_disabled,
            )
            params.overlay_image_opacity = st.slider(
                tr("Overlay Opacity"),
                min_value=0.0,
                max_value=1.0,
                value=float(
                    getattr(params, "overlay_image_opacity", 0.85) or 0.85
                ),
                step=0.05,
                key="overlay_image_opacity_slider",
                disabled=overlay_disabled,
            )


def _render_generation_controls(
    params, uploaded_files, uploaded_audio_file, uploaded_bgm_file, voice_mode
):
    """
    校验生成依赖、提交任务，并渲染日志与成片结果。

    返回本次页面执行是否成功提交了新任务。提交前已经请求非阻塞保存，调用方
    据此跳过页面末尾的重复请求。主脚本必须及时结束，定时 Fragment 才能持续
    刷新进度和任务日志。
    """
    restore_upload_requirements = st.session_state.get(
        "task_restore_upload_requirements", {}
    )
    has_local_materials = bool(
        uploaded_files or st.session_state.get("local_video_materials", [])
    )
    has_custom_audio = bool(uploaded_audio_file)
    unmet_restore_requirements = _get_unmet_restore_upload_requirements(
        restore_upload_requirements,
        video_source=params.video_source,
        voice_name=params.voice_name or "",
        has_local_materials=has_local_materials,
        has_custom_audio=has_custom_audio,
        voice_mode=voice_mode,
    )
    if "local_materials" in unmet_restore_requirements:
        st.warning(tr("Task Restore Local Materials Warning"))
    if "custom_audio" in unmet_restore_requirements:
        st.warning(tr("Task Restore Custom Audio Warning"))
    if restore_upload_requirements and not unmet_restore_requirements:
        # 用户已重新上传文件，或主动切换了素材来源/音色。此时历史任务的上传依赖
        # 已经得到明确处理，清除标记，避免后续普通生成继续显示旧提示。
        st.session_state.pop("task_restore_upload_requirements", None)

    # 生成任务在后台线程运行数分钟。任务运行期间或冷却窗口内禁用按钮，
    # 避免用户重复点击提交多个任务，也避免排队 rerun 把同一任务提交两次。
    generation_busy = _generation_in_progress() or not _action_ready("generate_video")
    start_button = st.button(
        tr("Generate Video"),
        use_container_width=True,
        type="primary",
        key="generate_video_button",
        on_click=_prepare_generation_task,
        disabled=generation_busy,
        help=tr("Generate Video Disabled Hint") if generation_busy else None,
    )
    render_onboarding_tour()
    if start_button:
        _save_runtime_config()
        task_id = st.session_state.get("pending_generation_task_id") or str(uuid4())
        _add_active_generation_task(
            task_id,
            subject=params.video_subject or params.video_script or task_id,
        )
        if not params.video_subject and not params.video_script:
            _remove_active_generation_task(task_id)
            st.error(tr("Video Script and Subject Cannot Both Be Empty"))
            st.stop()

        if params.video_source not in [
            "auto",
            "custom_api",
            "pexels",
            "pixabay",
            "coverr",
            "web_scrape",
            "pollinations",
            "local",
        ]:
            _remove_active_generation_task(task_id)
            st.error(tr("Please Select a Valid Video Source"))
            st.stop()

        if params.video_source == "pexels" and not config.app.get(
            "pexels_api_keys", ""
        ):
            _remove_active_generation_task(task_id)
            st.error(tr("Please Enter the Pexels API Key"))
            st.stop()

        if params.video_source == "pixabay" and not config.app.get(
            "pixabay_api_keys", ""
        ):
            _remove_active_generation_task(task_id)
            st.error(tr("Please Enter the Pixabay API Key"))
            st.stop()

        if params.video_source == "coverr" and not config.app.get(
            "coverr_api_keys", ""
        ):
            _remove_active_generation_task(task_id)
            st.error(tr("Please Enter the Coverr API Key"))
            st.stop()

        if params.video_source == "custom_api" and not (
            custom_media_service.is_custom_api_configured()
        ):
            _remove_active_generation_task(task_id)
            st.error(tr("Please Enter the Custom API URL and Key"))
            st.stop()

        if params.video_source == "auto" and not (
            config.app.get("pexels_api_keys")
            or config.app.get("pixabay_api_keys")
            or config.app.get("coverr_api_keys")
            or config.app.get("enable_web_scraping", False)
            or custom_media_service.is_custom_api_configured()
            or custom_media_service.is_pollinations_enabled()
        ):
            # auto 只查询已配置的供应商。一个 Key 都没配时任务必然在素材阶段
            # 失败，这里在启动前拦截并提示需要配置的入口。
            _remove_active_generation_task(task_id)
            st.error(tr("No Video Source Provider Configured"))
            st.stop()

        if (
            params.bgm_type == "sonilo"
            and bgm_service.should_use_bgm(params.bgm_type, params.bgm_volume)
            and not sonilo_service.is_enabled()
        ):
            _remove_active_generation_task(task_id)
            st.error(tr("Sonilo API Key Required"))
            st.stop()

        if (
            params.bgm_type == "elevenlabs"
            and bgm_service.should_use_bgm(params.bgm_type, params.bgm_volume)
            and not elevenlabs_music_service.is_enabled()
        ):
            _remove_active_generation_task(task_id)
            st.error(tr("ElevenLabs API Key Required"))
            st.stop()

        if params.video_source == "local" and not has_local_materials:
            # 本地素材为空时继续执行会先产生 TTS/字幕，最后才在素材预处理阶段失败。
            # 在任务启动前拦截，可以避免无意义的 API 调用和中间文件。
            _remove_active_generation_task(task_id)
            st.error(tr("Please Upload Local Materials First"))
            st.stop()

        if voice_mode == VOICE_MODE_UPLOAD and not uploaded_audio_file:
            # 上传音频是用户显式选择的配音方式，缺少文件时不能静默退回 TTS。
            # 在任务启动前拦截，避免产生与用户选择不一致的成片。
            _remove_active_generation_task(task_id)
            st.error(tr("Please Upload Voiceover File First"))
            st.stop()

        if "custom_audio" in unmet_restore_requirements:
            # 历史自定义音频不能自动回填。用户尚未重新上传且也没有主动更换音色时，
            # 必须阻止静默退回 TTS，否则重新生成的结果会与原任务语音不一致。
            _remove_active_generation_task(task_id)
            st.error(tr("Task Restore Custom Audio Warning"))
            st.stop()

        if uploaded_bgm_file and bgm_service.should_use_bgm(
            params.bgm_type, params.bgm_volume
        ):
            try:
                saved_bgm_name = bgm_service.save_bgm_upload(
                    uploaded_bgm_file.name, uploaded_bgm_file
                )
            except bgm_service.BgmUploadError as exc:
                _remove_active_generation_task(task_id)
                logger.warning(f"WebUI background music upload rejected: {str(exc)}")
                st.error(tr("Invalid Background Music"))
                st.stop()
            except bgm_service.BgmServiceError as exc:
                _remove_active_generation_task(task_id)
                logger.error(f"WebUI background music upload failed: {str(exc)}")
                st.error(tr("Background Music Validation Failed"))
                st.stop()
            # 保存成功后只把文件名写入任务参数。视频服务会在两个 BGM 白名单
            # 目录中重新解析，避免把服务器绝对路径持久化或展示给用户。
            params.bgm_file = saved_bgm_name
        elif uploaded_bgm_file:
            # 0 音量时视频服务不会使用任何 BGM，因此不再把已经预览的上传文件
            # 持久化到 storage。用户之后调高音量时可直接再次点击生成完成保存。
            params.bgm_file = ""

        if uploaded_audio_file:
            task_dir = utils.task_dir(task_id)
            try:
                custom_audio_path = _build_uploaded_file_path(
                    uploaded_audio_file,
                    task_dir,
                    CUSTOM_AUDIO_EXTENSIONS,
                    "custom-audio",
                )
            except ValueError:
                _remove_active_generation_task(task_id)
                st.error(tr("Unsupported Upload File Type"))
                st.stop()
            with open(custom_audio_path, "wb") as f:
                f.write(uploaded_audio_file.getbuffer())
            params.custom_audio_file = custom_audio_path

        if uploaded_files:
            local_videos_dir = utils.storage_dir("local_videos", create=True)
            # 每次重新上传时都以本次选择的素材为准，避免旧素材不断重复追加。
            params.video_materials = []
            persisted_local_materials = []
            for file in uploaded_files:
                try:
                    file_path = _build_uploaded_file_path(
                        file,
                        local_videos_dir,
                        LOCAL_MATERIAL_EXTENSIONS,
                        "material",
                    )
                except ValueError:
                    _remove_active_generation_task(task_id)
                    st.error(tr("Unsupported Upload File Type"))
                    st.stop()
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
                    m = MaterialInfo()
                    m.provider = "local"
                    m.url = file_path
                    params.video_materials.append(m)
                    persisted_local_materials.append(
                        {
                            "provider": m.provider,
                            "url": m.url,
                            "duration": m.duration,
                        }
                    )
            # 将已上传并保存到本地的视频素材写入会话，供后续只改文案时直接复用。
            st.session_state["local_video_materials"] = persisted_local_materials
        elif (
            params.video_source == "local" and st.session_state["local_video_materials"]
        ):
            # 当用户没有重新上传文件时，复用最近一次已经保存到磁盘的本地素材列表。
            params.video_materials = []
            for material in st.session_state["local_video_materials"]:
                m = MaterialInfo()
                m.provider = material.get("provider", "local")
                m.url = material.get("url", "")
                m.duration = material.get("duration", 0)
                if m.url:
                    params.video_materials.append(m)

        reusable_voice_preview = _get_reusable_full_voice_preview(
            params,
            voice_mode,
        )
        if reusable_voice_preview:
            # 试听缓存只存在当前 Streamlit 会话。提交前把音频写入目标任务目录，
            # 后台线程随后只读取任务自己的文件；即使页面 rerun、浏览器关闭或
            # 用户试听其它音色，也不会影响已经入队的生成任务。
            preview_audio_file = os.path.join(
                utils.task_dir(task_id),
                "audio.mp3",
            )
            with open(preview_audio_file, "wb") as file:
                file.write(reusable_voice_preview.pop("audio_bytes"))
            reusable_voice_preview["audio_file"] = preview_audio_file
            logger.info(
                f"reuse full voice preview for task: "
                f"task_id={task_id}, duration={reusable_voice_preview['duration']:.2f}s"
            )

        try:
            st.toast(tr("Generating Video"))
            logger.info(tr("Start Generating Video"))
            logger.info(utils.to_json(params))
            # 提交前进入冷却窗口：排队 rerun 里重复的点击会被吞掉，配合
            # 任务运行期间的 disabled 状态，从根上避免重复提交。
            _action_mark_triggered("generate_video")
            webui_task.submit_generation(
                task_id=task_id,
                params=params,
                capture_logs=not config.ui.get("hide_log", False),
                voice_preview=reusable_voice_preview,
            )
        except Exception:
            _remove_active_generation_task(task_id)
            st.error(tr("Video Generation Failed"))
            st.stop()

        st.session_state["current_generation_task_id"] = task_id
        logger.info(f"WebUI generation task submitted: task_id={task_id}")

    _render_current_generation_task()
    return start_button


def _verify_schema_fields():
    """启动自检：捕获“WebUI 更新后未完全重启进程”导致的模块过期。

    Streamlit 热重载只重新执行 Main.py，不会重新导入已经加载的
    app.models.schema。如果 schema.py 在进程启动之后新增过字段（例如内容
    智能相关字段），内存中的 VideoParams 仍是旧类，页面赋值时会抛出
    "object has no field" 的 ValueError。这里在渲染前提前检测，用明确的
    指引替代原始 traceback，检测到缺失时直接停止渲染。
    """
    expected_fields = (
        "agentic_planning",
        "content_profile",
        "automation_level",
        "platform",
        "content_format",
        "content_goal",
        "niche",
        "sub_niche",
        "audience",
        "tone",
        "research_depth",
        "fact_check_level",
        "trend_preference",
        "reference_channels",
        "sources",
        "script_style",
        "video_duration_seconds",
        "material_media_type",
        "image_motion_effect",
    )
    missing = [
        name for name in expected_fields if name not in VideoParams.model_fields
    ]
    if missing:
        st.error(
            "Outdated app modules detected: the WebUI was updated while it was "
            "running, so the loaded VideoParams model is missing fields "
            f"({', '.join(missing)}). Fully stop the WebUI (close the webui.bat "
            "window or press Ctrl+C) and start it again — a browser refresh is "
            "NOT enough. / 检测到应用模块过期：WebUI 在运行期间被更新，内存中的 "
            "VideoParams 缺少字段。请完全停止 WebUI（关闭 webui.bat 窗口或按 "
            "Ctrl+C）后重新启动，仅刷新浏览器无法解决。"
        )
        st.stop()


def _render_application():
    """按固定顺序渲染顶部栏、弹窗、生成表单和任务结果。"""
    _verify_schema_fields()
    _render_top_bar()

    if st.session_state.get("settings_dialog_open", False):
        _render_settings_dialog()

    restore_applied = _apply_pending_task_restore()
    restore_candidate_id = st.session_state.get("task_restore_candidate_id")
    if restore_candidate_id:
        _render_task_restore_dialog(restore_candidate_id)
    restore_succeeded = st.session_state.pop("task_restore_succeeded", False)
    if restore_applied or restore_succeeded:
        st.success(tr("Task Configuration Loaded"))

    with st.container(key="main_settings_grid"):
        main_cols = st.columns([1.1, 2.2])
    left_panel = main_cols[0]
    with main_cols[1]:
        tabs = st.tabs(
            [tr("Video Settings"), tr("Audio Settings"), tr("Subtitle Settings")]
        )
        middle_panel = tabs[0]
        audio_panel = tabs[1]
        right_panel = tabs[2]

    params = VideoParams(video_subject="")
    params.match_materials_to_script = bool(
        st.session_state.get("match_materials_to_script", False)
    )
    _render_script_settings(left_panel, params)

    uploaded_files = _render_video_settings(middle_panel, params)
    uploaded_audio_file, uploaded_bgm_file, voice_mode = _render_audio_settings(
        audio_panel, params
    )

    _render_subtitle_settings(right_panel, params)

    generation_submitted = _render_generation_controls(
        params,
        uploaded_files,
        uploaded_audio_file,
        uploaded_bgm_file,
        voice_mode,
    )

    # 生成分支在启动后台线程前已经请求过保存。普通控件交互继续请求非阻塞保存；
    # 如果后台任务正在使用配置，配置层会在任务结束时自动应用并落盘最新值。
    if not generation_submitted:
        _save_runtime_config()


_render_application()
