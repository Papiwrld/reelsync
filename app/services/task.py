import math
import os
import re
import socket
import threading
import time
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from functools import partial
from os import path
from uuid import uuid4

from loguru import logger

from app.config import config
from app.models import const
from app.models.schema import VideoConcatMode, VideoParams
from app.services import bgm as bgm_service
from app.services import (
    agentic,
    elevenlabs_music,
    llm,
    material,
    sonilo,
    subtitle,
    task_artifacts,
    twelvelabs,
    video,
    voice,
)
from app.services import upload_post
from app.services import state as sm
from app.utils import file_security, utils

_URL_USERINFO_RE = re.compile(
    r"((?:https?|wss?)://)([^/\s?#@]*:[^/\s?#@]*@)", re.IGNORECASE
)
_SENSITIVE_QUERY_RE = re.compile(
    r"([?&](?:api[_-]?key|access[_-]?token|token|key|secret|password)=)([^&#\s]+)",
    re.IGNORECASE,
)


def _sanitize_pipeline_error(error: object) -> str:
    """
    清理写入任务状态的失败信息。

    各服务通常已自行脱敏，这里是兜底：任何服务异常里若夹带
    ``https://user:pass@host`` 或 ``?api_key=xxx`` 形式的凭据，在写入
    WebUI 可见的任务状态前统一抹掉。
    """
    message = str(error)
    message = _URL_USERINFO_RE.sub(r"\1***:***@", message)
    message = _SENSITIVE_QUERY_RE.sub(r"\1***", message)
    return message


# 发布请求最长可等待数分钟，不能继续占用视频生成任务的并发名额。
# 固定大小的线程池将发布吞吐限制在可控范围内，同时让视频产物生成后
# 立即进入完成状态。
_cross_post_executor = ThreadPoolExecutor(
    max_workers=2,
    thread_name_prefix="mpt-cross-post",
)
_cross_post_max_pending_tasks = max(
    1,
    int(config.app.get("upload_post_max_pending_tasks", 10)),
)
_cross_post_slots = threading.BoundedSemaphore(_cross_post_max_pending_tasks)
_cross_post_registry_lock = threading.RLock()
_cross_post_futures: dict[str, Future] = {}
_cross_post_process_owner = f"{socket.gethostname()}:{os.getpid()}:{uuid4().hex}"
_generation_process_owner = f"{socket.gethostname()}:{os.getpid()}:{uuid4().hex[:8]}"
_ACTIVE_CROSS_POST_STATES = {
    const.CROSS_POST_STATE_PENDING,
    const.CROSS_POST_STATE_PROCESSING,
}
_CROSS_POST_STATE_WRITE_ATTEMPTS = 3
_CROSS_POST_STATE_RETRY_DELAY_SECONDS = 0.1
_INTERRUPTED_CROSS_POST_ERROR = (
    "cross-posting was interrupted before the process completed"
)
_INTERRUPTED_GENERATION_ERROR = (
    "generation was interrupted by a process restart; please re-run this task"
)
# 视频配乐服务只需实现 ``is_enabled`` 和 ``generate_bgm``。供应商差异集中在
# 文件扩展名、领域异常和 WebUI 警告代码；任务编排、0 音量短路及失败降级
# 全部复用同一路径，避免后续新增供应商时维护多份相似流程。
_VIDEO_MUSIC_PROVIDERS = {
    "sonilo": {
        "service": sonilo,
        "error_type": sonilo.SoniloError,
        "suffix": ".m4a",
        "warning_code": "sonilo_bgm_failed",
        "display_name": "Sonilo",
    },
    "elevenlabs": {
        "service": elevenlabs_music,
        "error_type": elevenlabs_music.ElevenLabsMusicError,
        "suffix": ".mp3",
        "warning_code": "elevenlabs_bgm_failed",
        "display_name": "ElevenLabs",
    },
}


def _get_video_music_prompt(params: VideoParams) -> str:
    """
    读取当前视频配乐供应商实际使用的提示词。

    新任务统一使用供应商无关字段；旧 Sonilo CLI 参数和历史任务仍可能只有
    ``sonilo_bgm_prompt``，因此仅在 Sonilo 通用字段为空时读取旧字段。
    """
    prompt = str(params.video_music_prompt or "").strip()
    if params.bgm_type == "sonilo" and not prompt:
        prompt = str(params.sonilo_bgm_prompt or "").strip()
    return prompt


def is_task_busy(task: dict | None) -> bool:
    """判断任务是否仍在生成或发布，供所有删除入口复用。"""
    if not task:
        return False

    state = task.get("state")
    try:
        state = int(state)
    except (TypeError, ValueError):
        pass

    # 视频生成和跨平台发布都可能继续读取任务目录。统一视为忙碌状态，
    # 可以避免 API 与 WebUI 分别维护规则后出现一个允许删除、另一个禁止
    # 删除的不一致行为。
    return (
        state == const.TASK_STATE_PROCESSING
        or task.get("cross_post_state") in _ACTIVE_CROSS_POST_STATES
    )


def _register_cross_post_future(task_id: str, future: Future) -> None:
    """登记当前进程持有的发布 Future，供启动恢复和测试判断真实运行状态。"""
    with _cross_post_registry_lock:
        _cross_post_futures[task_id] = future


def _unregister_cross_post_future(task_id: str, future: Future | None = None) -> None:
    """仅移除匹配的 Future，避免旧回调误删同任务后续注册的新工作。"""
    with _cross_post_registry_lock:
        current = _cross_post_futures.get(task_id)
        if current is None or (future is not None and current is not future):
            return
        _cross_post_futures.pop(task_id, None)


def _is_cross_post_active_in_process(task_id: str) -> bool:
    """判断当前进程是否仍持有未结束的发布任务。"""
    with _cross_post_registry_lock:
        future = _cross_post_futures.get(task_id)
        return future is not None and not future.done()


def _is_windows_process_alive(process_id: int) -> bool:
    """通过只读 Win32 API 判断进程状态，避免用 os.kill 误终止进程。"""
    import ctypes

    process_query_limited_information = 0x1000
    still_active = 259
    error_access_denied = 5
    error_invalid_parameter = 87
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    # ctypes 默认把未声明的返回值当作 32 位 int。Windows 64 位进程句柄可能
    # 因此被截断，必须显式声明 Win32 函数签名后再调用。
    kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.GetExitCodeProcess.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_ulong),
    ]
    kernel32.GetExitCodeProcess.restype = ctypes.c_int
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = ctypes.c_int
    handle = kernel32.OpenProcess(
        process_query_limited_information,
        False,
        process_id,
    )
    if not handle:
        error_code = ctypes.get_last_error()
        if error_code == error_invalid_parameter:
            return False
        if error_code == error_access_denied:
            # 进程存在但当前用户无查询权限时，必须保守地视为存活，避免错误
            # 回收其它账户正在执行的发布任务。
            return True
        logger.warning(
            "failed to open cross-post owner process on Windows, "
            f"process_id: {process_id}, error_code: {error_code}"
        )
        return True

    try:
        exit_code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            error_code = ctypes.get_last_error()
            logger.warning(
                "failed to read cross-post owner process state on Windows, "
                f"process_id: {process_id}, error_code: {error_code}"
            )
            return True
        return exit_code.value == still_active
    finally:
        kernel32.CloseHandle(handle)


def _is_owner_alive(owner: str | None) -> bool:
    """通用进程存活探测，供生成与发布任务复用。

    复用与跨发布相同的探测逻辑：hostname 不一致视为存活（无法探测远端）、
    同进程视为已中断（重启后无活跃 Future/线程）、否则通过 Windows
    只读 API 或 POSIX os.kill(pid,0) 判断。
    """
    if not owner:
        return False

    try:
        hostname, process_id_text, _ = owner.split(":", 2)
        process_id = int(process_id_text)
    except (TypeError, ValueError):
        logger.warning(f"invalid owner metadata: {owner}")
        return False

    # 无法可靠探测其它主机上的进程。共享 Redis 的多主机部署中必须保守地
    # 视为仍在运行，避免当前节点误删另一节点正在执行的任务。
    if hostname != socket.gethostname():
        return True

    # 当前进程内若 PID 一致但无活跃 Future/线程，视为已中断。生成与发布
    # 在启动时均无正在执行的任务可检查，此分支确保重启后能正确回收。
    if process_id == os.getpid():
        return False

    # Windows 的 os.kill(pid, 0) 与 POSIX 语义不同，可能直接终止目标进程。
    # 使用只申请查询权限的 Win32 API，不向目标进程发送任何信号。
    if os.name == "nt":
        return _is_windows_process_alive(process_id)

    try:
        os.kill(process_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        logger.warning(
            f"failed to inspect owner process, owner: {owner}, error: {exc}"
        )
        return True
    return True


def _is_cross_post_owner_alive(owner: str | None) -> bool:
    """判断持久化发布任务的本机进程是否仍存在。"""
    return _is_owner_alive(owner)


def _is_generation_owner_alive(owner: str | None) -> bool:
    """判断视频生成任务的本机进程是否仍存在。复用与跨发布相同的探测逻辑。"""
    return _is_owner_alive(owner)


def _mark_task_failed(task_id: str, stage: str, error: str) -> dict:
    """记录结构化失败信息，并保留任务失败前已经到达的进度。"""
    existing_task = None
    try:
        existing_task = sm.state.get_task(task_id)
    except Exception as exc:
        logger.warning(f"failed to read task state before failure update: {exc}")

    # 具体服务函数通常比编排层拥有更准确的错误原因。后续的空结果检查
    # 不能再用通用文案覆盖它，否则 API 调用方仍然只能看到模糊信息。
    if (
        existing_task
        and existing_task.get("state") == const.TASK_STATE_FAILED
        and existing_task.get("error")
    ):
        return existing_task

    message = str(error or "unknown task error").strip()
    progress = int((existing_task or {}).get("progress", 0) or 0)
    logger.error(f"task failed, task_id: {task_id}, stage: {stage}, error: {message}")
    failure = {
        "task_id": task_id,
        "state": const.TASK_STATE_FAILED,
        "progress": progress,
        "failed_stage": stage,
        "error": message,
    }
    # Preserve generation ownership so caller snapshot matches persisted state
    # and recovery logic can correctly identify the owning process.
    if existing_task:
        for _owner_key in ("generation_owner", "owner"):
            if _owner_key in existing_task:
                failure[_owner_key] = existing_task[_owner_key]
    sm.state.update_task(
        task_id,
        state=failure["state"],
        progress=failure["progress"],
        failed_stage=failure["failed_stage"],
        error=failure["error"],
    )
    return failure


def generate_script(task_id, params):
    logger.info("\n\n## generating video script")
    video_script = params.video_script.strip()
    if not video_script:
        if getattr(params, "agentic_planning", False):
            video_script = _generate_agentic_script(task_id, params)
        else:
            video_script = llm.generate_script(
                video_subject=params.video_subject,
                language=params.video_language,
                paragraph_number=params.paragraph_number,
                video_script_prompt=params.video_script_prompt,
                custom_system_prompt=params.custom_system_prompt,
                script_style=getattr(params, "script_style", "") or "",
                target_duration_seconds=getattr(params, "video_duration_seconds", 0) or 0,
            )
    else:
        logger.debug(f"video script: \n{video_script}")

    if not video_script:
        # 上游（如 QA 门禁）可能已经把任务标记为失败并写入了具体阶段；
        # 此时不能再用泛化的 "script" 失败覆盖，否则会丢失真正的失败原因。
        try:
            existing = sm.state.get_task(task_id)
            already_failed = bool(
                existing and existing.get("state") == const.TASK_STATE_FAILED
            )
        except Exception:
            already_failed = False
        if not already_failed:
            _mark_task_failed(task_id, "script", "failed to generate video script")
        return None

    return video_script


def _generate_agentic_script(task_id, params):
    """Run the agentic planning graph and persist its structured state.

    Returns the approved script (or "" so the caller marks the task failed).
    If the planning graph itself fails, fall back to the linear script
    generation so a dead strategy step never kills the whole pipeline. The
    fallback is recorded in the agentic state artifact so operators can tell
    a degraded run from a real agentic run.
    """
    try:
        state = agentic.plan_video_content_from_params(params, task_id=task_id)
    except agentic.AgenticError as exc:
        logger.error(f"agentic planning failed, falling back to linear script: {exc}")
        task_artifacts.write_agentic_state(
            task_id,
            {
                "user_input": params.video_subject,
                "profile_name": getattr(params, "content_profile", "") or "",
                "fallback_used": "linear_script",
                "fallback_reason": str(exc),
            },
        )
        return llm.generate_script(
            video_subject=params.video_subject,
            language=params.video_language,
            paragraph_number=params.paragraph_number,
            video_script_prompt=params.video_script_prompt,
            custom_system_prompt=params.custom_system_prompt,
            script_style=getattr(params, "script_style", "") or "",
            target_duration_seconds=getattr(params, "video_duration_seconds", 0) or 0,
        )
    task_artifacts.write_agentic_state(task_id, state.model_dump())
    # 最终评审门禁：QA 报告判定为 blocked（存在 CRITICAL 问题）时，不能
    # 让视频继续生成——critical 错误必须能真正阻断发布，而不是只在状态
    # 工件里留一条记录。
    final_review = (state.final_review or {}).get("verdict", "approved")
    if final_review == "blocked":
        summary = (state.qa_report or {}).get("summary", "QA blocked")
        logger.error(
            f"agentic planning blocked by QA, task_id={task_id}: {summary}"
        )
        _mark_task_failed(task_id, "qa", summary)
        return None
    return state.script or ""


def _agentic_scene_search_terms(task_id: str) -> list[str]:
    """Extract concrete per-scene search terms from the agentic scene plan.

    The Visual Director writes one search term per scene (objects, places,
    actions — never abstract concepts), which match stock footage far better
    than global topic keywords. Returns an empty list when no agentic state or
    scene plan is available, so callers fall back to LLM term generation.
    """
    try:
        state = task_artifacts.load_agentic_state(task_id)
        scenes = state.get("scenes") or state.get("scene_plan", {}).get("scenes") or []
        if not scenes:
            return []
        terms: list[str] = []
        for scene in scenes:
            if not isinstance(scene, dict):
                continue
            scene_terms = scene.get("search_terms") or []
            if isinstance(scene_terms, str):
                scene_terms = [scene_terms]
            for term in scene_terms:
                if isinstance(term, str) and term.strip():
                    terms.append(term.strip())
        return terms
    except Exception:  # noqa: BLE001 - planning data must never block generation
        return []


def generate_terms(task_id, params, video_script):
    logger.info("\n\n## generating video terms")
    video_terms = params.video_terms
    if not video_terms:
        # 先尝试使用 Agentic 场景规划生成的逐场景搜索词：视觉导演为每个
        # 场景产出“具体可视对象/地点/动作”的搜索词，比全局主题词更能命中
        # 与文案真正匹配的画面。只有场景词缺失或非空时才会回退到 LLM 词。
        scene_terms = _agentic_scene_search_terms(task_id)
        if scene_terms:
            video_terms = scene_terms
            logger.info(
                f"using {len(video_terms)} scene-plan search terms from agentic state"
            )
        else:
            # 开启素材按文案顺序匹配后，关键词本身也必须按脚本叙事顺序生成；
            # 否则后续即使顺序下载和顺序拼接，也只能复用一组全局主题词，
            # 无法改善“后面内容的画面提前出现”的问题。
            video_terms = llm.generate_terms(
                video_subject=params.video_subject,
                video_script=video_script,
                amount=8 if params.match_materials_to_script else 5,
                match_script_order=params.match_materials_to_script,
            )
    else:
        if isinstance(video_terms, str):
            video_terms = [term.strip() for term in re.split(r"[,，]", video_terms)]
        elif isinstance(video_terms, list):
            video_terms = [term.strip() for term in video_terms]
        else:
            raise ValueError("video_terms must be a string or a list of strings.")

        logger.debug(f"video terms: {utils.to_json(video_terms)}")

    if not video_terms:
        _mark_task_failed(
            task_id,
            "terms",
            "failed to generate video search terms",
        )
        return None

    # 可选的 TwelveLabs Marengo 语义重排：未启用时返回原顺序，无任何副作用。
    # 顺序匹配模式下关键词顺序本身就是脚本叙事顺序，必须保持原样，故跳过。
    if not params.match_materials_to_script:
        video_terms = twelvelabs.rerank_terms_by_subject(
            video_subject=params.video_subject,
            search_terms=video_terms,
        )

    return video_terms


def save_script_data(task_id, video_script, video_terms, params):
    script_data = {
        "script": video_script,
        "search_terms": video_terms,
        "params": params,
    }
    task_artifacts.write_script_data(task_id, script_data)


def resolve_custom_audio_file(task_id: str, custom_audio_file: str | None) -> str:
    requested_file = (custom_audio_file or "").strip()
    if not requested_file:
        return ""

    task_dir = utils.task_dir(task_id)
    try:
        return file_security.resolve_path_within_directory(
            task_dir,
            requested_file,
        )
    except ValueError as exc:
        task_dir_error = exc

    server_audio_file = path.realpath(
        requested_file
        if path.isabs(requested_file)
        else path.join(utils.root_dir(), requested_file)
    )
    if not path.isabs(requested_file):
        project_root = path.realpath(utils.root_dir())
        try:
            if path.commonpath([project_root, server_audio_file]) != project_root:
                raise ValueError(
                    "relative custom audio paths must stay within the project directory"
                )
        except ValueError as exc:
            raise ValueError(
                "custom audio file must be task-local or an existing server-side file"
            ) from exc

    if not path.isfile(server_audio_file):
        raise ValueError(
            "custom audio file does not exist or is not a file"
        ) from task_dir_error

    return server_audio_file


def _resolve_reusable_voice_preview(
    task_id: str,
    params,
    video_script: str,
    voice_preview: dict | None,
) -> tuple[str, float, object] | None:
    """
    校验并解析 WebUI 提交的完整试听缓存。

    该载荷不是公开 API 参数，只能来自当前进程的 WebUI。即便如此，后台任务
    仍重新核对文案和全部配音参数，并限制音频位于当前任务目录；任何不一致都
    回退普通 TTS，不让过期试听污染正式成片。
    """
    if not voice_preview:
        return None

    expected_values = {
        "script": str(video_script or "").strip(),
        "voice_name": params.voice_name,
        "voice_rate": float(params.voice_rate),
        "voice_volume": float(params.voice_volume),
    }
    if not math.isclose(float(params.voice_volume), 1.0) or any(
        voice_preview.get(key) != value for key, value in expected_values.items()
    ):
        logger.info(
            f"skip stale voice preview cache, task_id: {task_id}, "
            "reason: voice parameters changed"
        )
        return None

    preview_file = path.realpath(str(voice_preview.get("audio_file") or ""))
    task_root = path.realpath(utils.task_dir(task_id))
    try:
        preview_is_task_local = path.commonpath([task_root, preview_file]) == task_root
    except ValueError:
        preview_is_task_local = False

    duration = voice_preview.get("duration")
    sub_maker = voice_preview.get("sub_maker")
    if (
        not preview_is_task_local
        or not path.isfile(preview_file)
        or not isinstance(duration, (int, float))
        or not math.isfinite(duration)
        or duration <= 0
        or sub_maker is None
    ):
        logger.warning(
            f"skip invalid voice preview cache, task_id: {task_id}, "
            f"audio_file: {preview_file or '<empty>'}"
        )
        return None

    logger.info(
        f"using full voice preview audio, task_id: {task_id}, duration: {duration:.2f}s"
    )
    return preview_file, math.ceil(duration), sub_maker


def generate_audio(task_id, params, video_script, voice_preview=None):
    """
    Generate audio for the video script.
    If a custom audio file is provided, it will be used directly.
    There will be no subtitle maker object returned in this case.
    Otherwise, TTS will be used to generate the audio.
    Returns:
        - audio_file: path to the generated or provided audio file
        - audio_duration: duration of the audio in seconds
        - sub_maker: subtitle maker object if TTS is used, None otherwise
    """
    logger.info("\n\n## generating audio")
    # /audio 和 /subtitle 请求模型不包含 custom_audio_file，
    # 这里统一做兼容读取，避免直调接口时抛属性错误。
    requested_custom_audio_file = getattr(params, "custom_audio_file", None)
    try:
        custom_audio_file = resolve_custom_audio_file(
            task_id, requested_custom_audio_file
        )
    except ValueError as exc:
        _mark_task_failed(
            task_id,
            "audio",
            f"invalid custom audio file: {exc}",
        )
        return None, None, None

    if not custom_audio_file:
        reusable_preview = _resolve_reusable_voice_preview(
            task_id,
            params,
            video_script,
            voice_preview,
        )
        if reusable_preview:
            return reusable_preview

        logger.info("no custom audio file provided, using TTS to generate audio.")
        audio_file = path.join(utils.task_dir(task_id), "audio.mp3")
        sub_maker = voice.tts(
            text=video_script,
            voice_name=voice.parse_voice_name(params.voice_name),
            voice_rate=params.voice_rate,
            voice_file=audio_file,
        )
        if sub_maker is None:
            _mark_task_failed(
                task_id,
                "audio",
                "failed to synthesize audio; verify the selected voice and TTS connectivity",
            )
            return None, None, None
        audio_duration = math.ceil(voice.get_audio_duration(sub_maker))
        if audio_duration == 0:
            _mark_task_failed(task_id, "audio", "generated audio duration is zero")
            return None, None, None
        return audio_file, audio_duration, sub_maker
    else:
        logger.info(f"using custom audio file: {custom_audio_file}")
        audio_duration = voice.get_audio_duration(custom_audio_file)
        if audio_duration == 0:
            _mark_task_failed(
                task_id,
                "audio",
                "custom audio duration is zero",
            )
            return None, None, None
        return custom_audio_file, audio_duration, None


def generate_subtitle(task_id, params, video_script, sub_maker, audio_file):
    """
    Generate subtitle for the video script.
    If subtitle generation is disabled or no subtitle maker is provided, it will return an empty string.
    Otherwise, it will generate the subtitle using the specified provider.
    Returns:
        - subtitle_path: path to the generated subtitle file
    """
    logger.info("\n\n## generating subtitle")
    if not params.subtitle_enabled:
        return ""

    subtitle_path = path.join(utils.task_dir(task_id), "subtitle.srt")
    subtitle_provider = config.app.get("subtitle_provider", "edge").strip().lower()
    logger.info(f"\n\n## generating subtitle, provider: {subtitle_provider}")

    if not subtitle_provider:
        logger.info("subtitle provider is empty, skip subtitle generation")
        return ""

    if sub_maker is None and subtitle_provider != "whisper":
        # 自定义音频不会经过 TTS，因此没有 Edge/Azure 等 TTS 返回的
        # sub_maker 时间轴。只有 Whisper 可以直接从音频文件转写字幕；
        # 其他字幕提供方继续保持原有行为，避免生成错误的空时间轴。
        logger.warning(
            "subtitle maker is missing, skip subtitle generation for provider: "
            f"{subtitle_provider}"
        )
        return ""

    if subtitle_provider == "edge":
        voice.create_subtitle(
            text=video_script, sub_maker=sub_maker, subtitle_file=subtitle_path
        )
        if not os.path.exists(subtitle_path):
            # Edge 字幕偶尔会因为时间轴与文案无法匹配而没有产出文件。这里不能
            # 自动切换到 Whisper，否则首次失败会在用户不知情的情况下下载数 GB
            # 的模型。只有显式配置 Whisper 时才允许加载模型，Edge 失败则保留
            # 无字幕视频并记录原因，避免意外的网络和磁盘开销。
            logger.warning(
                "edge subtitle generation did not produce a subtitle file; "
                "skip subtitles without falling back to whisper"
            )
            return ""

        # 词时间轴 sidecar：从 TTS 的 SubMaker 提取逐词时间戳，供渲染层的
        # 逐词高亮使用。缺失/失败不影响字幕本身，只影响高亮功能。
        try:
            from app.services.subtitle_engine.timing import (
                extract_word_timings_from_submaker,
                word_timings_to_json,
            )

            word_timings = extract_word_timings_from_submaker(sub_maker)
            if word_timings:
                word_timings_to_json(word_timings, subtitle_path + ".words.json")
        except Exception as exc:
            logger.warning(
                f"failed to write word timings sidecar: "
                f"error={type(exc).__name__}, detail={exc}"
            )

    if subtitle_provider == "whisper":
        subtitle.create(audio_file=audio_file, subtitle_file=subtitle_path)
        logger.info("\n\n## correcting subtitle")
        subtitle.correct(subtitle_file=subtitle_path, video_script=video_script)

    # 无论 provider，最终统一应用 2~4 词短语分句：把 Whisper/Edge 按标点切出
    # 的单字帧合并成自然短语，同时把整句过长的帧按时间比例拆小，避免单字
    # 快速闪烁和整句一帧难以阅读。分句失败不影响成片，仅保留原字幕。
    try:
        subtitle.chunk_subtitle_file(subtitle_path)
    except Exception as exc:
        logger.warning(
            f"failed to chunk subtitle into phrases: "
            f"error={type(exc).__name__}, detail={exc}"
        )

    subtitle_lines = subtitle.file_to_subtitles(subtitle_path)
    if not subtitle_lines:
        logger.warning(f"subtitle file is invalid: {subtitle_path}")
        return ""

    return subtitle_path


def resolve_auto_clip_duration(params, term_count, audio_duration=0.0) -> int:
    """
    解析自动素材时长。

    video_clip_duration == 0 时按目标成片时长和素材数量推导每段素材时长：
      每段素材 ≈ 目标时长 / 素材数，落在 [2, 12] 秒区间。
    目标时长优先用 video_duration_seconds，其次 target_duration_seconds，
    都没有时用音频时长兜底。用户显式指定时长时原样返回。
    """
    clip_duration = params.video_clip_duration
    if clip_duration and clip_duration > 0:
        return clip_duration
    target = getattr(params, "video_duration_seconds", 0) or 0
    if not target or target <= 0:
        target = getattr(params, "target_duration_seconds", 0) or 0
    if not target or target <= 0:
        target = audio_duration or 0
    if not target or target <= 0:
        return 5
    count = max(int(term_count or 1), 1)
    per_clip = target / count
    return int(max(2, min(12, round(per_clip))))


def get_video_materials(task_id, params, video_terms, audio_duration):
    if params.video_source == "local":
        logger.info("\n\n## preprocess local materials")
        materials = video.preprocess_video(
            materials=params.video_materials,
            clip_duration=params.video_clip_duration,
            image_effect=getattr(params, "image_motion_effect", "") or "kenburns",
        )
        if not materials:
            _mark_task_failed(
                task_id,
                "materials",
                "no valid local video materials were found",
            )
            return None
        return [material_info.url for material_info in materials]
    else:
        logger.info(f"\n\n## downloading videos from {params.video_source}")
        # 顺序匹配模式只在用户显式开启时生效。这里强制素材下载按关键词顺序
        # 轮询，避免某个早期关键词下载太多素材，把后续脚本主题挤出最终时间线。
        downloaded_videos = material.download_videos(
            task_id=task_id,
            search_terms=video_terms,
            source=params.video_source,
            video_aspect=params.video_aspect,
            video_concat_mode=(
                VideoConcatMode.sequential
                if params.match_materials_to_script
                else params.video_concat_mode
            ),
            audio_duration=audio_duration * params.video_count,
            max_clip_duration=params.video_clip_duration,
            match_script_order=params.match_materials_to_script,
            allow_images=(
                getattr(params, "material_media_type", "images_videos")
                != "videos_only"
            ),
        )
        if not downloaded_videos:
            _mark_task_failed(
                task_id,
                "materials",
                "no video material could be downloaded for any search term",
            )
            return None

        # 生成式 AI 供应商（Gemini / Pollinations）返回的是静态图片，
        # 合成引擎按视频片段处理素材，因此先把图片素材转成 Ken Burns
        # 推镜短视频，再进入合并阶段。
        downloaded_videos = video.convert_image_materials_to_videos(
            downloaded_videos,
            clip_duration=params.video_clip_duration,
            image_effect=getattr(params, "image_motion_effect", "") or "kenburns",
        )
        return downloaded_videos


def generate_final_videos(
    task_id, params, downloaded_videos, audio_file, subtitle_path, audio_duration
):
    final_video_paths = []
    combined_video_paths = []
    warnings = []
    video_music_provider = _VIDEO_MUSIC_PROVIDERS.get(params.bgm_type)
    video_music_requested = (
        video_music_provider is not None
        and bgm_service.should_use_bgm(params.bgm_type, params.bgm_volume)
    )
    # 多视频生成默认会打散素材以增加差异；但“按文案顺序匹配素材”追求的是
    # 时间线稳定性和可解释性，所以开启后所有输出都使用顺序拼接。
    if params.match_materials_to_script:
        video_concat_mode = VideoConcatMode.sequential
    elif params.video_count == 1:
        video_concat_mode = params.video_concat_mode
    else:
        video_concat_mode = VideoConcatMode.random
    video_transition_mode = params.video_transition_mode

    _progress = 50
    for i in range(params.video_count):
        index = i + 1
        combined_video_path = path.join(
            utils.task_dir(task_id), f"combined-{index}.mp4"
        )
        final_video_path = path.join(utils.task_dir(task_id), f"final-{index}.mp4")
        logger.info(f"\n\n## combining video: {index} => {combined_video_path}")
        try:
            video.combine_videos(
                combined_video_path=combined_video_path,
                video_paths=downloaded_videos,
                audio_file=audio_file,
                video_aspect=params.video_aspect,
                video_concat_mode=video_concat_mode,
                video_transition_mode=video_transition_mode,
                max_clip_duration=params.video_clip_duration,
                threads=params.n_threads,
                clip_speed=params.video_clip_speed,
                mix_overlap_duration=params.mix_overlap_duration or 1.0,
            )

            _progress += 50 / params.video_count / 2
            sm.state.update_task(task_id, progress=_progress)

            # 视频配乐模式先明确禁用默认 BGM 解析，避免旧任务残留的 bgm_file 被
            # 误用。只有音量大于 0 才生成代理并调用付费 API；0 音量统一跳过。
            bgm_file_override = "" if video_music_provider else None
            if video_music_requested:
                service = video_music_provider["service"]
                display_name = video_music_provider["display_name"]
                warning_code = video_music_provider["warning_code"]
                generated_bgm_path = path.join(
                    utils.task_dir(task_id),
                    (f"{params.bgm_type}-bgm-{index}{video_music_provider['suffix']}"),
                )
                try:
                    service.generate_bgm(
                        video_path=combined_video_path,
                        output_path=generated_bgm_path,
                        video_duration=audio_duration,
                        prompt=_get_video_music_prompt(params),
                    )
                    bgm_file_override = generated_bgm_path
                except video_music_provider["error_type"] as exc:
                    # 视频、旁白和字幕都已生成时，第三方配乐临时失败不应浪费整条
                    # 任务。当前视频明确禁用 BGM，并把降级结果返回 WebUI 提醒用户。
                    logger.warning(
                        f"{display_name} BGM generation failed: task_id={task_id}, "
                        f"video_index={index}, error={exc}"
                    )
                    bgm_file_override = ""
                    warnings.append({"code": warning_code, "video_index": index})

            logger.info(f"\n\n## generating video: {index} => {final_video_path}")
            bgm_mix_succeeded = video.generate_video(
                video_path=combined_video_path,
                audio_path=audio_file,
                subtitle_path=subtitle_path,
                output_file=final_video_path,
                params=params,
                bgm_file_override=bgm_file_override,
            )
            if (
                video_music_provider is not None
                and bgm_file_override
                and not bgm_mix_succeeded
            ):
                # 第三方已成功返回并通过 FFmpeg 校验，但 MoviePy 最终混音仍可能
                # 因运行环境失败。视频服务会保留无 BGM 成片；API 生成失败时
                # override 为空，因此不会重复追加警告。
                warnings.append(
                    {
                        "code": video_music_provider["warning_code"],
                        "video_index": index,
                    }
                )
        except Exception:
            # 单个视频合成失败不应留下之前成功片段的误导性产物，也不应让任务
            # 以模糊的 pipeline 阶段失败。清理本片段的半成品并把失败归因到
            # 视频合成阶段。
            for partial_path in (
                combined_video_path,
                final_video_path,
                f"{final_video_path}.tmp.mp4",
            ):
                video.delete_files([partial_path])
            sm.state.update_task(
                task_id,
                progress=min(90, int(_progress + 50 / params.video_count / 2)),
            )
            raise

        _progress += 50 / params.video_count / 2
        sm.state.update_task(task_id, progress=_progress)

        final_video_paths.append(final_video_path)
        combined_video_paths.append(combined_video_path)

    # 所有视频都合成完成后，清理本任务由图片素材生成的 Ken Burns 中间 .mp4，
    # 避免素材目录里堆积孤儿文件；原始图片保留。清理失败不影响任务结果。
    try:
        video.delete_image_material_clips(downloaded_videos)
    except Exception as exc:
        logger.warning(
            f"failed to clean image material clips: task_id={task_id}, error={exc}"
        )

    return final_video_paths, combined_video_paths, warnings


def _patch_cross_post_state(task_id: str, **kwargs) -> bool | None:
    """安全更新发布字段；短暂状态后端故障时有限重试。"""
    for attempt in range(1, _CROSS_POST_STATE_WRITE_ATTEMPTS + 1):
        try:
            return sm.state.patch_task(task_id, **kwargs)
        except Exception as exc:
            # Redis 短暂断连不应让任务永久停留在 pending/processing。发布状态
            # 写入频率很低，这里使用固定次数和短等待即可覆盖瞬时故障，同时
            # 避免后台线程无限阻塞。最后一次失败保留完整堆栈便于定位。
            if attempt >= _CROSS_POST_STATE_WRITE_ATTEMPTS:
                logger.exception(
                    f"failed to update cross-post state after retries, "
                    f"task_id: {task_id}, fields: {', '.join(kwargs)}, "
                    f"attempts: {attempt}, error: {exc}"
                )
                return None

            logger.warning(
                f"retry cross-post state update, task_id: {task_id}, "
                f"fields: {', '.join(kwargs)}, attempt: {attempt}, error: {exc}"
            )
            time.sleep(_CROSS_POST_STATE_RETRY_DELAY_SECONDS)

    return None


def _record_cross_post_failure(
    task_id: str,
    error: Exception,
    results: list[dict] | None = None,
) -> None:
    """尽最大努力保存发布失败；状态后端不可用时由日志保留诊断信息。"""
    updated = _patch_cross_post_state(
        task_id,
        cross_post_state=const.CROSS_POST_STATE_FAILED,
        cross_post_results=results or None,
        cross_post_error=str(error),
        cross_post_owner=None,
    )
    if updated is False:
        logger.warning(f"discard cross-post failure for missing task: {task_id}")


def _ensure_cross_post_terminal_state(task_id: str) -> None:
    """Future 结束后把仍处于活动态的任务收敛为失败。"""
    try:
        task = sm.state.get_task(task_id)
    except Exception as exc:
        # 此处已经是 Future 的最终回调，没有后续同步调用方可以处理异常。
        # 状态后端恢复后，下一次进程启动仍会通过恢复逻辑处理遗留状态。
        logger.exception(
            f"failed to verify final cross-post state, task_id: {task_id}, error: {exc}"
        )
        return

    if not task or task.get("cross_post_state") not in _ACTIVE_CROSS_POST_STATES:
        return

    logger.warning(
        f"cross-post worker ended without terminal state, task_id: {task_id}, "
        f"state: {task.get('cross_post_state')}"
    )
    _record_cross_post_failure(
        task_id,
        RuntimeError("cross-post worker ended without persisting a terminal state"),
        task.get("cross_post_results"),
    )


def recover_interrupted_cross_posts(page_size: int = 100) -> int | None:
    """
    将进程重启后无法恢复的发布任务标记为失败。

    跨平台发布使用当前进程内的线程池，不是持久化任务队列。进程启动时，
    Redis 中残留的 pending/processing 不会自动继续执行；如果继续把它们视为
    运行中，用户将永久无法删除任务。这里分页扫描状态，只处理当前进程没有
    对应 Future 的活动记录，并保留已经生成的视频结果。
    """
    recovered = 0
    page = 1

    while True:
        try:
            tasks, total = sm.state.get_all_tasks(page, page_size)
        except Exception as exc:
            logger.exception(f"failed to recover interrupted cross-post tasks: {exc}")
            return None

        for task in tasks:
            task_id = str(task.get("task_id") or "")
            if (
                not task_id
                or task.get("cross_post_state") not in _ACTIVE_CROSS_POST_STATES
                or _is_cross_post_active_in_process(task_id)
                or _is_cross_post_owner_alive(task.get("cross_post_owner"))
            ):
                continue

            updated = _patch_cross_post_state(
                task_id,
                cross_post_state=const.CROSS_POST_STATE_FAILED,
                cross_post_error=_INTERRUPTED_CROSS_POST_ERROR,
                cross_post_owner=None,
            )
            if updated is True:
                recovered += 1

        if page * page_size >= total or not tasks:
            break
        page += 1

    if recovered:
        logger.warning(f"recovered interrupted cross-post tasks: {recovered}")
    return recovered


def recover_interrupted_generation_tasks(page_size: int = 100) -> int | None:
    """
    将进程重启后残留的“生成中”任务收敛为失败。

    视频生成由当前进程内的工作线程执行（InMemory/Redis 任务队列都不是
    持久化调度器）。进程异常退出（OOM、SIGKILL、断电）后，Redis 中的任务
    会永久停在 PROCESSING 状态，API/WebUI 上无法删除、无法重试。启动时
    把这类任务标记为失败，保留已生成的视频结果，让用户看到明确终态。

    默认 MemoryState 启动即为空，此函数不会产生副作用；只有在 Redis 状态
    持久化时才有实际作用。支持多副本部署：任务记录中若存在 generation_owner
    /owner 字段，则仅当所属进程已不可达时才回收，避免滚动发布误删对端
    正在执行的任务；无 owner 的历史任务沿用旧行为直接回收以兼容旧数据。
    """
    recovered = 0
    page = 1

    while True:
        try:
            tasks, total = sm.state.get_all_tasks(page, page_size)
        except Exception as exc:
            logger.exception(f"failed to recover interrupted generation tasks: {exc}")
            return None

        for task in tasks:
            task_id = str(task.get("task_id") or "")
            if not task_id or task.get("state") != const.TASK_STATE_PROCESSING:
                continue
            # 多副本滚动发布时，需避免误删对端仍在执行的任务。仅当 owner 存在
            # 且探测为存活时跳过回收；无 owner 的历史数据保持旧行为直接回收。
            owner = task.get("generation_owner") or task.get("owner")
            if owner and _is_generation_owner_alive(owner):
                continue
            updated = sm.state.patch_task(
                task_id,
                state=const.TASK_STATE_FAILED,
                failed_stage="restart",
                error=_INTERRUPTED_GENERATION_ERROR,
            )
            if updated is True:
                recovered += 1

        if page * page_size >= total or not tasks:
            break
        page += 1

    if recovered:
        logger.warning(f"recovered interrupted generation tasks: {recovered}")
    return recovered


def _run_cross_post(
    task_id: str,
    video_paths: tuple[str, ...],
    video_subject: str,
    video_script: str,
    video_language: str,
    platforms: tuple[str, ...],
    youtube_privacy_status: str,
) -> None:
    """后台执行跨平台发布，并只补充发布相关的任务字段。"""
    results = []
    try:
        state_updated = _patch_cross_post_state(
            task_id,
            cross_post_state=const.CROSS_POST_STATE_PROCESSING,
            cross_post_error=None,
            cross_post_owner=_cross_post_process_owner,
        )
        if state_updated is not True:
            # False 表示任务已删除，None 表示状态后端暂时不可用。两种情况都
            # 不应继续调用第三方接口，否则用户无法查询或控制这次发布。
            if state_updated is False:
                logger.warning(f"skip cross-post for missing task: {task_id}")
            else:
                _record_cross_post_failure(
                    task_id,
                    RuntimeError("failed to persist cross-post processing state"),
                )
            return

        logger.info(
            f"cross-post started, task_id: {task_id}, platforms: {', '.join(platforms)}"
        )
        youtube_extra = None
        if any(platform.startswith("youtube") for platform in platforms):
            metadata = llm.generate_social_metadata(
                video_subject=video_subject,
                video_script=video_script,
                language=video_language or "",
                platform="youtube_shorts",
            )
            youtube_extra = {
                "youtube_title": metadata.get("title", video_subject),
                "youtube_description": metadata.get("caption", ""),
                "tags": metadata.get("hashtags", []),
                "privacyStatus": youtube_privacy_status,
                "containsSyntheticMedia": True,
            }

        for video_path in video_paths:
            result = upload_post.cross_post_video(
                video_path=video_path,
                title=video_subject or "Check out this video! #shorts #viral",
                platforms=list(platforms),
                youtube_extra=youtube_extra,
            )
            if not isinstance(result, dict):
                result = {
                    "success": False,
                    "error": "Upload-Post returned an invalid response",
                }
            results.append(result)

        failures = [result for result in results if not result.get("success")]
        if failures:
            error_messages = [
                str(
                    result.get("error")
                    or result.get("message")
                    or "unknown upload error"
                )
                for result in failures
            ]
            cross_post_state = const.CROSS_POST_STATE_FAILED
            cross_post_error = "; ".join(error_messages)
            logger.warning(
                f"cross-post completed with failures, task_id: {task_id}, "
                f"failed: {len(failures)}, total: {len(results)}"
            )
        else:
            cross_post_state = const.CROSS_POST_STATE_COMPLETE
            cross_post_error = None
            logger.success(
                f"cross-post completed, task_id: {task_id}, videos: {len(results)}"
            )

        state_updated = _patch_cross_post_state(
            task_id,
            cross_post_state=cross_post_state,
            cross_post_results=results,
            cross_post_error=cross_post_error,
            cross_post_owner=None,
        )
        if state_updated is False:
            logger.warning(f"discard cross-post result for missing task: {task_id}")
        elif state_updated is None:
            # 上传已经结束但结果没有持久化时，不能继续保留 processing。
            # 失败状态写入会再次经过有限重试，至少让调用方得到明确终态。
            _record_cross_post_failure(
                task_id,
                RuntimeError("failed to persist final cross-post result"),
                results,
            )
    except Exception as exc:
        # 发布失败只影响发布状态，不能反向覆盖已经完成的视频任务。
        # 异常原文写入任务状态，API 调用方无需访问服务端日志也能定位问题。
        logger.exception(f"cross-post failed, task_id: {task_id}, error: {exc}")
        _record_cross_post_failure(task_id, exc, results)


def _run_cross_post_with_slot(*args) -> None:
    """执行发布任务，并确保成功、失败或异常时都会归还队列容量。"""
    try:
        _run_cross_post(*args)
    except Exception as exc:
        # _run_cross_post 已处理预期异常；这里是最后一道保护，避免未来新增
        # 逻辑抛出的异常只保存在无人读取的 Future 中。
        task_id = str(args[0]) if args else "unknown"
        logger.exception(f"cross-post worker crashed, task_id: {task_id}, error: {exc}")
        if args:
            _record_cross_post_failure(task_id, exc)
    finally:
        _cross_post_slots.release()


def _finalize_cross_post_future(task_id: str, future: Future) -> None:
    """清理 Future 注册，并确保取消、异常和状态写入失败都能收敛。"""
    _unregister_cross_post_future(task_id, future)

    try:
        error = future.exception()
    except CancelledError:
        logger.warning(f"cross-post future was cancelled, task_id: {task_id}")
        # Future 在开始执行前被取消时，worker 的 finally 不会运行，因此需要
        # 在回调中归还队列容量，并把持久化状态改为失败。
        _cross_post_slots.release()
        _record_cross_post_failure(
            task_id,
            RuntimeError("cross-post job was cancelled before execution"),
        )
        return
    except Exception as exc:
        logger.exception(
            f"failed to inspect cross-post future, task_id: {task_id}, error: {exc}"
        )
        _ensure_cross_post_terminal_state(task_id)
        return

    if error is not None:
        logger.error(
            f"cross-post future failed, task_id: {task_id}, "
            f"error: {type(error).__name__}: {error}"
        )

    _ensure_cross_post_terminal_state(task_id)


def _schedule_cross_post(
    task_id: str,
    video_paths: list[str],
    params: VideoParams,
    video_script: str,
    platforms: list[str],
    youtube_privacy_status: str,
) -> str | None:
    """提交后台发布任务；成功返回 None，调度失败返回可查询的错误原因。"""
    if not _cross_post_slots.acquire(blocking=False):
        error = "cross-post queue is full; publishing was skipped"
        logger.warning(
            f"skip cross-post because queue is full, task_id: {task_id}, "
            f"capacity: {_cross_post_max_pending_tasks}"
        )
        _patch_cross_post_state(
            task_id,
            cross_post_state=const.CROSS_POST_STATE_FAILED,
            cross_post_error=error,
            cross_post_owner=None,
        )
        return error

    try:
        future = _cross_post_executor.submit(
            _run_cross_post_with_slot,
            task_id,
            tuple(video_paths),
            params.video_subject or "",
            video_script,
            params.video_language or "",
            tuple(platforms),
            youtube_privacy_status,
        )
        _register_cross_post_future(task_id, future)
        future.add_done_callback(partial(_finalize_cross_post_future, task_id))
    except RuntimeError as exc:
        _unregister_cross_post_future(task_id)
        _cross_post_slots.release()
        logger.exception(
            f"failed to schedule cross-post, task_id: {task_id}, error: {exc}"
        )
        _patch_cross_post_state(
            task_id,
            cross_post_state=const.CROSS_POST_STATE_FAILED,
            cross_post_error=f"failed to schedule cross-post: {exc}",
            cross_post_owner=None,
        )
        return f"failed to schedule cross-post: {exc}"

    return None


def _run_pipeline(
    task_id,
    params: VideoParams,
    stop_at: str = "video",
    voice_preview: dict | None = None,
):
    logger.info(f"start task: {task_id}, stop_at: {stop_at}")
    sm.state.update_task(
        task_id,
        state=const.TASK_STATE_PROCESSING,
        progress=5,
        generation_owner=_generation_process_owner,
        owner=_generation_process_owner,
    )

    # 只有完整成片流程需要视频配乐供应商。尽早阻止缺少 Key 的完整任务，避免
    # 先消耗 LLM、TTS 和素材服务额度；中间产物接口仍可独立使用。
    video_music_provider = _VIDEO_MUSIC_PROVIDERS.get(params.bgm_type)
    video_music_enabled = (
        stop_at == "video"
        and video_music_provider is not None
        and bgm_service.should_use_bgm(params.bgm_type, params.bgm_volume)
    )
    if video_music_enabled:
        service = video_music_provider["service"]
        display_name = video_music_provider["display_name"]
        if not service.is_enabled():
            return _mark_task_failed(
                task_id,
                "preflight",
                f"{display_name} background music requires an API key",
            )

        # WebUI 会限制输入长度，但 API、CLI 和历史任务可以绕过前端控件。
        # 在生成脚本、配音和素材之前按供应商上限再次校验，避免完整视频合成后
        # 才由第三方请求拒绝。服务层仍保留同一校验，作为直接调用时的最后防线。
        music_prompt = _get_video_music_prompt(params)
        max_prompt_length = int(getattr(service, "MAX_PROMPT_LENGTH", 0) or 0)
        if max_prompt_length and len(music_prompt) > max_prompt_length:
            return _mark_task_failed(
                task_id,
                "preflight",
                (f"{display_name} music prompt exceeds {max_prompt_length} characters"),
            )

        # 供应商可以选择提供不计费的账号前置检查。检查函数只应抛出确定性
        # 错误；网络波动或权限范围无法确认时由服务层记录警告并继续实际生成。
        validate_access = getattr(service, "validate_generation_access", None)
        if callable(validate_access):
            try:
                validate_access()
            except video_music_provider["error_type"] as exc:
                return _mark_task_failed(task_id, "preflight", str(exc))

    # 1. Generate script
    video_script = generate_script(task_id, params)
    if not video_script or "Error: " in video_script:
        error = (
            video_script.removeprefix("Error: ").strip()
            if isinstance(video_script, str) and "Error: " in video_script
            else "failed to generate video script"
        )
        return _mark_task_failed(task_id, "script", error)

    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=10)

    if stop_at == "script":
        sm.state.update_task(
            task_id, state=const.TASK_STATE_COMPLETE, progress=100, script=video_script
        )
        return {"script": video_script}

    # 2. Generate terms
    video_terms = ""
    if params.video_source != "local":
        video_terms = generate_terms(task_id, params, video_script)
        if not video_terms:
            return _mark_task_failed(
                task_id,
                "terms",
                "failed to generate video search terms",
            )

    save_script_data(task_id, video_script, video_terms, params)

    # 素材时长 0 = 自动：根据目标时长与素材数量推导，并回写到 params，
    # 让下载、图片转视频和合并阶段共用同一份解析结果。
    if getattr(params, "video_clip_duration", 0) == 0:
        params.video_clip_duration = resolve_auto_clip_duration(
            params,
            term_count=len(video_terms) if isinstance(video_terms, list) else 0,
        )
        logger.info(
            f"auto clip duration resolved to {params.video_clip_duration}s "
            f"(terms={len(video_terms) if isinstance(video_terms, list) else 0})"
        )

    if stop_at == "terms":
        sm.state.update_task(
            task_id, state=const.TASK_STATE_COMPLETE, progress=100, terms=video_terms
        )
        return {"script": video_script, "terms": video_terms}

    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=20)

    # 3. Generate audio
    audio_file, audio_duration, sub_maker = generate_audio(
        task_id,
        params,
        video_script,
        voice_preview=voice_preview,
    )
    if not audio_file:
        return _mark_task_failed(
            task_id,
            "audio",
            "failed to prepare narration audio",
        )

    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=30)

    if stop_at == "audio":
        sm.state.update_task(
            task_id,
            state=const.TASK_STATE_COMPLETE,
            progress=100,
            audio_file=audio_file,
        )
        return {"audio_file": audio_file, "audio_duration": audio_duration}

    # 4. Generate subtitle
    subtitle_path = generate_subtitle(
        task_id, params, video_script, sub_maker, audio_file
    )

    if stop_at == "subtitle":
        sm.state.update_task(
            task_id,
            state=const.TASK_STATE_COMPLETE,
            progress=100,
            subtitle_path=subtitle_path,
        )
        return {"subtitle_path": subtitle_path}

    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=40)

    # 5. Get video materials
    downloaded_videos = get_video_materials(
        task_id, params, video_terms, audio_duration
    )
    if not downloaded_videos:
        return _mark_task_failed(
            task_id,
            "materials",
            "failed to prepare video materials",
        )

    if stop_at == "materials":
        sm.state.update_task(
            task_id,
            state=const.TASK_STATE_COMPLETE,
            progress=100,
            materials=downloaded_videos,
        )
        return {"materials": downloaded_videos}

    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=50)

    # 仅完整视频生成流程才需要处理视频拼接模式；
    # 这样可以避免 /subtitle 和 /audio 这类请求访问不存在的字段。
    if type(params.video_concat_mode) is str:
        params.video_concat_mode = VideoConcatMode(params.video_concat_mode)

    # 6. Generate final videos
    final_video_paths, combined_video_paths, generation_warnings = (
        generate_final_videos(
            task_id,
            params,
            downloaded_videos,
            audio_file,
            subtitle_path,
            audio_duration,
        )
    )

    if not final_video_paths:
        return _mark_task_failed(
            task_id,
            "video",
            "failed to generate final video",
        )

    logger.success(
        f"task {task_id} finished, generated {len(final_video_paths)} videos."
    )

    # 6.5 复制成片到用户自定义目录（若配置）。原始任务目录始终保留，
    # WebUI/API 继续从 tasks/<id>/ 提供预览与下载；自定义目录仅为
    # 额外副本，失败不影响任务状态。
    custom_output_dir = ""
    try:
        # API 每请求优先，WebUI/默认走全局配置
        param_output = getattr(params, "output_dir", "") or ""
        cfg_output = config.app.get("output_dir", "") or ""
        custom_output_dir = (param_output.strip() if param_output.strip() else cfg_output.strip())
    except Exception:
        custom_output_dir = ""
    output_copies: list[str] | None = None
    if custom_output_dir:
        try:
            copied = utils.copy_final_to_custom_output(final_video_paths, custom_output_dir)
            if copied:
                logger.info(f"copied {len(copied)} final videos to custom output: {custom_output_dir}")
                output_copies = copied
        except Exception as exc:
            logger.warning(f"failed to copy to custom output_dir {custom_output_dir}: {exc}")

    # 7. 先完成视频生成任务，再按需提交跨平台发布。第三方上传可能耗时
    # 数分钟，不应阻塞视频结果返回，也不能反向影响已经生成的成片。
    cross_post_enabled = (
        upload_post.upload_post_service.is_configured()
        and upload_post.upload_post_service.auto_upload
    )
    platforms = (
        list(upload_post.upload_post_service.platforms) if cross_post_enabled else []
    )
    should_cross_post = cross_post_enabled and bool(platforms)
    if cross_post_enabled and not platforms:
        logger.warning(
            f"skip cross-post because no platforms are configured, task_id: {task_id}"
        )
    cross_post_state = const.CROSS_POST_STATE_PENDING if should_cross_post else None

    kwargs = {
        "videos": final_video_paths,
        "combined_videos": combined_video_paths,
        "script": video_script,
        "terms": video_terms,
        "audio_file": audio_file,
        "audio_duration": audio_duration,
        "subtitle_path": subtitle_path,
        "materials": downloaded_videos,
        "cross_post_state": cross_post_state,
        "cross_post_results": None,
        "cross_post_error": None,
        "cross_post_owner": _cross_post_process_owner if should_cross_post else None,
        "warnings": generation_warnings or None,
        "output_dir": custom_output_dir or None,
        "output_copies": output_copies,
    }
    sm.state.update_task(
        task_id, state=const.TASK_STATE_COMPLETE, progress=100, **kwargs
    )

    if should_cross_post:
        scheduling_error = _schedule_cross_post(
            task_id=task_id,
            video_paths=final_video_paths,
            params=params,
            video_script=video_script,
            platforms=platforms,
            youtube_privacy_status=(
                upload_post.upload_post_service.youtube_privacy_status
            ),
        )
        # 队列满或线程池关闭属于同步可知的调度失败。任务状态已经由调度函数
        # 更新，这里同步修正返回快照，避免调用方收到与后续查询不一致的 pending。
        if scheduling_error:
            kwargs["cross_post_state"] = const.CROSS_POST_STATE_FAILED
            kwargs["cross_post_error"] = scheduling_error
            kwargs["cross_post_owner"] = None

    return kwargs


def start(
    task_id,
    params: VideoParams,
    stop_at: str = "video",
    voice_preview: dict | None = None,
):
    """执行任务流水线，并确保未预期异常也会转换成可查询的失败状态。"""
    try:
        # 任务不再在整个流水线期间持有 runtime_config_lock，否则并发任务会
        # 在这把进程级锁上串行等待。改为在任务开始时短暂持锁，深拷贝一份配置
        # 快照后立即释放，再通过线程局部 overlay 让整条流水线读取这份一致的
        # 配置。WebUI 仍可在任务期间非阻塞地更新全局配置，且不影响当前任务。
        snapshot = config.snapshot_config_for_task()
        config.begin_task_config(snapshot)
        try:
            return _run_pipeline(
                task_id,
                params,
                stop_at=stop_at,
                voice_preview=voice_preview,
            )
        finally:
            config.end_task_config()
    except Exception as exc:
        logger.exception(
            f"unexpected task pipeline failure, task_id: {task_id}, error: {exc}"
        )
        return _mark_task_failed(
            task_id,
            "pipeline",
            f"{type(exc).__name__}: {_sanitize_pipeline_error(exc)}",
        )


if __name__ == "__main__":
    task_id = "task_id"
    params = VideoParams(
        video_subject="金钱的作用",
        voice_name="zh-CN-XiaoyiNeural-Female",
        voice_rate=1.0,
    )
    start(task_id, params, stop_at="video")
