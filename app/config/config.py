import copy
import errno
import os
import shutil
import socket
import tempfile
import threading
from contextlib import contextmanager

import toml
from loguru import logger

from app import __version__
from app.utils.secrets import get_secret

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
config_file = f"{root_dir}/config.toml"
_CONTAINER_CGROUP_MARKERS = ("docker", "containerd", "kubepods", "libpod", "podman")
_DOCKER_HOST_GATEWAY_NAME = "host.docker.internal"
_config_save_lock = threading.RLock()
_runtime_config_depth = threading.local()
_runtime_config_overlay = threading.local()
_pending_config_lock = threading.RLock()
_pending_config_updates = {}
_pending_config_save_requested = False
_pending_config_flush_scheduled = False
_MISSING = object()
_DELETE = object()

# 记录启动时从环境变量或系统凭据管理器解析出的“密钥类”配置项。
# 这些值不应回写到 config.toml（避免把凭据管理器中的秘密以明文落盘）。
# 格式为 (section, key) 的集合。
_secret_sourced_keys: set[tuple[str, str]] = set()


def _is_secret_key(key: str) -> bool:
    """Return True if *key* holds a credential that should not be written to config.toml."""
    return key.endswith("_key") or key.endswith("_keys") or key in (
        "speech_key", "password", "api_key", "username",
    )


def _strip_secret_sourced_keys(config_to_save: dict):
    """Blank out env/keyring-sourced secrets before serializing to config.toml."""
    for section, key in _secret_sourced_keys:
        section_dict = config_to_save.get(section)
        if not isinstance(section_dict, dict):
            continue
        if key.endswith("_keys"):
            section_dict[key] = []
        else:
            section_dict[key] = ""


class _SynchronizedConfig(dict):
    """保持 dict 使用方式不变，同时让运行期配置写操作服从同一把锁。"""

    def __init__(self, *args, section_name=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._section_name = section_name

    def __getitem__(self, key):
        overlay = getattr(_runtime_config_overlay, "sections", None)
        if (
            overlay is not None
            and self._section_name
            and self._section_name in overlay
        ):
            section = overlay[self._section_name]
            if key in section:
                return section[key]
        return dict.__getitem__(self, key)

    def get(self, key, default=None):
        overlay = getattr(_runtime_config_overlay, "sections", None)
        if (
            overlay is not None
            and self._section_name
            and self._section_name in overlay
        ):
            section = overlay[self._section_name]
            if key in section:
                return section[key]
        return dict.get(self, key, default)

    def __setitem__(self, key, value):
        # Streamlit 每次整页 rerun 都会把当前控件值重新写回配置。视频任务持有
        # runtime_config_lock 时，如果值没有变化，这次写入没有任何副作用，也
        # 不应让刷新后的页面卡在表单中途。真正改变配置的写入仍进入下方锁，
        # 因而不能在正在生成的视频中途切换 Provider、密钥或其它全局设置。
        current = super().get(key, _MISSING)
        if current is not _MISSING and current == value:
            return
        with _config_save_lock:
            super().__setitem__(key, value)

    def __delitem__(self, key):
        with _config_save_lock:
            super().__delitem__(key)

    def clear(self):
        if not self:
            return
        with _config_save_lock:
            super().clear()

    def pop(self, key, default=_MISSING):
        # ``pop(key, default)`` 在 key 不存在时同样不会改变配置。WebUI 使用
        # 这种写法表达“采用默认策略”，刷新时必须允许它直接完成。
        if key not in self:
            if default is _MISSING:
                raise KeyError(key)
            return default
        with _config_save_lock:
            if default is _MISSING:
                return super().pop(key)
            return super().pop(key, default)

    def setdefault(self, key, default=None):
        # 与 __setitem__ 相同，已存在 key 的 setdefault 是只读操作。提前返回
        # 可以让只读取默认配置的页面刷新不受长任务配置锁影响。
        current = super().get(key, _MISSING)
        if current is not _MISSING:
            return current
        with _config_save_lock:
            return super().setdefault(key, default)

    def update(self, *args, **kwargs):
        changes = dict(*args, **kwargs)
        if all(
            (current := dict.get(self, key, _MISSING)) is not _MISSING
            and current == value
            for key, value in changes.items()
        ):
            return
        with _config_save_lock:
            super().update(changes)


def _pending_update_key(config_section, key):
    """为进程内固定配置分区生成待更新键。"""
    return id(config_section), key


def update_config_nonblocking(config_section, key, value):
    """
    非阻塞更新 WebUI 的运行期配置。

    视频生成会持有 ``runtime_config_lock``，确保同一任务不会在执行中途切换
    Provider、密钥或语音配置。Streamlit 控件发生变化时不能等待这把长任务锁，
    否则浏览器会表现为页面冻结。锁空闲时立即更新；锁繁忙时只保留每个配置项
    的最新值，并在当前任务释放锁时统一应用。

    返回 True 表示值已经生效，False 表示已进入待更新队列。
    """
    # 所有更新都先进入同一队列，再尝试获取配置锁。这样多个页面同时修改同一
    # 配置项时，写入队列的先后顺序就是最终顺序，不会出现较早线程在获取锁后
    # 把较新线程已经排队的值误删掉。
    with _pending_config_lock:
        _pending_config_updates[_pending_update_key(config_section, key)] = (
            config_section,
            key,
            copy.deepcopy(value),
        )

    acquired = _config_save_lock.acquire(blocking=False)
    if not acquired:
        # 调用方通常会在本次 Streamlit rerun 末尾请求保存，但不能依赖这一步
        # 一定执行。例如页面中途异常或更新恰好发生在任务退出保存阶段时，仍需
        # 有后台刷新线程保证排队值最终生效。
        _schedule_deferred_config_flush()
        return False

    try:
        _apply_pending_config_updates_locked()
        return config_section.get(key, _MISSING) == value
    finally:
        _config_save_lock.release()


def delete_config_nonblocking(config_section, key):
    """
    非阻塞删除 WebUI 配置项。

    “使用默认值”需要真正移除配置项，而不是写入空字符串。视频任务占用配置
    锁时，删除意图会覆盖同一配置项之前排队的更新，并在任务结束后执行。
    """
    with _pending_config_lock:
        _pending_config_updates[_pending_update_key(config_section, key)] = (
            config_section,
            key,
            _DELETE,
        )

    acquired = _config_save_lock.acquire(blocking=False)
    if not acquired:
        _schedule_deferred_config_flush()
        return False

    try:
        _apply_pending_config_updates_locked()
        return key not in config_section
    finally:
        _config_save_lock.release()


def _apply_pending_config_updates_locked():
    """在持有配置写锁时应用 WebUI 暂存的最新配置值。"""
    with _pending_config_lock:
        updates = list(_pending_config_updates.values())
        _pending_config_updates.clear()
        # 应用配置时继续持有待更新锁。读取“当前值 + 待更新值”快照的线程由此
        # 只能看到应用前或应用后的完整状态，不会读到只更新了一半的配置集合。
        for config_section, key, value in updates:
            if value is _DELETE:
                config_section.pop(key, None)
            else:
                config_section[key] = value
    return bool(updates)


def snapshot_config_with_pending(config_section):
    """
    返回配置分区的有效快照，并合并尚未应用的 WebUI 更新。

    视频任务持锁期间不能改写全局配置，但用户仍可准备下一条内容。LLM 请求
    使用这个快照后，界面中刚选择的 Provider、模型和密钥会参与新请求，同时
    不会改变正在执行的视频任务。
    """
    with _pending_config_lock:
        snapshot = dict(config_section)
        section_id = id(config_section)
        for (pending_section_id, key), (_, _, value) in _pending_config_updates.items():
            if pending_section_id != section_id:
                continue
            if value is _DELETE:
                snapshot.pop(key, None)
            else:
                snapshot[key] = copy.deepcopy(value)
    return snapshot


def snapshot_config_for_task() -> dict:
    """
    在短锁下取全部分区的深拷贝，供单个任务在整条流水线期间使用。

    视频生成不再在整个流水线期间持有 ``runtime_config_lock``，改为在任务开始
    时短暂获取锁、应用并保存待处理的 WebUI 更新，然后一次性深拷贝所有配置分区
    并立即释放。后续流水线通过线程局部 overlay 读取这份快照，从而允许多个任务
    并发执行，同时保证每个任务使用的 Provider、密钥等配置前后一致。
    """
    with _config_save_lock:
        _flush_pending_config_locked(suppress_save_errors=True)
        return {
            "app": copy.deepcopy(dict(app)),
            "azure": copy.deepcopy(dict(azure)),
            "siliconflow": copy.deepcopy(dict(siliconflow)),
            "minimax_tts": copy.deepcopy(dict(minimax_tts)),
            "elevenlabs": copy.deepcopy(dict(elevenlabs)),
            "chatterbox": copy.deepcopy(dict(chatterbox)),
            "ui": copy.deepcopy(dict(ui)),
            "agentic": copy.deepcopy(dict(agentic)),
            "research": copy.deepcopy(dict(research)),
        }


def begin_task_config(snapshot: dict):
    """为当前线程的流水线安装配置快照 overlay。"""
    _runtime_config_overlay.sections = snapshot


def end_task_config():
    """移除当前线程的配置快照 overlay，恢复读取全局配置。"""
    try:
        del _runtime_config_overlay.sections
    except AttributeError:
        pass


def _flush_pending_config_locked(*, suppress_save_errors):
    """在持有配置写锁时应用并保存当前所有待处理配置。"""
    global _pending_config_save_requested

    updates_applied = _apply_pending_config_updates_locked()
    with _pending_config_lock:
        save_requested = _pending_config_save_requested
        _pending_config_save_requested = False

    if not updates_applied and not save_requested:
        return True

    try:
        save_config()
        return True
    except Exception as exc:
        # 内存中的配置已经成功应用，保存失败时只保留待保存标记。视频任务不应
        # 因配置文件暂时不可写而被改判失败；下一次页面交互会再次触发保存。
        with _pending_config_lock:
            _pending_config_save_requested = True
        if not suppress_save_errors:
            raise
        logger.exception(f"failed to save deferred runtime config: {exc}")
        return False


def _run_deferred_config_flush():
    """等待长任务释放配置锁，并可靠清空期间积累的配置更新。"""
    global _pending_config_flush_scheduled

    while True:
        with _config_save_lock:
            flush_succeeded = _flush_pending_config_locked(suppress_save_errors=True)

        with _pending_config_lock:
            has_pending_work = bool(
                _pending_config_updates or _pending_config_save_requested
            )
            if not flush_succeeded or not has_pending_work:
                _pending_config_flush_scheduled = False
                return


def _schedule_deferred_config_flush():
    """保证同一时间最多只有一个后台线程等待刷新配置。"""
    global _pending_config_flush_scheduled

    with _pending_config_lock:
        if _pending_config_flush_scheduled:
            return
        _pending_config_flush_scheduled = True

    threading.Thread(
        target=_run_deferred_config_flush,
        name="mpt-config-flush",
        daemon=True,
    ).start()


def try_save_config():
    """
    非阻塞保存 WebUI 配置，锁繁忙时交由当前长任务结束后保存。

    普通 API、CLI 和维护脚本仍可调用 ``save_config`` 获得原来的阻塞写入语义；
    只有 Streamlit rerun 使用本函数，避免页面为等待视频任务而长时间无响应。
    """
    global _pending_config_save_requested

    with _pending_config_lock:
        _pending_config_save_requested = True

    acquired = _config_save_lock.acquire(blocking=False)
    if not acquired:
        _schedule_deferred_config_flush()
        return False

    try:
        return _flush_pending_config_locked(suppress_save_errors=False)
    finally:
        _config_save_lock.release()


@contextmanager
def runtime_config_lock():
    """
    在一次依赖全局配置的完整操作期间阻止其它 WebUI 会话改写配置。

    当前项目默认绑定本地回环地址，配置仍然是单用户全局配置。这个轻量锁主要
    保护生成、试听等长操作，避免另一个标签页在操作中途切换 Provider 或密钥。

    ``tm.start`` 内部也会获取同一把锁，为 API 任务提供与 WebUI 相同的保护。
    WebUI 工作线程已经持锁时，这里的线程局部深度计数会让嵌套获取直接放行，
    不会重复刷新待应用配置——否则等待中的 WebUI 配置更新会在生成中途被应用，
    恰好破坏锁想保证的配置一致性。
    """
    depth = getattr(_runtime_config_depth, "depth", 0)
    if depth > 0:
        _runtime_config_depth.depth = depth + 1
        try:
            yield
        finally:
            _runtime_config_depth.depth = depth
        return

    with _config_save_lock:
        # 如果上一个短操作释放锁时后台刷新线程尚未获得调度，新任务必须在读取
        # Provider、密钥等全局配置前先应用队列，不能继续使用旧配置执行整条流水线。
        _flush_pending_config_locked(suppress_save_errors=True)
        _runtime_config_depth.depth = 1
        try:
            yield
        finally:
            _runtime_config_depth.depth = 0
            _flush_pending_config_locked(suppress_save_errors=True)


@contextmanager
def try_runtime_config_lock():
    """
    尝试获取运行期配置锁，并立即返回是否成功。

    WebUI 试听属于用户主动触发的短操作，不应在后台视频任务持锁时等待数分钟。
    调用方可以在未获取锁时就近提示用户稍后重试；成功获取后仍能保证试听期间
    Provider、密钥和模型配置不会被其它会话修改。
    """
    acquired = _config_save_lock.acquire(blocking=False)
    try:
        if acquired:
            _flush_pending_config_locked(suppress_save_errors=True)
        yield acquired
    finally:
        if acquired:
            _flush_pending_config_locked(suppress_save_errors=True)
            _config_save_lock.release()


def is_running_in_container(
    dockerenv_path: str = "/.dockerenv",
    containerenv_path: str = "/run/.containerenv",
    cgroup_path: str = "/proc/1/cgroup",
) -> bool:
    """
    判断当前进程是否运行在容器内。

    这个判断主要用于 Ollama 默认地址选择：
    - 普通本机运行时，`localhost` 指向用户机器本身；
    - Docker 容器内，`localhost` 指向容器自己，访问宿主机 Ollama
      通常需要使用 `host.docker.internal`。

    不能只判断 `/proc/1/cgroup` 是否存在，因为普通 Linux 也会有这个文件。
    这里只在检测到明确的容器标记时返回 True，避免误伤非 Docker Linux 用户。
    参数保留为可注入路径，便于单元测试覆盖不同运行环境。
    """
    if os.path.isfile(dockerenv_path) or os.path.isfile(containerenv_path):
        return True

    try:
        with open(cgroup_path, mode="r", encoding="utf-8") as fp:
            cgroup_content = fp.read().lower()
    except OSError:
        return False

    return any(marker in cgroup_content for marker in _CONTAINER_CGROUP_MARKERS)


def _can_resolve_hostname(hostname: str) -> bool:
    try:
        socket.gethostbyname(hostname)
    except OSError:
        return False
    return True


def _decode_linux_route_gateway(hex_gateway: str) -> str:
    # /proc/net/route 里的 Gateway 是 16 进制小端序，例如 010011AC 表示
    # 172.17.0.1。这里单独解析，是为了在原生 Linux Docker 没有
    # host.docker.internal DNS 记录时，还能尝试访问容器默认网关上的宿主机。
    if len(hex_gateway) != 8:
        raise ValueError("invalid gateway length")

    octets = [
        str(int(hex_gateway[index : index + 2], 16)) for index in range(6, -1, -2)
    ]
    return ".".join(octets)


def get_container_default_gateway_ip(route_path: str = "/proc/net/route") -> str:
    """
    读取 Linux 容器里的默认网关 IP。

    Docker Desktop 通常提供 `host.docker.internal`，但原生 Linux Docker
    默认不一定提供这个 DNS 名称。默认网关通常可以作为访问宿主机服务的
    兜底地址；如果用户的 Ollama 只监听 127.0.0.1，则仍需要用户让
    Ollama 监听宿主机网卡或手动配置 `ollama_base_url`。
    """
    try:
        with open(route_path, mode="r", encoding="utf-8") as fp:
            route_lines = fp.readlines()
    except OSError:
        return ""

    for line in route_lines[1:]:
        fields = line.strip().split()
        if len(fields) < 3:
            continue

        destination = fields[1]
        gateway = fields[2]
        if destination != "00000000" or gateway == "00000000":
            continue

        try:
            return _decode_linux_route_gateway(gateway)
        except ValueError:
            logger.warning(f"invalid container gateway route entry: {line.strip()}")
            return ""

    return ""


def get_default_ollama_base_url() -> str:
    """
    返回 Ollama 的默认 OpenAI-compatible base_url。

    用户显式配置 `ollama_base_url` 时不会走这里；这里只处理“未配置时的
    最佳默认值”。容器内默认指向宿主机，普通本机运行默认指向 localhost。
    """
    if not is_running_in_container():
        return "http://localhost:11434/v1"

    if _can_resolve_hostname(_DOCKER_HOST_GATEWAY_NAME):
        return f"http://{_DOCKER_HOST_GATEWAY_NAME}:11434/v1"

    gateway_ip = get_container_default_gateway_ip()
    if gateway_ip:
        logger.info(
            "host.docker.internal is not resolvable, fallback to container "
            f"default gateway for Ollama: {gateway_ip}"
        )
        return f"http://{gateway_ip}:11434/v1"

    logger.warning(
        "failed to resolve host.docker.internal and container default gateway; "
        "fallback to host.docker.internal for Ollama"
    )
    return f"http://{_DOCKER_HOST_GATEWAY_NAME}:11434/v1"


def load_config():
    # fix: IsADirectoryError: [Errno 21] Is a directory: '/ReelSync/config.toml'
    if os.path.isdir(config_file):
        shutil.rmtree(config_file)

    if not os.path.isfile(config_file):
        example_file = f"{root_dir}/config.example.toml"
        if os.path.isfile(example_file):
            shutil.copyfile(example_file, config_file)
            logger.info("copy config.example.toml to config.toml")

    logger.info(f"load config from file: {config_file}")

    try:
        _config_ = toml.load(config_file)
    except Exception as e:
        logger.warning(f"load config failed: {str(e)}, try to load as utf-8-sig")
        with open(config_file, mode="r", encoding="utf-8-sig") as fp:
            _cfg_content = fp.read()
            _config_ = toml.loads(_cfg_content)

    # Apply environment variable overrides
    _apply_env_overrides(_config_)
    return _config_


def _apply_env_overrides(config: dict):
    """Apply environment variable overrides to config. Env vars take precedence over TOML."""
    # App section
    env_mapping = {
        "PEXELS_API_KEY": ("app", "pexels_api_keys"),
        "PIXABAY_API_KEY": ("app", "pixabay_api_keys"),
        "COVERR_API_KEY": ("app", "coverr_api_keys"),
        "CUSTOM_API_URL": ("app", "custom_api_url"),
        "CUSTOM_API_KEY": ("app", "custom_api_key"),
        "CUSTOM_API_METHOD": ("app", "custom_api_method"),
        "CUSTOM_API_RESPONSE_FORMAT": ("app", "custom_api_response_format"),
        "CUSTOM_API_EXTRA_HEADERS": ("app", "custom_api_extra_headers"),
        "CUSTOM_API_EXTRA_BODY": ("app", "custom_api_extra_body"),
        "TWELVELABS_API_KEY": ("app", "twelvelabs_api_keys"),
        "SONILO_API_KEY": ("app", "sonilo_api_key"),
        "SONILO_BASE_URL": ("app", "sonilo_base_url"),
        "LLM_PROVIDER": ("app", "llm_provider"),
        "MOONSHOT_API_KEY": ("app", "moonshot_api_key"),
        "MOONSHOT_BASE_URL": ("app", "moonshot_base_url"),
        "MOONSHOT_MODEL_NAME": ("app", "moonshot_model_name"),
        "OPENAI_API_KEY": ("app", "openai_api_key"),
        "OPENAI_BASE_URL": ("app", "openai_base_url"),
        "OPENAI_MODEL_NAME": ("app", "openai_model_name"),
        "GEMINI_API_KEY": ("app", "gemini_api_key"),
        "GEMINI_BASE_URL": ("app", "gemini_base_url"),
        "GEMINI_MODEL_NAME": ("app", "gemini_model_name"),
        "DEEPSEEK_API_KEY": ("app", "deepseek_api_key"),
        "DEEPSEEK_BASE_URL": ("app", "deepseek_base_url"),
        "DEEPSEEK_MODEL_NAME": ("app", "deepseek_model_name"),
        "QWEN_API_KEY": ("app", "qwen_api_key"),
        "QWEN_MODEL_NAME": ("app", "qwen_model_name"),
        "AZURE_API_KEY": ("app", "azure_api_key"),
        "AZURE_BASE_URL": ("app", "azure_base_url"),
        "AZURE_MODEL_NAME": ("app", "azure_model_name"),
        "VOLCENGINE_API_KEY": ("app", "volcengine_api_key"),
        "VOLCENGINE_BASE_URL": ("app", "volcengine_base_url"),
        "VOLCENGINE_MODEL_NAME": ("app", "volcengine_model_name"),
        "GROK_API_KEY": ("app", "grok_api_key"),
        "GROK_BASE_URL": ("app", "grok_base_url"),
        "GROK_MODEL_NAME": ("app", "grok_model_name"),
        "MINIMAX_API_KEY": ("app", "minimax_api_key"),
        "MINIMAX_BASE_URL": ("app", "minimax_base_url"),
        "MINIMAX_MODEL_NAME": ("app", "minimax_model_name"),
        "MIMO_API_KEY": ("app", "mimo_api_key"),
        "MIMO_BASE_URL": ("app", "mimo_base_url"),
        "MIMO_MODEL_NAME": ("app", "mimo_model_name"),
        "CLOUDFLARE_API_KEY": ("app", "cloudflare_api_key"),
        "CLOUDFLARE_ACCOUNT_ID": ("app", "cloudflare_account_id"),
        "CLOUDFLARE_GATEWAY_ID": ("app", "cloudflare_gateway_id"),
        "CLOUDFLARE_MODEL_NAME": ("app", "cloudflare_model_name"),
        "MODELSCOPE_API_KEY": ("app", "modelscope_api_key"),
        "MODELSCOPE_BASE_URL": ("app", "modelscope_base_url"),
        "MODELSCOPE_MODEL_NAME": ("app", "modelscope_model_name"),
        "AIHUBMIX_API_KEY": ("app", "aihubmix_api_key"),
        "AIHUBMIX_BASE_URL": ("app", "aihubmix_base_url"),
        "AIHUBMIX_MODEL_NAME": ("app", "aihubmix_model_name"),
        "AIMLAPI_API_KEY": ("app", "aimlapi_api_key"),
        "AIMLAPI_BASE_URL": ("app", "aimlapi_base_url"),
        "AIMLAPI_MODEL_NAME": ("app", "aimlapi_model_name"),
        "EVOLINK_API_KEY": ("app", "evolink_api_key"),
        "EVOLINK_BASE_URL": ("app", "evolink_base_url"),
        "EVOLINK_MODEL_NAME": ("app", "evolink_model_name"),
        "OLLAMA_BASE_URL": ("app", "ollama_base_url"),
        "OLLAMA_MODEL_NAME": ("app", "ollama_model_name"),
        "ONEAPI_API_KEY": ("app", "oneapi_api_key"),
        "ONEAPI_BASE_URL": ("app", "oneapi_base_url"),
        "ONEAPI_MODEL_NAME": ("app", "oneapi_model_name"),
        "LITELLM_MODEL_NAME": ("app", "litellm_model_name"),
        "GROQ_API_KEY": ("app", "groq_api_key"),
        "GROQ_BASE_URL": ("app", "groq_base_url"),
        "GROQ_MODEL_NAME": ("app", "groq_model_name"),
        "POLLINATIONS_API_KEY": ("app", "pollinations_api_key"),
        "POLLINATIONS_BASE_URL": ("app", "pollinations_base_url"),
        "POLLINATIONS_MODEL_NAME": ("app", "pollinations_model_name"),
        "ENABLE_REDIS": ("app", "enable_redis"),
        "REDIS_HOST": ("app", "redis_host"),
        "REDIS_PORT": ("app", "redis_port"),
        "REDIS_DB": ("app", "redis_db"),
        "REDIS_PASSWORD": ("app", "redis_password"),
        "MAX_CONCURRENT_TASKS": ("app", "max_concurrent_tasks"),
        "MAX_QUEUED_TASKS": ("app", "max_queued_tasks"),
        "UPLOAD_POST_ENABLED": ("app", "upload_post_enabled"),
        "UPLOAD_POST_API_KEY": ("app", "upload_post_api_key"),
        "UPLOAD_POST_USERNAME": ("app", "upload_post_username"),
        "UPLOAD_POST_PLATFORMS": ("app", "upload_post_platforms"),
        "UPLOAD_POST_AUTO_UPLOAD": ("app", "upload_post_auto_upload"),
        "UPLOAD_POST_YOUTUBE_PRIVACY_STATUS": ("app", "upload_post_youtube_privacy_status"),
        "UPLOAD_POST_MAX_PENDING_TASKS": ("app", "upload_post_max_pending_tasks"),
        "ENABLE_WEB_SCRAPING": ("app", "enable_web_scraping"),
        "CUSTOM_API_VIDEO_URL": ("app", "custom_api_video_url"),
        "CUSTOM_API_IMAGE_URL": ("app", "custom_api_image_url"),
        "CUSTOM_API_PROVIDER_PRESET": ("app", "custom_api_provider_preset"),
        "CUSTOM_API_VIDEO_MODEL": ("app", "custom_api_video_model"),
        "CUSTOM_API_IMAGE_MODEL": ("app", "custom_api_image_model"),
    }
    for env_var, (section, key) in env_mapping.items():
        value = get_secret(env_var)
        if value is not None:
            # Handle list-type configs (comma-separated)
            if key.endswith("_keys") or key == "upload_post_platforms" or key == "auto_providers":
                config.setdefault(section, {})[key] = [v.strip() for v in value.split(",") if v.strip()]
            elif key in ("enable_redis", "upload_post_auto_upload", "enable_web_scraping", "enable_pollinations", "hybrid_video_mode", "twelvelabs_rerank_terms", "match_materials_to_script"):
                config.setdefault(section, {})[key] = value.lower() in ("true", "1", "yes", "on")
            elif key in ("redis_port", "redis_db", "max_concurrent_tasks", "max_queued_tasks", "upload_post_max_pending_tasks", "sonilo_timeout"):
                config.setdefault(section, {})[key] = int(value)
            else:
                config.setdefault(section, {})[key] = value
            if _is_secret_key(key):
                _secret_sourced_keys.add((section, key))

    # Azure section
    azure = config.get("azure", {})
    if get_secret("AZURE_SPEECH_KEY"):
        azure["speech_key"] = get_secret("AZURE_SPEECH_KEY")
        _secret_sourced_keys.add(("azure", "speech_key"))
    if get_secret("AZURE_SPEECH_REGION"):
        azure["speech_region"] = get_secret("AZURE_SPEECH_REGION")
    config["azure"] = azure

    # SiliconFlow section
    siliconflow = config.get("siliconflow", {})
    if get_secret("SILICONFLOW_API_KEY"):
        siliconflow["api_key"] = get_secret("SILICONFLOW_API_KEY")
        _secret_sourced_keys.add(("siliconflow", "api_key"))
    config["siliconflow"] = siliconflow

    # Minimax TTS section
    minimax_tts = config.get("minimax_tts", {})
    if get_secret("MINIMAX_TTS_API_KEY"):
        minimax_tts["api_key"] = get_secret("MINIMAX_TTS_API_KEY")
        _secret_sourced_keys.add(("minimax_tts", "api_key"))
    if get_secret("MINIMAX_TTS_BASE_URL"):
        minimax_tts["base_url"] = get_secret("MINIMAX_TTS_BASE_URL")
    if get_secret("MINIMAX_TTS_MODEL_ID"):
        minimax_tts["model_id"] = get_secret("MINIMAX_TTS_MODEL_ID")
    if get_secret("MINIMAX_TTS_VOICE_ID"):
        minimax_tts["voice_id"] = get_secret("MINIMAX_TTS_VOICE_ID")
    config["minimax_tts"] = minimax_tts

    # ElevenLabs section
    elevenlabs = config.get("elevenlabs", {})
    if get_secret("ELEVENLABS_API_KEY"):
        elevenlabs["api_key"] = get_secret("ELEVENLABS_API_KEY")
        _secret_sourced_keys.add(("elevenlabs", "api_key"))
    if get_secret("ELEVENLABS_MODEL_ID"):
        elevenlabs["model_id"] = get_secret("ELEVENLABS_MODEL_ID")
    config["elevenlabs"] = elevenlabs

    # Chatterbox section
    chatterbox = config.get("chatterbox", {})
    if get_secret("CHATTERBOX_BASE_URL"):
        chatterbox["base_url"] = get_secret("CHATTERBOX_BASE_URL")
    if get_secret("CHATTERBOX_API_KEY"):
        chatterbox["api_key"] = get_secret("CHATTERBOX_API_KEY")
        _secret_sourced_keys.add(("chatterbox", "api_key"))
    if get_secret("CHATTERBOX_MODEL_ID"):
        chatterbox["model_id"] = get_secret("CHATTERBOX_MODEL_ID")
    config["chatterbox"] = chatterbox

    # Research section
    research = config.get("research", {})
    research_env_mapping = {
        "RESEARCH_PROVIDER": "provider",
        "RESEARCH_BASE_URL": "base_url",
        "RESEARCH_API_KEY": "api_key",
        "RESEARCH_TTL_HOURS": "ttl_hours",
        "RESEARCH_ZERO_KEY_ENABLED": "zero_key_enabled",
        "RESEARCH_CACHE_ENABLED": "cache_enabled",
        "RESEARCH_DEDUPLICATION_ENABLED": "deduplication_enabled",
        "RESEARCH_BATCHING_ENABLED": "batching_enabled",
        "RESEARCH_MAX_EXTERNAL_REQUESTS": "max_external_requests",
        "RESEARCH_MAX_REQUESTS_PER_PROVIDER": "max_requests_per_provider",
        "RESEARCH_USER_AGENT": "user_agent",
        "RESEARCH_CONTACT_EMAIL": "contact_email",
        "OPENALEX_API_KEY": "openalex_api_key",
        "NASA_API_KEY": "nasa_api_key",
        "OPENMETEO_TTL_MINUTES": "openmeteo_ttl_minutes",
        "NOMINATIM_ENABLED": "nominatim_enabled",
        "ENABLE_SPARQL": "enable_sparql",
    }
    for env_var, key in research_env_mapping.items():
        value = get_secret(env_var)
        if value is not None:
            if key in ("zero_key_enabled", "cache_enabled", "deduplication_enabled", "batching_enabled", "nominatim_enabled", "enable_sparql"):
                research[key] = value.lower() in ("true", "1", "yes", "on")
            elif key in ("ttl_hours", "max_external_requests", "max_requests_per_provider", "openmeteo_ttl_minutes"):
                research[key] = int(value)
            else:
                research[key] = value
            if _is_secret_key(key):
                _secret_sourced_keys.add(("research", key))
    config["research"] = research


def save_config():
    """
    原子保存运行时配置。

    Streamlit 的不同会话可能在相近时间触发配置保存。直接覆盖 config.toml 时，
    另一个线程可能读取到只写了一部分的 TOML 内容。这里使用进程内可重入锁串行化
    保存，并先写入同目录临时文件，再通过 os.replace 原子替换目标文件。

    Docker Desktop 单文件 bind mount 会把 config.toml 本身作为挂载点，
    Linux 内核不允许通过 rename/replace 替换挂载点，因此会返回 EBUSY。
    该场景下只能在锁内原地覆盖文件；其它异常仍然抛出，避免掩盖权限、磁盘
    或路径错误。

    这仍然保留项目现有的单用户全局配置语义，不额外引入复杂的多用户配置系统；
    主要用于避免多标签页或快速 rerun 时损坏配置文件。
    """
    with _config_save_lock:
        config_to_save = dict(_cfg)
        config_to_save["app"] = dict(app)
        config_to_save["azure"] = dict(azure)
        config_to_save["siliconflow"] = dict(siliconflow)
        config_to_save["minimax_tts"] = dict(minimax_tts)
        config_to_save["elevenlabs"] = dict(elevenlabs)
        config_to_save["chatterbox"] = dict(chatterbox)
        config_to_save["ui"] = dict(ui)
        config_to_save["research"] = dict(research)
        # 不要把来自环境变量 / 系统凭据管理器的秘密回写到 config.toml。
        # 启动时解析出的密钥保留在运行期配置中，但落盘时必须置空。
        _strip_secret_sourced_keys(config_to_save)
        serialized_config = toml.dumps(config_to_save)

        # WebUI 完整 rerun 结束时会调用保存。内容没有变化时直接返回，避免每次
        # 点击普通控件都产生一次磁盘写入和 fsync。
        try:
            with open(config_file, mode="r", encoding="utf-8") as f:
                if f.read() == serialized_config:
                    _cfg.clear()
                    _cfg.update(config_to_save)
                    return
        except (OSError, UnicodeError):
            pass

        temp_path = ""
        try:
            fd, temp_path = tempfile.mkstemp(
                prefix=".config-",
                suffix=".toml.tmp",
                dir=root_dir,
            )
            with os.fdopen(fd, mode="w", encoding="utf-8") as f:
                f.write(serialized_config)
                f.flush()
                os.fsync(f.fileno())
            try:
                os.replace(temp_path, config_file)
            except OSError as exc:
                if exc.errno != errno.EBUSY:
                    raise

                logger.warning(
                    "atomic config replacement is unavailable for the mounted "
                    f"file, fallback to in-place write: {config_file}"
                )
                with open(config_file, mode="w", encoding="utf-8") as f:
                    f.write(serialized_config)
                    f.flush()
                    os.fsync(f.fileno())
            _cfg.clear()
            _cfg.update(config_to_save)
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)


_cfg = load_config()
app = _SynchronizedConfig(_cfg.get("app", {}), section_name="app")
whisper = _cfg.get("whisper", {})
proxy = _cfg.get("proxy", {})
azure = _SynchronizedConfig(_cfg.get("azure", {}), section_name="azure")
siliconflow = _SynchronizedConfig(
    _cfg.get("siliconflow", {}), section_name="siliconflow"
)
minimax_tts = _SynchronizedConfig(
    _cfg.get("minimax_tts", {}), section_name="minimax_tts"
)
elevenlabs = _SynchronizedConfig(
    _cfg.get("elevenlabs", {}), section_name="elevenlabs"
)
chatterbox = _SynchronizedConfig(
    _cfg.get("chatterbox", {}), section_name="chatterbox"
)
ui = _SynchronizedConfig(
    _cfg.get(
        "ui",
        {
            "hide_log": False,
        },
    ),
    section_name="ui",
)
agentic = _SynchronizedConfig(
    _cfg.get(
        "agentic",
        {
            "max_script_revisions": 2,
        },
    ),
    section_name="agentic",
)
research = _SynchronizedConfig(
    _cfg.get(
        "research",
        {
            # Provider: "" (model knowledge + user notes) or "web_search".
            "provider": "",
            "base_url": "",
            "api_key": "",
            "ttl_hours": 24,
            # Zero-key research layer (public, keyless data sources).
            "zero_key_enabled": False,
            "cache_enabled": True,
            "deduplication_enabled": True,
            "batching_enabled": True,
            "max_external_requests": 20,
            "max_requests_per_provider": 5,
            "user_agent": "",
            "contact_email": "",
            "openalex_api_key": "",
            "nasa_api_key": "",
            "openmeteo_ttl_minutes": 60,
            "nominatim_enabled": True,
            "enable_sparql": False,
        },
    ),
    section_name="research",
)

hostname = socket.gethostname()

log_level = _cfg.get("log_level", "DEBUG")
listen_host = _cfg.get("listen_host", "0.0.0.0")
listen_port = _cfg.get("listen_port", 8080)
project_name = _cfg.get("project_name", "ReelSync")
project_description = _cfg.get(
    "project_description",
    "<a href='https://github.com/Papiwrld/reelsync'>https://github.com/Papiwrld/reelsync</a>",
)
project_version = _cfg.get("project_version", __version__)
reload_debug = False

app["redis_host"] = os.getenv(
    "MPT_APP_REDIS_HOST",
    os.getenv("REDIS_HOST", app.get("redis_host", "localhost")),
)

ffmpeg_path = app.get("ffmpeg_path", "")
if ffmpeg_path and os.path.isfile(ffmpeg_path):
    os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg_path

logger.info(f"{project_name} v{project_version}")
