"""Application implementation - ASGI."""

import hmac
import os
import threading
import time
from collections import deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from app.config import config
from app.models.exception import HttpException
from app.router import root_api_router
from app.utils import utils

# Simple in-memory per-IP token bucket for 9.0+ rate limiting
_rate_limit_store: dict[str, deque] = {}
_rate_limit_lock = threading.Lock()

# Redis 后端（enable_redis 时启用）：多 worker / 多实例共享限流计数。
# 连接失败按 30s 退避回退到进程内计数，不阻塞请求路径。
_rate_limit_redis_client = None
_rate_limit_redis_checked_at = 0.0
_RATE_LIMIT_REDIS_RETRY_SECONDS = 30.0


def _get_rate_limit_redis():
    global _rate_limit_redis_client, _rate_limit_redis_checked_at
    if not config.app.get("enable_redis"):
        return None
    if _rate_limit_redis_client is not None:
        return _rate_limit_redis_client
    now = time.time()
    if now - _rate_limit_redis_checked_at < _RATE_LIMIT_REDIS_RETRY_SECONDS:
        return None
    _rate_limit_redis_checked_at = now
    try:
        import redis as _redis_mod

        client = _redis_mod.Redis(
            host=config.app.get("redis_host", "localhost"),
            port=int(config.app.get("redis_port", 6379)),
            db=int(config.app.get("redis_db", 0)),
            password=config.app.get("redis_password") or None,
            socket_timeout=0.2,
            socket_connect_timeout=0.2,
        )
        client.ping()
        _rate_limit_redis_client = client
        logger.info("rate limiter using Redis backend (distributed counters)")
        return client
    except Exception as exc:  # noqa: BLE001 - 限流后端不可用不能影响请求
        logger.debug(f"redis rate limiter unavailable, using in-memory: {exc}")
        return None


def _check_rate_limit_redis(client, key: str, limit: int) -> tuple[bool, int]:
    """固定窗口计数（INCR+EXPIRE 原子），返回 (是否限流, 重试等待秒数)。"""
    redis_key = f"reelsync:ratelimit:{key}"
    count = client.incr(redis_key)
    if count == 1:
        try:
            client.expire(redis_key, 60)
        except Exception:  # noqa: BLE001
            pass
    if count > limit:
        retry_after = 60
        try:
            ttl = client.ttl(redis_key)
            if isinstance(ttl, int) and ttl > 0:
                retry_after = ttl
        except Exception:  # noqa: BLE001
            pass
        return True, retry_after
    return False, 0


def _check_rate_limit(ip: str, path: str) -> tuple[bool, int]:
    now = time.time()
    window = 60.0
    if path.startswith("/api/v1/videos"):
        limit = 10
        key = f"{ip}:videos"
    elif path.startswith("/api/v1/"):
        limit = 30
        key = f"{ip}:api"
    else:
        return False, 0

    client = _get_rate_limit_redis()
    if client is not None:
        try:
            return _check_rate_limit_redis(client, key, limit)
        except Exception as exc:  # noqa: BLE001
            global _rate_limit_redis_client, _rate_limit_redis_checked_at
            _rate_limit_redis_client = None
            _rate_limit_redis_checked_at = time.time()
            logger.debug(f"redis rate limit check failed, fallback to memory: {exc}")

    with _rate_limit_lock:
        dq = _rate_limit_store.get(key)
        if dq is None:
            dq = deque()
            _rate_limit_store[key] = dq
        while dq and now - dq[0] >= window:
            dq.popleft()
        if len(dq) >= limit:
            retry_after = int(window - (now - dq[0])) + 1
            if retry_after < 1:
                retry_after = 1
            return True, retry_after
        dq.append(now)
        return False, 0


def _get_client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for") or request.headers.get("X-Forwarded-For") or ""
    if xff:
        return xff.split(",")[0].strip() or (request.client.host if request.client else "unknown")
    return request.client.host if request.client else "unknown"


@asynccontextmanager
async def application_lifespan(_: FastAPI):
    """集中处理 API 进程启动恢复和关闭日志。"""
    logger.info("startup event")

    # 视频生成由进程内工作线程执行，服务重启后无法恢复；发布同样如此。
    # 启动时把 Redis 中确认已失去执行进程的活动状态收敛为失败，避免任务
    # 永久卡在“生成中/发布中”而无法删除或重试。
    from app.services import task as task_service

    task_service.recover_interrupted_generation_tasks()
    task_service.recover_interrupted_cross_posts()
    try:
        yield
    finally:
        logger.info("shutdown event")


def exception_handler(request: Request, e: HttpException):
    return JSONResponse(
        status_code=e.status_code,
        content=utils.get_response(e.status_code, e.data, e.message),
    )


def validation_exception_handler(request: Request, e: RequestValidationError):
    return JSONResponse(
        status_code=400,
        content=utils.get_response(
            status=400, data=e.errors(), message="field required"
        ),
    )


def get_application() -> FastAPI:
    """Initialize FastAPI application.

    Returns:
       FastAPI: Application object instance.

    """
    instance = FastAPI(
        title=config.project_name,
        description=config.project_description,
        version=config.project_version,
        debug=False,
        lifespan=application_lifespan,
    )
    instance.include_router(root_api_router)
    instance.add_exception_handler(HttpException, exception_handler)
    instance.add_exception_handler(RequestValidationError, validation_exception_handler)
    return instance


app = get_application()

# Configures the CORS middleware for the FastAPI app
cors_allowed_origins_str = os.getenv("CORS_ALLOWED_ORIGINS", "")
origins = cors_allowed_origins_str.split(",") if cors_allowed_origins_str else ["*"]
allow_credentials = "*" not in origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def verify_tasks_auth(request: Request, call_next):
    path = request.url.path
    if (path == "/tasks" or path.startswith("/tasks/")) and request.method != "OPTIONS":
        expected = config.app.get("api_key")
        if expected:
            token = request.headers.get("x-api-key") or ""
            if not hmac.compare_digest(token, expected):
                return JSONResponse(status_code=401, content=utils.get_response(401, None, "invalid token"))
    return await call_next(request)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    path = request.url.path
    if path.startswith("/api/v1/"):
        ip = _get_client_ip(request)
        limited, retry_after = _check_rate_limit(ip, path)
        if limited:
            return JSONResponse(
                status_code=429,
                content=utils.get_response(429, None, "rate limit exceeded"),
                headers={"Retry-After": str(retry_after)},
            )
    return await call_next(request)


task_dir = utils.task_dir()
app.mount(
    "/tasks", StaticFiles(directory=task_dir, html=False, follow_symlink=False), name=""
)

public_dir = utils.public_dir()
app.mount("/", StaticFiles(directory=public_dir, html=False, follow_symlink=False), name="")
