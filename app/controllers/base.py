from uuid import uuid4

from fastapi import Depends, Request

from app.config import config
from app.models.exception import HttpException


def get_task_id(request: Request):
    task_id = request.headers.get("x-task-id")
    if not task_id:
        task_id = uuid4()
    return str(task_id)


def get_api_key(request: Request):
    api_key = request.headers.get("x-api-key")
    return api_key


def verify_token(request: Request):
    token = get_api_key(request)
    if token != config.app.get("api_key", ""):
        request_id = get_task_id(request)
        request_url = request.url
        user_agent = request.headers.get("user-agent")
        raise HttpException(
            task_id=request_id,
            status_code=401,
            message=f"invalid token: {request_url}, {user_agent}",
        )


def auth_dependencies():
    """Router-level auth dependency list, enforced only when an api_key is set.

    未配置 api_key 时保持开放（与历史行为一致，避免默认配置直接锁死 API）；
    一旦用户配置了 api_key，所有 /api/v1 路由都会要求 `x-api-key` 头匹配。
    """
    if config.app.get("api_key"):
        return [Depends(verify_token)]
    return None
