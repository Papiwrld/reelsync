"""全局测试隔离：禁止任何测试写入真实的系统凭据管理器。

背景：凭据现在持久化在 OS keyring（Windows Credential Manager 等）。
测试若触发 _maybe_persist_secret / 迁移逻辑，会把 Mock 值写进真实凭据库，
甚至覆盖用户真实密钥（已发生过一次）。此 fixture 对整个测试套件生效：
- set_secret / delete_secret 一律替换为无副作用的成功桩；
- 启动迁移默认关闭（REELSYNC_SKIP_SECRET_MIGRATION=1）。
需要验证 keyring 行为的测试可在自己的用例内重新 patch 同一目标。
"""

import pytest


@pytest.fixture(autouse=True)
def _isolate_system_keyring(monkeypatch):
    monkeypatch.setattr(
        "app.config.config.set_secret", lambda name, value: True
    )
    monkeypatch.setattr(
        "app.config.config.delete_secret", lambda name: True
    )
    monkeypatch.setenv("REELSYNC_SKIP_SECRET_MIGRATION", "1")
    yield
