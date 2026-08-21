"""凭据安全存储：WebUI 保存的密钥进入系统凭据管理器，不落盘 config.toml。"""

import unittest
from unittest.mock import patch

from app.config import config


class TestMaybePersistSecret(unittest.TestCase):
    """_maybe_persist_secret：WebUI 凭据写入 keyring 并标记不得落盘。"""

    def setUp(self):
        self._original_sourced = set(config._secret_sourced_keys)
        config._secret_sourced_keys.clear()

    def tearDown(self):
        config._secret_sourced_keys.clear()
        config._secret_sourced_keys.update(self._original_sourced)

    def test_stores_secret_and_marks_sourced(self):
        with (
            patch.object(config, "set_secret", return_value=True) as set_mock,
            patch.object(config, "app", {"openai_api_key": "sk-test"}),
        ):
            config._maybe_persist_secret(config.app, "openai_api_key", "sk-test")
        set_mock.assert_called_once_with("OPENAI_API_KEY", "sk-test")
        self.assertIn(("app", "openai_api_key"), config._secret_sourced_keys)

    def test_list_secret_serialized_comma_separated(self):
        with (
            patch.object(config, "set_secret", return_value=True) as set_mock,
            patch.object(config, "app", {"pexels_api_keys": []}),
        ):
            config._maybe_persist_secret(
                config.app, "pexels_api_keys", ["k1 ", " k2", ""]
            )
        set_mock.assert_called_once_with("PEXELS_API_KEY", "k1,k2")

    def test_empty_value_deletes_from_keyring(self):
        with (
            patch.object(config, "delete_secret", return_value=True) as del_mock,
            patch.object(config, "set_secret") as set_mock,
            patch.object(config, "app", {"openai_api_key": ""}),
        ):
            config._maybe_persist_secret(config.app, "openai_api_key", "")
        del_mock.assert_called_once_with("OPENAI_API_KEY")
        set_mock.assert_not_called()

    def test_keyring_unavailable_keeps_plaintext_fallback(self):
        """keyring 不可用时不标记 sourced —— 宁可明文也不能丢密钥。"""
        with (
            patch.object(config, "set_secret", return_value=False),
            patch.object(config, "app", {"openai_api_key": "sk-test"}),
        ):
            config._maybe_persist_secret(config.app, "openai_api_key", "sk-test")
        self.assertNotIn(("app", "openai_api_key"), config._secret_sourced_keys)

    def test_non_secret_keys_ignored(self):
        with patch.object(config, "set_secret") as set_mock:
            config._maybe_persist_secret(config.app, "llm_provider", "openai")
        set_mock.assert_not_called()
        self.assertNotIn(("app", "llm_provider"), config._secret_sourced_keys)

    def test_unknown_section_ignored(self):
        with patch.object(config, "set_secret") as set_mock:
            config._maybe_persist_secret({"api_key": "x"}, "api_key", "x")
        set_mock.assert_not_called()


class TestApplyPendingUpdatesPersistsSecrets(unittest.TestCase):
    """WebUI 更新路径（pending updates）应触发凭据持久化。"""

    def setUp(self):
        self._original_sourced = set(config._secret_sourced_keys)
        config._secret_sourced_keys.clear()

    def tearDown(self):
        config._secret_sourced_keys.clear()
        config._secret_sourced_keys.update(self._original_sourced)

    def test_update_flow_persists_to_keyring(self):
        section = {"groq_api_key": ""}
        with (
            patch.object(config, "set_secret", return_value=True) as set_mock,
            patch.dict(
                "app.config.config.__dict__", {"app": section}
            ),
        ):
            config.update_config_nonblocking(section, "groq_api_key", "gsk-123")
            config._apply_pending_config_updates_locked()
        set_mock.assert_called_once_with("GROQ_API_KEY", "gsk-123")
        self.assertEqual(section["groq_api_key"], "gsk-123")
        self.assertIn(("app", "groq_api_key"), config._secret_sourced_keys)

    def test_delete_flow_removes_from_keyring(self):
        section = {"groq_api_key": "old"}
        config._secret_sourced_keys.add(("app", "groq_api_key"))
        with (
            patch.object(config, "delete_secret", return_value=True) as del_mock,
            patch.dict(
                "app.config.config.__dict__", {"app": section}
            ),
        ):
            config.delete_config_nonblocking(section, "groq_api_key")
            config._apply_pending_config_updates_locked()
        del_mock.assert_called_once_with("GROQ_API_KEY")
        self.assertNotIn("groq_api_key", section)


class TestPlaintextMigration(unittest.TestCase):
    """启动迁移：config.toml 明文凭据自动进入 keyring。"""

    def setUp(self):
        self._original_sourced = set(config._secret_sourced_keys)
        config._secret_sourced_keys.clear()

    def tearDown(self):
        config._secret_sourced_keys.clear()
        config._secret_sourced_keys.update(self._original_sourced)

    def test_migrates_plaintext_and_marks_sourced(self):
        cfg = {"app": {"openai_api_key": "sk-legacy"}}
        with (
            patch.object(config, "set_secret", return_value=True) as set_mock,
            patch.dict(os_environ := __import__("os").environ, {}, clear=False),
        ):
            migrated = config._migrate_plaintext_secrets_to_keyring(cfg)
        self.assertEqual(migrated, 1)
        set_mock.assert_called_once_with("OPENAI_API_KEY", "sk-legacy")
        self.assertIn(("app", "openai_api_key"), config._secret_sourced_keys)

    def test_skips_when_env_flag_set(self):
        import os

        cfg = {"app": {"openai_api_key": "sk-legacy"}}
        env = dict(os.environ)
        env["REELSYNC_SKIP_SECRET_MIGRATION"] = "1"
        with (
            patch.dict(os.environ, env, clear=True),
            patch.object(config, "set_secret") as set_mock,
        ):
            migrated = config._migrate_plaintext_secrets_to_keyring(cfg)
        self.assertEqual(migrated, 0)
        set_mock.assert_not_called()

    def test_already_sourced_keys_not_migrated_twice(self):
        cfg = {"app": {"openai_api_key": "sk-legacy"}}
        config._secret_sourced_keys.add(("app", "openai_api_key"))
        with patch.object(config, "set_secret") as set_mock:
            migrated = config._migrate_plaintext_secrets_to_keyring(cfg)
        self.assertEqual(migrated, 0)
        set_mock.assert_not_called()

    def test_keyring_unavailable_leaves_plaintext(self):
        cfg = {"app": {"openai_api_key": "sk-legacy"}}
        with patch.object(config, "set_secret", return_value=False):
            migrated = config._migrate_plaintext_secrets_to_keyring(cfg)
        self.assertEqual(migrated, 0)
        self.assertNotIn(("app", "openai_api_key"), config._secret_sourced_keys)


class TestSaveStripsWebuiSecrets(unittest.TestCase):
    """经 WebUI 保存的凭据在落盘时被清空（值已在 keyring 中）。"""

    def setUp(self):
        self._original_sourced = set(config._secret_sourced_keys)

    def tearDown(self):
        config._secret_sourced_keys.clear()
        config._secret_sourced_keys.update(self._original_sourced)

    def test_saved_toml_blanks_persisted_secret(self):
        import tomllib
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.toml"
            with (
                patch.object(config, "root_dir", tmp),
                patch.object(config, "config_file", str(config_path)),
                patch.object(
                    config,
                    "_secret_sourced_keys",
                    {("app", "groq_api_key")},
                ),
                patch.object(
                    config,
                    "app",
                    {"groq_api_key": "gsk-live", "llm_provider": "groq"},
                ),
            ):
                config.save_config()
            saved = tomllib.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["app"]["groq_api_key"], "")
            self.assertEqual(saved["app"]["llm_provider"], "groq")


if __name__ == "__main__":
    unittest.main()
