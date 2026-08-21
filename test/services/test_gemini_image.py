"""Nano Banana（Gemini 图片）高质量免费素材源 + quota 回退 pollinations。"""

import unittest
from unittest.mock import patch

from app.config import config
from app.services import custom_media


class TestClassifyGeminiFailure(unittest.TestCase):
    """HTTP 失败状态码到可操作失败原因的分类。"""

    def test_429_is_quota(self):
        self.assertEqual(custom_media._classify_gemini_failure(429, "rate limit"), "quota")

    def test_401_is_invalid_key(self):
        self.assertEqual(custom_media._classify_gemini_failure(401, "API key not valid"), "invalid_key")

    def test_500_is_unavailable(self):
        self.assertEqual(custom_media._classify_gemini_failure(503, "model overloaded"), "unavailable")

    def test_other_is_empty(self):
        self.assertEqual(custom_media._classify_gemini_failure(404, "not found"), "empty")


class TestIsGeminiImageConfigured(unittest.TestCase):
    """是否启用 Nano Banana 图片源。"""

    def setUp(self):
        self._orig = dict(config.app)
        config.app.clear()

    def tearDown(self):
        config.app.clear()
        config.app.update(self._orig)

    def test_gemini_api_key_enables(self):
        config.app["gemini_api_key"] = "AIza-xyz"
        self.assertTrue(custom_media.is_gemini_image_configured())

    def test_gemini_format_custom_api_enables(self):
        config.app.update(
            {
                "custom_api_key": "k",
                "custom_api_image_url": "https://example.com/i",
                "custom_api_response_format": "gemini",
            }
        )
        self.assertTrue(custom_media.is_gemini_image_configured())

    def test_nothing_configured_disables(self):
        self.assertFalse(custom_media.is_gemini_image_configured())

    def test_custom_api_standard_format_disables(self):
        config.app.update(
            {"custom_api_key": "k", "custom_api_url": "https://example.com", "custom_api_response_format": "standard"}
        )
        self.assertFalse(custom_media.is_gemini_image_configured())


class TestSearchMediaGeminiImage(unittest.TestCase):
    """search_media_gemini_image：成功返回、quota 回退 pollinations。"""

    def setUp(self):
        self._orig = dict(config.app)
        config.app.clear()
        config.app["gemini_api_key"] = "AIza-xyz"

    def tearDown(self):
        config.app.clear()
        config.app.update(self._orig)
        custom_media._gemini_clear_failure()

    def test_success_marks_provider_gemini_image(self):
        item = custom_media.MaterialInfo()
        item.provider = "custom_api"
        item.url = "/tmp/img.jpg"
        item.source_info = {"provider": "custom_api"}
        with patch.object(custom_media, "_search_gemini", return_value=[item]):
            items = custom_media.search_media_gemini_image("cat", 5)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].provider, "gemini_image")
        self.assertEqual(items[0].source_info["provider"], "gemini_image")

    def test_quota_failure_falls_back_to_pollinations(self):
        with (
            patch.object(custom_media, "_search_gemini", return_value=[]),
            patch.object(
                custom_media, "gemini_last_failure_reason", return_value="quota"
            ) as reason_mock,
            patch.object(
                custom_media, "search_media_pollinations", return_value=[1]
            ) as poll_mock,
        ):
            items = custom_media.search_media_gemini_image("cat", 5)
        reason_mock.assert_called_once()
        poll_mock.assert_called_once()
        self.assertEqual(items, [1])

    def test_invalid_key_falls_back_to_pollinations(self):
        with (
            patch.object(custom_media, "_search_gemini", return_value=[]),
            patch.object(
                custom_media, "gemini_last_failure_reason", return_value="invalid_key"
            ),
            patch.object(
                custom_media, "search_media_pollinations", return_value=[1]
            ) as poll_mock,
        ):
            items = custom_media.search_media_gemini_image("cat", 5)
        poll_mock.assert_called_once()
        self.assertEqual(items, [1])

    def test_empty_result_does_not_fallback(self):
        """Gemini 正常返回但无图（empty）时不烧 pollinations 额度。"""
        with (
            patch.object(custom_media, "_search_gemini", return_value=[]),
            patch.object(
                custom_media, "gemini_last_failure_reason", return_value="empty"
            ),
            patch.object(
                custom_media, "search_media_pollinations"
            ) as poll_mock,
        ):
            items = custom_media.search_media_gemini_image("cat", 5)
        poll_mock.assert_not_called()
        self.assertEqual(items, [])

    def test_fallback_passes_save_dir(self):
        with (
            patch.object(custom_media, "_search_gemini", return_value=[]),
            patch.object(
                custom_media, "gemini_last_failure_reason", return_value="quota"
            ),
            patch.object(
                custom_media, "search_media_pollinations"
            ) as poll_mock,
        ):
            custom_media.search_media_gemini_image("cat", 5, save_dir="/tmp/out")
        self.assertEqual(poll_mock.call_args[0][3], "/tmp/out")

    def test_uses_gemini_api_key_when_set(self):
        item = custom_media.MaterialInfo()
        with (
            patch.object(custom_media, "_get_custom_api_cfg") as cfg_mock,
            patch.object(custom_media, "_search_gemini", return_value=[item]) as g_mock,
        ):
            cfg_mock.return_value = {"key": "old-key", "extra_headers": ""}
            custom_media.search_media_gemini_image("cat", 5)
        args = g_mock.call_args[0]
        cfg_arg = args[2]
        self.assertEqual(cfg_arg["key"], "AIza-xyz")


class TestMaterialAutoProvidersGemini(unittest.TestCase):
    """material._auto_provider_configs 在配置 Gemini 后纳入 gemini_image。"""

    def setUp(self):
        self._orig = dict(config.app)
        config.app.clear()

    def tearDown(self):
        config.app.clear()
        config.app.update(self._orig)

    def test_gemini_image_in_auto_providers_when_configured(self):
        from app.services import material

        config.app["gemini_api_key"] = "AIza-xyz"
        providers = material._auto_provider_configs()
        names = [name for name, _ in providers]
        self.assertIn("gemini_image", names)


if __name__ == "__main__":
    unittest.main()
