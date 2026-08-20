import base64
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.config import config
from app.services import custom_media

_ONE_PX_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)

_MATERIAL_KEYS = (
    "custom_api_url",
    "custom_api_video_url",
    "custom_api_image_url",
    "custom_api_key",
    "custom_api_method",
    "custom_api_response_format",
    "custom_api_extra_headers",
    "custom_api_extra_body",
)


class TestCustomApiConfiguration(unittest.TestCase):
    def setUp(self):
        self.original_app_config = dict(config.app)
        for key in _MATERIAL_KEYS:
            config.app.pop(key, None)

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)

    def test_legacy_url_with_key_is_configured(self):
        config.app["custom_api_url"] = "https://api.example/v1"
        config.app["custom_api_key"] = "secret"
        self.assertTrue(custom_media.is_custom_api_configured())

    def test_video_url_alone_with_key_is_configured(self):
        config.app["custom_api_video_url"] = "https://api.example/videos"
        config.app["custom_api_key"] = "secret"
        self.assertTrue(custom_media.is_custom_api_configured())

    def test_image_url_alone_with_key_is_configured(self):
        config.app["custom_api_image_url"] = "https://api.example/images"
        config.app["custom_api_key"] = "secret"
        self.assertTrue(custom_media.is_custom_api_configured())

    def test_url_without_key_is_not_configured(self):
        config.app["custom_api_url"] = "https://api.example/v1"
        self.assertFalse(custom_media.is_custom_api_configured())

    def test_nothing_configured_is_not_configured(self):
        self.assertFalse(custom_media.is_custom_api_configured())


class TestCustomApiVideoImageFallback(unittest.TestCase):
    def setUp(self):
        self.original_app_config = dict(config.app)
        for key in _MATERIAL_KEYS:
            config.app.pop(key, None)
        config.app["custom_api_key"] = "secret"
        config.app["custom_api_video_url"] = "https://api.example/videos"
        config.app["custom_api_image_url"] = "https://api.example/images"
        self.calls = []

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)

    def _mock_post(self, responses):
        def fake_post(url, **kwargs):
            self.calls.append(url)
            index = len(self.calls) - 1
            body = responses[index]
            if isinstance(body, Exception):
                raise body
            return SimpleNamespace(status_code=200, text="", json=lambda: body)

        return fake_post

    def test_video_items_returned_without_image_call(self):
        responses = [{"videos": [{"url": "https://cdn/v1.mp4", "duration": 5}]}]
        with patch(
            "app.services.custom_media.requests.post", side_effect=self._mock_post(responses)
        ):
            items = custom_media.search_media_custom_api(
                "AI healthcare", minimum_duration=3
            )
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].url, "https://cdn/v1.mp4")
        self.assertEqual(items[0].source_info["media_type"], "video")
        self.assertEqual(self.calls, ["https://api.example/videos"])

    def test_empty_video_falls_back_to_image(self):
        responses = [
            {"videos": []},
            {"images": [{"url": "https://cdn/poster.jpg", "duration": 5}]},
        ]
        with patch(
            "app.services.custom_media.requests.post", side_effect=self._mock_post(responses)
        ):
            items = custom_media.search_media_custom_api(
                "AI healthcare", minimum_duration=3
            )
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].url, "https://cdn/poster.jpg")
        self.assertEqual(items[0].source_info["media_type"], "image")
        self.assertEqual(
            self.calls,
            ["https://api.example/videos", "https://api.example/images"],
        )

    def test_failed_video_falls_back_to_image(self):
        responses = [
            SimpleNamespace(status_code=404, text="not found", json=lambda: {}),
            {"images": [{"url": "https://cdn/poster.jpg"}]},
        ]
        with patch(
            "app.services.custom_media.requests.post", side_effect=self._mock_post(responses)
        ):
            items = custom_media.search_media_custom_api(
                "AI healthcare", minimum_duration=3
            )
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].source_info["media_type"], "image")

    def test_video_exception_falls_back_to_image(self):
        responses = [
            __import__("requests").Timeout("boom"),
            {"images": [{"url": "https://cdn/poster.jpg"}]},
        ]
        with patch(
            "app.services.custom_media.requests.post", side_effect=self._mock_post(responses)
        ):
            items = custom_media.search_media_custom_api(
                "AI healthcare", minimum_duration=3
            )
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].source_info["media_type"], "image")

    def test_same_url_does_not_double_call(self):
        config.app.pop("custom_api_image_url")
        config.app.pop("custom_api_video_url")
        config.app["custom_api_url"] = "https://api.example/v1"
        responses = [{"videos": []}]
        with patch(
            "app.services.custom_media.requests.post", side_effect=self._mock_post(responses)
        ):
            items = custom_media.search_media_custom_api(
                "AI healthcare", minimum_duration=3
            )
        self.assertEqual(items, [])
        self.assertEqual(self.calls, ["https://api.example/v1"])

    def test_both_endpoints_empty_returns_empty(self):
        responses = [{"videos": []}, {"images": []}]
        with patch(
            "app.services.custom_media.requests.post", side_effect=self._mock_post(responses)
        ):
            items = custom_media.search_media_custom_api(
                "AI healthcare", minimum_duration=3
            )
        self.assertEqual(items, [])

    def test_image_fallback_duration_is_clamped_to_minimum(self):
        responses = [
            {"videos": []},
            {"images": [{"url": "https://cdn/poster.jpg", "duration": 2}]},
        ]
        with patch(
            "app.services.custom_media.requests.post", side_effect=self._mock_post(responses)
        ):
            items = custom_media.search_media_custom_api(
                "AI healthcare", minimum_duration=5
            )
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].duration, 5)

    def test_image_endpoint_only_is_used_directly(self):
        config.app.pop("custom_api_video_url")
        responses = [{"images": [{"url": "https://cdn/poster.jpg"}]}]
        with patch(
            "app.services.custom_media.requests.post", side_effect=self._mock_post(responses)
        ):
            items = custom_media.search_media_custom_api(
                "AI healthcare", minimum_duration=3
            )
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].source_info["media_type"], "image")
        self.assertEqual(self.calls, ["https://api.example/images"])


class TestCustomApiGeminiFormat(unittest.TestCase):
    _GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"

    def setUp(self):
        self.original_app_config = dict(config.app)
        for key in _MATERIAL_KEYS:
            config.app.pop(key, None)
        config.app["custom_api_key"] = "secret"
        config.app["custom_api_response_format"] = "gemini"
        config.app["custom_api_video_url"] = self._GEMINI_URL
        config.app["custom_api_image_url"] = self._GEMINI_URL
        self.tmp = tempfile.TemporaryDirectory()
        self.calls = []

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)
        self.tmp.cleanup()

    def _image_response(self, b64=None):
        return {
            "id": "int_1",
            "status": "completed",
            "steps": [
                {
                    "type": "model_output",
                    "content": [
                        {
                            "type": "image",
                            "mime_type": "image/png",
                            "data": base64.b64encode(b64 or _ONE_PX_PNG).decode(),
                        }
                    ],
                }
            ],
        }

    def _video_uri_response(self, uri="https://gemini/files/v1:download?alt=media"):
        return {
            "id": "int_2",
            "status": "completed",
            "steps": [
                {
                    "type": "model_output",
                    "content": [
                        {"type": "video", "mime_type": "video/mp4", "uri": uri}
                    ],
                }
            ],
        }

    def _video_data_response(self, data=b"FAKEVIDEO"):
        return {
            "id": "int_3",
            "status": "completed",
            "steps": [
                {
                    "type": "model_output",
                    "content": [
                        {
                            "type": "video",
                            "mime_type": "video/mp4",
                            "data": base64.b64encode(data).decode(),
                        }
                    ],
                }
            ],
        }

    def _fake_post(self, responses):
        def fake_post(url, **kwargs):
            self.calls.append((url, kwargs))
            index = len(self.calls) - 1
            body = responses[index]
            if isinstance(body, Exception):
                raise body
            return SimpleNamespace(status_code=200, text="", json=lambda: body)

        return fake_post

    def _fake_post_with_status(self, responses):
        def fake_post(url, **kwargs):
            self.calls.append((url, kwargs))
            index = len(self.calls) - 1
            item = responses[index % len(responses)]
            if isinstance(item, Exception):
                raise item
            status, body = item
            return SimpleNamespace(status_code=status, text="", json=lambda: body)

        return fake_post

    def _empty_response(self):
        return {"id": "int_empty", "status": "completed", "steps": []}

    def test_gemini_image_writes_local_file(self):
        responses = [self._empty_response(), self._image_response()]
        with patch(
            "app.services.custom_media.requests.post",
            side_effect=self._fake_post(responses),
        ):
            items = custom_media.search_media_custom_api(
                "AI healthcare", minimum_duration=3, save_dir=self.tmp.name
            )
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].source_info["media_type"], "image")
        self.assertTrue(os.path.isfile(items[0].url))
        self.assertEqual(items[0].duration, 5)

    def test_gemini_image_body_uses_model_and_portrait_ratio(self):
        responses = [self._empty_response(), self._image_response()]
        with patch(
            "app.services.custom_media.requests.post",
            side_effect=self._fake_post(responses),
        ):
            custom_media.search_media_custom_api(
                "AI healthcare", minimum_duration=3, save_dir=self.tmp.name
            )
        _, kwargs = self.calls[1]
        body = kwargs["json"]
        self.assertEqual(body["model"], "gemini-3.1-flash-image")
        self.assertEqual(body["input"], "AI healthcare")
        self.assertEqual(body["response_format"]["type"], "image")
        self.assertEqual(body["response_format"]["aspect_ratio"], "9:16")

    def test_gemini_video_uses_uri_delivery_and_downloads(self):
        def fake_get(url, **kwargs):
            return SimpleNamespace(status_code=200, content=b"REALVIDEO")

        with (
            patch(
                "app.services.custom_media.requests.post",
                side_effect=self._fake_post([self._video_uri_response()]),
            ),
            patch("app.services.custom_media.requests.get", side_effect=fake_get),
        ):
            items = custom_media.search_media_custom_api(
                "AI smart home", minimum_duration=3, save_dir=self.tmp.name
            )
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].source_info["media_type"], "video")
        self.assertTrue(os.path.isfile(items[0].url))
        with open(items[0].url, "rb") as f:
            self.assertEqual(f.read(), b"REALVIDEO")
        _, kwargs = self.calls[0]
        self.assertEqual(kwargs["json"]["model"], "gemini-omni-flash-preview")
        self.assertEqual(
            kwargs["json"]["response_format"]["delivery"], "uri"
        )

    def test_gemini_video_base64_block_decoded(self):
        with patch(
            "app.services.custom_media.requests.post",
            side_effect=self._fake_post([self._video_data_response()]),
        ):
            items = custom_media.search_media_custom_api(
                "AI education", minimum_duration=3, save_dir=self.tmp.name
            )
        self.assertEqual(len(items), 1)
        self.assertTrue(os.path.isfile(items[0].url))

    def test_gemini_video_empty_falls_back_to_image_same_url(self):
        responses = [
            {"id": "int_4", "status": "completed", "steps": []},
            self._image_response(),
        ]
        with patch(
            "app.services.custom_media.requests.post",
            side_effect=self._fake_post(responses),
        ):
            items = custom_media.search_media_custom_api(
                "AI healthcare", minimum_duration=3, save_dir=self.tmp.name
            )
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].source_info["media_type"], "image")
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.calls[0][1]["json"]["model"], "gemini-omni-flash-preview")
        self.assertEqual(self.calls[1][1]["json"]["model"], "gemini-3.1-flash-image")

    def test_gemini_video_failure_falls_back_to_image(self):
        responses = [
            SimpleNamespace(status_code=403, text="forbidden", json=lambda: {}),
            self._image_response(),
        ]
        with patch(
            "app.services.custom_media.requests.post",
            side_effect=self._fake_post(responses),
        ):
            items = custom_media.search_media_custom_api(
                "AI healthcare", minimum_duration=3, save_dir=self.tmp.name
            )
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].source_info["media_type"], "image")

    def test_gemini_custom_models_are_used(self):
        config.app["custom_api_video_model"] = "veo-3.1-generate-001"
        config.app["custom_api_image_model"] = "gemini-3-pro-image"
        responses = [self._empty_response(), self._image_response()]
        with patch(
            "app.services.custom_media.requests.post",
            side_effect=self._fake_post(responses),
        ):
            custom_media.search_media_custom_api(
                "AI healthcare", minimum_duration=3, save_dir=self.tmp.name
            )
        self.assertEqual(self.calls[0][1]["json"]["model"], "veo-3.1-generate-001")
        self.assertEqual(self.calls[1][1]["json"]["model"], "gemini-3-pro-image")

    def test_gemini_image_retries_transient_503_then_succeeds(self):
        """503 高峰过载必须按退避重试，而不是一次抖动就让素材空手而归。"""
        responses = [
            (503, {"error": {"code": 503, "message": "high demand"}}),
            (200, self._image_response()),
        ]
        with (
            patch(
                "app.services.custom_media.requests.post",
                side_effect=self._fake_post_with_status(responses),
            ),
            patch("app.services.custom_media.time.sleep") as sleep,
        ):
            items = custom_media.search_media_custom_api(
                "AI healthcare", minimum_duration=3, save_dir=self.tmp.name
            )
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].source_info["media_type"], "image")
        # 视频端点 1 次 503 + 1 次成功(空结果)，图片端点 1 次 503 + 1 次成功。
        self.assertEqual(len(self.calls), 4)
        sleep.assert_called()

    def test_gemini_persistent_503_returns_empty(self):
        """持续 503 时有限次重试后放弃，失败快速回落给其它供应商。"""
        responses = [(503, {"error": {"code": 503, "message": "high demand"}})]
        with (
            patch(
                "app.services.custom_media.requests.post",
                side_effect=self._fake_post_with_status(responses),
            ),
            patch("app.services.custom_media.time.sleep"),
        ):
            items = custom_media.search_media_custom_api(
                "AI healthcare", minimum_duration=3, save_dir=self.tmp.name
            )
        self.assertEqual(len(items), 0)
        # 视频 + 图片端点各执行 1 次初始请求 + 3 次退避重试。
        self.assertEqual(len(self.calls), 8)

    def test_gemini_download_polls_through_503(self):
        """文件下载遇到瞬时 503 时应继续轮询，而不是立刻判定失败。"""
        statuses = iter([503, 200])

        def fake_get(url, **kwargs):
            status = next(statuses)
            return SimpleNamespace(
                status_code=status,
                text="" if status == 503 else "ok",
                content=b"" if status == 503 else b"REALVIDEO",
            )

        with patch("app.services.custom_media.requests.get", side_effect=fake_get):
            with patch("app.services.custom_media.time.sleep"):
                raw = custom_media._download_gemini_uri("https://gemini/file", "k")
        self.assertEqual(raw, b"REALVIDEO")


def _tiny_jpeg_bytes() -> bytes:
    """生成一张最小可用 JPEG，用于 PIL 校验通过的缓存/下载测试。"""
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (200, 30, 30)).save(buf, format="JPEG")
    return buf.getvalue()


class TestPollinationsEnabled(unittest.TestCase):
    """Pollinations 免费来源的启用策略。"""

    def setUp(self):
        self.original_app_config = dict(config.app)
        for key in (
            "enable_pollinations",
            "pexels_api_keys",
            "pixabay_api_keys",
            "coverr_api_keys",
            "custom_api_url",
            "custom_api_key",
            "enable_web_scraping",
        ):
            config.app.pop(key, None)

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)

    def test_enabled_by_default_when_no_provider_configured(self):
        self.assertTrue(custom_media.is_pollinations_enabled())

    def test_disabled_when_stock_provider_configured(self):
        config.app["pexels_api_keys"] = ["k"]
        self.assertFalse(custom_media.is_pollinations_enabled())

    def test_disabled_when_web_scraping_enabled(self):
        config.app["enable_web_scraping"] = True
        self.assertFalse(custom_media.is_pollinations_enabled())

    def test_explicit_enable_overrides_stock_providers(self):
        config.app["pexels_api_keys"] = ["k"]
        config.app["enable_pollinations"] = True
        self.assertTrue(custom_media.is_pollinations_enabled())

    def test_explicit_disable_overrides_free_mode(self):
        config.app["enable_pollinations"] = False
        self.assertFalse(custom_media.is_pollinations_enabled())


class TestPollinationsImageUrl(unittest.TestCase):
    def test_url_encodes_prompt_and_params(self):
        url = custom_media._pollinations_image_url(
            "sunset beach 4k", 1080, 1920, "flux", 12345
        )
        self.assertTrue(url.startswith(custom_media._POLLINATIONS_BASE_URL))
        self.assertIn("width=1080", url)
        self.assertIn("height=1920", url)
        self.assertIn("model=flux", url)
        self.assertIn("nologo=true", url)
        self.assertIn("seed=12345", url)
        self.assertNotIn("sunset beach 4k", url)  # 原文被百分号编码


class TestGeneratedMediaCache(unittest.TestCase):
    """生成图片缓存：命中、过期、损坏、清理与原子写入。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.original_app_config = dict(config.app)
        patcher = patch.object(
            custom_media, "_generated_media_cache_dir", return_value=self.tmp.name
        )
        self._dir_patcher = patcher
        self._dir_patcher.start()

    def tearDown(self):
        self._dir_patcher.stop()
        config.app.clear()
        config.app.update(self.original_app_config)
        self.tmp.cleanup()

    def _cache_path(self, provider="pollinations", prompt="test term"):
        return custom_media._generated_media_cache_path(
            provider, prompt, 1080, 1920, "flux"
        )

    def test_cache_key_is_deterministic_and_sensitive_to_params(self):
        k1 = custom_media._generated_media_cache_key(
            "pollinations", "test term", 1080, 1920, "flux"
        )
        k2 = custom_media._generated_media_cache_key(
            "pollinations", "test term", 1080, 1920, "flux"
        )
        k3 = custom_media._generated_media_cache_key(
            "pollinations", "test term", 1080, 1920, "turbo"
        )
        self.assertEqual(k1, k2)  # 相同参数幂等
        self.assertEqual(len(k1), 64)
        self.assertNotEqual(k2, k3)

    def test_load_miss_returns_empty(self):
        self.assertEqual(
            custom_media._load_generated_media_cache(
                "pollinations", "missing", 1080, 1920, "flux"
            ),
            "",
        )

    def test_load_fresh_valid_file_returns_path(self):
        path = self._cache_path()
        Path(path).write_bytes(_tiny_jpeg_bytes())
        loaded = custom_media._load_generated_media_cache(
            "pollinations", "test term", 1080, 1920, "flux"
        )
        self.assertEqual(loaded, path)

    def test_load_expired_file_is_deleted(self):
        path = self._cache_path()
        Path(path).write_bytes(_tiny_jpeg_bytes())
        old_mtime = time.time() - custom_media.GENERATED_IMAGE_CACHE_TTL_SECONDS - 60
        os.utime(path, (old_mtime, old_mtime))
        loaded = custom_media._load_generated_media_cache(
            "pollinations", "test term", 1080, 1920, "flux"
        )
        self.assertEqual(loaded, "")
        self.assertFalse(os.path.exists(path))

    def test_load_corrupt_file_is_deleted(self):
        path = self._cache_path()
        Path(path).write_bytes(b"not an image at all")
        loaded = custom_media._load_generated_media_cache(
            "pollinations", "test term", 1080, 1920, "flux"
        )
        self.assertEqual(loaded, "")
        self.assertFalse(os.path.exists(path))

    def test_store_valid_bytes_returns_path(self):
        stored = custom_media._store_generated_media_cache(
            "pollinations", "test term", 1080, 1920, "flux", _tiny_jpeg_bytes()
        )
        self.assertEqual(stored, self._cache_path())
        self.assertTrue(os.path.isfile(stored))

    def test_store_invalid_bytes_returns_empty_and_no_file(self):
        stored = custom_media._store_generated_media_cache(
            "pollinations", "test term", 1080, 1920, "flux", b"garbage"
        )
        self.assertEqual(stored, "")
        self.assertFalse(os.path.exists(self._cache_path()))

    def test_cleanup_only_removes_expired_sha256_jpgs(self):
        fresh = self._cache_path("pollinations", "fresh term")
        Path(fresh).write_bytes(_tiny_jpeg_bytes())
        stale = self._cache_path("pollinations", "stale term")
        Path(stale).write_bytes(_tiny_jpeg_bytes())
        old_mtime = time.time() - custom_media.GENERATED_IMAGE_CACHE_TTL_SECONDS - 60
        os.utime(stale, (old_mtime, old_mtime))
        Path(os.path.join(self.tmp.name, "user-file.jpg")).write_bytes(
            _tiny_jpeg_bytes()
        )
        Path(os.path.join(self.tmp.name, "note.txt")).write_text("keep me")

        deleted = custom_media._cleanup_expired_generated_media_cache(
            now=time.time(), force=True
        )
        self.assertEqual(deleted, 1)
        self.assertFalse(os.path.exists(stale))
        self.assertTrue(os.path.exists(fresh))
        self.assertTrue(os.path.exists(os.path.join(self.tmp.name, "user-file.jpg")))
        self.assertTrue(os.path.exists(os.path.join(self.tmp.name, "note.txt")))

    def test_cleanup_rate_limited_without_force(self):
        self.assertEqual(
            custom_media._cleanup_expired_generated_media_cache(force=True), 0
        )
        # 一小时内再次调用被限流，直接返回 0 且不触碰目录。
        self.assertEqual(
            custom_media._cleanup_expired_generated_media_cache(), 0
        )


class TestPollinationsDownload(unittest.TestCase):
    """_download_pollinations_image 的流式下载、重试与校验。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out_path = os.path.join(self.tmp.name, "gen.jpg")
        patcher = patch.object(
            custom_media, "_generated_media_cache_dir", return_value=self.tmp.name
        )
        self._dir_patcher = patcher
        self._dir_patcher.start()

    def tearDown(self):
        self._dir_patcher.stop()
        self.tmp.cleanup()

    @staticmethod
    def _response(status_code, raw=b""):
        class _Resp:
            def __init__(self):
                self.status_code = status_code
                self._raw = raw

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def iter_content(self, chunk_size=65536):
                yield self._raw

        return _Resp()

    def test_success_returns_cache_path(self):
        jpeg = _tiny_jpeg_bytes()
        with patch.object(
            custom_media.requests,
            "get",
            return_value=self._response(200, jpeg),
        ):
            result = custom_media._download_pollinations_image(
                "https://image.pollinations.ai/prompt/x", self.out_path, "flux"
            )
        self.assertEqual(result, self.out_path)
        self.assertTrue(os.path.isfile(result))

    def test_retries_then_succeeds_on_5xx(self):
        jpeg = _tiny_jpeg_bytes()
        calls = {"n": 0}

        def fake_get(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return self._response(500)
            return self._response(200, jpeg)

        with (
            patch.object(custom_media.requests, "get", side_effect=fake_get),
            patch.object(custom_media.time, "sleep"),
        ):
            result = custom_media._download_pollinations_image(
                "https://image.pollinations.ai/prompt/x", self.out_path, "flux"
            )
        self.assertEqual(result, self.out_path)
        self.assertEqual(calls["n"], 2)

    def test_persistent_failure_returns_empty(self):
        with (
            patch.object(
                custom_media.requests,
                "get",
                return_value=self._response(503),
            ),
            patch.object(custom_media.time, "sleep"),
        ):
            result = custom_media._download_pollinations_image(
                "https://image.pollinations.ai/prompt/x", self.out_path, "flux"
            )
        self.assertEqual(result, "")

    def test_network_exception_retries_then_returns_empty(self):
        def fake_get(*args, **kwargs):
            raise custom_media.requests.ConnectionError("boom")

        with (
            patch.object(custom_media.requests, "get", side_effect=fake_get),
            patch.object(custom_media.time, "sleep"),
        ):
            result = custom_media._download_pollinations_image(
                "https://image.pollinations.ai/prompt/x", self.out_path, "flux"
            )
        self.assertEqual(result, "")
        self.assertEqual(os.listdir(self.tmp.name), [])

    def test_non_image_body_fails_validation(self):
        with (
            patch.object(
                custom_media.requests,
                "get",
                return_value=self._response(200, b"<html>not an image</html>"),
            ),
        ):
            result = custom_media._download_pollinations_image(
                "https://image.pollinations.ai/prompt/x", self.out_path, "flux"
            )
        self.assertEqual(result, "")
        self.assertEqual(os.listdir(self.tmp.name), [])


class TestSearchMediaPollinations(unittest.TestCase):
    """search_media_pollinations 的素材条目与缓存交互。"""

    _PROVIDER_KEYS = (
        "enable_pollinations",
        "pexels_api_keys",
        "pixabay_api_keys",
        "coverr_api_keys",
        "custom_api_url",
        "custom_api_key",
        "enable_web_scraping",
        "pollinations_image_model",
    )

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.original_app_config = dict(config.app)
        for key in self._PROVIDER_KEYS:
            config.app.pop(key, None)
        patcher = patch.object(
            custom_media, "_generated_media_cache_dir", return_value=self.tmp.name
        )
        self._dir_patcher = patcher
        self._dir_patcher.start()

    def tearDown(self):
        self._dir_patcher.stop()
        config.app.clear()
        config.app.update(self.original_app_config)
        self.tmp.cleanup()

    def test_disabled_returns_empty_without_download(self):
        with (
            patch.object(custom_media, "is_pollinations_enabled", return_value=False),
            patch.object(custom_media, "_download_pollinations_image") as dl,
        ):
            items = custom_media.search_media_pollinations("nature", 5)
        self.assertEqual(items, [])
        dl.assert_not_called()

    def test_cache_hit_reuses_file_without_download(self):
        cached = custom_media._generated_media_cache_path(
            "pollinations", "nature", 1080, 1920, "flux"
        )
        Path(cached).write_bytes(_tiny_jpeg_bytes())
        with patch.object(custom_media, "_download_pollinations_image") as dl:
            items = custom_media.search_media_pollinations("nature", 5)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].url, cached)
        dl.assert_not_called()

    def test_cache_miss_downloads_into_real_cache_dir(self):
        calls = {}

        def fake_get(url, *args, **kwargs):
            calls["url"] = url
            return TestPollinationsDownload._response(200, _tiny_jpeg_bytes())

        with (
            patch.object(custom_media.requests, "get", side_effect=fake_get),
            patch.object(custom_media.time, "sleep"),
        ):
            items = custom_media.search_media_pollinations("nature", 5)
        self.assertEqual(len(items), 1)
        self.assertTrue(
            os.path.isfile(items[0].url),
            f"downloaded file missing: {items[0].url}",
        )
        self.assertTrue(items[0].url.startswith(self.tmp.name))
        self.assertTrue(items[0].url.endswith(".jpg"))
        self.assertIn("nature", calls["url"])

    def test_download_failure_returns_empty(self):
        with patch.object(
            custom_media, "_download_pollinations_image", return_value=""
        ):
            items = custom_media.search_media_pollinations("nature", 5)
        self.assertEqual(items, [])

    def test_item_fields_are_neutral_and_image_typed(self):
        with patch.object(
            custom_media,
            "_download_pollinations_image",
            return_value=os.path.join(self.tmp.name, "gen.jpg"),
        ):
            items = custom_media.search_media_pollinations("nature", 5)
        item = items[0]
        self.assertEqual(item.provider, "pollinations")
        self.assertEqual(item.duration, 5)
        self.assertNotIn("title", item.source_info)
        self.assertNotIn("tags", item.source_info)
        self.assertEqual(item.source_info["media_type"], "image")
        self.assertEqual(item.source_info["width"], 1080)
        self.assertEqual(item.source_info["height"], 1920)

    def test_portrait_resolution_and_minimum_duration(self):
        with patch.object(
            custom_media,
            "_download_pollinations_image",
            return_value=os.path.join(self.tmp.name, "gen.jpg"),
        ):
            items = custom_media.search_media_pollinations("nature", 12)
        self.assertEqual(items[0].duration, 12)


if __name__ == "__main__":
    unittest.main()