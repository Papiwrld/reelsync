import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import patch

from app import asgi
from app.services import subtitle
from app.services import web_scrape as ws


class TestCheckRateLimit(unittest.TestCase):
    """app.asgi._check_rate_limit 的额度窗口与 IP 隔离行为。"""

    def setUp(self):
        # 每个用例独立计数：清空进程级内存限流桶，避免用例之间相互污染。
        asgi._rate_limit_store.clear()

    def test_videos_path_allows_10_then_blocks_11th(self):
        ip = "203.0.113.1"
        for _ in range(10):
            limited, retry_after = asgi._check_rate_limit(
                ip, "/api/v1/videos/generate"
            )
            self.assertFalse(limited)
            self.assertEqual(retry_after, 0)
        limited, retry_after = asgi._check_rate_limit(
            ip, "/api/v1/videos/generate"
        )
        self.assertTrue(limited)
        self.assertGreaterEqual(retry_after, 1)

    def test_videos_window_resets_after_60_seconds(self):
        ip = "203.0.113.2"
        t0 = 1_000_000.0
        with patch.object(asgi.time, "time", return_value=t0):
            for _ in range(10):
                asgi._check_rate_limit(ip, "/api/v1/videos/generate")
            limited, _ = asgi._check_rate_limit(ip, "/api/v1/videos/generate")
            self.assertTrue(limited)
        # 60s 窗口过后旧计数全部过期，下一次请求应被放行。
        with patch.object(asgi.time, "time", return_value=t0 + 60.0):
            limited, retry_after = asgi._check_rate_limit(
                ip, "/api/v1/videos/generate"
            )
            self.assertFalse(limited)
            self.assertEqual(retry_after, 0)

    def test_scripts_path_allows_30_then_blocks_31st(self):
        ip = "203.0.113.3"
        for _ in range(30):
            limited, retry_after = asgi._check_rate_limit(
                ip, "/api/v1/scripts/generate"
            )
            self.assertFalse(limited)
            self.assertEqual(retry_after, 0)
        limited, retry_after = asgi._check_rate_limit(
            ip, "/api/v1/scripts/generate"
        )
        self.assertTrue(limited)
        self.assertGreaterEqual(retry_after, 1)

    def test_non_api_v1_path_never_limited(self):
        for path in ("/", "/tasks", "/api/v2/videos", "/static/x"):
            for _ in range(5):
                limited, retry_after = asgi._check_rate_limit("203.0.113.4", path)
                self.assertEqual((limited, retry_after), (False, 0))

    def test_ips_have_independent_counters(self):
        ip_a = "203.0.113.10"
        ip_b = "203.0.113.11"
        for _ in range(10):
            asgi._check_rate_limit(ip_a, "/api/v1/videos/generate")
        limited, _ = asgi._check_rate_limit(ip_a, "/api/v1/videos/generate")
        self.assertTrue(limited)
        # 不同 IP 的计数互不影响：B 仍有完整额度。
        limited, retry_after = asgi._check_rate_limit(
            ip_b, "/api/v1/videos/generate"
        )
        self.assertFalse(limited)
        self.assertEqual(retry_after, 0)


class TestDecideWebSearchPlatform(unittest.TestCase):
    """web 搜索平台选择：关键词启发式优先，否则交给 LLM，失败回退 youtube。"""

    _LLM_APP_CONFIG: ClassVar[dict] = {
        "llm_provider": "openai",
        "openai_api_key": "test-key",
    }

    def test_tiktok_keywords_short_circuit_without_llm(self):
        for term in ("dance", "viral", "tiktok", "challenge", "trend", "Dance Battle"):
            with (
                self.subTest(term=term),
                patch("app.config.config.app", self._LLM_APP_CONFIG),
                patch("app.services.llm._generate_response") as llm,
            ):
                    self.assertEqual(
                        ws._decide_web_search_platform(term), "tiktok"
                    )
                    llm.assert_not_called()

    def test_llm_chooses_instagram(self):
        with (
            patch("app.config.config.app", self._LLM_APP_CONFIG),
            patch("app.services.llm._generate_response", return_value="instagram"),
        ):
            self.assertEqual(ws._decide_web_search_platform("cats"), "instagram")

    def test_llm_chooses_youtube(self):
        with (
            patch("app.config.config.app", self._LLM_APP_CONFIG),
            patch("app.services.llm._generate_response", return_value="youtube"),
        ):
            self.assertEqual(ws._decide_web_search_platform("cats"), "youtube")

    def test_llm_error_falls_back_to_youtube(self):
        with (
            patch("app.config.config.app", self._LLM_APP_CONFIG),
            patch(
                "app.services.llm._generate_response",
                return_value="Error: provider failed",
            ),
        ):
            self.assertEqual(ws._decide_web_search_platform("cats"), "youtube")


class TestWebSearchPrefix(unittest.TestCase):
    """平台名到 yt-dlp 搜索前缀的映射。"""

    def test_tiktok_uses_tiktoksearch5_prefix(self):
        self.assertTrue(
            ws._web_search_prefix("tiktok", "x").startswith("tiktoksearch5:")
        )

    def test_youtube_uses_ytsearch5_prefix(self):
        self.assertTrue(
            ws._web_search_prefix("youtube", "x").startswith("ytsearch5:")
        )


class TestSearchVideosWebScrapeFallback(unittest.TestCase):
    """tiktoksearch 前缀失败（Unsupported url scheme）时回退到 ytsearch。"""

    class _FakePopen:
        """最小 Popen 替身：只提供 returncode / stdout / stderr。"""

        def __init__(self, stdout="", stderr="", returncode=0):
            self.stdout = stdout
            self.stderr = stderr
            self.returncode = returncode

        def communicate(self, timeout=None):
            return self.stdout, self.stderr

    def test_tiktok_prefix_failure_falls_back_to_ytsearch(self):
        payload = json.dumps(
            {
                "webpage_url": "https://example.com/v",
                "duration": 10,
                "width": 720,
                "height": 1280,
                "title": "dance clip",
            }
        )
        failed = self._FakePopen(stderr="Unsupported url scheme", returncode=1)
        succeeded = self._FakePopen(stdout=payload + "\n")
        with (
            patch.object(
                ws, "_rewrite_query_for_web_search", return_value="dance battle"
            ),
            patch.object(
                ws.subprocess, "Popen", side_effect=[failed, succeeded]
            ),
            patch.object(ws, "_search_videos_via_html_search", return_value=[]),
            patch.object(ws.logger, "warning") as warn,
        ):
            results = ws.search_videos_web_scrape(
                "dance battle", 5, ws.VideoAspect.portrait
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].url, "https://example.com/v")
        messages = " ".join(
            str(arg) for call in warn.call_args_list for arg in call.args
        )
        self.assertIn("falling back to ytsearch", messages)


class TestResolveWhisperModelSize(unittest.TestCase):
    """whisper 模型尺寸解析：默认值、别名归一化与原样透传。"""

    def test_none_or_empty_returns_default_turbo(self):
        self.assertEqual(
            subtitle._resolve_whisper_model_size(None, "cpu"), "large-v3-turbo"
        )
        self.assertEqual(
            subtitle._resolve_whisper_model_size("", "cpu"), "large-v3-turbo"
        )

    def test_turbo_aliases_normalize_to_large_v3_turbo(self):
        for raw in (
            "large-v3-turbo",
            "whisper-large-v3-turbo",
            "large-v3-turbo-int8",
        ):
            with self.subTest(raw=raw):
                self.assertEqual(
                    subtitle._resolve_whisper_model_size(raw, "cpu"),
                    "large-v3-turbo",
                )

    def test_other_sizes_pass_through(self):
        self.assertEqual(
            subtitle._resolve_whisper_model_size("medium", "cpu"), "medium"
        )


class TestResolveWhisperDevice(unittest.TestCase):
    """whisper 设备解析：auto/cuda 都按实际 CUDA 可用性收敛到 cuda 或 cpu。"""

    def test_explicit_cpu(self):
        self.assertEqual(subtitle._resolve_whisper_device("cpu"), "cpu")

    def test_auto_with_cuda_available(self):
        with patch.object(subtitle, "_is_cuda_available", return_value=True):
            self.assertEqual(subtitle._resolve_whisper_device("auto"), "cuda")

    def test_auto_without_cuda(self):
        with patch.object(subtitle, "_is_cuda_available", return_value=False):
            self.assertEqual(subtitle._resolve_whisper_device("auto"), "cpu")

    def test_cuda_without_cuda_falls_back_to_cpu(self):
        with patch.object(subtitle, "_is_cuda_available", return_value=False):
            self.assertEqual(subtitle._resolve_whisper_device("cuda"), "cpu")


class TestWhisperOomFallback(unittest.TestCase):
    """large-v3-turbo 加载 OOM 时应自动回退到 medium 并继续转写。"""

    def test_oom_falls_back_to_medium(self):
        class _FakeWhisperModel:
            def __init__(self, **kwargs):
                self.init_kwargs = kwargs

            def transcribe(self, audio_file, **kwargs):
                info = SimpleNamespace(language="en", language_probability=0.99)
                return [], info

        state = {"calls": 0}

        def _whisper_factory(*args, **kwargs):
            state["calls"] += 1
            if state["calls"] == 1:
                raise RuntimeError("out of memory")
            return _FakeWhisperModel(*args, **kwargs)

        with tempfile.TemporaryDirectory() as tmp_dir:
            subtitle_file = Path(tmp_dir) / "out.srt"
            with (
                patch.object(subtitle, "model", None),
                patch.object(subtitle, "model_size", "large-v3-turbo"),
                patch.object(subtitle.utils, "root_dir", return_value=tmp_dir),
                patch.object(
                    subtitle, "WhisperModel", side_effect=_whisper_factory
                ) as whisper_mock,
            ):
                subtitle.create("audio.mp3", str(subtitle_file))
                self.assertTrue(subtitle_file.exists())

        self.assertEqual(whisper_mock.call_count, 2)
        self.assertEqual(
            whisper_mock.call_args_list[1].kwargs["model_size_or_path"], "medium"
        )


if __name__ == "__main__":
    unittest.main()