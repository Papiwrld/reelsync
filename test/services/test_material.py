import base64
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import requests

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.config import config
from app.services import material


class TestMaterialTlsVerification(unittest.TestCase):
    def setUp(self):
        self.original_app_config = dict(config.app)
        self.original_proxy_config = dict(config.proxy)

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)
        config.proxy.clear()
        config.proxy.update(self.original_proxy_config)

    def test_search_pexels_uses_tls_verification_by_default(self):
        """
        默认路径必须开启 TLS 校验，避免素材 API key 和返回的素材 URL
        在公共网络或不可信代理环境中被中间人攻击截获或篡改。
        """
        config.app["pexels_api_keys"] = ["pexels-key"]
        config.app.pop("tls_verify", None)
        config.proxy.clear()

        fake_response = SimpleNamespace(
            json=lambda: {
                "videos": [
                    {
                        "id": 321,
                        "url": "https://www.pexels.com/video/example-321/?token=drop",
                        "duration": 8,
                        "user": {
                            "id": 654,
                            "name": "Pexels Creator",
                            "url": "https://www.pexels.com/@creator/?key=drop",
                        },
                        "video_files": [
                            {
                                "id": 987,
                                "width": 1080,
                                "height": 1920,
                                "link": "https://example.com/video.mp4",
                            }
                        ],
                    }
                ]
            }
        )

        with patch(
            "app.services.material.requests.get", return_value=fake_response
        ) as get:
            results = material.search_videos_pexels("cat", minimum_duration=1)

        self.assertEqual(len(results), 1)
        self.assertTrue(get.call_args.kwargs["verify"])
        self.assertEqual(results[0].source_info["asset_id"], "321")
        self.assertEqual(
            results[0].source_info["source_page"],
            "https://www.pexels.com/video/example-321/",
        )
        self.assertEqual(
            results[0].source_info["creator"]["profile_page"],
            "https://www.pexels.com/@creator/",
        )
        self.assertEqual(results[0].source_info["rendition"]["id"], "987")

    def test_search_pixabay_allows_explicit_tls_disable_for_proxy(self):
        """
        少数企业代理会使用自签证书。该场景必须显式配置关闭 TLS 校验，
        不能再由代码硬编码默认关闭。
        """
        config.app["pixabay_api_keys"] = ["pixabay-key"]
        config.app["tls_verify"] = False
        config.proxy.clear()

        fake_response = SimpleNamespace(
            status_code=200,
            headers={"content-type": "application/json"},
            text="",
            json=lambda: {
                "hits": [
                    {
                        "duration": 8,
                        "videos": {
                            "large": {
                                "width": 1920,
                                "height": 1080,
                                "url": "https://example.com/video.mp4",
                            }
                        },
                    }
                ]
            },
        )

        with patch(
            "app.services.material.requests.get", return_value=fake_response
        ) as get:
            results = material.search_videos_pixabay(
                "cat",
                minimum_duration=1,
                video_aspect=material.VideoAspect.landscape,
            )

        self.assertEqual(len(results), 1)
        self.assertFalse(get.call_args.kwargs["verify"])

    def test_remote_searches_only_return_requested_orientation(self):
        """
        三个素材源都必须只返回目标方向的素材，避免竖屏任务混入横屏素材后
        通过 letterbox 产生明显黑边。Pexels 使用远端参数并在本地校验，
        Pixabay 和 Coverr 使用响应尺寸做本地过滤。
        """
        config.app["pexels_api_keys"] = ["pexels-key"]
        config.app["pixabay_api_keys"] = ["pixabay-key"]
        config.app["coverr_api_keys"] = ["coverr-key"]
        config.proxy.clear()

        pexels_response = SimpleNamespace(
            json=lambda: {
                "videos": [
                    {
                        "id": 1,
                        "duration": 8,
                        "video_files": [
                            {
                                "id": 11,
                                "width": 1920,
                                "height": 1080,
                                "link": "https://example.com/landscape.mp4",
                            }
                        ],
                    },
                    {
                        "id": 2,
                        "duration": 8,
                        "video_files": [
                            {
                                "id": 22,
                                "width": 1080,
                                "height": 1920,
                                "link": "https://example.com/portrait.mp4",
                            }
                        ],
                    },
                ]
            }
        )
        pixabay_response = SimpleNamespace(
            status_code=200,
            headers={"content-type": "application/json"},
            text="",
            json=lambda: {
                "hits": [
                    {
                        "id": 1,
                        "duration": 8,
                        "videos": {
                            "large": {
                                "width": 1920,
                                "height": 1080,
                                "url": "https://example.com/landscape.mp4",
                            }
                        },
                    },
                    {
                        "id": 2,
                        "duration": 8,
                        "videos": {
                            "large": {
                                "width": 1080,
                                "height": 1920,
                                "url": "https://example.com/portrait.mp4",
                            }
                        },
                    },
                ]
            },
        )
        coverr_response = SimpleNamespace(
            json=lambda: {
                "hits": [
                    {
                        "id": "landscape",
                        "duration": 8,
                        "max_width": 1920,
                        "max_height": 1080,
                        "urls": {"mp4_download": "https://example.com/landscape.mp4"},
                    },
                    {
                        "id": "portrait",
                        "duration": 8,
                        "max_width": 1080,
                        "max_height": 1920,
                        "urls": {"mp4_download": "https://example.com/portrait.mp4"},
                    },
                    {
                        "id": "unknown",
                        "duration": 8,
                        "urls": {"mp4_download": "https://example.com/unknown.mp4"},
                    },
                ]
            }
        )

        with patch(
            "app.services.material.requests.get",
            return_value=pexels_response,
        ) as get:
            pexels_results = material.search_videos_pexels(
                "city",
                minimum_duration=1,
                video_aspect=material.VideoAspect.portrait,
            )
            pexels_url = get.call_args.args[0]
        with patch(
            "app.services.material.requests.get",
            return_value=pixabay_response,
        ):
            pixabay_results = material.search_videos_pixabay(
                "city",
                minimum_duration=1,
                video_aspect=material.VideoAspect.portrait,
            )
        with patch(
            "app.services.material.requests.get",
            return_value=coverr_response,
        ) as get:
            coverr_results = material.search_videos_coverr(
                "city",
                minimum_duration=1,
                video_aspect=material.VideoAspect.portrait,
            )
            coverr_url = get.call_args.args[0]

        self.assertIn("/v1/videos/search?", pexels_url)
        self.assertIn("orientation=portrait", pexels_url)
        self.assertIn("page_size=20", coverr_url)
        self.assertIn("filter=is_vertical%3Atrue", coverr_url)
        for results in (pexels_results, pixabay_results, coverr_results):
            self.assertEqual(
                [item.url for item in results],
                ["https://example.com/portrait.mp4"],
            )

    def test_video_aspect_matching_rejects_unknown_dimensions(self):
        """无法确认方向的素材不能进入严格的横竖屏候选列表。"""
        self.assertTrue(
            material._matches_video_aspect(
                1080,
                1920,
                material.VideoAspect.portrait,
            )
        )
        self.assertFalse(
            material._matches_video_aspect(
                1920,
                1080,
                material.VideoAspect.portrait,
            )
        )
        self.assertTrue(
            material._matches_video_aspect(
                None,
                None,
                material.VideoAspect.portrait,
                is_vertical=True,
            )
        )
        self.assertFalse(
            material._matches_video_aspect(
                None,
                None,
                material.VideoAspect.portrait,
            )
        )
        self.assertTrue(
            material._matches_video_aspect(
                1080,
                1080,
                material.VideoAspect.square,
            )
        )
        self.assertFalse(
            material._matches_video_aspect(
                1080,
                1920,
                material.VideoAspect.square,
            )
        )

    def test_coverr_passes_orientation_filter_to_remote_search(self):
        """Coverr 横竖屏搜索应在服务端筛选，方形素材继续使用本地尺寸校验。"""
        config.app["coverr_api_keys"] = ["coverr-key"]
        config.proxy.clear()
        fake_response = SimpleNamespace(json=lambda: {"hits": []})
        cases = (
            (material.VideoAspect.portrait, "filter=is_vertical%3Atrue"),
            (material.VideoAspect.landscape, "filter=is_vertical%3Afalse"),
            (material.VideoAspect.square, None),
        )

        for aspect, expected_filter in cases:
            with (
                self.subTest(aspect=aspect),
                patch(
                    "app.services.material.requests.get",
                    return_value=fake_response,
                ) as get,
            ):
                material.search_videos_coverr(
                    "city",
                    minimum_duration=1,
                    video_aspect=aspect,
                )
                request_url = get.call_args.args[0]

            self.assertIn("page_size=20", request_url)
            if expected_filter:
                self.assertIn(expected_filter, request_url)
            else:
                self.assertNotIn("filter=", request_url)

    def test_square_search_preserves_crop_compatible_materials(self):
        """
        Pixabay 和 Coverr 很少提供原生方形视频。方形输出必须继续接受可裁剪的
        横屏素材，否则选择这两个来源时会在搜索阶段直接得到空列表。
        """
        config.app["pixabay_api_keys"] = ["pixabay-key"]
        config.app["coverr_api_keys"] = ["coverr-key"]
        config.proxy.clear()
        pixabay_response = SimpleNamespace(
            status_code=200,
            headers={"content-type": "application/json"},
            text="",
            json=lambda: {
                "hits": [
                    {
                        "id": 1,
                        "duration": 8,
                        "videos": {
                            "large": {
                                "width": 1920,
                                "height": 1080,
                                "url": "https://example.com/pixabay-landscape.mp4",
                            }
                        },
                    }
                ]
            },
        )
        coverr_response = SimpleNamespace(
            json=lambda: {
                "hits": [
                    {
                        "id": "landscape",
                        "duration": 8,
                        "max_width": 1920,
                        "max_height": 1080,
                        "urls": {
                            "mp4_download": "https://example.com/coverr-landscape.mp4"
                        },
                    }
                ]
            }
        )

        with patch(
            "app.services.material.requests.get",
            return_value=pixabay_response,
        ):
            pixabay_results = material.search_videos_pixabay(
                "city",
                minimum_duration=1,
                video_aspect=material.VideoAspect.square,
            )
        with patch(
            "app.services.material.requests.get",
            return_value=coverr_response,
        ):
            coverr_results = material.search_videos_coverr(
                "city",
                minimum_duration=1,
                video_aspect=material.VideoAspect.square,
            )

        self.assertEqual(
            [item.url for item in pixabay_results],
            ["https://example.com/pixabay-landscape.mp4"],
        )
        self.assertEqual(
            [item.url for item in coverr_results],
            ["https://example.com/coverr-landscape.mp4"],
        )

    def test_search_pixabay_does_not_log_api_key(self):
        config.app["pixabay_api_keys"] = ["pixabay-secret-key"]
        config.proxy.clear()

        fake_response = SimpleNamespace(
            status_code=200,
            headers={"content-type": "application/json"},
            text="",
            json=lambda: {"hits": []},
        )

        with (
            patch("app.services.material.requests.get", return_value=fake_response),
            patch("app.services.material.logger.info") as log,
        ):
            material.search_videos_pixabay("cat", minimum_duration=1)

        logged_messages = " ".join(str(call.args[0]) for call in log.call_args_list)
        self.assertNotIn("pixabay-secret-key", logged_messages)

    def test_search_pixabay_reports_cloudflare_challenge(self):
        """
        Cloudflare Challenge 返回的是 HTML，不是 Pixabay API 的 JSON。
        应直接说明服务端拦截原因，避免用户只看到没有上下文的 JSON 解析错误。
        """
        config.app["pixabay_api_keys"] = ["pixabay-secret-key"]
        config.proxy.clear()

        fake_response = SimpleNamespace(
            status_code=429,
            headers={
                "content-type": "text/html; charset=UTF-8",
                "cf-mitigated": "challenge",
                "cf-ray": "test-ray",
            },
            text="<html><title>Just a moment...</title></html>",
        )

        with (
            patch("app.services.material.requests.get", return_value=fake_response),
            patch("app.services.material.logger.error") as log,
        ):
            results = material.search_videos_pixabay("nature", minimum_duration=1)

        logged_messages = " ".join(str(call.args[0]) for call in log.call_args_list)
        self.assertEqual(results, [])
        self.assertIn("Cloudflare challenge", logged_messages)
        self.assertIn("cf_ray=test-ray", logged_messages)
        self.assertNotIn("pixabay-secret-key", logged_messages)
        self.assertNotIn("Just a moment", logged_messages)

    def test_search_pixabay_reports_api_rate_limit(self):
        """
        Pixabay 自身的 429 限流与 Cloudflare HTML Challenge 是不同问题。
        保留 Retry-After 可以帮助用户判断何时重试，同时不记录响应正文。
        """
        config.app["pixabay_api_keys"] = ["pixabay-key"]
        config.proxy.clear()

        fake_response = SimpleNamespace(
            status_code=429,
            headers={
                "content-type": "text/plain; charset=UTF-8",
                "retry-after": "60",
            },
            text="API rate limit exceeded",
        )

        with (
            patch("app.services.material.requests.get", return_value=fake_response),
            patch("app.services.material.logger.error") as log,
        ):
            results = material.search_videos_pixabay("nature", minimum_duration=1)

        logged_messages = " ".join(str(call.args[0]) for call in log.call_args_list)
        self.assertEqual(results, [])
        self.assertIn("API rate limit exceeded", logged_messages)
        self.assertIn("retry_after=60", logged_messages)

    def test_search_pixabay_reports_non_json_response(self):
        """
        即使状态码为 200，上游代理也可能返回登录页或其他非 JSON 内容。
        该场景应记录响应类型，而不是向外暴露底层 JSONDecodeError。
        """
        config.app["pixabay_api_keys"] = ["pixabay-key"]
        config.proxy.clear()

        def raise_invalid_json():
            raise ValueError("Expecting value: line 1 column 1")

        fake_response = SimpleNamespace(
            status_code=200,
            headers={"content-type": "text/plain"},
            text="unexpected response",
            json=raise_invalid_json,
        )

        with (
            patch("app.services.material.requests.get", return_value=fake_response),
            patch("app.services.material.logger.error") as log,
        ):
            results = material.search_videos_pixabay("nature", minimum_duration=1)

        logged_messages = " ".join(str(call.args[0]) for call in log.call_args_list)
        self.assertEqual(results, [])
        self.assertIn("unexpected non-JSON response", logged_messages)
        self.assertNotIn("Expecting value", logged_messages)

    def test_search_pixabay_redacts_api_key_from_network_error(self):
        """
        requests 的连接异常可能回显完整请求 URL。异常详情仍应保留用于排查，
        但 URL 查询参数中的 Pixabay API Key 必须在写入日志前脱敏。
        """
        api_key = "pixabay-secret-key"
        config.app["pixabay_api_keys"] = [api_key]
        config.proxy.clear()
        error = requests.ConnectionError(
            f"request failed for https://pixabay.com/api/videos/?q=nature&key={api_key}"
        )

        with (
            patch("app.services.material.requests.get", side_effect=error),
            patch("app.services.material.logger.error") as log,
        ):
            results = material.search_videos_pixabay("nature", minimum_duration=1)

        logged_messages = " ".join(str(call.args[0]) for call in log.call_args_list)
        self.assertEqual(results, [])
        self.assertIn("ConnectionError", logged_messages)
        self.assertIn("key=***", logged_messages)
        self.assertNotIn(api_key, logged_messages)

    def test_search_pixabay_redacts_proxy_credentials_from_network_error(self):
        """
        代理连接异常可能回显含认证信息的完整代理 URL。日志应保留异常类型，
        但不能把代理用户名和密码持久化到日志文件。
        """
        proxy_url = "http://proxy-user:proxy-password@proxy.example.com:8080"
        config.app["pixabay_api_keys"] = ["pixabay-key"]
        config.proxy.clear()
        config.proxy["http"] = proxy_url
        error = requests.exceptions.ProxyError(
            f"failed to connect to proxy {proxy_url}"
        )

        with (
            patch("app.services.material.requests.get", side_effect=error),
            patch("app.services.material.logger.error") as log,
        ):
            results = material.search_videos_pixabay("nature", minimum_duration=1)

        logged_messages = " ".join(str(call.args[0]) for call in log.call_args_list)
        self.assertEqual(results, [])
        self.assertIn("ProxyError", logged_messages)
        self.assertNotIn("proxy-user", logged_messages)
        self.assertNotIn("proxy-password", logged_messages)

    def test_save_video_uses_tls_verification_by_default(self):
        config.app.pop("tls_verify", None)
        config.proxy.clear()

        fake_response = SimpleNamespace(
            content=b"fake-video",
            status_code=200,
            iter_content=lambda chunk_size=8192: [b"fake-video"],
        )

        class FakeVideoFileClip:
            duration = 1
            fps = 24

            def __init__(self, path):
                self.path = path

            def close(self):
                return None

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch(
                    "app.services.material.requests.get", return_value=fake_response
                ) as get,
                patch("app.services.material.VideoFileClip", FakeVideoFileClip),
            ):
                video_path = material.save_video(
                    "https://example.com/video.mp4?token=abc", save_dir=temp_dir
                )

            self.assertTrue(os.path.exists(video_path))
            self.assertTrue(get.call_args.kwargs["verify"])

    def test_save_video_cleans_up_empty_file_on_200_response(self):
        """When a 200 response returns empty content (0 bytes), the file should be deleted and not left behind."""
        config.app.pop("tls_verify", None)
        config.proxy.clear()

        fake_response = SimpleNamespace(
            content=b"",
            status_code=200,
            iter_content=lambda chunk_size=8192: [],
        )

        class FakeVideoFileClip:
            duration = 1
            fps = 24

            def __init__(self, path):
                self.path = path

            def close(self):
                return None

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch(
                    "app.services.material.requests.get", return_value=fake_response
                ) as get,
                patch("app.services.material.VideoFileClip", FakeVideoFileClip),
            ):
                video_path = material.save_video(
                    "https://example.com/video.mp4?token=abc", save_dir=temp_dir
                )

            self.assertEqual(video_path, "")
            self.assertFalse(os.path.exists(video_path))
            self.assertEqual(os.listdir(temp_dir), [])

    def test_download_videos_accepts_plain_string_concat_mode(self):
        """
        download_videos 可能被服务层或测试直接传入字符串模式，而不是
        VideoConcatMode 枚举。这里用空搜索词避免真实网络请求，只验证
        字符串 "random" 不会再因为访问 `.value` 抛 AttributeError。
        """
        result = material.download_videos(
            task_id="string-concat-mode",
            search_terms=[],
            video_concat_mode="random",
        )

        self.assertEqual(result, [])

    def test_material_source_record_uses_public_whitelist(self):
        """
        任务清单只应包含可追溯的公开字段，不能写入签名参数、下载地址、
        调用方传入的额外字段或本机绝对路径。
        """
        item = material.MaterialInfo(
            provider="pixabay",
            url="https://cdn.example.com/video.mp4?token=secret",
            duration=12,
            source_info={
                "provider": "pixabay",
                "search_term": "city",
                "asset_id": 123,
                "source_page": "https://pixabay.com/videos/city-123/?key=secret",
                "creator": {
                    "id": 456,
                    "name": "Creator",
                    "profile_page": "https://pixabay.com/users/creator/?token=secret",
                    "email": "private@example.com",
                },
                "rendition": {
                    "id": "large",
                    "width": 1920,
                    "height": 1080,
                    "download_url": "https://cdn.example.com/private",
                },
                "api_key": "must-not-persist",
            },
        )

        record = material._material_source_record(
            item,
            "/Users/example/private/task/vid-123.mp4",
        )
        serialized = str(record)

        self.assertEqual(record["local_file"], "vid-123.mp4")
        self.assertEqual(
            record["source_page"],
            "https://pixabay.com/videos/city-123/",
        )
        self.assertEqual(
            record["creator"]["profile_page"],
            "https://pixabay.com/users/creator/",
        )
        self.assertEqual(
            record["rendition"],
            {"id": "large", "width": 1920, "height": 1080},
        )
        self.assertNotIn("secret", serialized)
        self.assertNotIn("/Users/example", serialized)
        self.assertNotIn("private@example.com", serialized)

    def test_download_videos_can_round_robin_terms_in_script_order(self):
        """
        开启按文案顺序匹配素材后，不能让第一个关键词的多个候选先把
        音频时长填满。这里模拟两个关键词各有多个候选，验证下载顺序是
        term1-第1个、term2-第1个、term1-第2个，贴近脚本叙事顺序。
        """
        search_results = {
            "opening city": [
                material.MaterialInfo(
                    provider="pexels",
                    url="https://v.example/a1.mp4",
                    duration=3,
                    source_info={
                        "provider": "pexels",
                        "search_term": "opening city",
                        "asset_id": "a1",
                    },
                ),
                material.MaterialInfo(
                    provider="pexels",
                    url="https://v.example/a2.mp4",
                    duration=3,
                    source_info={
                        "provider": "pexels",
                        "search_term": "opening city",
                        "asset_id": "a2",
                    },
                ),
            ],
            "middle office": [
                material.MaterialInfo(
                    provider="pexels",
                    url="https://v.example/b1.mp4",
                    duration=3,
                    source_info={
                        "provider": "pexels",
                        "search_term": "middle office",
                        "asset_id": "b1",
                    },
                ),
                material.MaterialInfo(
                    provider="pexels",
                    url="https://v.example/b2.mp4",
                    duration=3,
                    source_info={
                        "provider": "pexels",
                        "search_term": "middle office",
                        "asset_id": "b2",
                    },
                ),
            ],
        }
        downloaded_urls = []

        def fake_search(search_term, minimum_duration, video_aspect):
            return search_results[search_term]

        def fake_save_video(video_url, save_dir=""):
            downloaded_urls.append(video_url)
            return f"/tmp/{video_url.rsplit('/', 1)[-1]}"

        with (
            patch.dict(config.app, {"material_directory": ""}),
            patch.object(material, "search_videos_pexels", side_effect=fake_search),
            patch.object(material, "save_video", side_effect=fake_save_video),
            patch.object(
                material.material_cache,
                "load_material_search_cache",
                return_value=None,
            ),
            patch.object(material.material_cache, "save_material_search_cache"),
            patch.object(
                material.task_artifacts,
                "patch_script_data",
                return_value=True,
            ) as patch_script,
        ):
            result = material.download_videos(
                task_id="ordered-materials",
                search_terms=["opening city", "middle office"],
                source="pexels",
                audio_duration=7,
                max_clip_duration=3,
                match_script_order=True,
            )

        self.assertEqual(
            downloaded_urls,
            [
                "https://v.example/a1.mp4",
                "https://v.example/b1.mp4",
                "https://v.example/a2.mp4",
            ],
        )
        self.assertEqual(result, ["/tmp/a1.mp4", "/tmp/b1.mp4", "/tmp/a2.mp4"])
        recorded_sources = patch_script.call_args.kwargs["material_sources"]
        self.assertEqual(
            [source["asset_id"] for source in recorded_sources],
            ["a1", "b1", "a2"],
        )
        self.assertEqual(
            [source["local_file"] for source in recorded_sources],
            ["a1.mp4", "b1.mp4", "a2.mp4"],
        )

    def test_material_source_persistence_failure_does_not_break_download(self):
        """辅助任务记录失败时，已经下载成功的素材仍应正常返回给成片主流程。"""
        item = material.MaterialInfo(
            provider="pexels",
            url="https://v.example/a1.mp4",
            duration=5,
            source_info={"provider": "pexels", "asset_id": "a1"},
        )

        with (
            patch.dict(config.app, {"material_directory": ""}),
            patch.object(material, "search_videos_pexels", return_value=[item]),
            patch.object(material, "save_video", return_value="/tmp/a1.mp4"),
            patch.object(
                material.material_cache,
                "load_material_search_cache",
                return_value=None,
            ),
            patch.object(material.material_cache, "save_material_search_cache"),
            patch.object(
                material.task_artifacts,
                "patch_script_data",
                side_effect=OSError("disk unavailable"),
            ),
            patch.object(material.logger, "warning") as warning,
        ):
            result = material.download_videos(
                task_id="persist-failure",
                search_terms=["city"],
                source="pexels",
                audio_duration=1,
                max_clip_duration=5,
            )

        self.assertEqual(result, ["/tmp/a1.mp4"])
        self.assertTrue(warning.called)


class TestScrapedVideoValidation(unittest.TestCase):
    """
    yt-dlp 返回成功不代表容器完好；_validate_scraped_video 必须在下载阶段
    就用 VideoFileClip 拦截损坏文件，避免最终渲染时整条任务炸掉。
    """

    class _ValidClip:
        duration = 3.0
        fps = 24.0

        def __init__(self, path):
            self.path = path

        def close(self):
            return None

    class _BrokenClip:
        def __init__(self, path):
            raise ValueError("not a video container")

        def close(self):
            return None

    class _ZeroClip:
        duration = 0.0
        fps = 0.0

        def __init__(self, path):
            self.path = path

        def close(self):
            return None

    def test_valid_container_returns_path_untouched(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = os.path.join(temp_dir, "scraped.mp4")
            Path(video_path).write_bytes(b"fake-but-probable-video")
            with patch("app.services.material.VideoFileClip", self._ValidClip):
                result = material._validate_scraped_video(video_path)

        self.assertEqual(result, video_path)

    def test_broken_container_returns_empty_and_removes_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = os.path.join(temp_dir, "scraped.mp4")
            Path(video_path).write_bytes(b"not really a video")
            with (
                patch("app.services.material.VideoFileClip", self._BrokenClip),
                patch.object(material.logger, "warning") as warning,
            ):
                result = material._validate_scraped_video(video_path)

            self.assertEqual(result, "")
            self.assertFalse(os.path.exists(video_path))
            self.assertTrue(warning.called)

    def test_zero_duration_container_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = os.path.join(temp_dir, "scraped.mp4")
            Path(video_path).write_bytes(b"empty-ish video")
            with patch("app.services.material.VideoFileClip", self._ZeroClip):
                result = material._validate_scraped_video(video_path)

            self.assertEqual(result, "")
            self.assertFalse(os.path.exists(video_path))

    def test_missing_or_empty_file_returns_empty(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing = os.path.join(temp_dir, "does-not-exist.mp4")
            self.assertEqual(material._validate_scraped_video(missing), "")

            empty = os.path.join(temp_dir, "empty.mp4")
            Path(empty).write_bytes(b"")
            self.assertEqual(material._validate_scraped_video(empty), "")

    def test_download_videos_skips_corrupt_scraped_download(self):
        """
        下载阶段拦截：download_web_video 返回 True 但文件容器损坏时，
        素材必须被丢弃而不是带着坏文件进入渲染阶段。
        """
        item = material.MaterialInfo(
            provider="web_scrape",
            url="https://example.com/video",
            duration=5,
            source_info={"provider": "web_scrape", "asset_id": "scrape-1"},
        )

        with tempfile.TemporaryDirectory() as temp_dir:

            def fake_search(search_term, minimum_duration, video_aspect, failed_providers=None):
                return [item]

            def fake_download_web_video(url, output_path):
                # yt-dlp 声称成功：确实写出了文件，但内容是不可解码的容器。
                Path(output_path).write_bytes(b"corrupt container content")
                return True

            with (
                patch.dict(config.app, {"material_directory": temp_dir}),
                patch.object(
                    material,
                    "_search_videos_auto_all_sources",
                    side_effect=fake_search,
                ),
                patch.object(
                    material.material_cache,
                    "load_material_search_cache",
                    return_value=None,
                ),
                patch.object(material.material_cache, "save_material_search_cache"),
                patch.object(
                    material.task_artifacts,
                    "patch_script_data",
                    return_value=True,
                ),
                patch("app.services.material.VideoFileClip", self._BrokenClip),
                patch(
                    "app.services.web_scrape.download_web_video",
                    side_effect=fake_download_web_video,
                ),
            ):
                result = material.download_videos(
                    task_id="scraped-corrupt",
                    search_terms=["nature"],
                    source="auto",
                    audio_duration=10,
                    max_clip_duration=5,
                )

            self.assertEqual(result, [])
            # 损坏文件已被探测逻辑清理，目录中不应残留脏文件。
            self.assertEqual(os.listdir(temp_dir), [])

    def test_download_videos_keeps_valid_scraped_download(self):
        """容器探测通过时，web_scrape 素材正常进入成片素材列表。"""
        item = material.MaterialInfo(
            provider="web_scrape",
            url="https://example.com/video",
            duration=5,
            source_info={"provider": "web_scrape", "asset_id": "scrape-2"},
        )

        with tempfile.TemporaryDirectory() as temp_dir:

            def fake_search(search_term, minimum_duration, video_aspect, failed_providers=None):
                return [item]

            def fake_download_web_video(url, output_path):
                Path(output_path).write_bytes(b"valid video bytes")
                return True

            with (
                patch.dict(config.app, {"material_directory": temp_dir}),
                patch.object(
                    material,
                    "_search_videos_auto_all_sources",
                    side_effect=fake_search,
                ),
                patch.object(
                    material.material_cache,
                    "load_material_search_cache",
                    return_value=None,
                ),
                patch.object(material.material_cache, "save_material_search_cache"),
                patch.object(
                    material.task_artifacts,
                    "patch_script_data",
                    return_value=True,
                ),
                patch("app.services.material.VideoFileClip", self._ValidClip),
                patch(
                    "app.services.web_scrape.download_web_video",
                    side_effect=fake_download_web_video,
                ),
            ):
                result = material.download_videos(
                    task_id="scraped-valid",
                    search_terms=["nature"],
                    source="auto",
                    audio_duration=10,
                    max_clip_duration=5,
                )

            self.assertEqual(len(result), 1)
            self.assertTrue(result[0].endswith(".mp4"))
            self.assertEqual(len(os.listdir(temp_dir)), 1)


class TestSkipFailedProviders(unittest.TestCase):
    """F2：auto 模式中抛异常的供应商被记录，后续搜索词不再查询它。"""

    def setUp(self):
        self.original_app_config = dict(config.app)
        for key in (
            "custom_api_url",
            "custom_api_key",
            "pexels_api_keys",
            "pixabay_api_keys",
            "coverr_api_keys",
            "enable_web_scraping",
            "enable_pollinations",
            "auto_providers",
        ):
            config.app.pop(key, None)

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)

    def test_failed_provider_skipped_for_later_terms(self):
        calls = {"pexels": 0, "pixabay": 0}

        def flaky_pexels(search_term, minimum_duration, video_aspect):
            calls["pexels"] += 1
            raise RuntimeError("pexels down")

        def good_pixabay(search_term, minimum_duration, video_aspect):
            calls["pixabay"] += 1
            return [
                material.MaterialInfo(
                    provider="pixabay",
                    url=f"https://v.example/{search_term}.mp4",
                    duration=5,
                    source_info={
                        "provider": "pixabay",
                        "search_term": search_term,
                        "asset_id": search_term,
                    },
                )
            ]

        with (
            patch.object(material, "search_videos_pexels", side_effect=flaky_pexels),
            patch.object(
                material, "search_videos_pixabay", side_effect=good_pixabay
            ),
            patch.dict(
                config.app,
                {
                    "pexels_api_keys": ["pexels-key"],
                    "pixabay_api_keys": ["pixabay-key"],
                },
            ),
            patch.object(
                material.material_cache,
                "load_material_search_cache",
                return_value=None,
            ),
            patch.object(material.material_cache, "save_material_search_cache"),
        ):
            failed: set[str] = set()
            first = material._search_videos_auto_all_sources(
                "term one", 5, material.VideoAspect.portrait, failed
            )
            second = material._search_videos_auto_all_sources(
                "term two", 5, material.VideoAspect.portrait, failed
            )

        self.assertIn("pexels", failed)
        self.assertEqual(calls["pexels"], 1)  # 只被查询一次，后续关键词跳过
        self.assertEqual(calls["pixabay"], 2)
        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 1)

    def test_empty_results_are_not_treated_as_failure(self):
        """正常返回空结果不算失败：供应商可能只是该关键词没有素材。"""
        calls = {"pexels": 0}

        def empty_pexels(search_term, minimum_duration, video_aspect):
            calls["pexels"] += 1
            return []

        with (
            patch.object(material, "search_videos_pexels", side_effect=empty_pexels),
            patch.dict(config.app, {"pexels_api_keys": ["pexels-key"]}),
            patch.object(
                material.material_cache,
                "load_material_search_cache",
                return_value=None,
            ),
            patch.object(material.material_cache, "save_material_search_cache"),
        ):
            failed: set[str] = set()
            material._search_videos_auto_all_sources(
                "term one", 5, material.VideoAspect.portrait, failed
            )
            material._search_videos_auto_all_sources(
                "term two", 5, material.VideoAspect.portrait, failed
            )

        self.assertNotIn("pexels", failed)
        self.assertEqual(calls["pexels"], 2)


class TestRankAndSelectBestMaterial(unittest.TestCase):
    """auto 来源的相关性排序：优先选择与脚本搜索词更贴合的素材。"""

    def _item(self, provider, duration=5, **source_info):
        si = {"provider": provider, "search_term": "nature landscape"}
        si.update(source_info)
        return material.MaterialInfo(
            provider=provider,
            url=f"https://example.com/{provider}",
            duration=duration,
            source_info=si,
        )

    def test_relevance_beats_source_priority(self):
        """更贴合搜索词的素材应胜过来源优先级更高但完全不符的素材。"""
        pexels = self._item(
            "pexels",
            title="city night traffic timelapse",
        )
        pixabay = self._item(
            "pixabay",
            tags="nature, landscape, forest, aerial",
        )
        ranked = material._rank_and_select_best_material(
            [pexels, pixabay], required_duration=5, video_aspect=material.VideoAspect.portrait
        )
        self.assertEqual(ranked[0], pixabay)

    def test_better_match_wins_within_same_source(self):
        """同一来源内，标题/描述与搜索词重合更多的素材排在最前。"""
        exact = self._item(
            "pexels",
            title="nature landscape mountain river",
        )
        partial = self._item(
            "pexels",
            title="nature forest",
        )
        unrelated = self._item(
            "pexels",
            title="business office desk",
        )
        ranked = material._rank_and_select_best_material(
            [unrelated, exact, partial],
            required_duration=5,
            video_aspect=material.VideoAspect.portrait,
        )
        self.assertEqual(ranked[0], exact)
        self.assertEqual(ranked[1], partial)
        self.assertEqual(ranked[2], unrelated)

    def test_missing_metadata_keeps_source_priority_order(self):
        """无文本元数据时保持中性，排序回退到来源优先级。"""
        pexels = self._item("pexels")
        pixabay = self._item("pixabay")
        coverr = self._item("coverr")
        ranked = material._rank_and_select_best_material(
            [coverr, pixabay, pexels],
            required_duration=5,
            video_aspect=material.VideoAspect.portrait,
        )
        self.assertEqual(
            [item.provider for item in ranked], ["pexels", "pixabay", "coverr"]
        )

    def test_duration_is_tiebreak_within_equal_relevance(self):
        """贴合度相同时，时长更接近目标的素材优先。"""
        close = self._item("pexels", duration=5, title="nature landscape")
        far = self._item("pexels", duration=20, title="nature landscape")
        ranked = material._rank_and_select_best_material(
            [far, close], required_duration=5, video_aspect=material.VideoAspect.portrait
        )
        self.assertEqual(ranked[0], close)

    def _aspect_item(self, provider, resolution):
        return material.MaterialInfo(
            provider=provider,
            url=f"https://example.com/{provider}-{resolution}",
            duration=5,
            resolution=resolution,
            source_info={
                "provider": provider,
                "search_term": "nature landscape",
                "title": "nature landscape",
            },
        )

    def test_orientation_match_breaks_ties_within_equal_resolution(self):
        """贴合度和像素相同时，方向与目标画幅一致的素材优先，避免成片黑边。"""
        portrait = self._aspect_item("pexels", "1080x1920")
        landscape = self._aspect_item("pexels", "1920x1080")
        ranked = material._rank_and_select_best_material(
            [landscape, portrait],
            required_duration=5,
            video_aspect=material.VideoAspect.portrait,
        )
        self.assertEqual(ranked[0], portrait)
        self.assertEqual(ranked[1], landscape)

    def test_resolution_still_beats_orientation_mismatch(self):
        """方向惩罚是 tiebreak：4K 横屏素材仍应胜过 720p 竖屏素材。"""
        landscape_4k = self._aspect_item("pexels", "3840x2160")
        portrait_720p = self._aspect_item("pexels", "720x1280")
        ranked = material._rank_and_select_best_material(
            [portrait_720p, landscape_4k],
            required_duration=5,
            video_aspect=material.VideoAspect.portrait,
        )
        self.assertEqual(ranked[0], landscape_4k)
        self.assertEqual(ranked[1], portrait_720p)

    def test_unknown_orientation_keeps_hd_advantage(self):
        """未知方向保持中性；已知方向的 1080p 横屏仍胜过无尺寸素材。"""
        landscape_hd = self._aspect_item("pexels", "1920x1080")
        unknown = self._item("pexels")
        ranked = material._rank_and_select_best_material(
            [unknown, landscape_hd],
            required_duration=5,
            video_aspect=material.VideoAspect.portrait,
        )
        self.assertEqual(ranked[0], landscape_hd)


class TestAutoProviderSelection(unittest.TestCase):
    """auto 来源只查询已完整配置的供应商，缺 Key 的来源不应被调用。"""

    _MATERIAL_KEYS = (
        "custom_api_url",
        "custom_api_key",
        "pexels_api_keys",
        "pixabay_api_keys",
        "coverr_api_keys",
        "enable_web_scraping",
        "enable_pollinations",
        "auto_providers",
    )

    def setUp(self):
        self.original_app_config = dict(config.app)
        for key in self._MATERIAL_KEYS:
            config.app.pop(key, None)

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)

    def _provider_names(self):
        return [name for name, _ in material._auto_provider_configs()]

    def test_custom_api_alone_is_enough(self):
        """只配置 custom_api（如 Google Veo）时，auto 只使用它。"""
        config.app["custom_api_url"] = "https://veo.example/v1"
        config.app["custom_api_key"] = "secret"
        self.assertEqual(self._provider_names(), ["custom_api"])

    def test_custom_api_without_key_is_skipped(self):
        """custom_api 缺 Key 时不参与 auto，即使 URL 已配置（仅剩免费模式）。"""
        config.app["custom_api_url"] = "https://veo.example/v1"
        self.assertEqual(self._provider_names(), ["pollinations"])

    def test_only_one_stock_key_enables_that_provider(self):
        """只提供 Pexels Key 时，auto 只使用 Pexels，不会调用未配置的来源。"""
        config.app["pexels_api_keys"] = ["pexels-key"]
        self.assertEqual(self._provider_names(), ["pexels"])

    def test_coverr_without_key_is_skipped(self):
        """Coverr 未配置 Key 时不再被无条件纳入 auto。"""
        config.app["pexels_api_keys"] = ["pexels-key"]
        self.assertEqual(self._provider_names(), ["pexels"])

    def test_all_configured_providers_are_included(self):
        """全部配置时按 custom_api → pexels → pixabay → coverr → web_scrape 顺序。"""
        config.app["custom_api_url"] = "https://veo.example/v1"
        config.app["custom_api_key"] = "secret"
        config.app["pexels_api_keys"] = ["pexels-key"]
        config.app["pixabay_api_keys"] = ["pixabay-key"]
        config.app["coverr_api_keys"] = ["coverr-key"]
        config.app["enable_web_scraping"] = True
        self.assertEqual(
            self._provider_names(),
            ["custom_api", "pexels", "pixabay", "coverr", "web_scrape"],
        )

    def test_no_provider_returns_free_mode_pollinations(self):
        """没有任何配置时返回 Pollinations：免费模式零 Key 也能出片。"""
        self.assertEqual(self._provider_names(), ["pollinations"])

    def test_pollinations_disabled_explicitly_returns_empty(self):
        """显式关闭 Pollinations 且无其他供应商时，auto 返回空列表。"""
        config.app["enable_pollinations"] = False
        self.assertEqual(self._provider_names(), [])

    def test_pollinations_enabled_explicitly_with_stock_keys(self):
        """显式开启 Pollinations 后，它作为最低优先级供应商加入 auto。"""
        config.app["enable_pollinations"] = True
        config.app["pexels_api_keys"] = ["pexels-key"]
        self.assertEqual(
            self._provider_names(), ["pexels", "pollinations"]
        )

    def test_auto_providers_filter_keeps_canonical_order(self):
        """F1：auto_providers 只决定参与来源，顺序仍按内置优先级。"""
        config.app["custom_api_url"] = "https://veo.example/v1"
        config.app["custom_api_key"] = "secret"
        config.app["pexels_api_keys"] = ["pexels-key"]
        config.app["pixabay_api_keys"] = ["pixabay-key"]
        config.app["coverr_api_keys"] = ["coverr-key"]
        config.app["enable_web_scraping"] = True
        # 用户只选 pexels 和 custom_api，且故意倒序传入。
        config.app["auto_providers"] = ["pexels", "custom_api"]
        self.assertEqual(self._provider_names(), ["custom_api", "pexels"])

    def test_auto_providers_filter_drops_unconfigured_providers(self):
        """F1：选中但未配置密钥的供应商（如 coverr）不会被强制加入。"""
        config.app["pexels_api_keys"] = ["pexels-key"]
        config.app["auto_providers"] = ["pexels", "coverr", "web_scrape"]
        self.assertEqual(self._provider_names(), ["pexels"])

    def test_auto_providers_empty_list_means_all(self):
        """auto_providers 为空列表时回退到默认行为（全部可用供应商）。"""
        config.app["pexels_api_keys"] = ["pexels-key"]
        config.app["auto_providers"] = []
        self.assertEqual(self._provider_names(), ["pexels"])


class TestCoverrProvider(unittest.TestCase):
    """
    Coverr 视频素材源(spec: 2026-06-09-coverr-video-provider-design.md)。
    全部用 unittest.mock 替换 requests，确保 CI 不依赖真实网络和真实 API key。
    """

    def setUp(self):
        self.original_app_config = dict(config.app)
        self.original_proxy_config = dict(config.proxy)

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)
        config.proxy.clear()
        config.proxy.update(self.original_proxy_config)

    # ---------------- Tests for search_videos_coverr ----------------

    def test_search_coverr_uses_mp4_download_url(self):
        """
        search_videos_coverr 应把每个 hit 转成 MaterialInfo，并把 urls.mp4_download
        直接作为 MaterialInfo.url。
        按 Coverr 官方文档 (api.coverr.co/docs/videos/#download-a-video),
        GET mp4_download 本身就被 Coverr 计入下载统计,无需额外 PATCH ping。
        同时验证 Authorization header 使用 Bearer scheme。
        """
        config.app["coverr_api_keys"] = ["coverr-key"]
        config.app.pop("tls_verify", None)
        config.proxy.clear()

        fake_response = SimpleNamespace(
            json=lambda: {
                "page": 0,
                "pages": 50,
                "page_size": 20,
                "total": 1,
                "hits": [
                    {
                        "id": "S1YbPl1NfI",
                        "duration": 11.625,
                        "aspect_ratio": "16:9",
                        "canonical_url": "https://coverr.co/videos/example?token=drop",
                        "creator": {
                            "id": "creator-1",
                            "name": "Coverr Creator",
                            "profile_url": "https://coverr.co/creators/example?key=drop",
                        },
                        "max_width": 3840,
                        "max_height": 2160,
                        "urls": {
                            "mp4": "https://storage.coverr.co/videos/abc?token=xyz",
                            "mp4_preview": "https://storage.coverr.co/videos/abc/preview?token=xyz",
                            "mp4_download": "https://storage.coverr.co/videos/abc/download?token=xyz",
                        },
                    }
                ],
            }
        )

        with patch(
            "app.services.material.requests.get", return_value=fake_response
        ) as get:
            results = material.search_videos_coverr(
                "nature",
                minimum_duration=5,
                video_aspect=material.VideoAspect.landscape,
            )

        self.assertEqual(len(results), 1)
        item = results[0]
        self.assertEqual(item.provider, "coverr")
        self.assertEqual(item.duration, 11)
        # url 字段就是 mp4_download URL,不再做 coverr://id|url 编码
        self.assertEqual(
            item.url, "https://storage.coverr.co/videos/abc/download?token=xyz"
        )
        self.assertEqual(item.source_info["asset_id"], "S1YbPl1NfI")
        self.assertEqual(
            item.source_info["source_page"],
            "https://coverr.co/videos/example",
        )
        self.assertEqual(
            item.source_info["creator"]["profile_page"],
            "https://coverr.co/creators/example",
        )
        # Bearer auth + TLS verify on by default
        self.assertEqual(
            get.call_args.kwargs["headers"]["Authorization"], "Bearer coverr-key"
        )
        self.assertTrue(get.call_args.kwargs["verify"])

    def test_search_coverr_uses_tls_verification_by_default(self):
        """与 pexels/pixabay 一致:未显式配置时 TLS 校验默认开启。"""
        config.app["coverr_api_keys"] = ["coverr-key"]
        config.app.pop("tls_verify", None)
        config.proxy.clear()

        fake_response = SimpleNamespace(json=lambda: {"hits": []})

        with patch(
            "app.services.material.requests.get", return_value=fake_response
        ) as get:
            material.search_videos_coverr("nature", minimum_duration=1)

        self.assertTrue(get.call_args.kwargs["verify"])

    def test_search_coverr_allows_explicit_tls_disable_for_proxy(self):
        """企业自签证书代理场景必须能显式关闭 TLS 校验。"""
        config.app["coverr_api_keys"] = ["coverr-key"]
        config.app["tls_verify"] = False
        config.proxy.clear()

        fake_response = SimpleNamespace(json=lambda: {"hits": []})

        with patch(
            "app.services.material.requests.get", return_value=fake_response
        ) as get:
            material.search_videos_coverr("nature", minimum_duration=1)

        self.assertFalse(get.call_args.kwargs["verify"])

    def test_search_coverr_filters_by_min_duration_and_accepts_string(self):
        """
        Coverr duration 字段在不同响应里可能是 number 或 string,
        两种格式都要接受;低于 minimum_duration 的应被过滤。
        """
        config.app["coverr_api_keys"] = ["coverr-key"]
        config.app.pop("tls_verify", None)
        config.proxy.clear()

        fake_response = SimpleNamespace(
            json=lambda: {
                "hits": [
                    {
                        "id": "shortvid",
                        "duration": 3,  # below minimum
                        "urls": {"mp4_download": "https://example.com/a.mp4"},
                    },
                    {
                        "id": "stringdur",
                        "duration": "10.500000",  # string accepted
                        "max_width": 1080,
                        "max_height": 1920,
                        "urls": {"mp4_download": "https://example.com/b.mp4"},
                    },
                ]
            }
        )

        with patch("app.services.material.requests.get", return_value=fake_response):
            results = material.search_videos_coverr("x", minimum_duration=5)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].duration, 10)
        self.assertEqual(results[0].url, "https://example.com/b.mp4")

    def test_search_coverr_skips_invalid_items(self):
        """缺 id 或缺 urls.mp4_download 的条目应被跳过,不应抛异常。"""
        config.app["coverr_api_keys"] = ["coverr-key"]
        config.app.pop("tls_verify", None)
        config.proxy.clear()

        fake_response = SimpleNamespace(
            json=lambda: {
                "hits": [
                    {  # missing urls.mp4_download
                        "id": "no-download",
                        "duration": 10,
                        "urls": {"mp4_preview": "https://example.com/preview.mp4"},
                    },
                    {  # missing id
                        "duration": 10,
                        "urls": {"mp4_download": "https://example.com/x.mp4"},
                    },
                    {  # valid baseline
                        "id": "good",
                        "duration": 10,
                        "max_width": 1080,
                        "max_height": 1920,
                        "urls": {"mp4_download": "https://example.com/good.mp4"},
                    },
                ]
            }
        )

        with patch("app.services.material.requests.get", return_value=fake_response):
            results = material.search_videos_coverr("x", minimum_duration=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].url, "https://example.com/good.mp4")

    def test_search_coverr_returns_empty_on_failure(self):
        """
        响应结构异常 / 网络异常时,函数必须返回 [] 而不是抛异常,
        与 pexels/pixabay 行为保持一致。
        """
        config.app["coverr_api_keys"] = ["coverr-key"]
        config.app.pop("tls_verify", None)
        config.proxy.clear()

        # Subtest A: malformed response (no "hits" key)
        with self.subTest("malformed response"):
            fake_response = SimpleNamespace(json=lambda: {"error": "rate limited"})
            with patch(
                "app.services.material.requests.get", return_value=fake_response
            ):
                results = material.search_videos_coverr("x", minimum_duration=1)
            self.assertEqual(results, [])

        # Subtest B: network exception bubbles up from requests.get
        with self.subTest("network exception"):
            with patch(
                "app.services.material.requests.get",
                side_effect=requests.ConnectionError("boom"),
            ):
                results = material.search_videos_coverr("x", minimum_duration=1)
            self.assertEqual(results, [])

    # ---------------- Tests for download_videos coverr branch ----------------

    def test_download_videos_passes_mp4_download_url_to_save_video(self):
        """
        在 source="coverr" 时:
          1. dispatch 到 search_videos_coverr
          2. coverr item 走通用下载路径:save_video 收到的就是 mp4_download URL
             (不再有 coverr://id|url 编码,也不再调用 PATCH ping)
          3. 返回保存路径
        """
        config.app["coverr_api_keys"] = ["coverr-key"]
        config.app.pop("tls_verify", None)
        config.app.pop("material_directory", None)
        config.proxy.clear()

        fake_item = material.MaterialInfo()
        fake_item.provider = "coverr"
        fake_item.url = "https://storage.coverr.co/videos/abc/download?token=xyz"
        fake_item.duration = 10

        with (
            patch(
                "app.services.material.search_videos_coverr",
                return_value=[fake_item],
            ) as search,
            patch(
                "app.services.material.save_video",
                return_value="/tmp/coverr-saved.mp4",
            ) as save,
            patch(
                "app.services.material.material_cache.load_material_search_cache",
                return_value=None,
            ),
            patch(
                "app.services.material.material_cache.save_material_search_cache",
            ),
        ):
            result = material.download_videos(
                task_id="t-coverr",
                search_terms=["nature"],
                source="coverr",
                audio_duration=5,
                max_clip_duration=5,
            )

        # 1. dispatch
        self.assertEqual(search.call_count, 1)

        # 2. save_video 收到的就是 mp4_download URL,原样传入
        save_url = save.call_args.kwargs.get("video_url") or save.call_args.args[0]
        self.assertEqual(
            save_url, "https://storage.coverr.co/videos/abc/download?token=xyz"
        )

        # 3. 返回值正确
        self.assertEqual(result, ["/tmp/coverr-saved.mp4"])


class TestSaveVideoLocalGeneratedMedia(unittest.TestCase):
    """save_video 对本地生成素材（Gemini 等）的快速通道。"""

    _ONE_PX_PNG = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_existing_local_image_is_returned(self):
        image_path = os.path.join(self.tmp.name, "gemini-image-1.jpg")
        with open(image_path, "wb") as f:
            f.write(self._ONE_PX_PNG)
        result = material.save_video(image_path, save_dir=self.tmp.name)
        self.assertEqual(result, image_path)

    def test_corrupt_local_file_is_rejected(self):
        corrupt_path = os.path.join(self.tmp.name, "gemini-image-bad.jpg")
        with open(corrupt_path, "wb") as f:
            f.write(b"not a real media file at all")
        result = material.save_video(corrupt_path, save_dir=self.tmp.name)
        self.assertEqual(result, "")


class TestRedactVideoUrl(unittest.TestCase):
    """下载地址日志脱敏：不能把签名令牌 / API Key 查询串写入日志。"""

    def test_strips_query_fragment_and_long_token_segments(self):
        self.assertEqual(
            material._redact_video_url(
                "https://storage.coverr.co/videos/abc/eyJhbGciOiJIUzI1NiJ9?signature=SECRET"
            ),
            "https://storage.coverr.co/videos/abc",
        )

    def test_keeps_host_and_first_two_path_segments(self):
        self.assertEqual(
            material._redact_video_url("https://cdn.example.com/a/b/c.mp4?k=1"),
            "https://cdn.example.com/a/b",
        )

    def test_unparseable_url_returns_placeholder(self):
        self.assertEqual(
            material._redact_video_url("https://[invalid"), "<unparseable url>"
        )


if __name__ == "__main__":
    unittest.main()
