import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.services import web_scrape as ws


class _FakePopen:
    """最小 Popen 替身：可模拟超时、产出文件和非零退出码。"""

    def __init__(
        self,
        pid=4242,
        returncode=0,
        timeout_on_first_call=False,
        write_output=None,
    ):
        self.pid = pid
        self.returncode = returncode
        self.write_output = write_output
        self._timeout_pending = timeout_on_first_call
        self.wait_calls = 0

    def poll(self):
        return None

    def communicate(self, timeout=None):
        if self._timeout_pending:
            self._timeout_pending = False
            raise subprocess.TimeoutExpired("yt-dlp", timeout)
        if self.write_output is not None:
            self.write_output()
        return ("", "")

    def wait(self):
        self.wait_calls += 1
        return self.returncode


class _MetadataPopen(_FakePopen):
    """Popen 替身：返回指定的 yt-dlp JSON 输出，或模拟超时。"""

    def __init__(self, stdout_text="", returncode=0, timeout=False):
        super().__init__(returncode=returncode)
        self.stdout_text = stdout_text
        self._timeout = timeout

    def communicate(self, timeout=None):
        if self._timeout:
            raise subprocess.TimeoutExpired("yt-dlp", timeout)
        return (self.stdout_text, "")


class TestFetchPageMetadata(unittest.TestCase):
    def test_extracts_safe_subset(self):
        payload = json.dumps(
            {
                "title": "Tech Channel",
                "description": "Gadget reviews",
                "channel": "Tech",
                "uploader": "Tech",
                "channel_id": "UC123",
                "channel_follower_count": 12345,
                "categories": ["Science & Technology"],
                "tags": ["tech", "reviews"],
                "webpage_url": "https://www.youtube.com/@tech",
                "id": "ignored-field",
            }
        )
        fake = _MetadataPopen(stdout_text=payload + "\n")
        with patch.object(ws.subprocess, "Popen", return_value=fake):
            result = ws.fetch_page_metadata("https://www.youtube.com/@tech")
        self.assertEqual(result["title"], "Tech Channel")
        self.assertEqual(result["description"], "Gadget reviews")
        self.assertEqual(result["channel"], "Tech")
        self.assertEqual(result["channel_id"], "UC123")
        self.assertEqual(result["channel_follower_count"], 12345)
        self.assertEqual(result["categories"], ["Science & Technology"])
        self.assertEqual(result["tags"], ["tech", "reviews"])
        self.assertNotIn("id", result)

    def test_non_zero_exit_returns_empty(self):
        fake = _MetadataPopen(returncode=1)
        with patch.object(ws.subprocess, "Popen", return_value=fake):
            result = ws.fetch_page_metadata("https://x.com/user")
        self.assertEqual(result, {})

    def test_timeout_returns_empty_and_kills_tree(self):
        fake = _MetadataPopen(stdout_text="", timeout=True)
        with (
            patch.object(ws.subprocess, "Popen", return_value=fake),
            patch.object(ws, "_kill_process_tree") as kill,
        ):
            result = ws.fetch_page_metadata("https://x.com/user")
        self.assertEqual(result, {})
        kill.assert_called_once()

    def test_non_http_url_refused_without_subprocess(self):
        with patch.object(ws.subprocess, "Popen") as popen:
            result = ws.fetch_page_metadata("file:///etc/passwd")
        self.assertEqual(result, {})
        popen.assert_not_called()


class TestKillProcessTree(unittest.TestCase):
    def test_posix_kills_entire_process_group(self):
        fake = _FakePopen(pid=4242)

        class _FakeSignal:
            SIGKILL = 9

        with (
            patch.object(ws, "_is_windows", return_value=False),
            # Windows 的 os/signal 模块没有 killpg/SIGKILL，注入替身断言。
            patch.object(ws.os, "killpg", create=True) as killpg,
            patch.object(ws, "signal", _FakeSignal()),
        ):
            ws._kill_process_tree(fake)
        killpg.assert_called_once_with(4242, 9)
        self.assertEqual(fake.wait_calls, 1)

    def test_windows_kills_process_tree_with_taskkill(self):
        fake = _FakePopen(pid=4242)
        with (
            patch.object(ws, "_is_windows", return_value=True),
            patch.object(ws.subprocess, "run") as taskkill,
        ):
            ws._kill_process_tree(fake)
        taskkill.assert_called_once()
        command = taskkill.call_args.args[0]
        self.assertEqual(command[:2], ["taskkill", "/PID"])
        self.assertIn("4242", command)
        self.assertIn("/T", command)
        self.assertIn("/F", command)
        self.assertEqual(fake.wait_calls, 1)

    def test_skips_when_process_has_already_exited(self):
        fake = _FakePopen()
        fake.poll = lambda: 0
        with (
            patch.object(ws, "_is_windows", return_value=True),
            patch.object(ws.subprocess, "run") as taskkill,
        ):
            ws._kill_process_tree(fake)
        taskkill.assert_not_called()


class TestRemovePartialFiles(unittest.TestCase):
    def test_removes_part_ytdl_and_fragments_keeps_final_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = os.path.join(temp_dir, "out.mp4")
            partials = ["out.mp4.part", "out.mp4.ytdl", "out.mp4.f137.mp4.part"]
            for name in partials:
                Path(temp_dir, name).write_bytes(b"partial")
            Path(output).write_bytes(b"final")

            ws._remove_partial_files(output)

            self.assertTrue(os.path.isfile(output))
            for name in partials:
                self.assertFalse(
                    os.path.exists(os.path.join(temp_dir, name)),
                    f"expected {name} to be removed",
                )

    def test_ignores_missing_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = os.path.join(temp_dir, "missing.mp4")
            # 不应抛异常，也不应创建任何文件。
            ws._remove_partial_files(output)
            self.assertEqual(os.listdir(temp_dir), [])


class TestSearchVideosWebScrape(unittest.TestCase):
    def test_timeout_returns_empty_and_kills_process_tree(self):
        fake = _FakePopen(timeout_on_first_call=True)
        with (
            patch.object(ws.subprocess, "Popen", return_value=fake),
            patch.object(ws, "_kill_process_tree") as kill,
        ):
            result = ws.search_videos_web_scrape("cats", 5, ws.VideoAspect.portrait)
        self.assertEqual(result, [])
        kill.assert_called_once_with(fake)


class TestDownloadWebVideo(unittest.TestCase):
    def test_success_returns_true_when_output_file_exists(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = os.path.join(temp_dir, "video.mp4")

            def write_output():
                Path(output).write_bytes(b"video-data")

            fake = _FakePopen(write_output=write_output)
            with (
                patch.object(ws.subprocess, "Popen", return_value=fake),
                patch.object(ws, "_remove_partial_files"),
            ):
                ok = ws.download_web_video("https://example.com/v", output)

            self.assertTrue(ok)

    def test_timeout_returns_false_kills_tree_and_cleans_partials(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = os.path.join(temp_dir, "video.mp4")
            Path(output + ".part").write_bytes(b"partial")
            fake = _FakePopen(timeout_on_first_call=True)

            with (
                patch.object(ws.subprocess, "Popen", return_value=fake),
                patch.object(ws, "_kill_process_tree") as kill,
            ):
                ok = ws.download_web_video("https://example.com/v", output)

            self.assertFalse(ok)
            kill.assert_called_once_with(fake)
            self.assertFalse(os.path.exists(output + ".part"))

    def test_nonzero_exit_returns_false_and_cleans_partials(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = os.path.join(temp_dir, "video.mp4")
            Path(output + ".ytdl").write_bytes(b"meta")
            fake = _FakePopen(returncode=1)

            with patch.object(ws.subprocess, "Popen", return_value=fake):
                ok = ws.download_web_video("https://example.com/v", output)

            self.assertFalse(ok)
            self.assertFalse(os.path.exists(output + ".ytdl"))

    def test_no_output_file_returns_false(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = os.path.join(temp_dir, "video.mp4")
            fake = _FakePopen(returncode=0)

            with (
                patch.object(ws.subprocess, "Popen", return_value=fake),
                patch.object(ws, "_remove_partial_files"),
            ):
                ok = ws.download_web_video("https://example.com/v", output)

            self.assertFalse(ok)

    def test_popen_failure_returns_false_without_raising(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = os.path.join(temp_dir, "video.mp4")
            with patch.object(
                ws.subprocess,
                "Popen",
                side_effect=OSError("yt-dlp not found"),
            ):
                ok = ws.download_web_video("https://example.com/v", output)
            self.assertFalse(ok)

    def test_rejects_non_http_url_without_running_popen(self):
        """file:// 和选项形输入一律拒绝，不触碰子进程（F4 安全护栏）。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output = os.path.join(temp_dir, "video.mp4")
            with (
                patch.object(ws.subprocess, "Popen") as popen,
                patch.object(ws, "_remove_partial_files"),
            ):
                self.assertFalse(
                    ws.download_web_video("file:///etc/passwd", output)
                )
                self.assertFalse(
                    ws.download_web_video("--download-archive=x", output)
                )
                self.assertFalse(ws.download_web_video("", output))
            popen.assert_not_called()

    def test_download_caps_resolution_and_filesize(self):
        """F3/F4：下载命令限制 1080p 与单文件体积，避免低端设备解码 4K。"""
        captured = {}

        def fake_popen(command, **_kwargs):
            captured["command"] = command
            return _FakePopen(write_output=lambda: Path(captured["out"]).write_bytes(b"x"))

        with tempfile.TemporaryDirectory() as temp_dir:
            output = os.path.join(temp_dir, "video.mp4")
            captured["out"] = output
            with (
                patch.object(ws.subprocess, "Popen", side_effect=fake_popen),
                patch.object(ws, "_remove_partial_files"),
            ):
                self.assertTrue(
                    ws.download_web_video("https://example.com/v", output)
                )

        format_arg = captured["command"][captured["command"].index("-f") + 1]
        self.assertIn("height<=1080", format_arg)
        self.assertIn("--max-filesize", captured["command"])
        self.assertEqual(
            captured["command"][
                captured["command"].index("--max-filesize") + 1
            ],
            ws._MAX_FILESIZE,
        )


class TestSearchWebScrapeAspectGate(unittest.TestCase):
    """搜索阶段过滤方向不符或过低清的 web 素材（横屏/360p 不得混入竖屏成片）。"""

    def _run_search(self, payloads):
        class _Popen:
            def __init__(self, *args, **kwargs):
                pass

            def communicate(self, timeout=None):
                return ("\n".join(payloads), "")

        with (
            patch.object(ws.subprocess, "Popen", return_value=_Popen()),
            patch.object(ws, "_search_videos_via_html_search", return_value=[]),
        ):
            return ws.search_videos_web_scrape("cats", 5, ws.VideoAspect.portrait)

    def _payload(self, url="https://youtube.com/watch?v=1", width=720, height=1280):
        return json.dumps(
            {
                "webpage_url": url,
                "duration": 10,
                "width": width,
                "height": height,
                "title": "test clip",
            }
        )

    def test_keeps_portrait_hd_and_drops_landscape(self):
        results = self._run_search(
            [
                self._payload("a", 720, 1280),
                self._payload("b", 1920, 1080),  # landscape -> rejected
            ]
        )
        self.assertEqual([r.url for r in results], ["a"])

    def test_drops_low_resolution_even_when_portrait(self):
        results = self._run_search(
            [
                self._payload("a", 480, 854),  # too low -> rejected
                self._payload("b", 720, 1280),
            ]
        )
        self.assertEqual([r.url for r in results], ["b"])

    def test_drops_results_without_dimensions(self):
        results = self._run_search([self._payload("a", None, None)])
        self.assertEqual(results, [])

    def test_square_target_keeps_square_only(self):
        class _Popen:
            def __init__(self, *args, **kwargs):
                pass

            def communicate(self, timeout=None):
                return (self._payloads, "")

            _payloads = "\n".join(
                [
                    json.dumps(
                        {
                            "webpage_url": "sq",
                            "duration": 10,
                            "width": 1080,
                            "height": 1080,
                            "title": "square",
                        }
                    ),
                    json.dumps(
                        {
                            "webpage_url": "port",
                            "duration": 10,
                            "width": 720,
                            "height": 1280,
                            "title": "portrait",
                        }
                    ),
                ]
            )

        with patch.object(ws.subprocess, "Popen", return_value=_Popen()):
            results = ws.search_videos_web_scrape("cats", 5, ws.VideoAspect.square)
        self.assertEqual([r.url for r in results], ["sq"])


class TestIsVideoHost(unittest.TestCase):
    """_is_video_host 判断 URL 是否来自已知可下载视频站点。"""

    def test_known_hosts_accepted(self):
        self.assertTrue(ws._is_video_host("https://www.youtube.com/watch?v=abc"))
        self.assertTrue(ws._is_video_host("https://youtu.be/abc"))
        self.assertTrue(ws._is_video_host("https://www.tiktok.com/@user/video/123"))
        self.assertTrue(ws._is_video_host("https://www.instagram.com/reel/abc"))
        self.assertTrue(ws._is_video_host("https://vimeo.com/123456"))
        self.assertTrue(ws._is_video_host("https://dailymotion.com/video/abc"))
        self.assertTrue(ws._is_video_host("https://fb.watch/abc"))
        self.assertTrue(ws._is_video_host("https://x.com/user/status/123"))

    def test_unknown_host_rejected(self):
        self.assertFalse(ws._is_video_host("https://example.com/video.mp4"))
        self.assertFalse(ws._is_video_host("https://someblog.com/posts/1"))
        self.assertFalse(ws._is_video_host(""))


class TestSearchVideosViaHtmlSearch(unittest.TestCase):
    """_search_videos_via_html_search 的 DuckDuckGo HTML 兜底搜索。"""

    def _mock_response(self, html: str):
        """返回一个模拟 requests.Response。"""
        mock = MagicMock()
        mock.text = html
        mock.raise_for_status = MagicMock()
        return mock

    def test_returns_empty_when_request_fails(self):
        with patch("requests.get", side_effect=Exception("net error")):
            result = ws._search_videos_via_html_search("test", "portrait", 3, "youtube")
        self.assertEqual(result, [])

    def test_parses_duckduckgo_redirect_links(self):
        """DuckDuckGo 结果的 uddg 重定向 URL 应被正确解析。"""
        html = """<html><body>
        <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3Dabc12345678&amp;rut=abc">title</a>
        <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.tiktok.com%2F%40user%2Fvideo%2F456&amp;rut=def">tiktok</a>
        </body></html>"""
        mock_resp = self._mock_response(html)
        with patch("requests.get", return_value=mock_resp):
            result = ws._search_videos_via_html_search("test", "portrait", 3, "youtube")
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].url, "https://www.youtube.com/watch?v=abc12345678")
        self.assertEqual(result[1].url, "https://www.tiktok.com/@user/video/456")

    def test_filters_non_video_hosts(self):
        """非白名单站点（如 blog）应被过滤。"""
        html = """<html><body>
        <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.blog.com%2Fpost&amp;rut=a">blog</a>
        </body></html>"""
        mock_resp = self._mock_response(html)
        with patch("requests.get", return_value=mock_resp):
            result = ws._search_videos_via_html_search("test", "portrait", 3, "youtube")
        self.assertEqual(result, [])

    def test_platform_tiktok_adds_site_filter_with_watermark_removal(self):
        """platform='tiktok' 且开启水印移除时请求参数应包含 site:tiktok.com。"""
        html = """<html><body>
        <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.tiktok.com%2F%40user%2Fvideo%2F1&amp;rut=a">tiktok</a>
        </body></html>"""
        mock_resp = self._mock_response(html)
        with (
            patch(
                "requests.get", return_value=mock_resp
            ) as mock_get,
            patch.dict(
                "app.config.config.app",
                {"enable_watermark_removal": True},
                clear=False,
            ),
        ):
            result = ws._search_videos_via_html_search("dance", "portrait", 3, "tiktok")
        self.assertEqual(len(result), 1)
        call_kwargs = mock_get.call_args.kwargs
        self.assertIn("site:tiktok.com", call_kwargs["params"]["q"])

    def test_tiktok_platform_without_watermark_removal_no_site_filter(self):
        """未开启水印移除时，platform='tiktok' 也不定向 site:tiktok.com。"""
        html = """<html><body>
        <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.tiktok.com%2F%40user%2Fvideo%2F1&amp;rut=a">tiktok</a>
        </body></html>"""
        mock_resp = self._mock_response(html)
        with (
            patch("requests.get", return_value=mock_resp) as mock_get,
            patch.dict(
                "app.config.config.app",
                {"enable_watermark_removal": False},
                clear=False,
            ),
        ):
            ws._search_videos_via_html_search("dance", "portrait", 3, "tiktok")
        call_kwargs = mock_get.call_args.kwargs
        self.assertNotIn("site:tiktok.com", call_kwargs["params"]["q"])

    def test_deduplicates_urls(self):
        """重复 URL 应被去重。"""
        html = """<html><body>
        <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3Dabc12345678&amp;rut=1">a</a>
        <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3Dabc12345678&amp;rut=2">a dup</a>
        </body></html>"""
        mock_resp = self._mock_response(html)
        with patch("requests.get", return_value=mock_resp):
            result = ws._search_videos_via_html_search("test", "portrait", 3, "youtube")
        self.assertEqual(len(result), 1)


class TestHtmlSearchIntegration(unittest.TestCase):
    """search_videos_web_scrape 的多源补充：yt-dlp 无结果时触发 HTML 搜索。"""

    def test_html_fallback_fills_empty_results(self):
        """yt-dlp 无结果且 platform 非 youtube 时，HTML 搜索补充结果。"""
        # yt-dlp 返回空 stdout
        class _EmptyPopen:
            def __init__(self, *args, **kwargs):
                pass

            def communicate(self, timeout=None):
                return ("", "")

            def poll(self):
                return None

        # HTML 搜索返回一个候选
        html_candidate = ws.MaterialInfo(
            provider="web_scrape",
            url="https://www.tiktok.com/@user/video/123",
            duration=3,
            source_info={"provider": "web_scrape", "search_term": "dance", "source": "html_video_search"},
        )

        with (
            patch.object(ws.subprocess, "Popen", return_value=_EmptyPopen()),
            patch.object(ws, "_search_videos_via_html_search", return_value=[html_candidate]),
            patch.object(ws, "_rank_web_results_by_relevance", side_effect=lambda r, *_a, **_kw: r),
        ):
            results = ws.search_videos_web_scrape("dance", 5, ws.VideoAspect.portrait)

        self.assertIn("https://www.tiktok.com/@user/video/123", [r.url for r in results])


class TestIsDirectVideoUrl(unittest.TestCase):
    """_is_direct_video_url 过滤 tag/主页/搜索页，只保留可直接下载的视频。"""

    def test_tiktok_direct_videos_accepted(self):
        self.assertTrue(ws._is_direct_video_url("https://www.tiktok.com/@user/video/7567888805643013384"))
        self.assertTrue(ws._is_direct_video_url("https://www.tiktok.com/@user/photo/123456"))

    def test_tiktok_listing_pages_rejected(self):
        self.assertFalse(ws._is_direct_video_url("https://www.tiktok.com/tag/dancetutorial/"))
        self.assertFalse(ws._is_direct_video_url("https://www.tiktok.com/@user"))
        self.assertFalse(ws._is_direct_video_url("https://www.tiktok.com/explore"))
        self.assertFalse(ws._is_direct_video_url("https://www.tiktok.com/search?q=dance"))

    def test_youtube_direct_accepted(self):
        self.assertTrue(ws._is_direct_video_url("https://www.youtube.com/watch?v=J---aiyznGQ"))
        self.assertTrue(ws._is_direct_video_url("https://youtu.be/J---aiyznGQ"))
        self.assertTrue(ws._is_direct_video_url("https://www.youtube.com/shorts/J---aiyznGQ"))

    def test_youtube_listing_rejected(self):
        self.assertFalse(ws._is_direct_video_url("https://www.youtube.com/@channel"))
        self.assertFalse(ws._is_direct_video_url("https://www.youtube.com/playlist?list=abc"))
        self.assertFalse(ws._is_direct_video_url("https://www.youtube.com/user/foo"))

    def test_instagram_direct_accepted(self):
        self.assertTrue(ws._is_direct_video_url("https://www.instagram.com/reel/abc123/"))
        self.assertTrue(ws._is_direct_video_url("https://www.instagram.com/p/xyz789"))

    def test_instagram_listing_rejected(self):
        self.assertFalse(ws._is_direct_video_url("https://www.instagram.com/explore/"))
        self.assertFalse(ws._is_direct_video_url("https://www.instagram.com/@user"))

    def test_vimeo_dailymotion(self):
        self.assertTrue(ws._is_direct_video_url("https://vimeo.com/123456"))
        self.assertTrue(ws._is_direct_video_url("https://www.dailymotion.com/video/x7abc"))
        self.assertFalse(ws._is_direct_video_url("https://vimeo.com/channels/abc"))

    def test_x_status_accepted(self):
        self.assertTrue(ws._is_direct_video_url("https://x.com/user/status/123456"))

    def test_html_search_filters_listing_pages(self):
        """HTML 搜索候选里混入的 tag/主页页应被过滤，只留直接视频。"""
        html = """<html><body>
        <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.tiktok.com%2Ftag%2Fdancetutorial%2F&amp;rut=1">tag page</a>
        <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.tiktok.com%2F%40user%2Fvideo%2F123&amp;rut=2">video</a>
        </body></html>"""
        mock_resp = MagicMock()
        mock_resp.text = html
        mock_resp.raise_for_status = MagicMock()
        with patch("requests.get", return_value=mock_resp):
            result = ws._search_videos_via_html_search("dance", "portrait", 3, "tiktok")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].url, "https://www.tiktok.com/@user/video/123")


if __name__ == "__main__":
    unittest.main()
    unittest.main()
