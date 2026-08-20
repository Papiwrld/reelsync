"""Tests for social profile inference (Content Intelligence convenience)."""

import unittest
from unittest.mock import patch

from app.services import social_profile
from app.services.social_profile import (
    SocialProfileInference,
    detect_platform,
    extract_handle,
    infer_content_context,
)


class TestDetectPlatform(unittest.TestCase):
    def test_known_platforms(self):
        cases = [
            ("https://www.tiktok.com/@user", ("tiktok", "TikTok")),
            ("https://www.instagram.com/user/", ("instagram_reels", "Instagram")),
            ("https://www.facebook.com/zuck", ("", "Facebook")),
            ("https://x.com/elonmusk", ("x", "X / Twitter")),
            ("https://twitter.com/user", ("x", "X / Twitter")),
            ("https://www.youtube.com/@channel", ("youtube", "YouTube")),
            ("https://www.bilibili.com/video/BV1", ("bilibili", "Bilibili")),
        ]
        for url, expected in cases:
            with self.subTest(url=url):
                self.assertEqual(detect_platform(url), expected)

    def test_unknown_or_invalid(self):
        self.assertEqual(detect_platform(""), ("", ""))
        self.assertEqual(detect_platform("https://example.com/blog"), ("", ""))
        self.assertEqual(detect_platform("not a url"), ("", ""))


class TestExtractHandle(unittest.TestCase):
    def test_handles(self):
        cases = [
            ("https://www.tiktok.com/@techreviews", "techreviews"),
            ("https://www.instagram.com/fitgirl/", "fitgirl"),
            ("https://x.com/elonmusk", "elonmusk"),
            ("https://www.youtube.com/@codingtutorials", "codingtutorials"),
            ("https://www.youtube.com/channel/UCabc123", "ucabc123"),
            ("https://www.facebook.com/zuck", "zuck"),
        ]
        for url, expected in cases:
            with self.subTest(url=url):
                self.assertEqual(extract_handle(url), expected)

    def test_video_links_do_not_produce_handles(self):
        self.assertEqual(extract_handle("https://www.tiktok.com/"), "")
        self.assertEqual(extract_handle("https://www.youtube.com/watch?v=abc"), "")
        self.assertEqual(extract_handle("https://www.instagram.com/reel/xyz/"), "")


class TestInferContentContext(unittest.TestCase):
    def test_invalid_url(self):
        result = infer_content_context("not-a-url")
        self.assertIsInstance(result, SocialProfileInference)
        self.assertIn("not a valid", result.note)
        self.assertEqual(result.niche, "")

    def test_unsupported_platform(self):
        result = infer_content_context("https://example.com/blog")
        self.assertEqual(result.platform_name, "")
        self.assertIn("unsupported platform", result.note)

    def test_model_knowledge_fallback_without_provider(self):
        with (
            patch.object(
                social_profile.WebSearchResearchProvider,
                "is_configured",
                return_value=False,
            ),
            patch.object(
                social_profile,
                "_llm_json",
                return_value={
                    "niche": "tech reviews",
                    "sub_niche": "AI gadgets",
                    "audience": "young adults into AI",
                    "tone": "energetic, educational",
                    "summary": "Appears to review consumer tech.",
                },
            ),
        ):
            result = infer_content_context("https://www.tiktok.com/@techreviews")
        self.assertEqual(result.platform, "tiktok")
        self.assertEqual(result.handle, "techreviews")
        self.assertEqual(result.niche, "tech reviews")
        self.assertEqual(result.sub_niche, "AI gadgets")
        self.assertEqual(result.audience, "young adults into AI")
        self.assertEqual(result.tone, "energetic, educational")
        self.assertFalse(result.used_external_info)
        self.assertEqual(result.provenance, "model_knowledge")
        self.assertIn("model knowledge only", result.note)

    def test_web_search_provider_used_when_configured(self):
        snippets = [
            {
                "title": "techreviews TikTok",
                "url": "https://www.tiktok.com/@techreviews",
                "tier": "social_media",
                "is_primary": False,
                "note": "Gadget reviews and unboxings.",
                "provenance": "web_search",
            }
        ]
        with (
            patch.object(
                social_profile.WebSearchResearchProvider,
                "is_configured",
                return_value=True,
            ),
            patch.object(
                social_profile.WebSearchResearchProvider,
                "discover",
                return_value=snippets,
            ),
            patch.object(
                social_profile,
                "_llm_json",
                return_value={
                    "niche": "tech reviews",
                    "sub_niche": "",
                    "audience": "gadget fans",
                    "tone": "casual",
                    "summary": "Reviews gadgets and unboxings.",
                },
            ),
        ):
            result = infer_content_context("https://www.tiktok.com/@techreviews")
        self.assertTrue(result.used_external_info)
        self.assertEqual(result.provenance, "web_search")
        self.assertIn("1 public search snippet", result.note)
        self.assertEqual(result.niche, "tech reviews")

    def test_llm_failure_degrades_to_empty_suggestion(self):
        with (
            patch.object(
                social_profile.WebSearchResearchProvider,
                "is_configured",
                return_value=False,
            ),
            patch.object(
                social_profile, "_llm_json", side_effect=RuntimeError("llm down")
            ),
        ):
            result = infer_content_context("https://www.instagram.com/fitgirl")
        self.assertEqual(result.platform, "instagram_reels")
        self.assertEqual(result.niche, "")
        self.assertIn("model knowledge only", result.note)

    def test_page_metadata_used_for_youtube_when_scraping_enabled(self):
        with (
            patch.object(
                social_profile.WebSearchResearchProvider,
                "is_configured",
                return_value=False,
            ),
            patch.object(social_profile.shutil, "which", return_value="yt-dlp"),
            patch.object(
                social_profile.web_scrape,
                "fetch_page_metadata",
                return_value={
                    "title": "Tech Channel",
                    "description": "Gadget reviews and unboxings",
                    "channel": "Tech",
                    "tags": ["tech", "reviews"],
                    "webpage_url": "https://www.youtube.com/@techchannel",
                },
            ),
            patch.object(
                social_profile,
                "_llm_json",
                return_value={
                    "niche": "tech reviews",
                    "sub_niche": "gadgets",
                    "audience": "gadget fans",
                    "tone": "casual",
                    "summary": "Reviews gadgets and unboxings.",
                },
            ),
        ):
            result = infer_content_context(
                "https://www.youtube.com/@techchannel",
                app_config={"enable_web_scraping": True},
            )
        self.assertTrue(result.used_external_info)
        self.assertEqual(result.provenance, "page_metadata")
        self.assertIn("public page metadata (yt-dlp)", result.note)
        self.assertEqual(result.niche, "tech reviews")

    def test_page_metadata_skipped_when_scraping_disabled(self):
        with (
            patch.object(
                social_profile.WebSearchResearchProvider,
                "is_configured",
                return_value=False,
            ),
            patch.object(
                social_profile.web_scrape,
                "fetch_page_metadata",
                side_effect=AssertionError("must not fetch when scraping is off"),
            ),
            patch.object(
                social_profile,
                "_llm_json",
                return_value={"niche": "", "sub_niche": "", "audience": "", "tone": "", "summary": ""},
            ),
        ):
            result = infer_content_context(
                "https://www.youtube.com/@techchannel",
                app_config={"enable_web_scraping": False},
            )
        self.assertFalse(result.used_external_info)
        self.assertEqual(result.provenance, "model_knowledge")
        self.assertIn("Web Video Scraping", result.note)

    def test_page_metadata_not_attempted_for_tiktok(self):
        with (
            patch.object(
                social_profile.WebSearchResearchProvider,
                "is_configured",
                return_value=False,
            ),
            patch.object(
                social_profile.web_scrape,
                "fetch_page_metadata",
                side_effect=AssertionError("tiktok is not a metadata host"),
            ),
            patch.object(
                social_profile,
                "_llm_json",
                return_value={"niche": "", "sub_niche": "", "audience": "", "tone": "", "summary": ""},
            ),
        ):
            result = infer_content_context(
                "https://www.tiktok.com/@techreviews",
                app_config={"enable_web_scraping": True},
            )
        self.assertFalse(result.used_external_info)
        self.assertEqual(result.provenance, "model_knowledge")

    def test_fetch_failure_explains_network_block(self):
        """yt-dlp 运行了但超时/失败时，提示应说明原因与解决方向（代理/搜索源）。"""
        with (
            patch.object(
                social_profile.WebSearchResearchProvider,
                "is_configured",
                return_value=False,
            ),
            patch.object(social_profile.shutil, "which", return_value="yt-dlp"),
            patch.object(
                social_profile.web_scrape,
                "fetch_page_metadata",
                return_value={},
            ),
            patch.object(
                social_profile,
                "_llm_json",
                return_value={"niche": "", "sub_niche": "", "audience": "", "tone": "", "summary": ""},
            ),
        ):
            result = infer_content_context(
                "https://www.youtube.com/@techchannel",
                app_config={"enable_web_scraping": True},
            )
        self.assertFalse(result.used_external_info)
        self.assertIn("could not fetch the public page", result.note)
        self.assertIn("proxy", result.note)
        self.assertIn("model knowledge only", result.note)

    def test_proxy_passed_to_ytdlp_fetch(self):
        """配置了 [proxy] 时，代理地址应转发给 yt-dlp。"""
        captured = {}

        def fake_fetch(url, proxy=""):
            captured["proxy"] = proxy
            return {"title": "Tech Channel", "description": "Gadget reviews", "channel": "Tech"}

        with (
            patch.object(
                social_profile.WebSearchResearchProvider,
                "is_configured",
                return_value=False,
            ),
            patch.object(social_profile.shutil, "which", return_value="yt-dlp"),
            patch.object(social_profile.config, "proxy", {"https": "http://127.0.0.1:1080"}),
            patch.object(
                social_profile.web_scrape, "fetch_page_metadata", side_effect=fake_fetch
            ),
            patch.object(
                social_profile,
                "_llm_json",
                return_value={"niche": "tech", "sub_niche": "", "audience": "", "tone": "", "summary": ""},
            ),
        ):
            infer_content_context(
                "https://www.youtube.com/@techchannel",
                app_config={"enable_web_scraping": True},
            )
        self.assertEqual(captured.get("proxy"), "http://127.0.0.1:1080")

    def test_prompt_features_bio_as_primary_signal(self):
        """简介/About 描述必须作为首要信号出现在提示词中，且排在搜索片段之前。"""
        captured = {}

        def fake_llm_json(prompt, fallback, app_config=None, tracker=None, agent=""):
            captured["prompt"] = prompt
            return {
                "niche": "film analysis",
                "sub_niche": "",
                "audience": "",
                "tone": "",
                "summary": "",
            }

        with (
            patch.object(
                social_profile.WebSearchResearchProvider,
                "is_configured",
                return_value=False,
            ),
            patch.object(social_profile.shutil, "which", return_value="yt-dlp"),
            patch.object(
                social_profile.web_scrape,
                "fetch_page_metadata",
                return_value={
                    "title": "CineTalk",
                    "description": "Film analysis and cinematic video essays",
                    "channel": "CineTalk",
                    "channel_follower_count": 250000,
                    "categories": ["Film & Animation"],
                    "webpage_url": "https://www.youtube.com/@cinetalk",
                },
            ),
            patch.object(social_profile, "_llm_json", side_effect=fake_llm_json),
        ):
            infer_content_context(
                "https://www.youtube.com/@cinetalk",
                app_config={"enable_web_scraping": True},
            )

        prompt = captured["prompt"]
        self.assertIn("PRIMARY signal", prompt)
        self.assertIn("Channel/About description (bio)", prompt)
        self.assertIn("Film analysis and cinematic video essays", prompt)
        self.assertIn("Public follower count: 250000", prompt)
        self.assertIn("Film & Animation", prompt)
        # 简介必须排在搜索片段之前（bio 优先于 snippet）
        self.assertLess(
            prompt.index("Channel/About description (bio)"),
            prompt.index("search snippets"),
        )
        # 推理规则明确按信任度排序，bio 是第一优先级
        self.assertLess(
            prompt.index("PRIMARILY on the channel/about"),
            prompt.index("Honesty rules"),
        )


if __name__ == "__main__":
    unittest.main()
