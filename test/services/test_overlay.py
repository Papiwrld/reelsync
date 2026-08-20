import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.models.schema import VideoParams
from app.services import overlay as ov


class TestOverlayDetection(unittest.TestCase):
    def test_detect_fact_percentage(self):
        self.assertTrue(ov.detect_fact("全行业增长 23.5%。"))
        self.assertTrue(ov.detect_fact("同比增长了 40 percent。"))

    def test_detect_fact_currency(self):
        self.assertTrue(ov.detect_fact("公司融资 $12 million。"))
        self.assertTrue(ov.detect_fact("成本降至 ¥3500。"))

    def test_detect_fact_year_and_large_number(self):
        self.assertTrue(ov.detect_fact("这场运动始于 2019 年。"))
        self.assertTrue(ov.detect_fact("累计吸引了 1,500,000 名观众。"))

    def test_detect_fact_ratio_and_citation(self):
        self.assertTrue(ov.detect_fact("销量增长 3 倍。"))
        self.assertTrue(ov.detect_fact("according to the latest report。"))

    def test_detect_fact_plain_sentence_is_not_fact(self):
        self.assertFalse(ov.detect_fact("春天来了，万物复苏。"))
        self.assertFalse(ov.detect_fact(""))

    def test_detect_callout_short_hook(self):
        self.assertTrue(ov.detect_callout("But here's the thing."))
        self.assertTrue(ov.detect_callout("This is the key point."))

    def test_detect_callout_too_long_or_factual(self):
        self.assertFalse(
            ov.detect_callout("But this is a very long sentence that keeps going and going and going.")
        )
        self.assertFalse(ov.detect_callout("But 23.5% of users churn."))
        self.assertFalse(ov.detect_callout(""))


class TestParseSubtitlePhrases(unittest.TestCase):
    def test_parse_srt_lines(self):
        lines = [
            [1, "00:00:00,100 --> 00:00:00,917", "It is hard to"],
            [2, "00:00:00,917 --> 00:00:01,735", "talk about the things"],
        ]
        phrases = ov.parse_subtitle_phrases(lines)
        self.assertEqual(len(phrases), 2)
        self.assertAlmostEqual(phrases[0][0], 0.1)
        self.assertAlmostEqual(phrases[0][1], 0.917)
        self.assertEqual(phrases[0][2], "It is hard to")

    def test_parse_skips_malformed(self):
        lines = [
            [1, "broken --> no timestamp", "bad"],
            [2, "00:00:00,100 --> 00:00:00,917", "good"],
            ["only two"],
        ]
        phrases = ov.parse_subtitle_phrases(lines)
        self.assertEqual(len(phrases), 1)
        self.assertEqual(phrases[0][2], "good")

    def test_parse_empty(self):
        self.assertEqual(ov.parse_subtitle_phrases([]), [])


class TestBuildOverlayPlan(unittest.TestCase):
    def _params(self, **overrides):
        base = dict(
            video_subject="都市通勤",
            overlay_enabled=True,
            overlay_style="title_fact",
            overlay_title_card=True,
            overlay_fact_cards=True,
            overlay_callouts=False,
        )
        base.update(overrides)
        return VideoParams(**base)

    def _phrases(self):
        return [
            (0.0, 1.5, "每天有 300 万人乘坐地铁。"),
            (1.5, 3.0, "这座城市的公共交通网络已经运营了 2019 年。"),
            (3.0, 4.5, "But the real challenge lies ahead."),
            (4.5, 6.0, "春天来了，万物复苏。"),
        ]

    def test_disabled_returns_empty(self):
        params = VideoParams(video_subject="主题", overlay_enabled=False)
        self.assertEqual(ov.build_overlay_plan(params, "主题", "", []), [])

    def test_title_card_first(self):
        params = self._params()
        items = ov.build_overlay_plan(params, "都市通勤", "", self._phrases())
        self.assertEqual(items[0].kind, "title")
        self.assertEqual(items[0].text, "都市通勤")
        self.assertEqual(items[0].start, 0.0)
        self.assertLessEqual(items[0].end, ov.TITLE_MAX_DURATION)

    def test_fact_cards_time_aligned(self):
        params = self._params()
        items = ov.build_overlay_plan(params, "都市通勤", "", self._phrases())
        facts = [i for i in items if i.kind == "fact"]
        self.assertEqual(len(facts), 2)
        self.assertEqual(facts[0].position, ov.FACT_POSITION)
        self.assertAlmostEqual(facts[0].start, 0.0)
        self.assertAlmostEqual(facts[0].end, 1.5)

    def test_callout_not_included_by_default(self):
        params = self._params()
        items = ov.build_overlay_plan(params, "都市通勤", "", self._phrases())
        callouts = [i for i in items if i.kind == "callout"]
        self.assertEqual(callouts, [])

    def test_full_style_includes_callouts(self):
        params = self._params(overlay_style="full")
        items = ov.build_overlay_plan(params, "都市通勤", "", self._phrases())
        kinds = [i.kind for i in items]
        self.assertIn("callout", kinds)
        callout = [i for i in items if i.kind == "callout"][0]
        self.assertEqual(callout.text, "But the real challenge lies ahead.")

    def test_title_only_style(self):
        params = self._params(overlay_style="title_only")
        items = ov.build_overlay_plan(params, "都市通勤", "", self._phrases())
        self.assertEqual([i.kind for i in items], ["title"])

    def test_facts_only_style(self):
        params = self._params(overlay_style="facts_only")
        items = ov.build_overlay_plan(params, "都市通勤", "", self._phrases())
        kinds = [i.kind for i in items]
        self.assertNotIn("title", kinds)
        self.assertIn("fact", kinds)

    def test_no_phrases_still_emits_title(self):
        params = self._params()
        items = ov.build_overlay_plan(params, "都市通勤", "", [], video_duration=10.0)
        self.assertEqual([i.kind for i in items], ["title"])
        self.assertEqual(items[0].text, "都市通勤")


if __name__ == "__main__":
    unittest.main()