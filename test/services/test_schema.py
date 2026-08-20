import sys
import unittest
from pathlib import Path

from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.models.schema import (
    ASPECT_RESOLUTIONS,
    VideoAspect,
    VideoParams,
    validate_video_aspect,
    video_aspect_from_string,
)


class TestVideoAspect(unittest.TestCase):
    def test_to_resolution_known_aspects(self):
        self.assertEqual(VideoAspect.landscape.to_resolution(), (1920, 1080))
        self.assertEqual(VideoAspect.portrait.to_resolution(), (1080, 1920))
        self.assertEqual(VideoAspect.square.to_resolution(), (1080, 1080))

    def test_to_resolution_rejects_unsupported_value(self):
        with self.assertRaises(ValueError):
            VideoAspect.to_resolution("4:5")

    def test_aspect_resolutions_map_covers_all_members(self):
        self.assertEqual(set(ASPECT_RESOLUTIONS), set(VideoAspect))

    def test_video_aspect_from_string_parses_ratios(self):
        self.assertEqual(video_aspect_from_string("16:9"), VideoAspect.landscape)
        self.assertEqual(video_aspect_from_string("9:16"), VideoAspect.portrait)
        self.assertEqual(video_aspect_from_string("1:1"), VideoAspect.square)

    def test_video_aspect_from_string_parses_aliases_case_insensitively(self):
        self.assertEqual(video_aspect_from_string("LANDSCAPE"), VideoAspect.landscape)
        self.assertEqual(video_aspect_from_string(" 1080x1920 "), VideoAspect.portrait)
        self.assertEqual(video_aspect_from_string("Square"), VideoAspect.square)

    def test_video_aspect_from_string_accepts_enum_member(self):
        self.assertEqual(
            video_aspect_from_string(VideoAspect.portrait), VideoAspect.portrait
        )

    def test_video_aspect_from_string_rejects_invalid(self):
        for invalid in ("4:5", "21:9", "", "portrait-mode", None, 123):
            with self.subTest(value=invalid):
                with self.assertRaises(ValueError):
                    video_aspect_from_string(invalid)

    def test_validate_video_aspect(self):
        self.assertTrue(validate_video_aspect("16:9"))
        self.assertTrue(validate_video_aspect("1080x1920"))
        self.assertFalse(validate_video_aspect("4:5"))
        self.assertFalse(validate_video_aspect(None))


class TestVideoParams(unittest.TestCase):
    def test_rejects_non_positive_generation_counts(self):
        for field_name in ("video_count",):
            for value in (0, -1, None):
                with self.subTest(field_name=field_name, value=value):
                    with self.assertRaises(ValidationError):
                        VideoParams(video_subject="Coffee", **{field_name: value})

    def test_rejects_negative_clip_duration(self):
        with self.assertRaises(ValidationError):
            VideoParams(video_subject="Coffee", video_clip_duration=-1)

    def test_accepts_zero_clip_duration_as_auto_mode(self):
        """video_clip_duration=0 表示自动推导素材时长，应被接受。"""
        params = VideoParams(video_subject="Coffee", video_clip_duration=0)
        self.assertEqual(params.video_clip_duration, 0)

    def test_accepts_positive_generation_counts(self):
        params = VideoParams(
            video_subject="Coffee", video_clip_duration=1, video_count=1
        )

        self.assertEqual(params.video_clip_duration, 1)
        self.assertEqual(params.video_count, 1)


if __name__ == "__main__":
    unittest.main()
