import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.models.schema import VideoParams
from app.services import cache_manager, material
from app.services import task as tm
from app.services.subtitle_engine.timing import WordTiming


class TestComputeSceneDurations(unittest.TestCase):
    """_compute_scene_durations_from_word_timings 的按场景时长推导。"""

    def _timings(self):
        return [
            WordTiming(text="hello", start=0.0, end=0.5),
            WordTiming(text="world", start=0.5, end=1.0),
            WordTiming(text="coffee", start=1.0, end=1.8),
            WordTiming(text="morning", start=1.8, end=2.6),
            WordTiming(text="routine", start=2.6, end=3.4),
        ]

    def test_empty_inputs_return_none(self):
        """空词时间轴或空场景列表都不能参与计算，返回 None 让上层回退。"""
        self.assertIsNone(tm._compute_scene_durations_from_word_timings([], ["a"], 5))
        self.assertIsNone(tm._compute_scene_durations_from_word_timings(self._timings(), [], 5))
        self.assertIsNone(tm._compute_scene_durations_from_word_timings([], [], 5))

    def test_two_scenes_proportional_to_token_count(self):
        """
        简单场景：2 个场景各 2/3 个词，返回 2 段正时长，
        总和归一化到完整音频时长，词更多的场景分得更长。
        """
        durations = tm._compute_scene_durations_from_word_timings(
            self._timings(),
            ["hello world", "coffee morning routine"],
            6.8,
        )
        self.assertEqual(len(durations), 2)
        self.assertTrue(all(d > 0 for d in durations))
        self.assertAlmostEqual(sum(durations), 6.8, places=5)
        self.assertGreater(durations[1], durations[0])
        self.assertAlmostEqual(durations[0], 2.0, places=5)
        self.assertAlmostEqual(durations[1], 4.8, places=5)

    def test_token_word_mismatch_still_returns_proportional_list(self):
        """
        词数与 token 数偏差超过 3 倍（CJK/英文混排）时不回退 None，
        仍按比例分配，保证素材时长不会因此丢失。
        """
        narrations = [
            "one two three four five six",
            "seven eight nine ten eleven twelve",
        ]
        timings = [
            WordTiming(text="one", start=0.0, end=0.6),
            WordTiming(text="two", start=0.6, end=1.2),
        ]
        durations = tm._compute_scene_durations_from_word_timings(
            timings, narrations, 12.0
        )
        self.assertIsNotNone(durations)
        self.assertEqual(len(durations), 2)
        self.assertTrue(all(d > 0 for d in durations))
        self.assertAlmostEqual(sum(durations), 12.0, places=5)

    def test_missing_or_zero_audio_duration_skips_normalization(self):
        """
        audio_duration 缺失或为 0 时不触发归一化（避免除零）也不崩溃：
        None 允许返回 None 让上层回退，0 返回原始时间跨度。
        """
        self.assertIsNone(
            tm._compute_scene_durations_from_word_timings(
                self._timings(),
                ["hello world", "coffee morning routine"],
                None,
            )
        )
        for audio_duration in (0, 0.0):
            with self.subTest(audio_duration=audio_duration):
                durations = tm._compute_scene_durations_from_word_timings(
                    self._timings(),
                    ["hello world", "coffee morning routine"],
                    audio_duration,
                )
                self.assertIsNotNone(durations)
                self.assertEqual(len(durations), 2)
                self.assertTrue(all(d > 0 for d in durations))
                self.assertAlmostEqual(sum(durations), 3.4, places=5)


class TestMaybeSetSceneDurations(unittest.TestCase):
    """_maybe_set_scene_durations 把词时间轴推导结果写入 params。"""

    def _timings(self):
        return [
            WordTiming(text="hello", start=0.0, end=0.5),
            WordTiming(text="world", start=0.5, end=1.0),
            WordTiming(text="coffee", start=1.0, end=1.8),
            WordTiming(text="morning", start=1.8, end=2.6),
            WordTiming(text="routine", start=2.6, end=3.4),
        ]

    def test_sets_params_and_persists_when_sidecar_available(self):
        """sidecar 词时间轴可用时写入 params.scene_durations 并持久化。"""
        params = VideoParams(
            video_subject="test",
            scene_search_terms=[["city"], ["coffee"]],
            scene_narrations=["hello world", "coffee morning routine"],
        )
        with (
            patch(
                "app.services.subtitle_engine.timing.load_word_timings_from_json",
                return_value=self._timings(),
            ),
            patch.object(
                tm.task_artifacts, "patch_script_data", return_value=True
            ) as patch_script,
        ):
            tm._maybe_set_scene_durations(
                "task-id", params, 6.8, "sub.srt", MagicMock()
            )

        self.assertEqual(len(params.scene_durations), 2)
        self.assertAlmostEqual(params.scene_durations[0], 2.0, places=6)
        self.assertAlmostEqual(params.scene_durations[1], 4.8, places=6)
        patch_script.assert_called_once()
        self.assertEqual(patch_script.call_args.args[0], "task-id")
        saved = patch_script.call_args.kwargs["scene_durations"]
        self.assertEqual(len(saved), 2)
        self.assertAlmostEqual(saved[0], 2.0, places=6)
        self.assertAlmostEqual(saved[1], 4.8, places=6)

    def test_sidecar_load_failure_leaves_params_unchanged(self):
        """sidecar 读取失败时静默跳过，不能崩溃也不能写脏数据。"""
        params = VideoParams(
            video_subject="test",
            scene_search_terms=[["city"], ["coffee"]],
            scene_narrations=["hello world", "coffee morning routine"],
        )
        with (
            patch(
                "app.services.subtitle_engine.timing.load_word_timings_from_json",
                side_effect=OSError("sidecar missing"),
            ),
            patch.object(tm.task_artifacts, "patch_script_data") as patch_script,
        ):
            tm._maybe_set_scene_durations(
                "task-id", params, 6.8, "sub.srt", None
            )

        self.assertIsNone(params.scene_durations)
        patch_script.assert_not_called()

    def test_skipped_without_scene_groups(self):
        """没有场景分组时直接返回，不尝试加载任何词时间轴。"""
        params = VideoParams(video_subject="test")
        with patch(
            "app.services.subtitle_engine.timing.load_word_timings_from_json"
        ) as load:
            tm._maybe_set_scene_durations(
                "task-id", params, 6.8, "sub.srt", None
            )
        load.assert_not_called()
        self.assertIsNone(params.scene_durations)


class TestDownloadVideosSceneDurationsPlumbing(unittest.TestCase):
    """download_videos 把 scene_durations 透传给分组下载。"""

    def test_forwards_scene_durations_to_grouped_download(self):
        with patch.object(
            material,
            "_download_videos_grouped",
            return_value=["s0.mp4", "s1.mp4"],
        ) as grouped:
            result = material.download_videos(
                task_id="scene-durations-plumbing",
                search_terms=["city", "coffee"],
                grouped_search_terms=[["city"], ["coffee"]],
                scene_narrations=["hello world", "coffee morning routine"],
                scene_durations=[2.0, 4.8],
                audio_duration=6.8,
                max_clip_duration=3,
            )

        self.assertEqual(result, ["s0.mp4", "s1.mp4"])
        self.assertEqual(grouped.call_args.kwargs["scene_durations"], [2.0, 4.8])
        self.assertEqual(
            grouped.call_args.kwargs["grouped_search_terms"], [["city"], ["coffee"]]
        )
        self.assertEqual(
            grouped.call_args.kwargs["scene_narrations"],
            ["hello world", "coffee morning routine"],
        )


class TestDownloadVideosGroupedSceneTargets(unittest.TestCase):
    """_download_videos_grouped 的每场景素材分配与时长来源。"""

    def _item(self, index):
        return material.MaterialInfo(
            provider="pexels",
            url=f"https://v.example/{index}.mp4",
            duration=3,
            source_info={"provider": "pexels", "asset_id": str(index)},
        )

    def _run_grouped(self, scene_durations=None, scene_narrations=None):
        captured = {}

        def fake_download(
            tasks,
            material_directory,
            max_clip_duration,
            audio_duration,
            video_paths,
            material_sources,
        ):
            captured["tasks"] = tasks
            for item, _term in tasks:
                video_paths.append(f"/tmp/{item.source_info['asset_id']}.mp4")
            return 0.0

        def fake_search(search_term, minimum_duration, video_aspect, failed_providers=None):
            if search_term == "city":
                return [self._item(i) for i in range(4)]
            return [self._item(i) for i in range(4, 8)]

        with (
            patch.object(
                material,
                "_download_with_concurrent_prefix",
                side_effect=fake_download,
            ),
            patch.object(material, "_persist_material_sources"),
            patch("app.services.twelvelabs.is_enabled", return_value=False),
        ):
            result = material._download_videos_grouped(
                task_id="grouped-math",
                grouped_search_terms=[["city"], ["coffee"]],
                scene_narrations=scene_narrations,
                search_videos=fake_search,
                video_aspect=material.VideoAspect.portrait,
                audio_duration=10,
                max_clip_duration=3,
                material_directory="",
                scene_durations=scene_durations,
            )
        return result, captured["tasks"]

    def test_word_timing_scene_durations_drive_allocation(self):
        """
        scene_durations=[2, 8] 时按词时间轴比例分配：
        场景 0 目标 2s 只保留 1 段，场景 1 目标 8s 保留 3 段。
        """
        result, tasks = self._run_grouped(
            scene_durations=[2.0, 8.0], scene_narrations=None
        )
        urls = [item.url for item, _ in tasks]
        self.assertEqual(len(tasks), 8)
        self.assertEqual(
            urls[:4],
            [
                "https://v.example/0.mp4",
                "https://v.example/4.mp4",
                "https://v.example/5.mp4",
                "https://v.example/6.mp4",
            ],
        )
        self.assertEqual(
            result,
            [
                "/tmp/0.mp4",
                "/tmp/4.mp4",
                "/tmp/5.mp4",
                "/tmp/6.mp4",
                "/tmp/1.mp4",
                "/tmp/2.mp4",
                "/tmp/3.mp4",
                "/tmp/7.mp4",
            ],
        )

    def test_char_ratio_fallback_when_no_scene_durations(self):
        """
        没有 scene_durations 时回退字符比例：文案 1:1 → 每场景目标 5s，
        各保留 2 段素材，与词时间轴分配明显不同。
        """
        result, tasks = self._run_grouped(
            scene_durations=None, scene_narrations=["ab", "cd"]
        )
        urls = [item.url for item, _ in tasks]
        self.assertEqual(len(tasks), 8)
        self.assertEqual(
            urls[:4],
            [
                "https://v.example/0.mp4",
                "https://v.example/1.mp4",
                "https://v.example/4.mp4",
                "https://v.example/5.mp4",
            ],
        )
        self.assertEqual(len(result), 8)


class TestGetSynonymTerms(unittest.TestCase):
    """_get_synonym_terms 的 LLM 响应解析与去重。"""

    def test_parses_json_array_from_llm(self):
        with patch(
            "app.services.llm._generate_response",
            return_value='["neon street", "rainy window"]',
        ) as generate:
            terms = material._get_synonym_terms("city night")

        self.assertEqual(terms, ["neon street", "rainy window"])
        generate.assert_called_once()

    def test_returns_empty_on_error_response(self):
        with patch(
            "app.services.llm._generate_response",
            return_value="Error: invalid api key",
        ):
            self.assertEqual(material._get_synonym_terms("city night"), [])

    def test_strips_code_fences(self):
        with patch(
            "app.services.llm._generate_response",
            return_value='```json\n["neon street", "rainy window"]\n```',
        ):
            self.assertEqual(
                material._get_synonym_terms("city night"),
                ["neon street", "rainy window"],
            )

    def test_dedupes_and_skips_original_term(self):
        with patch(
            "app.services.llm._generate_response",
            return_value='["city night", "neon street", "neon street"]',
        ):
            self.assertEqual(material._get_synonym_terms("city night"), ["neon street"])


class TestSynonymRetryInDownload(unittest.TestCase):
    """download_videos 首轮搜索为空时用同义词重试一次。"""

    def test_retries_once_with_synonym_when_first_search_is_empty(self):
        item = material.MaterialInfo(
            provider="pexels",
            url="https://v.example/syn.mp4",
            duration=5,
            source_info={"provider": "pexels", "asset_id": "syn"},
        )
        searched = []

        def fake_cached_search(
            provider, search_videos, search_term, minimum_duration, video_aspect
        ):
            searched.append(search_term)
            if len(searched) == 1:
                return []
            return [item]

        with (
            patch.object(
                material,
                "_search_videos_with_cache",
                side_effect=fake_cached_search,
            ),
            patch.object(material, "_get_synonym_terms", return_value=["alt term"]),
            patch.object(
                material, "save_video", return_value="/tmp/syn.mp4"
            ) as save,
            patch.object(material, "_enforce_video_cache_limit_quietly"),
            patch.object(material, "_persist_material_sources"),
        ):
            result = material.download_videos(
                task_id="synonym-retry",
                search_terms=["city night"],
                source="pexels",
                audio_duration=5,
                max_clip_duration=5,
            )

        self.assertEqual(searched, ["city night", "alt term"])
        self.assertEqual(result, ["/tmp/syn.mp4"])
        save.assert_called_once()
        self.assertEqual(save.call_args.kwargs["video_url"], "https://v.example/syn.mp4")


class TestEnforceCacheLimit(unittest.TestCase):
    """enforce_cache_limit 的 LRU 清理与无操作保护。"""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.cache_dir = Path(self.temp_dir.name)
        self.storage_patch = patch.object(
            cache_manager.utils,
            "storage_dir",
            return_value=str(self.cache_dir),
        )
        self.storage_patch.start()

    def tearDown(self):
        self.storage_patch.stop()
        self.temp_dir.cleanup()

    def _create_cache_file(self, digest: str, size: int, mtime: float) -> Path:
        path = self.cache_dir / f"vid-{digest}.mp4"
        path.write_bytes(b"x" * size)
        os.utime(path, (mtime, mtime))
        return path

    def test_deletes_oldest_until_under_file_limit(self):
        now = 2_000_000_000.0
        for i in range(6):
            self._create_cache_file(f"{i:032x}", 100, now - (6 - i) * 3600)

        result = cache_manager.enforce_cache_limit(max_size_gb=10, max_files=4)

        self.assertEqual(result.deleted_count, 2)
        self.assertEqual(result.deleted_size, 200)
        self.assertEqual(result.failed_count, 0)
        remaining = sorted(p.name for p in self.cache_dir.iterdir())
        self.assertEqual(
            remaining, [f"vid-{i:032x}.mp4" for i in range(2, 6)]
        )

    def test_deletes_oldest_until_under_size_limit(self):
        now = 2_000_000_000.0
        for i in range(6):
            self._create_cache_file(f"{i:032x}", 1024 * 1024, now - (6 - i) * 3600)
        max_size_gb = (5 * 1024**2) / (1024**3)

        result = cache_manager.enforce_cache_limit(
            max_size_gb=max_size_gb, max_files=500
        )

        self.assertEqual(result.deleted_count, 1)
        self.assertEqual(result.deleted_size, 1024 * 1024)
        self.assertFalse((self.cache_dir / f"vid-{0:032x}.mp4").exists())
        self.assertTrue((self.cache_dir / f"vid-{1:032x}.mp4").exists())

    def test_noop_when_already_under_limits(self):
        now = 2_000_000_000.0
        for i in range(3):
            self._create_cache_file(f"{i:032x}", 100, now - (3 - i) * 3600)

        result = cache_manager.enforce_cache_limit(max_size_gb=10, max_files=500)

        self.assertEqual(result.deleted_count, 0)
        self.assertEqual(len(list(self.cache_dir.iterdir())), 3)

    def test_zero_or_falsy_limits_are_noop(self):
        now = 2_000_000_000.0
        for i in range(4):
            self._create_cache_file(f"{i:032x}", 100, now - (4 - i) * 3600)

        for kwargs in ({"max_size_gb": 0}, {"max_files": 0}, {"max_size_gb": 0.0}):
            with self.subTest(kwargs=kwargs):
                result = cache_manager.enforce_cache_limit(**kwargs)
                self.assertEqual(result.deleted_count, 0)

        self.assertEqual(len(list(self.cache_dir.iterdir())), 4)


if __name__ == "__main__":
    unittest.main()