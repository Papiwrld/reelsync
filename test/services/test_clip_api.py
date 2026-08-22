import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.controllers.v1 import video as video_controller
from app.models.exception import HttpException
from app.models.schema import ClipRequest
from app.services import clip_generator
from app.utils import utils


class TestClipGeneratorPipeline(unittest.TestCase):
    def test_generate_clips_returns_empty_for_missing_source(self):
        """源文件不存在时应返回空列表而不是崩溃。"""
        self.assertEqual(clip_generator.generate_clips("/nonexistent/video.mp4"), [])

    def test_heuristic_moments_produces_count_moments(self):
        """启发式选段应产出恰好 count 个且 end > start 的时刻。"""
        moments = clip_generator._heuristic_moments(
            [10.0, 30.0, 50.0], count=3
        )
        self.assertEqual(len(moments), 3)
        for moment in moments:
            self.assertGreater(moment["end"], moment["start"])

    def test_select_moments_parses_llm_json(self):
        """LLM 返回合法 JSON 数组时应解析并过滤出有效时刻。"""
        transcript = [{"msg": "hello", "start_time": 0.0, "end_time": 2.0}]
        scenes = [10.0, 20.0]
        raw = (
            '[{"start": 1, "end": 20, "title": "t", "reason": "r"},'
            ' {"start": 25, "end": 40, "title": "t2", "reason": "r2"}]'
        )
        with patch(
            "app.services.llm._generate_response", return_value=raw
        ):
            moments = clip_generator.select_moments(transcript, scenes, count=2)
        self.assertEqual(len(moments), 2)
        for moment in moments:
            self.assertGreater(moment["end"], moment["start"])

    def test_select_moments_falls_back_to_heuristic_on_error(self):
        """LLM 返回 Error: 前缀时应回退到启发式选段。"""
        transcript = [{"msg": "x", "start_time": 0.0, "end_time": 1.0}]
        with patch(
            "app.services.llm._generate_response", return_value="Error: boom"
        ):
            moments = clip_generator.select_moments(transcript, [], count=3)
        self.assertEqual(len(moments), 3)
        for moment in moments:
            self.assertGreater(moment["end"], moment["start"])

    def test_detect_scenes_returns_empty_on_subprocess_failure(self):
        """子进程（ffmpeg）抛异常时场景检测应返回空列表。"""
        with patch(
            "app.services.clip_generator.subprocess.run",
            side_effect=OSError("ffmpeg missing"),
        ):
            times = clip_generator.detect_scenes("whatever.mp4")
        self.assertEqual(times, [])

    def test_extract_clip_success_and_failure(self):
        """returncode=0 且输出文件存在时返回 True，否则返回 False。"""
        proc_ok = SimpleNamespace(returncode=0, stderr="")
        proc_fail = SimpleNamespace(returncode=1, stderr="boom")
        with tempfile.TemporaryDirectory() as temp_dir:
            out_ok = os.path.join(temp_dir, "ok.mp4")
            out_fail = os.path.join(temp_dir, "fail.mp4")
            Path(out_ok).write_bytes(b"x" * 2048)
            with patch(
                "app.services.clip_generator.subprocess.run",
                return_value=proc_ok,
            ):
                self.assertTrue(
                    clip_generator.extract_clip("src.mp4", 1.0, 10.0, out_ok)
                )
            with patch(
                "app.services.clip_generator.subprocess.run",
                return_value=proc_fail,
            ):
                self.assertFalse(
                    clip_generator.extract_clip("src.mp4", 1.0, 10.0, out_fail)
                )


class TestClipApiEndpoint(unittest.TestCase):
    @staticmethod
    def _request():
        return SimpleNamespace(headers={"x-task-id": "request-123"})

    def test_create_clips_rejects_missing_local_source(self):
        """本地源文件不存在时应返回 400，且不排队任何任务（不能是 500）。"""
        missing = os.path.join(utils.storage_dir("local_videos"), "missing.mp4")
        body = ClipRequest(source_video=missing)

        with (
            patch.object(
                video_controller.utils, "get_uuid", return_value="task-clip-1"
            ),
            patch.object(video_controller.sm.state, "update_task"),
            patch.object(video_controller.task_manager, "add_task") as add_task,
        ):
            with self.assertRaises(HttpException) as raised:
                video_controller.create_clips(self._request(), body)

        self.assertEqual(raised.exception.status_code, 400)
        add_task.assert_not_called()

    def test_create_clips_queues_task_for_local_file(self):
        """受信任目录内的本地文件应入队 clip 任务并返回 task_id。"""
        local_file = os.path.join(
            utils.storage_dir("local_videos"), "clipgen-source.mp4"
        )
        os.makedirs(os.path.dirname(local_file), exist_ok=True)
        Path(local_file).write_bytes(b"video")
        body = ClipRequest(source_video=local_file)
        request = SimpleNamespace(headers={"x-task-id": "request-123"})

        try:
            with (
                patch.object(
                    video_controller.utils, "get_uuid", return_value="task-clip-2"
                ),
                patch.object(
                    video_controller.sm.state, "update_task"
                ) as update_task,
                patch.object(
                    video_controller.task_manager, "add_task"
                ) as add_task,
            ):
                response = video_controller.create_clips(request, body)
        finally:
            os.remove(local_file)

        self.assertEqual(response["status"], 200)
        self.assertEqual(response["data"]["task_id"], "task-clip-2")
        self.assertEqual(response["data"]["request_id"], "request-123")
        self.assertEqual(
            response["data"]["params"]["source_video"],
            os.path.realpath(local_file),
        )
        update_task.assert_called_once_with("task-clip-2", stop_at="clips")
        add_task.assert_called_once_with(
            video_controller.clip_generator.start_clip_generation,
            task_id="task-clip-2",
            params=response["data"]["params"],
        )


class TestClipTaskWebhook(unittest.TestCase):
    def test_start_clip_generation_fires_webhook_on_success(self):
        """生成成功时应写入完成状态并触发 terminal webhook。"""
        clips = [
            {
                "path": "/tmp/clip-01.mp4",
                "start": 1.0,
                "end": 20.0,
                "title": "t",
                "reason": "r",
            }
        ]
        params = {"source_video": "/tmp/src.mp4", "count": 1}

        with (
            patch(
                "app.services.clip_generator.generate_clips", return_value=clips
            ),
            patch("app.services.state.state.update_task") as update_task,
            patch(
                "app.services.webhooks.notify_task_terminal"
            ) as notify,
        ):
            result = clip_generator.start_clip_generation("task-clip-3", params)

        self.assertEqual(result["state"], 1)
        self.assertEqual(result["progress"], 100)
        self.assertEqual(result["clips"], clips)
        update_task.assert_called_once_with(
            "task-clip-3",
            state=1,
            progress=100,
            clips=clips,
        )
        notify.assert_called_once_with("task-clip-3")


if __name__ == "__main__":
    unittest.main()
