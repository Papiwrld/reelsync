"""Webhook 与 MCP：任务终态回调签名、负载与 MCP tools/list 契约。"""

import hashlib
import hmac
import threading
import unittest
from unittest.mock import patch

from app.services import webhooks


class TestWebhookPayload(unittest.TestCase):
    def test_completed_event_for_completed_state(self):
        payload = webhooks._build_payload(
            {"task_id": "abc", "state": "1", "progress": 100, "videos": ["/tmp/v.mp4"]}
        )
        self.assertEqual(payload["event"], "task.completed")
        self.assertEqual(payload["task_id"], "abc")
        self.assertEqual(payload["videos"], ["/tmp/v.mp4"])

    def test_failed_event_for_failed_state(self):
        payload = webhooks._build_payload(
            {"task_id": "abc", "state": "-1", "failed_stage": "script", "error": "boom"}
        )
        self.assertEqual(payload["event"], "task.failed")
        self.assertEqual(payload["failed_stage"], "script")
        self.assertEqual(payload["error"], "boom")


class TestWebhookSigning(unittest.TestCase):
    def test_signature_matches_hmac_sha256(self):
        body = b'{"a": 1}'
        secret = "s3cret"
        sig = webhooks._sign_body(body, secret)
        expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
        self.assertEqual(sig, expected)


class TestWebhookDelivery(unittest.TestCase):
    def test_is_configured_false_by_default(self):
        self.assertFalse(webhooks.is_webhook_configured({}))

    def _mock_urlopen(self, status_code):
        def _factory():
            class _Response:
                status = status_code
            class _Ctx:
                def __enter__(self):
                    return _Response()
                def __exit__(self, *a):
                    return False
            return _Ctx()
        return patch(
            "app.services.webhooks.urllib.request.urlopen",
            side_effect=lambda *a, **k: _factory(),
        )

    def test_deliver_success(self):
        with self._mock_urlopen(200):
            self.assertTrue(webhooks._deliver_once("https://x/h", "s", {"a": 1}))

    def test_deliver_non_2xx_returns_false(self):
        with self._mock_urlopen(500):
            self.assertFalse(webhooks._deliver_once("https://x/h", "s", {"a": 1}))

    def test_deliver_transport_error_returns_false(self):
        with patch(
            "app.services.webhooks.urllib.request.urlopen",
            side_effect=OSError("conn refused"),
        ):
            self.assertFalse(webhooks._deliver_once("https://x/h", "s", {"a": 1}))


class TestWebhookNotify(unittest.TestCase):
    def test_notify_skips_when_not_configured(self):
        with (
            patch.object(webhooks, "_webhook_config", return_value=("", "")),
            patch.object(webhooks, "threading") as thr,
        ):
            webhooks.notify_task_terminal("t1", {})
            thr.Thread.assert_not_called()

    def test_notify_spawns_delivery_thread(self):
        from app.services import state as sm

        with (
            patch.object(
                sm.state, "get_task", return_value={"task_id": "t1", "state": "1"}
            ),
            patch.object(sm.state, "update_task") as update_task,
            patch.object(
                webhooks, "_webhook_config", return_value=("https://x/h", "s")
            ),
            patch.object(threading, "Thread") as thread_mock,
        ):
            webhooks.notify_task_terminal("t1", {})
            thread_mock.assert_called_once()
            thread_mock.return_value.start.assert_called_once()
            update_task.assert_not_called()

    def test_notify_does_not_record_when_no_url(self):
        """没有配置 webhook_url 时不写任务状态（避免污染每个任务）。"""
        from app.services import state as sm

        with (
            patch.object(
                sm.state,
                "get_task",
                return_value={
                    "task_id": "t1",
                    "state": 1,
                    "progress": 100,
                    "warnings": [],
                },
            ),
            patch.object(sm.state, "update_task") as update_task,
            patch.object(webhooks, "_webhook_config", return_value=("", "")),
        ):
            webhooks.notify_task_terminal("t1", {})

        update_task.assert_not_called()

    def test_send_with_retries_records_delivered(self):
        """投递成功后应把 delivered 写回任务状态。"""
        from app.services import state as sm

        with (
            patch.object(
                sm.state,
                "get_task",
                return_value={
                    "task_id": "t1",
                    "state": 1,
                    "progress": 100,
                    "warnings": [],
                },
            ),
            patch.object(sm.state, "update_task") as update_task,
            patch.object(webhooks, "_deliver_once", return_value=True),
        ):
            self.assertTrue(
                webhooks._send_with_retries("t1", "https://x/h", "s", {"a": 1})
            )

        update_task.assert_called_once()
        self.assertEqual(
            update_task.call_args.kwargs["webhook_state"], "delivered"
        )
        self.assertEqual(
            update_task.call_args.kwargs["warnings"], ["webhook delivered"]
        )

    def test_send_with_retries_records_failed(self):
        """多次重试都失败后应把 failed 写回任务状态。"""
        from app.services import state as sm

        with (
            patch.object(
                sm.state,
                "get_task",
                return_value={
                    "task_id": "t1",
                    "state": -1,
                    "progress": 100,
                    "warnings": [],
                },
            ),
            patch.object(sm.state, "update_task") as update_task,
            patch.object(webhooks, "_deliver_once", return_value=False),
            patch.object(webhooks.time, "sleep"),
        ):
            self.assertFalse(
                webhooks._send_with_retries("t1", "https://x/h", "s", {"a": 1})
            )

        update_task.assert_called_once()
        self.assertEqual(update_task.call_args.kwargs["webhook_state"], "failed")
        self.assertIn(
            "webhook delivery failed", update_task.call_args.kwargs["warnings"][0]
        )


class TestMcpToolsContract(unittest.TestCase):
    """MCP 暴露的 tools 契约。"""

    def test_tools_are_exposed(self):
        from app.mcp_server import _MCP

        names = set()
        for tool in _MCP._tool_manager.list_tools():
            names.add(tool.name)
        self.assertEqual(
            names,
            {"create_video", "get_task_status", "list_tasks", "generate_script"},
        )


if __name__ == "__main__":
    unittest.main()
