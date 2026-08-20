import unittest

from app.config import config
from app.services import media_utils


class TestSafePublicUrl(unittest.TestCase):
    def test_strips_query_and_fragment(self):
        self.assertEqual(
            media_utils.safe_public_url(
                "https://example.com/page?token=SECRET#frag"
            ),
            "https://example.com/page",
        )

    def test_keeps_public_https_url(self):
        self.assertEqual(
            media_utils.safe_public_url("https://example.com/photo.png"),
            "https://example.com/photo.png",
        )

    def test_rejects_non_http_and_credentials(self):
        self.assertIsNone(media_utils.safe_public_url("ftp://example.com/a"))
        self.assertIsNone(
            media_utils.safe_public_url("https://user:pass@example.com/a")
        )

    def test_rejects_blank_or_non_string(self):
        self.assertIsNone(media_utils.safe_public_url(""))
        self.assertIsNone(media_utils.safe_public_url(None))
        self.assertIsNone(media_utils.safe_public_url(123))


class TestTlsVerify(unittest.TestCase):
    def setUp(self):
        self.original_app_config = dict(config.app)

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)

    def test_defaults_to_true(self):
        config.app.pop("tls_verify", None)
        self.assertTrue(media_utils.get_tls_verify())

    def test_string_false_variants_disable(self):
        for raw in ("0", "false", "no", "off", "False"):
            with self.subTest(raw=raw):
                config.app["tls_verify"] = raw
                self.assertFalse(media_utils.get_tls_verify())

    def test_true_variants_enable(self):
        config.app["tls_verify"] = "true"
        self.assertTrue(media_utils.get_tls_verify())


class TestRedactSecret(unittest.TestCase):
    def test_redacts_raw_and_urlencoded_secret(self):
        self.assertEqual(
            media_utils.redact_secret("key=my secret", "my secret"),
            "key=***",
        )
        self.assertEqual(
            media_utils.redact_secret("key=my+secret", "my secret"),
            "key=***",
        )

    def test_empty_secret_is_noop(self):
        self.assertEqual(media_utils.redact_secret("plain", ""), "plain")

    def test_redact_request_error_removes_secrets(self):
        error = RuntimeError("failed https://x/?api_key=TOKEN")
        self.assertNotIn("TOKEN", media_utils.redact_request_error(error, "TOKEN"))


if __name__ == "__main__":
    unittest.main()