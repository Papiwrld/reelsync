import unittest
from unittest.mock import patch

from app.utils import secrets


class TestSecrets(unittest.TestCase):
    def test_get_secret_env_var_takes_precedence_over_keyring(self):
        with (
            patch.object(secrets, "_keyring_available", return_value=True),
            patch.object(secrets._keyring, "get_password", return_value="keyring-value") as get_pw,
            patch.dict("os.environ", {"MPT_TEST_SECRET": "env-value"}),
        ):
            self.assertEqual(secrets.get_secret("MPT_TEST_SECRET"), "env-value")
        get_pw.assert_not_called()

    def test_get_secret_falls_back_to_keyring_when_no_env_var(self):
        with (
            patch.object(secrets, "_keyring_available", return_value=True),
            patch.object(secrets._keyring, "get_password", return_value="keyring-value") as get_pw,
            patch.dict("os.environ", {}, clear=False),
        ):
            self.assertEqual(secrets.get_secret("MPT_TEST_SECRET"), "keyring-value")
        get_pw.assert_called_once_with("reelsync", "MPT_TEST_SECRET")

    def test_get_secret_returns_none_when_keyring_unavailable(self):
        with (
            patch.object(secrets, "_keyring_available", return_value=False),
            patch.dict("os.environ", {}, clear=False),
        ):
            self.assertIsNone(secrets.get_secret("MPT_TEST_SECRET"))

    def test_get_secret_ignores_keyring_errors(self):
        with (
            patch.object(secrets, "_keyring_available", return_value=True),
            patch.object(
                secrets._keyring,
                "get_password",
                side_effect=RuntimeError("no backend"),
            ),
            patch.dict("os.environ", {}, clear=False),
        ):
            self.assertIsNone(secrets.get_secret("MPT_TEST_SECRET"))

    def test_set_secret_stores_in_keyring(self):
        with (
            patch.object(secrets, "_keyring_available", return_value=True),
            patch.object(secrets._keyring, "set_password") as set_pw,
        ):
            self.assertTrue(secrets.set_secret("MPT_TEST_SECRET", "abc"))
        set_pw.assert_called_once_with("reelsync", "MPT_TEST_SECRET", "abc")

    def test_set_secret_returns_false_when_keyring_unavailable(self):
        with patch.object(secrets, "_keyring_available", return_value=False):
            self.assertFalse(secrets.set_secret("MPT_TEST_SECRET", "abc"))


if __name__ == "__main__":
    unittest.main()
