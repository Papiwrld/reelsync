import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts import manage_secrets  # noqa: E402


class TestManageSecretsCli(unittest.TestCase):
    def test_get_found_prints_value(self):
        with patch.object(manage_secrets, "has_keyring", return_value=True):
            with patch.object(manage_secrets, "get_secret", return_value="found"):
                self.assertEqual(manage_secrets.main(["get", "OPENAI_API_KEY"]), 0)

    def test_get_not_found_returns_error(self):
        with patch.object(manage_secrets, "has_keyring", return_value=True):
            with patch.object(manage_secrets, "get_secret", return_value=None):
                self.assertEqual(manage_secrets.main(["get", "OPENAI_API_KEY"]), 1)

    def test_set_stores_secret(self):
        with patch.object(manage_secrets, "has_keyring", return_value=True):
            with patch("getpass.getpass", return_value="secret-value") as prompt:
                with patch.object(manage_secrets, "set_secret", return_value=True) as set_pw:
                    self.assertEqual(manage_secrets.main(["set", "OPENAI_API_KEY"]), 0)
        prompt.assert_called_once_with("OPENAI_API_KEY: ")
        set_pw.assert_called_once_with("OPENAI_API_KEY", "secret-value")

    def test_set_empty_value_returns_error(self):
        with patch.object(manage_secrets, "has_keyring", return_value=True):
            with patch("getpass.getpass", return_value=""):
                self.assertEqual(manage_secrets.main(["set", "OPENAI_API_KEY"]), 1)

    def test_set_storage_failure_returns_error(self):
        with patch.object(manage_secrets, "has_keyring", return_value=True):
            with patch("getpass.getpass", return_value="secret-value"):
                with patch.object(manage_secrets, "set_secret", return_value=False):
                    self.assertEqual(manage_secrets.main(["set", "OPENAI_API_KEY"]), 1)

    def test_delete_always_succeeds(self):
        with patch.object(manage_secrets, "has_keyring", return_value=True):
            with patch.object(manage_secrets, "delete_secret") as del_pw:
                self.assertEqual(manage_secrets.main(["delete", "OPENAI_API_KEY"]), 0)
        del_pw.assert_called_once_with("OPENAI_API_KEY")

    def test_list_prints_known_secrets(self):
        with patch.object(manage_secrets, "has_keyring", return_value=True):
            self.assertEqual(manage_secrets.main(["list"]), 0)

    def test_no_keyring_returns_error(self):
        with patch.object(manage_secrets, "has_keyring", return_value=False):
            self.assertEqual(manage_secrets.main(["list"]), 1)


if __name__ == "__main__":
    unittest.main()
