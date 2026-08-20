"""Cross-platform secret storage backed by the OS credential manager.

On Windows this uses Credential Manager (DPAPI-encrypted, per-user). On Linux
and macOS it falls back to the platform keyring (libsecret / Keychain). If no
keyring backend is available (e.g. headless CI), values silently fall back to
environment variables, preserving current behavior.
"""

from __future__ import annotations

import os
from typing import Final

from loguru import logger

try:
    import keyring as _keyring
    from keyring.errors import KeyringError
except Exception:  # pragma: no cover - import failure on exotic platforms
    _keyring = None
    KeyringError = Exception

SERVICE_NAME: Final = "reelsync"


def _keyring_available() -> bool:
    if _keyring is None:
        return False
    try:
        return not _keyring.get_keyring().__class__.__name__.startswith("Fail")
    except Exception:
        return False


def get_secret(name: str) -> str | None:
    """Return the secret stored under ``name`` or None.

    Precedence: environment variable, then OS credential manager. This follows
    the 12-factor convention where a process-local env var can override a
    stored credential for a single run; the keyring is the persistent store.
    """
    env_value = os.getenv(name)
    if env_value:
        return env_value
    if _keyring_available():
        try:
            value = _keyring.get_password(SERVICE_NAME, name)
            if value:
                return value
        except (KeyringError, Exception):
            logger.debug(f"keyring lookup failed for {name}")
    return None


def set_secret(name: str, value: str) -> bool:
    """Store ``value`` in the OS credential manager. Returns False if unavailable."""
    if _keyring_available():
        try:
            _keyring.set_password(SERVICE_NAME, name, value)
            return True
        except (KeyringError, Exception):
            logger.warning(f"failed to store secret {name} in the system keyring")
    return False


def delete_secret(name: str) -> bool:
    """Remove a stored secret. Returns True if the keyring handled it."""
    if _keyring_available():
        try:
            _keyring.delete_password(SERVICE_NAME, name)
            return True
        except _keyring.errors.PasswordDeleteError:
            return True
        except (KeyringError, Exception):
            logger.warning(f"failed to delete secret {name} from the system keyring")
    return False


def has_keyring() -> bool:
    """True if a real OS credential manager backend is active."""
    return _keyring_available()
