"""
Shared utilities for media provider services.
Extracted here to avoid circular imports between material.py and custom_media.py.
"""

from typing import Any
from urllib.parse import quote_plus, urlsplit, urlunsplit

from loguru import logger

from app.config import config


def safe_public_url(value: Any) -> str | None:
    """
    Only keep publicly displayable HTTP(S) page addresses, strip query params
    and credentials. Mirrors material._safe_public_url.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = urlsplit(value.strip())
    except ValueError:
        return None
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        return None
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def get_tls_verify() -> bool:
    tls_verify = config.app.get("tls_verify", True)
    if isinstance(tls_verify, str):
        tls_verify = tls_verify.strip().lower() not in ("0", "false", "no", "off")
    if not tls_verify:
        logger.warning(
            "TLS certificate verification is disabled by config.app.tls_verify=false. "
            "Only use this in trusted proxy environments."
        )
    return bool(tls_verify)


def redact_secret(message: str, secret: str) -> str:
    safe_message = str(message)
    if not secret:
        return safe_message
    safe_message = safe_message.replace(secret, "***")
    encoded_secret = quote_plus(secret)
    if encoded_secret != secret:
        safe_message = safe_message.replace(encoded_secret, "***")
    return safe_message


def redact_request_error(error: Exception, *secrets: str) -> str:
    safe_message = str(error)
    for secret in secrets:
        safe_message = redact_secret(safe_message, str(secret or ""))
    for proxy_url in config.proxy.values():
        safe_message = redact_secret(safe_message, str(proxy_url))
    return safe_message
